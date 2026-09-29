"""Serves the annotated view to a browser, as MJPEG.

WHAT: a tiny HTTP server with two routes - a page to look at, and the stream
      it embeds.
WHY:  the analysis runs in a container, which has no screen, so cv2.imshow is
      not an option. MJPEG needs no plugin, no JavaScript and no WebSocket: an
      <img> tag pointed at a multipart response is enough, and it works from
      another machine on the network without anything installed.
HOW:  multipart/x-mixed-replace - the browser keeps the connection open and
      replaces the image every time a new part arrives.

MJPEG is not efficient - every frame is a complete JPEG, with no compression
between frames. That is the right trade here: it is a handful of viewers on a
LAN, and the alternative costs a video pipeline and a player to save bandwidth
nobody is short of.
"""
import errno
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

log = logging.getLogger(__name__)

BOUNDARY = "frameboundary"
JPEG_QUALITY = 80

# Cap on how often frames are pushed to a browser. The engine runs slower than
# this anyway; the limit just stops a fast client spinning on repeats of a frame
# it already has.
MAX_STREAM_FPS = 20

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live view</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; background: #111; color: #ddd;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    padding: 16px; box-sizing: border-box;
  }}
  h1 {{ font-size: 15px; font-weight: 600; margin: 0; color: #999; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; background: #000; }}
  p {{ margin: 0; color: #777; font-size: 13px; text-align: center; }}
  code {{ color: #9c9; }}
</style>
</head>
<body>
  <h1>{name} &middot; live</h1>
  <img src="/stream.mjpg" alt="live camera view">
  <p>A picture is saved to <code>evidence/live_view/</code> whenever something is detected. No video is recorded.</p>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Two routes and nothing else. `frame_source` and `camera_name` are set on
    the class by serve() below, because BaseHTTPRequestHandler instantiates per
    request and gives us no way to pass anything in."""

    frame_source = None
    camera_name = "camera"

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_page()
        elif self.path == "/stream.mjpg":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_page(self) -> None:
        body = PAGE.format(name=self.camera_name).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        minimum_gap = 1.0 / MAX_STREAM_FPS
        last_sent = 0.0

        try:
            while True:
                frame = self.frame_source()
                if frame is None:
                    time.sleep(0.05)
                    continue

                gap = time.monotonic() - last_sent
                if gap < minimum_gap:
                    time.sleep(minimum_gap - gap)
                last_sent = time.monotonic()

                ok, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ok:
                    continue

                data = buffer.tobytes()
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            pass        # the tab was closed - entirely normal, not worth logging

    def handle_one_request(self):
        """Treats a client vanishing mid-request as normal, because it is.

        Closing the tab, reloading, or a phone going to sleep all reset the
        connection while the base handler is blocked reading it, which it
        reports as an unhandled exception with a full traceback. Left alone,
        routine browsing fills the log with alarming-looking noise."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    def log_message(self, format, *args):
        """Silences the default per-request line on stderr.

        An MJPEG viewer opens one long-lived request, so the access log says
        nothing useful and buries the engine's own output."""
        return


def serve(frame_source, camera_name: str, host: str = "0.0.0.0", port: int = 8001):
    """Starts the viewer in a background thread and returns the server.

    frame_source is called for each frame to send and should return the latest
    annotated frame, or None if there isn't one yet."""
    _Handler.frame_source = staticmethod(frame_source)
    _Handler.camera_name = camera_name

    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as e:
        # Almost always an earlier copy of this still running, and the stack
        # trace for it says nothing a person can act on. Say what to do instead.
        if e.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"\nPort {port} is already in use - something else is on it.\n\n"
                f"  Most likely an earlier live view is still running:\n"
                f"      pkill -f 'app.live'\n\n"
                f"  To see what has it:\n"
                f"      lsof -nP -iTCP:{port} -sTCP:LISTEN\n\n"
                f"  Or just use a different one:\n"
                f"      python -m app.live --port {port + 1}\n"
            ) from None
        raise

    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="live-server")
    thread.start()

    log.info("live view on http://localhost:%d", port)
    return server
