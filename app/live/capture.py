"""Holds the RTSP connection open and always hands back the newest frame.

WHAT: one background thread per camera, reading frames as fast as they arrive.
WHY:  OpenCV buffers frames internally, so reading on demand gives you the
      OLDEST buffered frame. A consumer slower than the camera then drifts
      further behind real time on every read - which is precisely the latency a
      live view exists to avoid.
HOW:  the thread reads continuously and keeps only the latest frame, dropping
      the rest. A consumer that falls behind therefore skips frames and stays
      current, instead of falling behind and staying behind.

Dropping frames is the right trade here: for a live view, a fresh frame beats a
complete sequence. Nothing downstream needs every frame - the detectors are
paced to a fixed rate anyway (see TARGET_FPS in __main__.py).
"""
import logging
import threading

import cv2

log = logging.getLogger(__name__)

# How long to wait before trying the camera again after it drops out. Long
# enough not to hammer a camera that is rebooting, short enough that a brief
# network blip recovers on its own without anyone noticing.
RECONNECT_DELAY_SECONDS = 3.0

# Consecutive failed reads before the connection is treated as dead. A handful
# of failures is normal on a lossy stream; dozens in a row means it is gone.
READ_FAIL_LIMIT = 30


class Capture:
    """A camera connection that always has the latest frame ready.

    Start it, call latest() whenever you want a frame, stop it when done. It
    reconnects by itself for as long as it is running, so a camera reboot or a
    dropped network shows up as a gap in the view rather than a crash."""

    def __init__(self, url: str, name: str = "cam"):
        self.url = url
        self.name = name

        self._frame = None
        self._seq = 0                 # bumped per frame, so a consumer can tell
                                      # a new frame from the one it already has
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"capture-{self.name}"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def latest(self):
        """The newest frame and its sequence number, as (frame, seq).

        frame is None until the first one arrives. The caller compares seq
        against the last one it handled to avoid processing a frame twice."""
        with self._lock:
            return self._frame, self._seq

    def _open(self):
        """Opens the stream over TCP, with the smallest buffer OpenCV allows."""
        # TCP rather than the UDP default: a partial frame from a dropped UDP
        # packet shows up as visible corruption, and the pose model then reads
        # keypoints off garbage pixels.
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        # Best-effort - not every backend honours it. Where it works, it stops
        # OpenCV holding a queue of stale frames ahead of the live one.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return capture

    def _run(self) -> None:
        """Connect, read until the stream dies, reconnect. Forever, until stopped."""
        while not self._stop.is_set():
            capture = self._open()

            if not capture.isOpened():
                capture.release()
                log.warning(
                    "[%s] cannot open the stream - retrying in %.0fs. Check the "
                    "URL, the username and that the camera is on this network.",
                    self.name, RECONNECT_DELAY_SECONDS,
                )
                self._stop.wait(RECONNECT_DELAY_SECONDS)
                continue

            log.info("[%s] connected", self.name)
            self.connected = True
            failures = 0

            while not self._stop.is_set():
                ok, frame = capture.read()

                if not ok:
                    failures += 1
                    if failures >= READ_FAIL_LIMIT:
                        log.warning("[%s] stream stopped responding - reconnecting", self.name)
                        break
                    continue

                failures = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1

            self.connected = False
            capture.release()

            if not self._stop.is_set():
                self._stop.wait(RECONNECT_DELAY_SECONDS)

        log.info("[%s] capture stopped", self.name)
