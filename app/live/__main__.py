"""Entry point for the live view: `python -m app.live`.

WHAT: connects to a camera, scores what it sees, and serves the annotated feed
      to a browser.
WHY:  this is the whole program; everything else in the package is a piece it
      wires together.
HOW:  four threads - one reading the camera, one doing pose and tracking, one
      scoring, one classifying fights - plus the HTTP server's own. The main
      thread only waits for a shutdown signal.

    Capture (thread) -> latest frame -> LiveEngine -> render -> MJPEG -> browser
    always newest,                        pose+track+score      overlay
    drops the rest                        + fight classifier             + snapshot

No video is ever recorded. On a detection it saves one annotated still to
evidence/live_view/. The camera is configured in camera.yaml.
"""
import argparse
import logging
import signal
import threading
import time

import cv2

from app.live import render, server
from app.live.capture import Capture
from app.live.engine import LiveEngine
from app.live.evidence import DEFAULT_DIR, Snapshotter
from app.live import config as config_module

# Frames are scaled to this width before anything looks at them. Measured on an
# 8-core CPU with no GPU, one pose pass on real camera frames costs:
#
#     640 wide    43 ms    23 fps
#     480 wide    40 ms    25 fps
#
# There is little to gain below 640 and real detail to lose, so 640 it is.
# Override with --width on a slower machine.
DEFAULT_WIDTH = 640

# Frames per second to run pose and tracking at, and the reason it is a fixed
# target rather than "as fast as possible".
#
# The detectors count history in records, not seconds: MAX_FILL_GAP_FRAMES is
# literally MAX_FILL_GAP_SECONDS * TRACK_FPS, with TRACK_FPS = 15. Feed them
# records at 8 a second and a gap the author meant as one second silently
# becomes two, so missing keypoints get invented across stretches they never
# intended. Feed them 23 a second and the same constant tightens to two thirds
# of a second instead.
#
# Either way the scores drift from what the recorded pipeline would give for the
# same footage, which defeats the point of sharing the detectors at all. So the
# loop paces itself to the rate they were written for. Measured across known
# clips, matching it took fall-1 from a wrong answer to 1.00, exactly what the
# recorded pipeline scores.
TARGET_FPS = 15.0

# 8000 and 8001 are avoided because both are common defaults for a local dev
# server - a PHP or node process already bound to 127.0.0.1:8001 wins over a
# wildcard bind on macOS, so the view would appear to start fine and then serve
# someone else's site. Override with --port.
DEFAULT_PORT = 8088

# How many CPU threads each inference may use.
#
# Three things run models here at once - the tracking loop, the pose scorer and
# the fight classifier - and PyTorch defaults to using every core for each. They
# then fight over the same cores and all get slower. Measured on 8 cores:
#
#     torch threads 8   259 ms/frame
#     torch threads 4   152 ms/frame
#     torch threads 2   140 ms/frame
#
# Fewer threads, faster result - the extra ones were pure contention. 4 leaves
# both workers room without either monopolising the machine, and keeps the host
# usable while the view is open. Override with --threads.
DEFAULT_THREADS = 4

# How often to print a line saying it is alive. Rare enough to ignore, frequent
# enough to tell "nothing is happening" from "it has quietly died".
HEARTBEAT_SECONDS = 30.0


def _resize(frame, target_width: int):
    """Scales a frame down to the working width, keeping its shape.

    Everything downstream - pose, tracking, scoring, drawing - then works in one
    coordinate space, so no result ever has to be scaled back up."""
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / width
    return cv2.resize(frame, (target_width, round(height * scale)), interpolation=cv2.INTER_AREA)


def _pick_camera(cameras: list, name: str | None):
    """The camera to watch: the one named, or the only one there is."""
    if name:
        for camera in cameras:
            if camera.name == name:
                return camera
        available = ", ".join(c.name for c in cameras)
        raise SystemExit(f"No camera called {name!r} in camera.yaml. Available: {available}")
    return cameras[0]


def main():
    parser = argparse.ArgumentParser(
        prog="python -m app.live",
        description="Live camera view with real-time detection of falls, fights, spitting and vandalism.",
    )
    parser.add_argument("--camera", help="which camera in camera.yaml (default: the first)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port for the viewer (default: {DEFAULT_PORT})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"working frame width - lower is faster (default: {DEFAULT_WIDTH})")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"CPU threads per inference (default: {DEFAULT_THREADS})")
    parser.add_argument("--fps", type=float, default=TARGET_FPS,
                        help=f"frames per second to track at (default: {TARGET_FPS})")
    parser.add_argument("--evidence-dir", default=DEFAULT_DIR,
                        help=f"where snapshots go (default: {DEFAULT_DIR})")
    parser.add_argument("--no-save", action="store_true",
                        help="watch only - write nothing at all")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("live")

    camera = _pick_camera(config_module.load(), args.camera)

    # Imported here rather than at module level: loading ultralytics takes a few
    # seconds, and an argument mistake should fail instantly with a usage
    # message instead of after the wait.
    import torch
    from ultralytics import YOLO

    # Must be set before any inference runs - see DEFAULT_THREADS.
    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))

    log.info("loading models (%d threads per inference)", args.threads)
    person_model = YOLO("models/yolo11n-pose.pt")

    engine = LiveEngine(person_model).start()

    # The latest annotated frame, written by the worker loop and read by every
    # browser connection. One slot, guarded - a viewer always gets the newest
    # frame rather than queueing up behind old ones.
    latest = {"frame": None}
    latest_lock = threading.Lock()

    def frame_source():
        with latest_lock:
            return latest["frame"]

    stop = threading.Event()

    def shutdown(signum, _frame):
        log.info("signal %s received, shutting down", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    capture = Capture(camera.url, camera.name).start()
    viewer = server.serve(frame_source, camera.name, port=args.port)

    snapshotter = None if args.no_save else Snapshotter(camera.name, args.evidence_dir)

    log.info("watching %s - open http://localhost:%d", camera.name, args.port)
    if snapshotter:
        log.info("saving a snapshot per detection to %s/", snapshotter.directory)
    else:
        log.info("--no-save: nothing will be written")

    last_seq = 0
    frame_stamps = []
    last_heartbeat = time.monotonic()
    minimum_gap = 1.0 / max(args.fps, 1.0)
    next_frame_due = time.monotonic()

    try:
        while not stop.is_set():
            # Pace to TARGET_FPS. The camera sends faster than this and the
            # machine can process faster than this - both of which would push
            # more records per second at the detectors than they are built for.
            # See TARGET_FPS above for why that matters.
            wait = next_frame_due - time.monotonic()
            if wait > 0:
                stop.wait(wait)
                continue
            next_frame_due = time.monotonic() + minimum_gap

            frame, seq = capture.latest()

            # Nothing new yet - the camera is slower than our pace, or is
            # reconnecting. Sleep briefly rather than spin.
            if frame is None or seq == last_seq:
                stop.wait(0.01)
                continue
            last_seq = seq

            started = time.monotonic()
            working = _resize(frame, args.width)
            people = engine.process(working)

            # The rate the DETECTORS actually get records at, measured between
            # frames that were really processed.
            #
            # This used to be 1 / (time one pass costs), which is the machine's
            # capacity, not the achieved rate - it ignored the pacing wait
            # entirely and so read 18.8 while the loop was deliberately capped at
            # 15. Two different numbers wearing one label, and the only one that
            # matters for accuracy - are the detectors being fed at the rate
            # their constants assume - was the one not being shown. See
            # TARGET_FPS.
            frame_stamps.append(started)
            del frame_stamps[:-31]          # keep the rolling average short and recent
            span = frame_stamps[-1] - frame_stamps[0]
            fps = (len(frame_stamps) - 1) / span if span > 0 else 0.0

            fight = engine.fight()
            annotated = render.draw(working, people, {
                "connected": capture.connected,
                "fps": fps,
                "people": len(people),
                # Say so on screen rather than leaving the reader to know that 15
                # is the number to compare against.
                "note": "SLOW - labels unreliable" if 0 < fps < args.fps * 0.85 else None,
            }, fight=fight)
            with latest_lock:
                latest["frame"] = annotated

            now = time.monotonic()

            # Taken every frame even with --no-save, so confirmed detections
            # never pile up unread. Each one carries its own best-moment frame,
            # which is what gets drawn and saved - see CONFIRM_SECONDS.
            events = engine.take_events()
            if snapshotter:
                snapshotter.update(annotated, people, now, fight=fight, events=events)
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                flagged = [p for p in people if p["action"]]
                log.info(
                    "%.1f fps, %d tracked%s%s%s",
                    fps, len(people),
                    "".join(
                        f", #{p['track_id']} {p['action']} {p['confidence'] * 100:.0f}%"
                        for p in flagged
                    ),
                    f", fight {fight * 100:.0f}%" if fight >= render.FIGHT_THRESHOLD else "",
                    f", {snapshotter.saved_count} saved" if snapshotter else "",
                )
                last_heartbeat = now

    finally:
        capture.stop()
        engine.stop()
        viewer.shutdown()
        log.info("stopped")


if __name__ == "__main__":
    main()
