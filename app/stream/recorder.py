"""Records one camera's live stream to disk as a series of fixed-length files.

WHAT: keeps an ffmpeg process running that writes spool/<camera>/*.mp4,
      a new file every `segment_seconds`.
WHY:  the detectors work on video files, not on a live stream, so the stream has
      to be chopped into files the analysis worker can pick up.
HOW:  ffmpeg's `segment` muxer does the chopping; this class just keeps ffmpeg
      alive and restarts it whenever the camera drops.

Why ffmpeg and not OpenCV: OpenCV's RTSP support drops frames silently, buffers
unpredictably, and has no clean way to reconnect. ffmpeg also copies the camera's
already-compressed video straight to disk without decoding it (`-c copy`), so
recording costs almost no CPU - which matters, because detection needs all of it.
"""
import logging
import os
import subprocess
import threading
import time

log = logging.getLogger(__name__)

# Segment filenames look like: cam-1_20260924-071830.mp4
# The timestamp means files sort chronologically by name (the worker relies on
# this) and any saved evidence can be traced back to a wall-clock moment.
# The %% are escaped because this string goes through Python's % formatting first.
FILENAME_PATTERN = "%s_%%Y%%m%%d-%%H%%M%%S.mp4"

# Reconnect backoff. Starts short so a brief network blip recovers almost
# instantly, then doubles - so a camera that is switched off overnight is retried
# once a minute instead of thousands of times.
RECONNECT_BACKOFF_START = 2.0
RECONNECT_BACKOFF_MAX = 60.0

# A connection that lasted at least this long counts as "it was working". The
# backoff resets after one, so a single glitch at 3am doesn't leave the camera
# stuck on a 60-second retry for the rest of the day.
STABLE_RUN_SECONDS = 60.0


class Recorder:
    """Keeps one camera recording, restarting ffmpeg whenever the stream drops.

    Runs its own thread, because ffmpeg has to be waited on and there is one of
    these per camera."""

    def __init__(self, camera, spool_dir: str, segment_seconds: int):
        self.camera = camera
        self.spool_dir = os.path.join(spool_dir, camera.name)
        self.segment_seconds = segment_seconds

        self._process = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"rec:{camera.name}", daemon=True
        )

    def start(self):
        os.makedirs(self.spool_dir, exist_ok=True)
        self._thread.start()

    def stop(self):
        """Stops recording and waits for ffmpeg to shut down cleanly."""
        self._stop.set()

        process = self._process
        if process and process.poll() is None:
            # terminate() (SIGTERM), not kill(): it gives ffmpeg the chance to
            # finish writing the file it is part-way through. A killed ffmpeg
            # leaves behind an unplayable, half-written mp4.
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()      # it ignored us; stop waiting

        self._thread.join(timeout=15)

    def _build_command(self) -> list:
        """The ffmpeg command line, with a note on why each flag is there."""
        output = os.path.join(self.spool_dir, FILENAME_PATTERN % self.camera.name)
        return [
            "ffmpeg",
            "-nostdin",                 # never try to read the keyboard; we are a service
            "-hide_banner",
            "-loglevel", "warning",     # errors only, so the logs stay readable

            # RTSP defaults to UDP, which loses packets on a busy network and
            # produces corrupt segments. TCP costs a little latency, which does
            # not matter at all when the destination is a file on disk.
            "-rtsp_transport", "tcp",

            "-i", self.camera.url,

            "-c:v", "copy",             # no re-encoding - see the module docstring

            # Drop audio entirely. Nothing here listens to it - every detector
            # works on pixels - and keeping it actively breaks recording: many
            # cameras (Tapo and most ONVIF ones) send G.711 a-law/mu-law, which
            # MP4 cannot store. Copying it produces exactly this, on every
            # segment, forever:
            #     Could not write header (incorrect codec parameters ?)
            # Re-encoding to AAC would also fix it, but spends CPU on a stream
            # that then gets thrown away.
            "-an",

            # The segment muxer: start a new output file every N seconds.
            "-f", "segment",
            "-segment_time", str(self.segment_seconds),
            "-segment_format", "mp4",

            # Without this, every segment keeps the running timestamps of the
            # live stream, so a file recorded an hour in claims to start at
            # 01:00:00. Players and OpenCV then read its duration wrongly.
            "-reset_timestamps", "1",

            "-strftime", "1",           # let ffmpeg expand the %Y%m%d in the filename
            output,
        ]

    def _run(self):
        """Thread body: run ffmpeg, and when it exits, reconnect - forever."""
        backoff = RECONNECT_BACKOFF_START

        while not self._stop.is_set():
            started = time.monotonic()
            log.info("[%s] connecting", self.camera.name)

            try:
                self._process = subprocess.Popen(
                    self._build_command(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,     # captured so we can log why it died
                    text=True,
                )
            except FileNotFoundError:
                # Nothing here can recover from a missing binary, so stop rather
                # than retry forever against a problem no retry will fix.
                raise SystemExit("ffmpeg is not installed or not on PATH.")

            # Blocks until ffmpeg exits - which, for a healthy camera, means the
            # stream dropped.
            _, errors = self._process.communicate()
            ran_for = time.monotonic() - started

            if self._stop.is_set():
                log.info("[%s] stopped", self.camera.name)
                return

            # ffmpeg's last stderr line is usually the actual reason it gave up.
            last_lines = (errors or "").strip().splitlines()
            reason = last_lines[-1] if last_lines else f"exit code {self._process.returncode}"
            log.warning("[%s] stream ended after %.0fs: %s", self.camera.name, ran_for, reason)

            if ran_for >= STABLE_RUN_SECONDS:
                backoff = RECONNECT_BACKOFF_START

            log.info("[%s] reconnecting in %.0fs", self.camera.name, backoff)

            # wait() returns True if stop() was called while we were sleeping, so
            # shutdown does not have to sit through the full backoff.
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
