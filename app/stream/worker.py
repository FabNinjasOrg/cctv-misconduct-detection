"""Sends recorded segments to the detection API and keeps whatever it flags.

WHAT: watches the spool folder, takes each finished segment, asks the API what is
      in it, and saves evidence for anything detected.
WHY:  this is the piece that turns "we have video files" into "we know something
      happened".
HOW:  a single loop - pick the oldest finished file, analyze or skip it, delete
      it, repeat.

Why HTTP instead of importing the detectors directly: the detection service
already has the models loaded in memory. Importing them here would load a second
copy and double the RAM for no benefit. The API also motion-gates each segment
itself, so a quiet stretch of footage costs about a second instead of thirty.

Files are always deleted after being handled - analyzed or skipped. The spool is
a queue, not an archive; anything worth keeping has already been copied into the
evidence folder by then.
"""
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

log = logging.getLogger(__name__)


def _recorded_at(segment_stem: str) -> str:
    """Turns a segment filename into the wall-clock time it was recorded.

    The recorder names files `<camera>_YYYYMMDD-HHMMSS.mp4` (see recorder.py),
    so the timestamp is already in the name and needs no extra bookkeeping.
    Returns an ISO 8601 string with the local UTC offset, e.g.
    2026-09-24T07:18:30+05:30 - set TZ in .env to control which zone that is.

    Falls back to None if the name is not in the expected shape, which should
    only happen for a file this program did not write."""
    try:
        stamp = datetime.strptime(segment_stem.rsplit("_", 1)[-1], "%Y%m%d-%H%M%S")
    except (ValueError, IndexError):
        return None
    return stamp.astimezone().isoformat(timespec="seconds")

# How long to sleep when there is nothing waiting. Short enough that a new
# segment is picked up promptly, long enough not to spin the CPU doing nothing.
POLL_SECONDS = 2.0

# If the API keeps failing on the same segment, give up on it. Retrying forever
# would block every segment behind it and the queue would never drain.
MAX_ATTEMPTS = 3

# The newest file in the spool is normally the one ffmpeg is still writing, so it
# is left alone. But if it hasn't been touched for this many segment-lengths, the
# recorder must have died - the file will never be "finished", so take it anyway.
STALE_SEGMENT_FACTOR = 3


class Worker:
    """Analyzes recorded segments, one at a time.

    Single-threaded on purpose: analysis already uses every CPU core, so running
    two in parallel would just make both slower and fix nothing."""

    def __init__(self, config):
        self.config = config

        # Counts every segment picked up per camera. Used for duty cycling in
        # _process - see analyze_one_in.
        self._seen_count = {camera.name: 0 for camera in config.cameras}

        # Failed API attempts per file path, so a broken segment can be dropped
        # after MAX_ATTEMPTS instead of retried forever.
        self._attempts = {}

    # ------------------------------------------------------------------
    # Finding work
    # ------------------------------------------------------------------

    def _finished_segments(self, camera) -> list:
        """Segments ffmpeg has finished writing, oldest first.

        Filenames carry a timestamp, so sorting by name sorts by time. ffmpeg
        only ever writes to the newest file, which is why everything except the
        last entry is safe to read."""
        folder = os.path.join(self.config.spool_dir, camera.name)
        if not os.path.isdir(folder):
            return []       # the recorder hasn't created it yet

        files = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".mp4")
        )
        if not files:
            return []

        # If the newest file has gone stale, the recorder is not running and
        # nothing will ever close it - so include it rather than wait forever.
        newest_age = time.time() - os.path.getmtime(files[-1])
        if newest_age > self.config.segment_seconds * STALE_SEGMENT_FACTOR:
            return files

        return files[:-1]

    def _trim_backlog(self, camera, pending: list) -> list:
        """Throws away the oldest segments when analysis has fallen too far behind.

        Two reasons this has to exist: unbounded, the spool grows until the disk
        is full; and a detection found in twenty-minute-old footage is useless as
        an alert anyway. Better to stay current and miss some."""
        excess = len(pending) - self.config.max_backlog_segments
        if excess <= 0:
            return pending

        log.warning(
            "[%s] %d segments waiting, limit is %d - dropping the %d oldest. "
            "Analysis is not keeping up: raise analyze_one_in or use faster hardware.",
            camera.name, len(pending), self.config.max_backlog_segments, excess,
        )
        for path in pending[:excess]:
            self._delete(path)
        return pending[excess:]

    def _delete(self, path: str) -> None:
        """Removes a segment from the spool once we are done with it."""
        self._attempts.pop(path, None)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass        # already gone; nothing to do

    # ------------------------------------------------------------------
    # Analyzing
    # ------------------------------------------------------------------

    def _call_api(self, path: str) -> dict:
        """Posts one segment to /analyze-action and returns the parsed verdict.

        `path` is sent as-is: the API resolves paths relative to the project root,
        and the spool folder is mounted into both containers at the same place,
        so the same string means the same file on both sides."""
        body = urllib.parse.urlencode({"filepath": path}).encode()
        request = urllib.request.Request(self.config.api_url, data=body)

        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return json.load(response)

    def _process(self, camera, path: str) -> None:
        """Handles one segment end to end: skip it, or analyze and record it."""
        seen = self._seen_count[camera.name]
        self._seen_count[camera.name] = seen + 1

        # Duty cycling. When detection is slower than the camera, analyzing an
        # evenly spaced sample beats falling further behind on every segment -
        # a detection from five minutes ago is not worth much. analyze_one_in=3
        # means analyze the 1st, skip the 2nd and 3rd, analyze the 4th, and so on.
        if seen % self.config.analyze_one_in != 0:
            self._delete(path)
            return

        started = time.monotonic()
        try:
            result = self._call_api(path)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            self._handle_failure(camera, path, e)
            return
        elapsed = time.monotonic() - started

        detected = (
            result.get("activity_detected")
            and result["confidence"] >= self.config.min_confidence
        )
        if detected:
            self._save_detection(camera, path, result)
        else:
            log.info(
                "[%s] %s clear (%.0fs, motion=%s)",
                camera.name, os.path.basename(path), elapsed, result.get("motion_detected"),
            )

        # If a segment takes longer to analyze than it took to record, the
        # backlog can only grow. Say so, rather than let it fail quietly.
        if elapsed > self.config.segment_seconds * self.config.analyze_one_in:
            log.warning(
                "[%s] analysis took %.0fs for %ds of footage - not keeping up in real time.",
                camera.name, elapsed, self.config.segment_seconds,
            )

        self._delete(path)

    def _handle_failure(self, camera, path: str, error) -> None:
        """Retries a failed segment a few times, then gives up on it.

        Usually means the detection service is restarting or briefly unreachable,
        in which case the next attempt succeeds. A segment that fails repeatedly
        is more likely corrupt, and is dropped so it stops blocking the queue."""
        attempts = self._attempts.get(path, 0) + 1
        self._attempts[path] = attempts
        name = os.path.basename(path)

        if attempts >= MAX_ATTEMPTS:
            log.error("[%s] giving up on %s after %d attempts: %s",
                      camera.name, name, attempts, error)
            self._delete(path)
        else:
            log.warning("[%s] analysis failed for %s (attempt %d): %s",
                        camera.name, name, attempts, error)

    # ------------------------------------------------------------------
    # Saving evidence
    # ------------------------------------------------------------------

    def _save_detection(self, camera, segment_path: str, result: dict) -> None:
        """Copies the evidence out of the spool and records the detection.

        Two things get saved: the annotated frame the detector produced (the one
        with the box and label burned in), and optionally the video it came from -
        a still rarely shows enough to judge what actually happened. Both are
        copied rather than moved, because the spool copy is about to be deleted.

        Also adds one entry to detections.json, which is the searchable record of
        what was found and when."""
        folder = os.path.join(self.config.evidence_dir, camera.name)
        os.makedirs(folder, exist_ok=True)

        # e.g. cam-1_20260924-071830_fall_70pct - keeps the recording time, the
        # action and the confidence together in a name that sorts chronologically.
        stem = os.path.splitext(os.path.basename(segment_path))[0]
        action = result["action"]
        percent = round(result["confidence"] * 100)
        base = f"{stem}_{action}_{percent}pct"

        saved_frame = None
        detector_frame = result.get("final_frame")
        if detector_frame and os.path.exists(detector_frame):
            saved_frame = os.path.join(folder, base + ".jpg")
            shutil.copy2(detector_frame, saved_frame)

        saved_clip = None
        if self.config.keep_segment_on_detection:
            saved_clip = os.path.join(folder, base + ".mp4")
            shutil.copy2(segment_path, saved_clip)

        self._append_to_log({
            # When it happened, taken from the recording's own filename. This is
            # the timestamp that matters - detected_at can be a minute or more
            # later, because it is when analysis finished, not when the event was.
            "recorded_at": _recorded_at(stem),
            "detected_at": datetime.now().astimezone().isoformat(timespec="seconds"),

            "camera": camera.name,
            "action": action,
            "confidence": result["confidence"],
            "confidence_percent": percent,

            "evidence_image": saved_frame,
            "evidence_clip": saved_clip,

            # every detector's score, not just the winner - useful when working
            # out why something was or wasn't flagged
            "scores": {
                name: score["confidence"]
                for name, score in (result.get("scores") or {}).items()
            },
            "source_segment": os.path.basename(segment_path),
        })

        # Logged at WARNING so a detection stands out among the routine INFO
        # lines when scrolling through `docker compose logs`.
        log.warning(
            "[%s] %s detected at %d%% -> %s",
            camera.name, action.upper(), percent, saved_frame or "(no frame)",
        )

    def _append_to_log(self, record: dict) -> None:
        """Adds one detection to detections.json.

        The file is a single JSON array, newest entry last, so it can be opened
        and read as-is by a person or loaded in one line by anything else
        (`json.load`, `jq`, a spreadsheet importer, a future dashboard).

        That means rewriting the whole file each time, which is fine here:
        detections are rare - a handful an hour at worst - and the rewrite costs
        microseconds next to the tens of seconds of analysis that produced the
        record. The write goes to a temporary file that is then renamed over the
        original, because os.replace is atomic: a crash mid-write leaves the
        previous complete file in place rather than a half-written one."""
        path = self.config.detections_log
        folder = os.path.dirname(path) or "."
        os.makedirs(folder, exist_ok=True)

        detections = self._read_log(path)
        detections.append(record)

        with tempfile.NamedTemporaryFile(
            "w", dir=folder, prefix=".detections-", suffix=".tmp", delete=False
        ) as f:
            json.dump(detections, f, indent=2)
            f.write("\n")
            temp_path = f.name

        os.replace(temp_path, path)

    @staticmethod
    def _read_log(path: str) -> list:
        """Loads the existing detections, or starts a new list.

        A damaged file is moved aside rather than overwritten - it is the only
        record of what was found, so losing it silently to recover from a parse
        error would be the worse failure."""
        try:
            with open(path) as f:
                detections = json.load(f)
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError) as e:
            salvage = f"{path}.corrupt-{int(time.time())}"
            log.error("%s is unreadable (%s); moving it to %s and starting a new log",
                      path, e, salvage)
            try:
                os.replace(path, salvage)
            except OSError:
                pass
            return []

        # Anything other than a list means the file is not ours; treat it the
        # same way rather than trying to append to it.
        return detections if isinstance(detections, list) else []

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self, stop_event) -> None:
        """Processes segments until stop_event is set.

        Handles one segment per pass so that shutdown and the backlog check both
        get a look in between each (slow) analysis."""
        log.info("worker ready, posting to %s", self.config.api_url)

        while not stop_event.is_set():
            found_work = False

            for camera in self.config.cameras:
                pending = self._trim_backlog(camera, self._finished_segments(camera))
                if pending:
                    found_work = True
                    self._process(camera, pending[0])    # oldest first
                if stop_event.is_set():
                    break

            # Only sleep when every camera was idle. Otherwise loop straight
            # round - there may be a backlog to work through.
            if not found_work:
                stop_event.wait(POLL_SECONDS)
