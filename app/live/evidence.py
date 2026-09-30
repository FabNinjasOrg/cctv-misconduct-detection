"""Saves a picture when the live view flags something.

WHAT: writes an annotated frame - box, skeleton and label drawn on - to
      evidence/live_view/<camera>/ once a detection is confirmed (see
      CONFIRM_SECONDS in engine.py). The picture is the detection's best moment,
      not the moment it was confirmed, which comes a second or more later.
WHY:  the live view is something you watch, and nobody watches all day. Without
      this, an event that happens while you are looking away leaves no trace at
      all.
HOW:  see the cooldown below. That is the whole of the interesting part.

Deliberately separate from the recorder's evidence/live_detections/. Two writers
in one folder would make it impossible to tell which found what, and they would
race over the same detections.json. Same building, different rooms.

The frame saved is the annotated one rather than the clean camera image: the
point of this file is to show what the system *thought*, so the box and the
label are the evidence, not decoration.

Each picture gets a .json beside it holding every detector's score and the
winning detector's own breakdown of how it got there. A picture alone shows
*that* something fired; only the numbers show *why*, and why is the only part
you can act on. The pair together is the regression set - real footage from the
real camera, which no clip collection can substitute for.
"""
import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np

from app.live import render

log = logging.getLogger(__name__)

# How long to wait before saving the same action for the same person again.
#
# This is the difference between a usable folder and an unusable one. Tracking
# runs at 15 frames a second and scoring re-runs every 0.3s, so a fall that
# lasts ten seconds would otherwise write about thirty near-identical pictures
# of the same moment. One per event is what anyone actually wants.
#
# Per person and per action, not per camera: two people can genuinely fall at
# once, and someone can genuinely spit and then be hit, and both of those are
# worth having.
COOLDOWN_SECONDS = 30.0

# JPEG quality for saved snapshots. Higher than the live stream's 80 - a stream
# frame is gone in a sixteenth of a second, this one gets looked at closely.
JPEG_QUALITY = 92

DEFAULT_DIR = "evidence/live_view"

# Matches the fight detector's own CONFIDENCE_MIN and render.FIGHT_THRESHOLD, so
# what gets saved is exactly what was shown on screen.
FIGHT_THRESHOLD = 0.5


def _unique(path: str) -> str:
    """The path, or the next free `-2`, `-3`... if something is already there.

    Names are built from a whole second, so two people flagged with the same
    action and the same rounded confidence inside one second would otherwise
    produce the same name and the second would silently erase the first. Rare,
    but "two people fell at once" is exactly the event you least want to lose
    half of.

    Kept out of the name in the common case on purpose: almost every snapshot
    gets the plain, readable form, and only a genuine clash pays for it."""
    if not os.path.exists(path):
        return path

    stem, extension = os.path.splitext(path)
    for suffix in range(2, 1000):
        candidate = f"{stem}-{suffix}{extension}"
        if not os.path.exists(candidate):
            return candidate
    return path     # 999 collisions in one second is not a real scenario


# Signals that exist only so the recorded pipeline can draw its own evidence
# frames. Live has already drawn and saved the picture, so here they are just
# hundreds of coordinates burying the dozen numbers worth reading.
BULK_SIGNALS = ("peak_keypoints", "peak_kpts_conf", "burst_frames", "records")


def _plain(value):
    """Turns a detector's signals into something json can write.

    They are full of numpy scalars and arrays - a box is a list of np.float32,
    a keypoint array is 17x2 - and json refuses all of it. Rather than teaching
    every detector to return plain Python, which would be a change to code the
    recorded pipeline shares, the conversion happens here at the edge."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        # NaN and infinity are valid floats and invalid json - a detector
        # dividing by a zero-height box produces them, and one of those would
        # otherwise write a file nothing can read back.
        return value if value == value and abs(value) != float("inf") else None
    return value


class Snapshotter:
    """Writes one picture per event, and then keeps quiet about it."""

    def __init__(self, camera_name: str, directory: str = DEFAULT_DIR,
                 cooldown: float = COOLDOWN_SECONDS):
        self.camera_name = camera_name
        self.directory = os.path.join(directory, camera_name)
        self.cooldown = cooldown

        # (track_id, action) -> the clock reading when it was last saved
        self._last_saved: dict = {}

        self.saved_count = 0

    def update(self, frame, people: list, now: float, fight: float = 0.0,
               events: list = ()) -> list:
        """Saves a snapshot for anything newly confirmed. Returns what it wrote.

        `events` are the engine's confirmed detections (LiveEngine.take_events),
        each carrying the clean frame and the person as they were at their best
        moment - that is what gets drawn and saved. `frame` is the annotated
        live frame, used only for fight. `now` is any steadily increasing clock -
        it is only ever compared against itself, so monotonic time is fine and
        is better here than wall time, which can jump."""
        saved = []

        for event in events:
            if event.get("frame") is None:
                continue
            action = event["action"]

            path = self._maybe_write(
                self._picture(event), now, (event["track_id"], action), action,
                event["confidence"], event["track_id"],
                detail={
                    "scope": "person",
                    # Every action's score, not just the winner - what *nearly*
                    # fired is half of what you need to judge whether the winner
                    # should have.
                    "scores": event.get("scores") or {},
                    "signals": {
                        name: value
                        for name, value in (event.get("signals") or {}).items()
                        if name not in BULK_SIGNALS
                    },
                    "history_records": event.get("history"),
                    # What the change gate measured for this person. Recorded on
                    # every detection so MIN_POSE_CHANGE can be confirmed against
                    # real events rather than left on reasoning alone.
                    "pose_change": event.get("pose_change"),
                    "box": event.get("box"),
                    # How the wait went - recorded on every detection so that
                    # CONFIRM_SECONDS and HOLD_THRESHOLD can be checked against
                    # real events, as the numbers behind them were only measured
                    # on test clips.
                    "confirmation": {
                        "held_for_s": event.get("held_for"),
                        "lowest_score_while_held": event.get("lowest_while_held"),
                        "left_view_before_confirmed": event.get("left_view"),
                    },
                },
                when=event.get("seen_at"),
            )
            if path:
                saved.append(path)

        # Fight is scene-level - MoViNet judges the whole picture, so there is
        # no track id to key the cooldown on. Keyed on the camera instead, which
        # is the right granularity anyway: one fight in the room is one alert.
        if fight >= FIGHT_THRESHOLD:
            path = self._maybe_write(
                frame, now, ("scene", "fight"), "fight", fight, None,
                detail={
                    # MoViNet returns one number and no breakdown, so there is
                    # nothing else to record. How many people were in shot is
                    # the one thing worth keeping: it is what the two-person
                    # guard turns on.
                    "scope": "scene",
                    "scores": {"fight": fight},
                    "signals": {},
                    "people_in_shot": len(people),
                },
            )
            if path:
                saved.append(path)

        self._forget_old(now)
        return saved

    @staticmethod
    def _picture(event: dict):
        """The event's best moment, drawn: its clean frame with just that person
        boxed and labelled, as the live view would have shown them had the
        alarm been called there and then."""
        person = {
            "track_id": event["track_id"],
            "box": event["box"],
            "kpts_xy": event["kpts_xy"],
            "kpts_conf": event["kpts_conf"],
            "scores": event.get("scores") or {},
            "action": event["action"],
            "confidence": event["confidence"],
            "history": event.get("history") or render.WARMING_UP_RECORDS,
        }
        return render.draw(event["frame"], [person])

    def _maybe_write(self, frame, now: float, key: tuple, action: str,
                     confidence: float, track_id, detail: dict, when: float = None):
        """Writes one snapshot unless this exact thing is still in cooldown."""
        last = self._last_saved.get(key)
        if last is not None and now - last < self.cooldown:
            return None

        path = self._write(frame, action, confidence, track_id, detail, when)
        if path:
            self._last_saved[key] = now
            self.saved_count += 1
        return path

    def _write(self, frame, action: str, confidence: float, track_id, detail: dict,
               when: float = None):
        """Writes one annotated frame and returns where it went.

        `when` is the wall-clock time the picture shows, if not now."""
        # Created here rather than at import, so a run that finds nothing leaves
        # no empty folders behind.
        os.makedirs(self.directory, exist_ok=True)

        # Wall clock for the name, because a filename is read by a person: the
        # time it happened is the whole point - which, for a confirmed detection,
        # is its best moment rather than the second or so later it was confirmed.
        taken = datetime.fromtimestamp(when) if when is not None else datetime.now()
        stamp = taken.strftime("%Y%m%d-%H%M%S")
        percent = round(confidence * 100)
        name = f"{self.camera_name}_{stamp}_{action}_{percent}pct.jpg"
        path = _unique(os.path.join(self.directory, name))

        try:
            ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        except Exception as e:
            log.warning("could not save snapshot to %s: %s", path, e)
            return None

        if not ok:
            log.warning("could not save snapshot to %s", path)
            return None

        self._write_sidecar(path, stamp, action, confidence, track_id, detail)

        who = f" (#{track_id})" if track_id is not None else ""
        log.warning("%s %d%%%s -> %s", action.upper(), percent, who, path)
        return path

    def _write_sidecar(self, image_path: str, stamp: str, action: str,
                       confidence: float, track_id, detail: dict) -> None:
        """Writes the numbers behind the picture, next to the picture.

        Failing to write this must never cost the picture, which is why it runs
        after the image is safely on disk and swallows its own errors - a
        snapshot with no sidecar is a small loss, a detection with no snapshot
        is the whole point gone."""
        record = {
            "camera": self.camera_name,
            "time": stamp,
            "action": action,
            "confidence": round(float(confidence), 3),
            "track_id": track_id,
            "image": os.path.basename(image_path),
            **_plain(detail),
        }
        path = os.path.splitext(image_path)[0] + ".json"
        try:
            with open(path, "w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
        except Exception as e:
            log.warning("could not save signals to %s: %s", path, e)

    def _forget_old(self, now: float) -> None:
        """Drops cooldown entries that can no longer block anything.

        Without this the dictionary grows for as long as the process runs - one
        entry per person per action, forever. Nobody would notice in an hour;
        over a week on a busy camera it is a leak."""
        expired = [
            key for key, when in self._last_saved.items()
            if now - when > self.cooldown * 2
        ]
        for key in expired:
            del self._last_saved[key]
