"""Turns a stream of frames into "who is in shot, and what are they doing".

WHAT: one pose pass per frame, feeding a rolling window of the last few seconds
      of each tracked person, scored continuously by the real detectors.
WHY:  the recorded pipeline answers that per 10-second file and starts from
      scratch on each one - tracker state, identities and history all reset at
      every boundary. A live view needs the answer now, and needs the same
      person followed *through* time rather than rediscovered every chunk.
HOW:  see the two sections below - tracking, then scoring.

Nothing here writes to disk.

Why the detectors are reused rather than reimplemented
------------------------------------------------------
Each pose detector already splits cleanly into two halves:

    compute_kinematics(records)   pure maths over one person's history
    score_event(...)              turns those features into a confidence

Both are fed a list of per-frame records, which is exactly what a rolling window
is. So this module builds the window and calls the detectors' own functions,
unmodified. A number shown on screen is therefore the number the recorded
pipeline would produce for the same footage - not a separate "live" estimate
that slowly drifts away from it as one side gets tuned and the other doesn't.

The one piece of glue is the frame adapter below.
"""
import logging
import threading
import time
from collections import deque

import numpy as np

from app.actions.fall import kinematic_detector as fall_det
from app.actions.fall import stgcn_detector as fall_model
from app.actions.fight import movinet_detector as fight_det
# from app.actions.spit import kinematic_detector as spit_det  # spit turned off
from app.actions.vandalism import object_action_detector as vandalism_det
from app.actions.vandalism.tamper_detector import TamperDetector
from app.shared.pose_tracking import PERSON_CLASS_ID, iou, pose_rejection_reason

log = logging.getLogger(__name__)

# How much of each person's recent history to keep and score over. The detectors
# reason in seconds - a held bow, a trunk angle flipping, a burst of strikes -
# and their longest look-back is fall's COLLAPSE_HOLD_SECONDS at 1.5s on top of
# a 1.0s baseline. Six seconds comfortably covers every one of them while
# staying small enough to score often.
WINDOW_SECONDS = 6.0

# Drop a person from tracking when they have not been seen for this long. Short
# enough that a new arrival gets a fresh identity rather than inheriting the
# history of whoever stood there before; long enough to ride out the odd frame
# where the pose model misses someone.
TRACK_EXPIRY_SECONDS = 2.0

# How often to re-score. Scoring is far more expensive than tracking, so it runs
# on its own clock rather than once per frame.
#
# 0.3s is not a comfort setting, it is the difference between seeing an event
# and missing it. Measured by replaying known clips through this engine:
#
#                     recorded pipeline    live @ 0.7s    live @ 0.3s
#     spit-1.mp4            0.777             0.23           0.77
#     fall-1.mp4            1.000             1.00           1.00
#
# A spit is over in well under a second, so scoring every 0.7s simply stepped
# across its peak. Widening the window instead changed nothing - 6s and 12s
# scored identically - because the problem was never how much history was kept,
# it was how often anyone looked at it.
SCORE_INTERVAL_SECONDS = 0.3

# A window shorter than this has too little history for the kinematics to mean
# anything - velocities and angle swings need several samples to exist at all.
MIN_RECORDS_TO_SCORE = 8

# How many recent frames to keep for the detectors to look back at (see the
# frame adapter). At roughly 8 fps this is about twelve seconds, comfortably
# more than WINDOW_SECONDS, so a peak frame is always still in hand.
FRAME_BUFFER_SIZE = 96

# Matching a detection to an existing person. Lower than the recorded
# pipeline's 0.3 because frames arrive further apart here - at ~8 fps a walking
# person moves noticeably between samples, so boxes overlap less.
IOU_MATCH_MIN = 0.2

# Shown on screen once a score reaches this. Matches the detectors' own
# DECISION_THRESHOLD, so the live view calls the same thing a detection that the
# recorded pipeline would.
DISPLAY_THRESHOLD = 0.6 #threshold

# ACTIONS = ("spit", "vandalism", "fall")
ACTIONS = ("vandalism", "fall")  # spit turned off

# --- wait and confirm --------------------------------------------------------
#
# A score reaching DISPLAY_THRESHOLD is not an alarm yet, only a candidate. The
# detectors look back over the window but never forward: they call the start of
# something the moment it looks right, before seeing how it ends - a head dipped
# at a phone reads as a spit's bow until the person is seen walking on with it.
# So a candidate is watched for CONFIRM_SECONDS, and becomes an alarm only if its
# score never falls below HOLD_THRESHOLD in that time.
#
# Measured by replaying every clip through this engine, with each clip's last
# frame held for 4s afterwards - the clips end 0-2s after their event, and live
# footage does not. "Real" is the 11 clips whose event the detectors can see
# (spit-1..8, fall-2..4); the longer waits are simulated from the same scores:
#
#                          false alarms    real events still alarmed (of 11)
#     alarm at once             26              11
#     wait 1.0s                  8              11
#     wait 1.5s                 ~6              10 - spit-6 lost
#     wait 3.0s                 ~4               9 - spit-6 and fall-4 lost
#
# Real events hold: every real spit and fall stayed above HOLD_THRESHOLD for as
# long as the footage lasted. Most false alarms do not - they fall back within
# 0.3-1.2s, the phone-walker in fall-1 among them. 1.0s is the longest wait that
# kept every real event; spit-6 held for exactly that long before dropping, so a
# longer wait is not free, whatever it saves.
#
# fall-1 is not in the 11. Its old "fall" alarms were a phantom box over an empty
# pit, and the man's real drop into it scores 0.00 in every version - the wait
# cancels the phantom, which is right, but the clip's fall was never detected.
#
# HOLD_THRESHOLD sits below the alarm line on purpose. Real spits hover at
# 0.60-0.70, and holding them to 0.6 itself would cancel a genuine event for
# dipping to 0.59; the false alarms that fall back fall a long way, to 0.0-0.3.
CONFIRM_SECONDS = 1.0
HOLD_THRESHOLD = 0.5

# Someone who leaves the picture during the wait is judged on what was seen, not
# cancelled - a vandal who kicks and runs, or someone who falls out of shot,
# leaves no "after" to watch. Only a score falling while the person is still in
# view cancels an alarm. Not seen for this long counts as out of view.
#
# PROVISIONAL - reasoned, not measured: no test clip has a real event whose
# person leaves during the wait. (fall-1 looks like one and is not: the man's
# drop into the pit scores 0.00 in every version, and its old "fall" alarms were
# a phantom box over the empty pit that inherited his track.) Every snapshot's
# .json records left_view_before_confirmed so this can be checked on real events.
LEFT_VIEW_SECONDS = 0.5

# --- the change gate ---------------------------------------------------------
#
# Only score a person whose posture has actually changed during the window.
#
# The recorded pipeline never scored a still scene: it found the stretches where
# something moved and scored only those, so the mere existence of a window
# already meant something had happened. Live there is no such filter - every
# tracked person is scored every SCORE_INTERVAL_SECONDS whatever they are doing,
# which is about 12,000 attempts per person per hour. Guards calibrated against
# a few dozen attempts per clip do not survive thousands: a rule that lets
# through one innocent posture in a thousand fires twelve times an hour.
#
# That filter was dropped when the recorder was removed and nothing replaced it.
# This is its replacement, and it is per-person rather than per-frame, which
# suits the question better - "has this body changed" rather than "did any pixel
# move".
#
# It is also the right shape for what the detectors look for. All four actions
# are transitions: a bow taken and released, a trunk flipping down, a limb
# swung. None of them is a posture held unchanged for six seconds. When the
# window holds no change there is no event in it, and whatever the detectors'
# two-level splits and peak-finders latch onto is noise being asked to explain
# itself.
POSE_CONF_FLOOR = 0.15      # matches the detectors - below this a joint is missing
MIN_KEYPOINT_SAMPLES = 4    # a joint seen fewer times than this says nothing about movement

# How much the body must have moved to be worth scoring, as the average joint's
# wander across the window in fractions of that person's own box height.
#
# PROVISIONAL - reasoned, not yet measured, unlike most constants in this
# project. The margin it leans on is wide: a spit's bow swings the head through
# 40-57 degrees, moving the nose something like 0.2-0.3 box heights, and a fall
# or a swing moves more still, while a held posture wanders by a fraction of
# that. 0.02 sits an order of magnitude below the smallest real action and just
# above the pose model's own jitter, so it should only ever remove someone
# sitting genuinely still.
#
# Every snapshot's .json records the measured value as `pose_change`, so the
# real distribution can be read off the evidence folder once there are genuine
# events in it - and this number confirmed or moved on measurement rather than
# on the reasoning above.
MIN_POSE_CHANGE = 0.02

# --- fall: two judges, both must agree ---------------------------------------
#
# A person's fall score is the LOWER of two independent answers:
#
#   the model   ST-GCN, trained on Le2i skeletons (fall/stgcn_detector.py) - did
#               the last 3 seconds move like a fall?
#   the rules   the hand-written kinematics (fall/kinematic_detector.py) - did the
#               trunk flip, the outline collapse, the hips really go down?
#
# Each fails in a different place, and the other covers it. Replayed over every
# clip with its last frame held 4s, confirmed fall alarms:
#
#                         real falls caught    other alarms
#     rules alone           fall-2, -3, -4     a desk clip, a fighter knocked down
#     model alone           fall-2, -3         48 - crowds, people in a brawl
#     both must agree       fall-2, -3         a fighter knocked down - really down
#
# The model learned from one person in a quiet room, and in a crowd its skeletons
# get stitched together from different people by the tracker - it calls that a
# fall. The rules never see that. The rules, on the other hand, called all six
# of the live camera's fall alarms on 2026-09-29, and every one was someone
# sitting down on the floor or a chair; the model scores 0.00 on all four
# minutes of this camera's recordings. Asking both is the one combination that
# kept every real fall the model can see.
#
# What it costs: fall-4. Its man is already lying down when the clip starts, so
# the model never sees a fall happen; only the rules called it. And a fall the
# rules cannot read at all is now missed too - when the model has been trained on
# this camera's own footage, it may be good enough to be trusted alone.

# --- fight -------------------------------------------------------------------
#
# Fight is the odd one out and has to be handled separately. The other three
# score a PERSON from their pose over time; MoViNet scores the PEOPLE TOGETHER
# over a second of video - one square crop around everyone in shot - and returns
# one number for the scene. There is no per-person answer to attach a label to.
#
# The crop is what the model was trained on (fight/clip_classifier.py), so each
# sampled frame keeps the person boxes the pose pass found in it - all of them,
# including the ones too close to measure, exactly as in training.
#
# It also cannot ride along on the pose pass: it is its own 3D CNN and costs
# about 156 ms a window, measured on 8 cores. So it runs on a slower duty cycle
# of its own, and the frame rate the pose detectors depend on is unaffected.
#
# The model it replaced saw the whole picture squeezed to 172 x 172. Replayed
# over every clip, windows called a fight (old -> new):
#
#     the seven fight clips          134 -> 216 of 310
#     your acted fight (real-fight-1)  0 -> 2 of 32   (the two moments of contact)
#     this camera, 4 min of normal     0 -> 0
#     other clips                      3 -> 13  - 11 in crowds (people rushing to
#                                                   someone who fell), 2 a lone man
#                                                   kicking a bike, which the
#                                                   two-person guard below discounts
#
# The model was trained on 8 frames sampled at 8 fps - one second of video - so
# it must be fed at that rate regardless of what the tracking loop runs at.
# Feeding it 15 fps frames would hand it half a second of footage and call it a
# second, which is exactly the class of mistake TARGET_FPS exists to avoid.
FIGHT_SAMPLE_FPS = 8.0
FIGHT_WINDOW_FRAMES = fight_det.WINDOW_FRAMES       # 8, i.e. one second
FIGHT_INTERVAL_SECONDS = 1.0                        # how often to classify

# A fight needs at least two people. MoViNet alone will happily call one person
# moving quickly a fight - vandalism-2's lone man kicking a bike scores 1.00 - so
# the recorded pipeline checks how many people were near the flagged moment and
# discounts it if there was only one. Live, the pose pass already found everyone
# in each frame of the window, so the same guard costs nothing. Counted from
# those boxes rather than from the tracker, which skips anyone too close to
# measure - two people grappling are exactly that.
FIGHT_MIN_PEOPLE = 2

# --- camera tamper -----------------------------------------------------------
#
# Vandalism of the camera itself - covered, sprayed, turned away. Judged on the
# whole picture, a couple of times a second, by vandalism/tamper_detector.py. It
# is cheap enough (a 160-pixel grey copy) to run in the frame loop.


class _FrameBuffer:
    """The last few seconds of frames, addressable by frame number.

    Exists to serve the frame adapter below. Frames are kept as-is rather than
    JPEG-encoded: spit inspects small crops around the mouth to decide whether
    anything actually left it, and compression artifacts at that scale are the
    same order as the signal it is measuring."""

    def __init__(self, size: int = FRAME_BUFFER_SIZE):
        self._frames = deque(maxlen=size)      # (frame_number, frame)
        self._lock = threading.Lock()

    def add(self, frame_number: int, frame) -> None:
        with self._lock:
            self._frames.append((frame_number, frame))

    def get(self, frame_number: int):
        """The frame with that number, or the nearest one still held.

        Never returns None once anything has been buffered. A detector asking
        for a frame that has already aged out should get the closest thing
        available rather than crash the view - being a few frames off makes a
        corroboration score slightly wrong, which is a far better outcome than
        a live view that dies."""
        with self._lock:
            if not self._frames:
                return None
            for number, frame in self._frames:
                if number == frame_number:
                    return frame
            return min(self._frames, key=lambda item: abs(item[0] - frame_number))[1]


def _pose_change(window: list) -> float:
    """How far the average joint wandered across the window, in box heights.

    Normalised by the person's own box height so the same real movement reads
    the same whether they are near the camera or far from it - the same trick
    the detectors use for every other distance they measure.

    Averaged over joints rather than taken at the loudest one, deliberately.
    Somebody typing moves two hands and nothing else; a spit, a fall or a swing
    moves most of the body. Taking the maximum would call the typist busy."""
    if len(window) < 2:
        return 0.0

    heights = [record["box_height"] for record in window if record["box_height"]]
    scale = float(np.median(heights)) if heights else 0.0
    if scale <= 0:
        return 0.0

    positions = {}
    for record in window:
        confidences = record["kpts_conf"]
        for index, point in enumerate(record["kpts_xy"]):
            if confidences[index] >= POSE_CONF_FLOOR:
                positions.setdefault(index, []).append(point)

    wanders = [
        float(np.hypot(*np.std(np.asarray(seen, dtype=float), axis=0)))
        for seen in positions.values()
        if len(seen) >= MIN_KEYPOINT_SAMPLES
    ]
    return float(np.mean(wanders)) / scale if wanders else 0.0


def _install_frame_adapters(buffer: _FrameBuffer) -> None:
    """Points the detectors' frame reads at memory instead of a video file.

    Both spit and vandalism fetch a frame or two while scoring - spit to measure
    the mouth region, vandalism to look for a tool near the swinging hand. They
    do it through two small module-level helpers that take (filepath,
    frame_number) and open the file with OpenCV.

    Live, the frames are already in hand and there is no file. Rather than
    changing those detectors - which would mean the recorded pipeline and this
    one no longer run identical code, the exact thing this design is trying to
    avoid - we swap the two helpers for versions that read the ring buffer. The
    filepath argument is ignored; the frame number is all that is needed, and it
    means the same thing on both sides.

    Called once, at startup."""

    def read_one(_filepath, frame_number):
        return buffer.get(frame_number)

    def read_many(_filepath, frame_numbers):
        return {number: buffer.get(number) for number in frame_numbers}

    # spit_det._read_frame_at = read_one       # spit turned off
    # spit_det._read_frames_at = read_many
    vandalism_det._read_frame_at = read_one


class LiveEngine:
    """Tracks everyone in view and keeps a running score of what they are doing.

    Call process(frame) for each new frame; read .people() for what to draw.
    Thread-confined: process() is expected to be called from one thread only."""

    def __init__(self, person_model, window_seconds: float = WINDOW_SECONDS):
        self.person_model = person_model
        self.window_seconds = window_seconds

        self._tracks: dict = {}         # track_id -> deque of per-frame records
        self._last_seen: dict = {}      # track_id -> time it was last matched
        self._last_box: dict = {}       # track_id -> its most recent box
        self._scores: dict = {}         # track_id -> {action: confidence}
        self._signals: dict = {}        # track_id -> {action: the detector's own breakdown}
        self._changes: dict = {}        # track_id -> how much they moved, see the change gate
        self._next_id = 1

        # Wait and confirm - see CONFIRM_SECONDS. All three are guarded by
        # _scores_lock, since the scorer writes them and the renderer reads them.
        self._candidates: dict = {}     # (track_id, action) -> being watched, not yet an alarm
        self._confirmed: dict = {}      # (track_id, action) -> an alarm, until its score falls
        self._new_events: list = []     # confirmed since the last take_events()

        self._frame_number = 0
        self._frames = _FrameBuffer()
        _install_frame_adapters(self._frames)

        self._last_scored_at = 0.0
        self._started_at = time.monotonic()

        # Set once the first frame arrives - the detectors need the frame size
        # to judge how close someone is to an edge.
        self._frame_width = 0
        self._frame_height = 0

        self._scores_lock = threading.Lock()    # guards _scores, read by the renderer
        self._tracks_lock = threading.Lock()    # guards the track dicts, which the
                                                # scoring thread snapshots while the
                                                # capture loop is still appending
        self._score_stop = threading.Event()
        self._score_thread = None

        # Fight: its own frame buffer at its own rate, and its own thread. Each
        # entry is (frame, the person boxes found in it) - the boxes are what the
        # model's crop is built from.
        self._fight_frames = deque(maxlen=FIGHT_WINDOW_FRAMES)
        self._fight_frames_lock = threading.Lock()
        self._fight_next_sample = 0.0
        self._fight_last_scored = 0.0
        self._fight_score = 0.0
        self._fight_detail = {}
        self._fight_thread = None

        # Camera tamper: fed from the frame loop, read by the renderer.
        self._tamper = TamperDetector()
        self._tamper_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Running scoring in the background
    #
    # Tracking has to hold ~15 frames a second, because the detectors count
    # history in records and their gap-filling constants are sized for that
    # rate - drop to 8 and a 1-second tolerance silently becomes 2. Scoring,
    # meanwhile, costs more the more people are in shot.
    #
    # Left in the same loop, a crowded frame would slow tracking below the rate
    # the detectors assume, and the scores would quietly get less accurate
    # exactly when the scene got interesting. So scoring runs on its own thread
    # over a snapshot, and a slow score costs latency on the label rather than
    # fidelity in the history it is reading.
    # ------------------------------------------------------------------

    def start(self):
        """Begins scoring in the background. Without this, nothing is labelled."""
        self._score_thread = threading.Thread(
            target=self._score_loop, daemon=True, name="live-scoring"
        )
        self._score_thread.start()

        # Fight gets its own thread rather than sharing the pose one: at ~156 ms
        # a window it would otherwise hold up the pose scores behind it.
        self._fight_thread = threading.Thread(
            target=self._fight_loop, daemon=True, name="live-fight"
        )
        self._fight_thread.start()
        return self

    def stop(self) -> None:
        self._score_stop.set()
        for thread in (self._score_thread, self._fight_thread):
            if thread:
                thread.join(timeout=5.0)

    def _score_loop(self) -> None:
        while not self._score_stop.is_set():
            started = time.monotonic()
            try:
                self._score_all(time.monotonic() - self._started_at)
            except Exception as e:      # never let the scorer take the view down
                log.debug("scoring pass failed: %s", e)

            # Pace from the end of the last pass, so a slow pass simply runs
            # less often rather than queueing up behind itself.
            remaining = SCORE_INTERVAL_SECONDS - (time.monotonic() - started)
            if remaining > 0:
                self._score_stop.wait(remaining)

    def _fight_loop(self) -> None:
        """Classifies the last second of video, over and over."""
        while not self._score_stop.is_set():
            started = time.monotonic()
            self._fight_once()

            remaining = FIGHT_INTERVAL_SECONDS - (time.monotonic() - started)
            if remaining > 0:
                self._score_stop.wait(remaining)

    def _fight_once(self) -> None:
        """Classifies the last second of video once, if a whole second is in hand."""
        with self._fight_frames_lock:
            window = list(self._fight_frames)

        if len(window) >= FIGHT_WINDOW_FRAMES:
            score, detail = self._score_fight(window)
            with self._scores_lock:
                self._fight_score = score
                self._fight_detail = detail

    def _score_fight(self, window: list) -> tuple:
        """One MoViNet window, with the same two-people guard the recorded path
        uses. Returns (score, detail) - detail is what a snapshot records."""
        frames = [frame for frame, _ in window]
        boxes = [frame_boxes for _, frame_boxes in window]
        try:
            score, crop = fight_det.fight_classifier(frames, boxes)
        except Exception as e:
            log.debug("fight scoring failed: %s", e)
            return 0.0, {}

        # A fight takes two. The recorded pipeline re-runs pose around the
        # flagged moment to count people; here every frame of the window already
        # has its boxes, so the same guard is free.
        people = max(len(frame_boxes) for frame_boxes in boxes)
        detail = {"model_score": round(score, 3), "crop": list(crop), "people_in_window": people,
                  "single_person_penalty_applied": people < FIGHT_MIN_PEOPLE}
        if people < FIGHT_MIN_PEOPLE:
            score *= fight_det.SINGLE_PERSON_PENALTY

        return score, detail

    # ------------------------------------------------------------------
    # Tracking - runs on every frame
    # ------------------------------------------------------------------

    def process(self, frame, now: float = None) -> list:
        """Takes one frame and returns everyone currently tracked, ready to draw.

        Each entry is {track_id, box, kpts_xy, kpts_conf, scores, signals,
        pose_change, action, confidence, checking, history} - what render.py
        needs to draw. `action` is a confirmed alarm only; `checking` is one still
        being watched - see CONFIRM_SECONDS. Confirmed alarms, with the picture
        they need, come from take_events().

        `now` is seconds since this engine started, and defaults to the wall
        clock - which is what live footage wants, since a frame really did
        arrive when it arrived. Pass it explicitly to replay a recording at its
        own frame rate: the detectors reason in seconds, so feeding a file
        faster than real time without it would compress every duration and make
        a two-second movement look like a flinch."""
        if now is None:
            now = time.monotonic() - self._started_at
        self._frame_height, self._frame_width = frame.shape[:2]

        self._frame_number += 1
        self._frames.add(self._frame_number, frame)

        detections = self._detect(frame)
        boxes = [det["box"] for det in detections]

        # Feed the fight window at ITS rate, not ours - MoViNet was trained on
        # 8 fps and a window has to be a real second of video. See
        # FIGHT_SAMPLE_FPS.
        #
        # On a fixed grid, 8 slots a second, like the ffmpeg fps=8 filter the
        # training clips went through. "At least 1/8s since the last sample"
        # would be wrong in a quiet way: at 15 fps it takes every other frame,
        # which is 7.5 a second, and a window becomes 1.07s of video.
        if now >= self._fight_next_sample:
            self._fight_next_sample = max(self._fight_next_sample + 1.0 / FIGHT_SAMPLE_FPS, now)
            with self._fight_frames_lock:
                self._fight_frames.append((frame, boxes))

        with self._tamper_lock:
            self._tamper.update(frame, boxes, now)

        with self._tracks_lock:
            self._match(detections, now)
            self._expire(now)
            self._trim(now)

        # Only score here when nothing is scoring in the background. start()
        # is the normal path; this keeps the engine usable synchronously, which
        # is what replaying a recording through it needs in order to be
        # repeatable.
        if self._score_thread is None and now - self._last_scored_at >= SCORE_INTERVAL_SECONDS:
            self._score_all(now)
            self._last_scored_at = now
        if self._fight_thread is None and now - self._fight_last_scored >= FIGHT_INTERVAL_SECONDS:
            self._fight_once()
            self._fight_last_scored = now

        return self.people()

    def _detect(self, frame) -> list:
        """Runs the pose model and keeps the people we can actually measure.

        Uses the same usable-pose gate as the recorded pipeline, so a person too
        close to be measured is skipped here exactly as there - showing a
        confident label built on unusable geometry would be worse than showing
        nothing."""
        result = self.person_model(frame, verbose=False)[0]

        detections = []
        for index, box in enumerate(result.boxes):
            if int(box.cls) != PERSON_CLASS_ID:
                continue
            det = {
                "box": box.xyxy[0].tolist(),
                "kpts_xy": result.keypoints.xy[index].cpu().numpy(),
                "kpts_conf": result.keypoints.conf[index].cpu().numpy(),
            }
            det["rejected"] = pose_rejection_reason(det, self._frame_height)
            detections.append(det)

        return detections

    def _match(self, detections, now: float) -> None:
        """Assigns each detection to the person it most overlaps, or a new id.

        The identities survive between frames indefinitely, which is the whole
        point of doing this live: the recorded pipeline throws them away at
        every 10-second boundary."""
        usable = [det for det in detections if det["rejected"] is None]
        assigned = set()

        for track_id, previous_box in list(self._last_box.items()):
            best_index, best_overlap = None, IOU_MATCH_MIN
            for index, det in enumerate(usable):
                if index in assigned:
                    continue
                overlap = iou(previous_box, det["box"])
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap

            if best_index is not None:
                assigned.add(best_index)
                self._append(track_id, usable[best_index], now)

        for index, det in enumerate(usable):
            if index not in assigned:
                track_id = self._next_id
                self._next_id += 1
                self._append(track_id, det, now)

    def _append(self, track_id, det: dict, now: float) -> None:
        """Adds one frame's detection to a person's history.

        The record shape is the one the detectors expect - identical to
        app/shared/pose_tracking._append_record, because their compute_kinematics
        functions read these exact keys."""
        x1, y1, x2, y2 = det["box"]
        self._tracks.setdefault(track_id, deque()).append({
            "t": now,
            "frame": self._frame_number,
            "box": det["box"],
            "box_height": y2 - y1,
            "box_width": x2 - x1,
            "kpts_xy": det["kpts_xy"],
            "kpts_conf": det["kpts_conf"],
        })
        self._last_box[track_id] = det["box"]
        self._last_seen[track_id] = now

    def _expire(self, now: float) -> None:
        """Forgets anyone who has been out of shot long enough to be a stranger."""
        gone = [
            track_id for track_id, seen in self._last_seen.items()
            if now - seen > TRACK_EXPIRY_SECONDS
        ]
        for track_id in gone:
            self._tracks.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._last_box.pop(track_id, None)
            with self._scores_lock:
                self._scores.pop(track_id, None)
                self._signals.pop(track_id, None)
                self._changes.pop(track_id, None)

    def _trim(self, now: float) -> None:
        """Drops history older than the window, so memory stays flat over days."""
        cutoff = now - self.window_seconds
        for records in self._tracks.values():
            while records and records[0]["t"] < cutoff:
                records.popleft()

    # ------------------------------------------------------------------
    # Scoring - runs on its own slower clock
    # ------------------------------------------------------------------

    def _score_all(self, now: float) -> None:
        """Re-scores every tracked person against the pose detectors in ACTIONS.

        Takes a snapshot of every track first and releases the lock before doing
        any real work: scoring three detectors takes tens of milliseconds, and
        holding the lock for that would stall the capture loop and cost exactly
        the frame rate this thread exists to protect. The copy is shallow - the
        per-frame records are never mutated once appended, only added and
        dropped - so it costs a list of references, not the keypoint arrays."""
        with self._tracks_lock:
            snapshot = {
                track_id: list(records)
                for track_id, records in self._tracks.items()
                if len(records) >= MIN_RECORDS_TO_SCORE
            }
            all_tracks = {track_id: list(records) for track_id, records in self._tracks.items()}
            last_seen = dict(self._last_seen)

        scores, signals = {}, {}
        changes = {track_id: _pose_change(window) for track_id, window in snapshot.items()}

        # The fall model scores everyone moving in one batch, ahead of the loop.
        moving = {track_id: window for track_id, window in snapshot.items()
                  if changes[track_id] >= MIN_POSE_CHANGE}
        fall_model_scores = self._fall_model_scores(moving)

        for track_id, window in snapshot.items():
            # Nothing in this window but somebody holding still. See the change
            # gate above: there is no event to find, so don't go looking - and
            # report zero rather than the last score, because "not moving" is
            # the truthful answer, not a stale one kept alive.
            if track_id not in moving:
                scores[track_id] = {action: 0.0 for action in ACTIONS}
                signals[track_id] = {}
                continue

            others = {
                other_id: records
                for other_id, records in all_tracks.items()
                if other_id != track_id
            }
            scored = {
                # "spit": self._score_one(spit_det, window, others, now),  # spit turned off
                "vandalism": self._score_one(vandalism_det, window, others, now),
                "fall": self._score_fall(window, others, now, fall_model_scores.get(track_id)),
            }
            scores[track_id] = {action: value for action, (value, _) in scored.items()}
            signals[track_id] = {action: detail for action, (_, detail) in scored.items()}

        with self._scores_lock:
            self._scores = scores
            self._signals = signals
            self._changes = changes
            self._update_events(scores, signals, changes, snapshot, last_seen, now)

    def _update_events(self, scores, signals, changes, windows, last_seen, now: float) -> None:
        """Moves each (person, action) from candidate to alarm, or drops it.

        See CONFIRM_SECONDS. A candidate starts at DISPLAY_THRESHOLD, is cancelled
        if its score falls below HOLD_THRESHOLD while the person is still in view,
        and becomes an alarm once it has held for CONFIRM_SECONDS. An alarm lasts
        until its score falls below HOLD_THRESHOLD again.

        What gets remembered for the picture is the candidate's best moment, not
        the moment it was confirmed: by then the spit is over and the person is
        walking away, and a picture of that proves nothing.

        Runs under _scores_lock, so it must not take _tracks_lock."""
        for track_id, person_scores in scores.items():
            in_view = now - last_seen.get(track_id, now) <= LEFT_VIEW_SECONDS

            for action, score in person_scores.items():
                key = (track_id, action)

                if key in self._confirmed:
                    if in_view and score < HOLD_THRESHOLD:
                        del self._confirmed[key]        # the event is over
                    continue

                candidate = self._candidates.get(key)
                if candidate is None:
                    if score >= DISPLAY_THRESHOLD and in_view:
                        candidate = self._candidates[key] = {"since": now, "confidence": 0.0,
                                                             "lowest": score, "left_view": False}
                    else:
                        continue
                elif not in_view:
                    # Out of view: the score now describes a frozen history, not
                    # the person, so it is neither held against them nor kept.
                    candidate["left_view"] = True
                    continue
                elif score < HOLD_THRESHOLD:
                    del self._candidates[key]           # it did not hold - not what it looked like
                    continue

                candidate["lowest"] = min(candidate["lowest"], score)
                if score > candidate["confidence"]:
                    self._remember_peak(candidate, track_id, action, score,
                                        scores, signals, changes, windows[track_id])

        # Confirm whatever has held long enough - including anyone the tracker has
        # since lost, who is judged on what was seen of them.
        for key, candidate in list(self._candidates.items()):
            if key[0] not in last_seen:
                candidate["left_view"] = True
            if now - candidate["since"] < CONFIRM_SECONDS:
                continue

            del self._candidates[key]
            if key[0] in last_seen:
                self._confirmed[key] = candidate["confidence"]
            self._new_events.append({
                **candidate["peak"],
                "held_for": round(now - candidate["since"], 2),
                "lowest_while_held": round(candidate["lowest"], 3),
                "left_view": candidate["left_view"],
            })

        for key in [key for key in self._confirmed if key[0] not in last_seen]:
            del self._confirmed[key]

    def _remember_peak(self, candidate, track_id, action, score,
                       scores, signals, changes, window) -> None:
        """Keeps what the picture of this candidate needs, taken at its best moment."""
        latest = window[-1]
        candidate["confidence"] = score
        candidate["peak"] = {
            "track_id": track_id,
            "action": action,
            "confidence": score,
            "scores": dict(scores[track_id]),
            "signals": (signals.get(track_id) or {}).get(action) or {},
            "pose_change": changes.get(track_id, 0.0),
            "history": len(window),
            "box": latest["box"],
            "kpts_xy": latest["kpts_xy"],
            "kpts_conf": latest["kpts_conf"],
            # The clean frame this person was last seen in. A reference, not a
            # copy - frames in the buffer are never drawn on.
            "frame": self._frames.get(latest["frame"]),
            "seen_at": time.time(),
            "t": latest["t"],       # on the engine's clock - in a replay, seconds into the clip
        }

    def take_events(self) -> list:
        """Alarms confirmed since the last call, each with its best moment attached.

        Must be drained regularly even when nothing is saved, or the list grows."""
        with self._scores_lock:
            events, self._new_events = self._new_events, []
        return events

    def _score_one(self, detector, window: list, others: dict, now: float) -> tuple:
        """Runs one detector over one person's window, returning (confidence, signals).

        The signals are the detector's own breakdown of how it got there -
        swing_deg, strike_count, which penalties fired. They cost nothing to keep
        and they are the difference between a snapshot you can diagnose and a
        snapshot you can only look at: without them, a false positive on disk
        says what was decided but never why.

        Every call is wrapped: these functions were written for complete clips
        with a definite end, and a rolling window is a shape they have not seen.
        A detector raising on some edge case should cost that one score, not
        take down the live view."""
        try:
            if detector is fall_det:
                # fall is the only one that needs to know when the window ends -
                # it checks whether someone is still down at that point. Live,
                # "the end" is now, which is exactly the question worth asking.
                features = detector.compute_kinematics(window, now)
            else:
                features = detector.compute_kinematics(window)

            if features is None:
                return 0.0, {}

            if detector is fall_det:
                confidence, detail = detector.score_event(
                    features, self._frame_width, self._frame_height, others
                )
            else:
                # The filepath is ignored - the frame adapter serves these from
                # memory. Passing a name rather than "" keeps any log line that
                # mentions it readable.
                confidence, detail = detector.score_event("<live>", features, others)

            return float(confidence), detail

        except Exception as e:
            log.debug("%s scoring failed: %s", detector.__name__.rsplit(".", 1)[-1], e)
            return 0.0, {}

    @staticmethod
    def _fall_model_scores(windows: dict) -> dict:
        """The ST-GCN's answer for everyone in `windows`, as {track_id: (score, signals)}.
        Wrapped like _score_one: a model error costs the fall score, not the view."""
        if not windows:
            return {}
        try:
            return fall_model.score_people(windows)
        except Exception as e:
            log.debug("fall model scoring failed: %s", e)
            return {}

    def _score_fall(self, window: list, others: dict, now: float, model_result) -> tuple:
        """The rules' fall score and the model's, and the lower of the two - see
        "fall: two judges" above. Both are kept in the signals, so a snapshot shows
        which one held it back."""
        rules, detail = self._score_one(fall_det, window, others, now)
        model, model_detail = model_result or (0.0, {})
        detail = {**detail, "rules_score": round(rules, 3), "model_score": round(model, 3),
                  "model": model_detail}
        return min(rules, model), detail

    # ------------------------------------------------------------------
    # What to draw
    # ------------------------------------------------------------------

    def fight(self) -> float:
        """How much the last second of video looks like a fight, 0 to 1.

        Scene-level rather than per-person: MoViNet classifies the people
        together, so unlike the other three this does not belong to anyone in
        particular."""
        with self._scores_lock:
            return self._fight_score

    def fight_detail(self) -> dict:
        """How the last fight score was reached: the crop the model looked at, how
        many people were in the window, whether the two-person guard applied."""
        with self._scores_lock:
            return dict(self._fight_detail)

    def tamper(self) -> dict:
        """Whether the camera itself looks tampered with: {kind, confidence,
        signals}, kind being covered / blurred / moved, or None."""
        with self._tamper_lock:
            return self._tamper.state()

    def people(self) -> list:
        """Everyone currently tracked, with their latest scores.

        Takes the two locks one after the other rather than one inside the
        other. _expire holds the track lock while dropping a score, so nesting
        them the other way round here is the classic recipe for a deadlock."""
        with self._scores_lock:
            scores = dict(self._scores)
            signals = dict(self._signals)
            changes = dict(self._changes)
            confirmed = dict(self._confirmed)
            checking = set(self._candidates)

        with self._tracks_lock:
            snapshot = [
                (track_id, records[-1], len(records))
                for track_id, records in self._tracks.items() if records
            ]

        people = []
        for track_id, latest, history in snapshot:
            person_scores = scores.get(track_id, {})

            # Only a confirmed alarm is an action - a candidate still being
            # watched is reported separately, as `checking`. See CONFIRM_SECONDS.
            alarms = {a: c for (tid, a), c in confirmed.items() if tid == track_id}
            action, confidence = max(alarms.items(), key=lambda item: item[1]) if alarms else (None, 0.0)
            watched = [a for (tid, a) in checking if tid == track_id and a not in alarms]
            watched.sort(key=lambda a: -person_scores.get(a, 0.0))

            people.append({
                "track_id": track_id,
                "box": latest["box"],
                "kpts_xy": latest["kpts_xy"],
                "kpts_conf": latest["kpts_conf"],
                "scores": person_scores,
                "signals": signals.get(track_id, {}),
                "pose_change": changes.get(track_id, 0.0),
                "action": action,
                "confidence": confidence,
                "checking": watched[0] if watched else None,
                "history": history,
            })

        return people
