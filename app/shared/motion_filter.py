# Shared step-1 motion pre-filter, used by every action detector (spit, vandalism,
# fall, fight) - cheap OpenCV background subtraction that finds the segments of a
# clip where something actually moved, so the far more expensive per-frame/per-clip
# models (pose keypoints, fine-tuned classifiers, MoViNet, ...) only ever look at
# those segments instead of scanning every static frame.
#
# This single copy is the one source of truth for the algorithm. Right now it's
# only actually called from analyze_action/action_analyzer.py, which runs it once
# per clip and shares the result across all four detectors instead of each one
# re-running its own copy. Each detector file keeps a commented-out import of this
# module at its own motion-gating call site (see e.g. app/actions/spit/kinematic_detector.py)
# documenting where per-detector gating used to happen / could be re-enabled later.

import itertools

import cv2
import numpy as np

MOTION_CHECKS_PER_SECOND = 5.0
MOTION_WARMUP_SECONDS = 0.4
MOTION_THRESHOLD = 0.003

MAX_TICK_GAP_SECONDS = 0.6
SEGMENT_PAD_SECONDS = 0.6
MIN_SEGMENT_SECONDS = 0.5


def detect_motion(filepath: str) -> list:
    """Scans the clip and returns a list of timestamps where something moved."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, round(fps / MOTION_CHECKS_PER_SECOND))
    warmup_checks = max(1, int(MOTION_WARMUP_SECONDS * MOTION_CHECKS_PER_SECOND))

    background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    events = []
    checks_done = 0

    for frame_number in itertools.count():
        ok, frame = video.read()
        if not ok:
            break
        if frame_number % frame_skip:
            continue

        checks_done += 1

        small_frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        motion_mask = background.apply(gray_frame)
        _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
        motion_level = float(np.count_nonzero(motion_mask)) / motion_mask.size

        if checks_done <= warmup_checks:
            continue

        if motion_level > MOTION_THRESHOLD:
            events.append({"t": round(frame_number / fps, 2), "frame": frame_number, "level": motion_level})

    video.release()
    return events


def group_motion_segments(events: list, fps: float, frame_count: int) -> list:
    """Groups nearby motion timestamps into single time windows (segments) to analyze."""
    if not events:
        return []

    events = sorted(events, key=lambda e: e["t"])
    runs, current = [], [events[0]]
    for prev, cur in zip(events, events[1:]):
        if cur["t"] - prev["t"] > MAX_TICK_GAP_SECONDS:
            runs.append(current)
            current = []
        current.append(cur)
    runs.append(current)

    clip_duration = frame_count / fps
    segments = []
    for run in runs:
        start_t = max(0.0, run[0]["t"] - SEGMENT_PAD_SECONDS)
        end_t = min(clip_duration, run[-1]["t"] + SEGMENT_PAD_SECONDS)
        if end_t - start_t >= MIN_SEGMENT_SECONDS:
            segments.append({
                "start_t": start_t,
                "end_t": end_t,
                "start_frame": int(start_t * fps),
                "end_frame": int(end_t * fps),
            })

    segments.sort(key=lambda s: s["start_t"])
    merged = []
    for seg in segments:
        if merged and seg["start_t"] <= merged[-1]["end_t"]:
            merged[-1]["end_t"] = max(merged[-1]["end_t"], seg["end_t"])
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], seg["end_frame"])
        else:
            merged.append(seg)
    return merged
