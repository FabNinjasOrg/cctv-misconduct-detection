import os

import cv2

from fall.kinematic_detector import detect_fall
from motion_filter import detect_motion, group_motion_segments
from fight.movinet_detector import detect_fighting_mil
from pose_tracking import track_clip
from spit.kinematic_detector import detect_activity
from vandalism.object_action_detector import detect_vandalism

FINAL_FRAMES_DIR = "analyze_action/evidence"
os.makedirs(FINAL_FRAMES_DIR, exist_ok=True)

BANNER_HEIGHT = 40
BANNER_COLOR = (0, 128, 0)          # green, distinct from each detector's own red box/label
BANNER_TEXT_COLOR = (255, 255, 255)


def analyze_action(filepath: str, person_model) -> dict:
    """Runs each action's primary detector on the clip (spit/kinematic,
    vandalism/object-action, fall/kinematic, fight/movinet) and reduces the four
    results to a single combined verdict: the highest-confidence activity found.
    Reuses whichever winning detector's own evidence jpg as-is (it already has a
    box/border + label burned in - see each detector's save_evidence_frame) and
    only adds one banner on top, so no detector file needs to change.

    Motion detection now runs once, here, as the single stage-1 gate for all four
    detectors (their own internal motion filters are commented out - see each
    file's detect_* function) instead of each detector re-running it separately."""
    if not _has_motion(filepath):
        no_motion_scores = {
            action: {"detected": False, "confidence": 0.0}
            for action in ("spit", "vandalism", "fall", "fight")
        }
        return {
            "activity_detected": False,
            "action": None,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "final_frame": None,
            "motion_detected": False,
            "scores": no_motion_scores,
        }

    # One pose pass for all three pose-based detectors. They ask for exactly the
    # same thing - YOLO11-pose at the same sampling rate, matched into tracks the
    # same way - so running it per detector meant putting every sampled frame
    # through the same model three times for an identical answer, which was most
    # of the cost of a combined verdict. Fight is unaffected: MoViNet classifies
    # raw frames and never touches these tracks.
    _, _, tracks_by_segment = track_clip(filepath, person_model)

    spit_result = detect_activity(filepath, person_model, tracks_by_segment)
    vandalism_result = detect_vandalism(filepath, person_model, tracks_by_segment)
    fall_result = detect_fall(filepath, person_model, tracks_by_segment)
    fight_result = detect_fighting_mil(filepath, person_model)

    candidates = [
        _best_spit(spit_result),
        _best_single_frame("vandalism", vandalism_result, "vandalism_detected"),
        _best_single_frame("fall", fall_result, "fall_detected"),
        _best_single_frame("fight", fight_result, "fighting_detected"),
    ]

    scores = {
        c["action"]: {"detected": c["detected"], "confidence": round(c["confidence"], 3)}
        for c in candidates
    }

    detected = [c for c in candidates if c["detected"]]
    winner = max(detected, key=lambda c: c["confidence"], default=None)

    final_frame = _save_final_frame(filepath, winner) if winner and winner["frame_path"] else None

    return {
        "activity_detected": winner is not None,
        "action": winner["action"] if winner else None,
        "confidence": round(winner["confidence"], 3) if winner else 0.0,
        "confidence_percent": round(winner["confidence"] * 100, 1) if winner else 0.0,
        "final_frame": final_frame,
        "motion_detected": True,
        "scores": scores,
    }


def _has_motion(filepath: str) -> bool:
    """The single stage-1 gate for the whole endpoint: if nothing in the clip ever
    moves, none of the four (expensive) detectors are worth running at all."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    motion_events = detect_motion(filepath)
    segments = group_motion_segments(motion_events, fps, frame_count)
    return len(segments) > 0


def _best_spit(result: dict) -> dict:
    """Spit/kinematic nests its evidence a level deeper than the others: each
    confirmed candidate holds a list of "frames", not a single frame_path."""
    candidates = result.get("evidence_frames") or []
    if not candidates:
        return {"action": "spit", "detected": False, "confidence": result.get("confidence", 0.0), "frame_path": None}

    best = max(candidates, key=lambda c: c["confidence"])
    frame_path = best["frames"][0]["frame_path"] if best.get("frames") else None
    return {
        "action": "spit",
        "detected": bool(result.get("spitting_detected")),
        "confidence": best["confidence"],
        "frame_path": frame_path,
    }


def _best_single_frame(action: str, result: dict, detected_key: str) -> dict:
    """Shared shape for vandalism, fall, and fight/movinet - each confirmed
    candidate already carries one "frame_path" directly."""
    candidates = result.get("evidence_frames") or []
    if not candidates:
        return {"action": action, "detected": False, "confidence": result.get("confidence", 0.0), "frame_path": None}

    best = max(candidates, key=lambda c: c["confidence"])
    return {
        "action": action,
        "detected": bool(result.get(detected_key)),
        "confidence": best["confidence"],
        "frame_path": best.get("frame_path"),
    }


def _save_final_frame(filepath: str, winner: dict) -> str | None:
    """Loads the winning detector's own already-boxed/labeled evidence jpg and adds
    one banner strip on top naming the combined verdict, so it reads as the
    pipeline's final answer rather than any single detector's raw output."""
    frame = cv2.imread(winner["frame_path"])
    if frame is None:
        return None

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, BANNER_HEIGHT), BANNER_COLOR, -1)
    label = f"{winner['action'].capitalize()} Detected - {winner['confidence'] * 100:.0f}%"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BANNER_TEXT_COLOR, 2)

    stem = os.path.splitext(os.path.basename(filepath))[0]
    pct = int(winner["confidence"] * 100)
    frame_path = os.path.join(FINAL_FRAMES_DIR, f"{stem}_{winner['action']}_{pct}pct.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path
