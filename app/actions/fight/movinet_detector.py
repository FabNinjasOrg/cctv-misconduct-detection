import math
import os

import cv2

# from app.shared.motion_filter import detect_motion, group_motion_segments  # motion filter now runs once in analyze_action/action_analyzer.py
from app.actions.fight.person_guard import SINGLE_PERSON_PENALTY, max_people_nearby
from app.actions.fight.vd_mil.movinet_classifier import MovinetViolenceClassifier

BACKBONE_WEIGHTS_DIR = "models/movinet/backbone"
HEAD_WEIGHTS_PATH = "models/movinet/model_20_state_dict.pt"  # CC BY-NC 4.0, internal/research use only - see models/movinet/NOTICE.md

TARGET_FPS = 8.0        # the model was trained on clips sampled at this rate
WINDOW_FRAMES = 8       # frames per classified clip
STRIDE_FRAMES = 4       # hop between consecutive windows
CONFIDENCE_MIN = 0.5    # ignore low-confidence guesses
TOP_FRAMES_LIMIT = 1    # only the single highest-confidence window gets saved as evidence

FIGHT_MIL_FRAMES_DIR = "debug/fight/movinet"

violence_classifier = MovinetViolenceClassifier(BACKBONE_WEIGHTS_DIR, HEAD_WEIGHTS_PATH)


def detect_fighting_mil(filepath: str, person_model) -> dict:
    """Motion-gates the clip first (see motion_filter.py) so the MoViNet clip
    classifier only runs its sliding window within segments that have actual movement -
    the classifier itself is unchanged. The window buffer resets at the start of each
    segment since frames on either side of a motion gap aren't temporally contiguous.
    person_model (shared YOLO11-pose) is used only for the single-person guard below,
    checking whether the flagged moment actually involved two people."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    frame_skip = max(1, round(fps / TARGET_FPS))
    # motion_events = detect_motion(filepath)  # motion filter now runs once in analyze_action/action_analyzer.py
    # segments = group_motion_segments(motion_events, fps, frame_count)
    segments = [{"start_t": 0.0, "end_t": frame_count / fps, "start_frame": 0, "end_frame": frame_count - 1}]

    windows_checked = 0
    best_confidence = 0.0
    top_matches = []  # up to TOP_FRAMES_LIMIT of {t, confidence, frame}, highest confidence first

    video = cv2.VideoCapture(filepath)
    for segment in segments:
        video.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])
        window = []  # sliding buffer of {"frame", "t"}, length capped at WINDOW_FRAMES
        frame_number = segment["start_frame"]
        while frame_number <= segment["end_frame"]:
            ok, frame = video.read()
            if not ok:
                break

            if (frame_number - segment["start_frame"]) % frame_skip == 0:
                window.append({"frame": frame, "t": round(frame_number / fps, 2)})
                if len(window) >= WINDOW_FRAMES:
                    windows_checked += 1
                    confidence = float(violence_classifier([[item["frame"] for item in window]])[0])
                    best_confidence = max(best_confidence, confidence)
                    center = window[WINDOW_FRAMES // 2]

                    if confidence >= CONFIDENCE_MIN:
                        top_matches.append({"t": center["t"], "confidence": confidence, "frame": center["frame"]})
                        top_matches.sort(key=lambda match: match["confidence"], reverse=True)
                        del top_matches[TOP_FRAMES_LIMIT:]

                    window = window[STRIDE_FRAMES:]

            frame_number += 1

    video.release()

    # --- a fight needs at least two people - if nobody else was ever visible near the
    # single strongest match, this is more likely a lone person's motion than a real
    # physical fight (see fight/person_guard.py) ---
    single_person = False
    if top_matches:
        peak = top_matches[0]
        people_nearby = max_people_nearby(filepath, round(peak["t"] * fps), fps, person_model)
        single_person = people_nearby < 2
        if single_person:
            for match in top_matches:
                match["confidence"] *= SINGLE_PERSON_PENALTY
            best_confidence *= SINGLE_PERSON_PENALTY
            top_matches = [m for m in top_matches if m["confidence"] >= CONFIDENCE_MIN]

    return {
        "fighting_detected": len(top_matches) > 0,
        "confidence": truncate_2dp(best_confidence),
        "segments_checked": len(segments),
        "windows_checked": windows_checked,
        "single_person_penalty_applied": single_person,
        "fight_timestamps": [m["t"] for m in top_matches],
        "evidence_frames": [save_evidence_frame(match) for match in top_matches],
    }


def truncate_2dp(value: float) -> float:
    """Truncates (not rounds) to 2 decimal places, so a near-1.0 score like 0.9997
    reads as 0.99 instead of inflating to a misleading 1.0."""
    return math.floor(value * 100) / 100


def save_evidence_frame(match: dict) -> dict:
    """Draws a red border around the frame (this model scores the whole clip, not a
    region, so there's no box to localize) + confidence label, saved under FIGHT_MIL_FRAMES_DIR."""
    outlined_frame = match["frame"].copy()
    height, width = outlined_frame.shape[:2]
    cv2.rectangle(outlined_frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)

    confidence = truncate_2dp(match["confidence"])
    label = f"Fight {confidence:.2f} | t={match['t']}s"
    cv2.putText(outlined_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    os.makedirs(FIGHT_MIL_FRAMES_DIR, exist_ok=True)  # only when there is something to save
    frame_path = os.path.join(FIGHT_MIL_FRAMES_DIR, f"fight_mil_t{match['t']}.jpg")
    cv2.imwrite(frame_path, outlined_frame)

    return {"t": match["t"], "confidence": confidence, "frame_path": frame_path}
