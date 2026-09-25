import os

import cv2
from ultralytics import YOLO

from app.shared.motion_filter import detect_motion, group_motion_segments
from app.actions.fight.person_guard import SINGLE_PERSON_PENALTY, max_people_nearby
from app.shared.gemini_review import review_clip

FIGHT_MODEL_PATH = "models/fight_yolov8n.pt"
VIOLENCE_CLASS_NAME = "violence"
CONFIDENCE_MIN = 0.5   # ignore low-confidence guesses
CHECKS_PER_SECOND = 4.0
TOP_FRAMES_LIMIT = 1   # only the single highest-confidence hit gets saved as evidence

FIGHT_QUESTION = (
    "You are reviewing a short CCTV clip flagged by an automated detector as a possible physical "
    "fight. Most flags are false positives - ordinary shoving in a crowd, playful roughhousing, "
    "sports, hugging, or people just walking close together. Only answer 'yes' if the clip clearly "
    "shows sustained, hostile physical aggression between people (real hitting, grappling, kicking) "
    "- not a single ambiguous or incidental moment of contact. Answer starting with exactly 'yes' "
    "or 'no', then a short reason."
)

FIGHT_FRAMES_DIR = "evidence/fight/yolo"
os.makedirs(FIGHT_FRAMES_DIR, exist_ok=True)

fight_model = YOLO(FIGHT_MODEL_PATH)


def detect_fighting(filepath: str, person_model) -> dict:
    """Motion-gates the clip first (see motion_filter.py) so the fine-tuned
    YOLOv8 model only scans frames within segments that have actual movement -
    the classifier itself is unchanged, it just never sees the static stretches.
    person_model (shared YOLO11-pose) is used only for the single-person guard
    below, checking whether the flagged moment actually involved two people."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    frame_skip = max(1, round(fps / CHECKS_PER_SECOND))
    motion_events = detect_motion(filepath)
    segments = group_motion_segments(motion_events, fps, frame_count)

    frames_checked = 0
    best_confidence = 0.0
    top_matches = []  # up to TOP_FRAMES_LIMIT of {t, frame_number, confidence, box, frame}, highest confidence first

    video = cv2.VideoCapture(filepath)
    for segment in segments:
        video.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])
        frame_number = segment["start_frame"]
        while frame_number <= segment["end_frame"]:
            ok, frame = video.read()
            if not ok:
                break

            if (frame_number - segment["start_frame"]) % frame_skip == 0:
                frames_checked += 1
                result = fight_model(frame, verbose=False)[0]

                for box in result.boxes:
                    if fight_model.names[int(box.cls)] != VIOLENCE_CLASS_NAME:
                        continue
                    confidence = float(box.conf)
                    best_confidence = max(best_confidence, confidence)
                    if confidence < CONFIDENCE_MIN:
                        continue

                    match = {
                        "t": round(frame_number / fps, 2),
                        "frame_number": frame_number,
                        "confidence": confidence,
                        "box": box.xyxy[0].tolist(),
                        "frame": frame,
                    }
                    top_matches.append(match)
                    top_matches.sort(key=lambda m: m["confidence"], reverse=True)
                    del top_matches[TOP_FRAMES_LIMIT:]

            frame_number += 1

    video.release()

    # --- a fight needs at least two people - if nobody else was ever visible near the
    # single strongest match, this is more likely a lone person's motion than a real
    # physical fight (see fight/person_guard.py) ---
    single_person = False
    if top_matches:
        peak = top_matches[0]
        people_nearby = max_people_nearby(filepath, peak["frame_number"], fps, person_model)
        single_person = people_nearby < 2
        if single_person:
            for match in top_matches:
                match["confidence"] *= SINGLE_PERSON_PENALTY
            best_confidence *= SINGLE_PERSON_PENALTY
            top_matches = [m for m in top_matches if m["confidence"] >= CONFIDENCE_MIN]

    confirmed = []
    for match in top_matches:
        # confirmed_by_gemini, reason = review_clip(filepath, match["frame_number"], fps, FIGHT_QUESTION)
        # if not confirmed_by_gemini:
        #     continue
        # match["gemini_reason"] = reason
        match["gemini_reason"] = 'none'
        confirmed.append(match)

    return {
        "fighting_detected": len(confirmed) > 0,
        "confidence": round(best_confidence, 3),
        "segments_checked": len(segments),
        "frames_checked": frames_checked,
        "single_person_penalty_applied": single_person,
        "fight_timestamps": [match["t"] for match in confirmed],
        "evidence_frames": [save_evidence_frame(match) for match in confirmed],
    }


def save_evidence_frame(match: dict) -> dict:
    """Draws the detected box + confidence label on the frame and saves it under FIGHT_FRAMES_DIR."""
    x1, y1, x2, y2 = [int(v) for v in match["box"]]
    outlined_frame = match["frame"].copy()
    cv2.rectangle(outlined_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    label = f"Fight {match['confidence']:.0%} | t={match['t']}s"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(outlined_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_path = os.path.join(FIGHT_FRAMES_DIR, f"fight_t{match['t']}.jpg")
    cv2.imwrite(frame_path, outlined_frame)

    return {
        "t": match["t"],
        "confidence": round(match["confidence"], 3),
        "frame_path": frame_path,
        "gemini_reason": match["gemini_reason"],
    }
