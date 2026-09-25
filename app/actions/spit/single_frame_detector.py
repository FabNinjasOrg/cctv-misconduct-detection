import itertools
import os

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6  # keypoint indices used to score head-bow/tilt
LEFT_WRIST, RIGHT_WRIST = 9, 10                # keypoint indices used to score hand-near-mouth

OBJECT_CONFIDENCE_MIN = 0.6  # a small, blurry head crop makes the object detector guess a lot - ignore weak guesses
REVIEW_LIMIT = 2             # how many highest-doubt-score events to save as evidence

SPIT_FRAMES_DIR = "evidence/spit/single_frame"
os.makedirs(SPIT_FRAMES_DIR, exist_ok=True)

object_model = YOLO("models/yolo11n.pt")  # general object detector, checked near the head


def detect_spitting(filepath: str, person_model) -> dict:
    events = detect_motion(filepath)
    events = detect_person(filepath, events, person_model)
    person_event_count = len(events)

    doubtful_events = save_doubtful_events(filepath, events)

    return {
        "person_event_count": person_event_count,
        "events": [
            {
                "t": event["t"],
                "confidence": round(event["doubt_score"], 3),
                "review_frame": event["review_frame"],
            }
            for event in doubtful_events
        ],
    }


def detect_motion(filepath: str, checks_per_second: float = 5.0) -> list:
    """Step 1: scan the clip with OpenCV, return timestamps where something moved."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, round(fps / checks_per_second))
    warmup_checks = int(2 * checks_per_second) # let the background model learn the scene first

    motion_threshold = 0.003  # low enough to catch a brief pause (like stopping to spit), not just a full walk

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
        _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)  # drop shadows
        motion_level = float(np.count_nonzero(motion_mask)) / motion_mask.size

        if checks_done <= warmup_checks:
            continue

        if motion_level > motion_threshold:
            events.append({"t": round(frame_number / fps, 2), "frame": frame_number})

    video.release()
    return events


def detect_person(filepath: str, events: list, person_model) -> list:
    """Step 2: for each motion event, keep it only if YOLO finds a real person there,
    and score how doubtful their posture looks - bowed forward, tilted sideways
    (either direction), a hand near the mouth, or any object near the head."""
    video = cv2.VideoCapture(filepath)
    confirmed_events = []

    for event in events:
        video.set(cv2.CAP_PROP_POS_FRAMES, event["frame"])
        ok, frame = video.read()
        if not ok:
            continue

        result = person_model(frame, verbose=False)[0]
        person_indices = [i for i, box in enumerate(result.boxes) if int(box.cls) == PERSON_CLASS_ID]
        if not person_indices:
            continue

        person_index = person_indices[0]
        box = result.boxes[person_index].xyxy[0].tolist()
        keypoints = result.keypoints.xy[person_index]
        x1, y1, x2, y2 = box
        box_height = y2 - y1

        nose_point = keypoints[NOSE]
        left_shoulder, right_shoulder = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]

        # a level pair of shoulders sits at the same height - a sideways tilt (either
        # direction) lifts one shoulder above the other in the image
        side_tilt = abs(float(left_shoulder[1] - right_shoulder[1])) / box_height if box_height else 0.0

        wrist_points = [keypoints[LEFT_WRIST], keypoints[RIGHT_WRIST]]
        wrist_to_nose = min(float(np.linalg.norm(wrist - nose_point)) for wrist in wrist_points)
        hand_near_mouth = wrist_to_nose / box_height if box_height else 1.0  # lower = hand closer to the mouth

        event["box"] = box
        event["box_height"] = box_height
        event["side_tilt"] = side_tilt
        event["hand_near_mouth"] = hand_near_mouth
        event["object_near_head"] = find_object_near_head(frame, box, box_height)
        confirmed_events.append(event)

    video.release()

    if confirmed_events:
        standing_height = max(event["box_height"] for event in confirmed_events)
        for event in confirmed_events:
            event["head_bow"] = 1 - (event["box_height"] / standing_height)  # higher = more bowed over
            event["doubt_score"] = (
                event["head_bow"]
                + event["side_tilt"]
                - event["hand_near_mouth"]
                + (1.0 if event["object_near_head"] else 0.0)
            )

    return confirmed_events


def find_object_near_head(frame, box, box_height):
    """Runs a general object detector on just the head area - any non-person object
    found there (a bottle, a cup, a phone, ...) is returned by name, else None."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1) - 40), max(0, int(y1))
    x2, y2 = min(width, int(x2) + 40), min(height, int(y1 + box_height * 0.35))
    head_region = frame[y1:y2, x1:x2]
    if head_region.size == 0:
        return None

    result = object_model(head_region, verbose=False)[0]
    for detected in result.boxes:
        if int(detected.cls) != PERSON_CLASS_ID and float(detected.conf) >= OBJECT_CONFIDENCE_MIN:
            return object_model.names[int(detected.cls)]
    return None


def save_doubtful_events(filepath: str, events: list) -> list:
    """Step 3: save the top-REVIEW_LIMIT most doubtful events (bowed head, tilted
    shoulders, hand near mouth, object near head) as evidence frames for review."""
    doubtful_events = sorted(events, key=lambda event: event["doubt_score"], reverse=True)[:REVIEW_LIMIT]

    for event in doubtful_events:
        event["review_frame"] = save_review_frame(filepath, event)

    return doubtful_events


def save_review_frame(filepath: str, event: dict) -> str:
    """Draws the person's box on the event's frame and saves it under SPIT_FRAMES_DIR."""
    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, event["frame"])
    ok, frame = video.read()
    video.release()
    if not ok:
        return None

    x1, y1, x2, y2 = [int(v) for v in event["box"]]
    outlined_frame = frame.copy()
    cv2.rectangle(outlined_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    confidence_pct = max(0.0, min(1.0, event["doubt_score"]))  # doubt_score is an unbounded heuristic, clamp for display
    label = f"Spit {confidence_pct:.0%} | t={event['t']}s"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(outlined_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    review_path = os.path.join(SPIT_FRAMES_DIR, f"t{event['t']}.jpg")
    cv2.imwrite(review_path, outlined_frame)
    return review_path
