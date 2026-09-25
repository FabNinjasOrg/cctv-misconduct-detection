# Plays a clip back with the vandalism-detection pose skeleton and confidence overlaid
# live, synced to playback time. Runs the existing object_action_detector pipeline once
# up front (no changes to its scoring) and keeps every frame's keypoints, not just the
# top evidence frames, then redraws them as the clip plays. See spit/playback.py and
# fight/playback.py for the other action counterparts.

import itertools
import sys

import cv2
from ultralytics import YOLO

from motion_filter import detect_motion, group_motion_segments
from vandalism.object_action_detector import (
    CONF_FLOOR,
    SKELETON_EDGES,
    compute_kinematics,
    score_event,
    track_segment,
)

WINDOW_NAME = "Vandalism Detection - Playback"


def analyze_for_playback(filepath: str, person_model) -> dict:
    """Runs the vandalism-detection pipeline once and returns frame_number -> list of
    tracked people in that frame, each with their keypoints and their track's confidence."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    motion_events = detect_motion(filepath)
    segments = group_motion_segments(motion_events, fps, frame_count)

    by_frame: dict = {}
    for segment in segments:
        tracks = track_segment(filepath, segment, person_model, fps)
        for track_id, records in tracks.items():
            features = compute_kinematics(records)
            if features is None:
                continue
            confidence, signals = score_event(filepath, features)
            for record in records:
                by_frame.setdefault(record["frame"], []).append({
                    "box": record["box"],
                    "kpts_xy": record["kpts_xy"],
                    "kpts_conf": record["kpts_conf"],
                    "confidence": confidence,
                    "action": signals["action"],
                })

    return by_frame


def _draw_person(frame, person: dict) -> None:
    x1, y1, x2, y2 = [int(v) for v in person["box"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    keypoints, kpts_conf = person["kpts_xy"], person["kpts_conf"]
    for a, b in SKELETON_EDGES:
        if kpts_conf[a] >= CONF_FLOOR and kpts_conf[b] >= CONF_FLOOR:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pa, pb, (0, 255, 255), 2)
    for idx, (x, y) in enumerate(keypoints):
        if kpts_conf[idx] >= CONF_FLOOR:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    label = f"Vandalism {person['confidence']:.0%} ({person['action']})"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def play_with_overlay(filepath: str, person_model) -> None:
    by_frame = analyze_for_playback(filepath, person_model)

    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, round(1000 / fps))

    print("Analysis done, playing back. Press 'q' in the window to quit.")
    for frame_number in itertools.count():
        ok, frame = video.read()
        if not ok:
            break

        for person in by_frame.get(frame_number, []):
            _draw_person(frame, person)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -m vandalism.playback <path-to-clip>")

    person_model = YOLO("models/yolo11n-pose.pt")
    play_with_overlay(sys.argv[1], person_model)
