# Shared "does this actually involve two people" guard for both fight detectors. A
# physical fight needs at least two people in frame, but neither the fine-tuned YOLOv8
# classifier nor the MoViNet clip classifier has any notion of "how many people are
# present" - they only see raw pixels/motion, so a single person's fast, aggressive-
# looking motion (kicking/tampering with an object, exercising, ...) can still score
# high as "violence". Checked against real footage (a lone person tampering with a
# parked motorcycle, no one else ever in frame) that scored 0.58 on the YOLOv8 model
# and 0.99 on MoViNet - this guard checks whether a second person was ever visible near
# the single strongest match, using the same shared YOLO11-pose model the other
# detectors already use, and discounts the score rather than a lone person's motion
# being misread as a two-person fight.

import cv2

PERSON_CLASS_ID = 0

SINGLE_PERSON_CHECK_WINDOW_SECONDS = 1.0  # look this far before/after the flagged moment
SINGLE_PERSON_CHECK_FPS = 5.0             # sparse enough to be cheap, dense enough not to miss a second person
SINGLE_PERSON_CONF_MIN = 0.4              # confidence floor for counting a detection as "a person"
SINGLE_PERSON_PENALTY = 0.3               # confidence multiplier when nobody else was ever visible nearby


def max_people_nearby(filepath: str, frame_number: int, fps: float, person_model) -> int:
    """Returns the most distinct people ever seen at once (via the shared YOLO11-pose
    person_model) within SINGLE_PERSON_CHECK_WINDOW_SECONDS of frame_number."""
    center_t = frame_number / fps
    start_frame = max(0, int((center_t - SINGLE_PERSON_CHECK_WINDOW_SECONDS) * fps))
    end_frame = int((center_t + SINGLE_PERSON_CHECK_WINDOW_SECONDS) * fps)
    check_skip = max(1, round(fps / SINGLE_PERSON_CHECK_FPS))

    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    max_people = 0
    current_frame = start_frame
    while current_frame <= end_frame:
        ok, frame = video.read()
        if not ok:
            break
        if (current_frame - start_frame) % check_skip == 0:
            result = person_model(frame, verbose=False)[0]
            count = sum(
                1 for box in result.boxes
                if int(box.cls) == PERSON_CLASS_ID and float(box.conf) >= SINGLE_PERSON_CONF_MIN
            )
            max_people = max(max_people, count)
        current_frame += 1

    video.release()
    return max_people
