# Plays a clip back with the fight-detection MoViNet confidence overlaid live, synced to
# playback time. Runs the same sliding-window classifier as movinet_detector.py's
# detect_fighting_mil once up front (no changes to its scoring), but keeps every window's
# score - not just the ones above threshold - then redraws the score live as the clip
# plays, updating each time a new window becomes available (same cadence a live camera
# feed would update at) instead of just off a couple of saved evidence frames.

import itertools
import sys

import cv2

from app.actions.fight.movinet_detector import CONFIDENCE_MIN, STRIDE_FRAMES, TARGET_FPS, WINDOW_FRAMES, truncate_2dp, violence_classifier

WINDOW_NAME = "Fight Detection - Playback"


def analyze_for_playback(filepath: str) -> list:
    """Runs the sliding-window scan once and returns [(frame_number, confidence), ...] -
    one entry per completed window, tagged with the frame it became available at."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, round(fps / TARGET_FPS))

    window = []  # sliding buffer of {"frame", "frame_number"}, length capped at WINDOW_FRAMES
    checkpoints = []  # (frame_number, confidence), in playback order

    for frame_number in itertools.count():
        ok, frame = video.read()
        if not ok:
            break
        if frame_number % frame_skip:
            continue

        window.append({"frame": frame, "frame_number": frame_number})
        if len(window) < WINDOW_FRAMES:
            continue

        confidence = truncate_2dp(float(violence_classifier([[item["frame"] for item in window]])[0]))
        checkpoints.append((frame_number, confidence))

        window = window[STRIDE_FRAMES:]

    video.release()
    return checkpoints


def play_with_overlay(filepath: str) -> None:
    checkpoints = analyze_for_playback(filepath)

    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, round(1000 / fps))

    confidence = 0.0
    next_checkpoint = 0

    print("Analysis done, playing back. Press 'q' in the window to quit.")
    for frame_number in itertools.count():
        ok, frame = video.read()
        if not ok:
            break

        # a window's score only becomes known once all its frames have played -
        # carry the last known score forward until the next checkpoint is reached
        while next_checkpoint < len(checkpoints) and checkpoints[next_checkpoint][0] <= frame_number:
            confidence = checkpoints[next_checkpoint][1]
            next_checkpoint += 1

        if confidence >= CONFIDENCE_MIN:
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)

        cv2.putText(frame, f"Fight: {confidence:.0%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -m fight.playback <path-to-clip>")

    play_with_overlay(sys.argv[1])
