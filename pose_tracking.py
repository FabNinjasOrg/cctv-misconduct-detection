"""Shared pose tracking: runs YOLO11-pose across a motion window once and follows
each person frame to frame.

Spit, vandalism and fall all need exactly the same thing here - the same model,
sampled at the same rate, with the same box-matching rule - and until this module
existed each of them carried its own byte-identical copy of it. Running all three
from analyze_action therefore meant putting every sampled frame through the same
pose model three times to get the same answer back, which was most of the cost of
a combined verdict. Now the pass happens once and the result is shared.

The detectors still work standalone: called without a pre-computed track set, each
one runs this itself, exactly as before.
"""
import cv2

PERSON_CLASS_ID = 0

TRACK_FPS = 15.0        # how many frames per second to run pose detection on - fast
                        # enough to catch a swing or a stumble, which can be under
                        # half a second, without paying for every frame of a 30fps feed
IOU_MATCH_MIN = 0.3     # how much two boxes must overlap to be treated as the same person


def iou(box_a, box_b) -> float:
    """How much two boxes overlap, from 0 (no overlap) to 1 (identical)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def whole_clip_segments(filepath: str):
    """The clip as the single segment the detectors work over, plus its fps.

    Motion gating runs once in analyze_action/action_analyzer.py rather than per
    detector, so there is nothing left to split the clip into here. Returns
    (fps, segments) so a caller that only wants the segment list does not have to
    open the video a second time for the frame rate."""
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    segments = [{
        "start_t": 0.0,
        "end_t": frame_count / fps,
        "start_frame": 0,
        "end_frame": frame_count - 1,
    }]
    return fps, segments


def track_clip(filepath: str, person_model):
    """Every person tracked across the whole clip - the shared pose pass.

    Returns (fps, segments, tracks_by_segment), where tracks_by_segment lines up
    index-for-index with segments so a detector can take its own segment's tracks
    without re-deriving anything."""
    fps, segments = whole_clip_segments(filepath)
    tracks_by_segment = [
        track_segment(filepath, segment, person_model, fps) for segment in segments
    ]
    return fps, segments, tracks_by_segment


def track_segment(filepath: str, segment: dict, person_model, fps: float) -> dict:
    """Runs pose detection across one motion window and follows each person frame-to-frame.

    Returns every person tracked, not just one - each gets scored independently by
    the detectors, so nobody's event is silently skipped just because someone else
    was also in frame."""
    frame_skip = max(1, round(fps / TRACK_FPS))

    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])

    tracks: dict = {}
    last_box: dict = {}
    next_id = 0
    frame_number = segment["start_frame"]

    while frame_number <= segment["end_frame"]:
        ok, frame = video.read()
        if not ok:
            break

        if (frame_number - segment["start_frame"]) % frame_skip == 0:
            result = person_model(frame, verbose=False)[0]
            detections = [
                {
                    "box": box.xyxy[0].tolist(),
                    "kpts_xy": result.keypoints.xy[i].cpu().numpy(),
                    "kpts_conf": result.keypoints.conf[i].cpu().numpy(),
                }
                for i, box in enumerate(result.boxes)
                if int(box.cls) == PERSON_CLASS_ID
            ]

            # match each detection to the closest-matching person from the last frame
            assigned = set()
            for track_id, prev_box in list(last_box.items()):
                best_index, best_iou = None, IOU_MATCH_MIN
                for i, det in enumerate(detections):
                    if i in assigned:
                        continue
                    overlap = iou(prev_box, det["box"])
                    if overlap > best_iou:
                        best_index, best_iou = i, overlap
                if best_index is not None:
                    assigned.add(best_index)
                    _append_record(tracks, track_id, frame_number, fps, detections[best_index])
                    last_box[track_id] = detections[best_index]["box"]

            # anyone left over is a new person, not seen before in this segment
            for i, det in enumerate(detections):
                if i not in assigned:
                    track_id = next_id
                    next_id += 1
                    _append_record(tracks, track_id, frame_number, fps, det)
                    last_box[track_id] = det["box"]

        frame_number += 1

    video.release()
    return tracks


def _append_record(tracks: dict, track_id, frame_number: int, fps: float, det: dict) -> None:
    """Adds one frame's detection to a person's track history.

    Carries box_width even though only the fall detector reads it: the record is
    shared by all three now, so it holds the union of what they need rather than
    each one getting a slightly different shape."""
    box = det["box"]
    x1, y1, x2, y2 = box
    tracks.setdefault(track_id, []).append({
        "t": frame_number / fps,
        "frame": frame_number,
        "box": box,
        "box_height": y2 - y1,
        "box_width": x2 - x1,
        "kpts_xy": det["kpts_xy"],
        "kpts_conf": det["kpts_conf"],
    })
