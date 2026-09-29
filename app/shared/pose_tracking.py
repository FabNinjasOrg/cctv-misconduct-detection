"""Shared pose tracking: runs YOLO11-pose across a motion window once and follows
each person frame to frame.

Spit, vandalism and fall all need exactly the same thing here - the same model,
sampled at the same rate, with the same box-matching rule - and until this module
existed each of them carried its own byte-identical copy of it. Running all three
from the combined analyzer therefore meant putting every sampled frame through the same
pose model three times to get the same answer back, which was most of the cost of
a combined verdict. Now the pass happens once and the result is shared.

The detectors still work standalone: called without a pre-computed track set, each
one runs this itself, exactly as before.
"""
import logging

import cv2

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0

TRACK_FPS = 15.0        # how many frames per second to run pose detection on - fast
                        # enough to catch a swing or a stumble, which can be under
                        # half a second, without paying for every frame of a 30fps feed
IOU_MATCH_MIN = 0.3     # how much two boxes must overlap to be treated as the same person


# ---------------------------------------------------------------------------
# Usable-pose gate
#
# Spit, vandalism and fall all measure a person against their own body: trunk
# angle from shoulders-to-hips, hand speed in box-heights per second, "lying
# flat" from box shape. Every one of those needs a person who is *wholly in
# frame at a distance* - a CCTV view. Given a close-up instead, the arithmetic
# does not fail loudly, it quietly returns nonsense:
#
#   - no hips  -> the trunk vector is invented from noise, so its angle swings
#                 wildly, which is exactly the signature a fall looks for
#   - too close -> the box is head-and-shoulders, perhaps a third of the real
#                 body, so "box heights per second" reads about 3x too fast and
#                 reaching for a phone clears the violent-strike bar
#
# Measured on a desk camera: person at 81% of frame height, box aspect 0.65
# (wider than tall), hips/knees/ankles all invisible - six false detections in
# thirteen minutes across all three detectors.
#
# So this is not a tuning knob layered on top. It is the precondition those
# calculations already silently assume, made explicit: if we cannot measure the
# body, we decline to judge it rather than judge it wrongly.
#
# Deliberately NOT gated on box aspect ratio: a person going from upright to
# horizontal is the fall signal itself, and rejecting wide boxes would throw
# away the very thing we are looking for.
# ---------------------------------------------------------------------------

REQUIRE_VISIBLE_HIPS = True   # set False to accept poses with no hip keypoints

L_HIP, R_HIP = 11, 12         # COCO pose keypoint indices, as the detectors use
KEYPOINT_CONF_FLOOR = 0.15    # matches CONF_FLOOR in all three detectors

# A person taller than this fraction of the frame is too close for their box to
# be a meaningful yardstick. Height only, never width: a fallen person's box is
# short and wide, so gating on width would reject real falls.
#
# The hips check above does the real work - measured across our clips, a genuine
# event had hips on 12/12 sampled frames. This is only a backstop for a pose so
# close that hips are found but the box still cannot be trusted, so it is set
# loosely on purpose.
#
# Twice it has been set too tight and thrown away real detections:
#   0.60 rejected spit-1.mp4, a genuine event measuring 0.62 with hips plainly
#        visible.
#   0.75 cost spit-3.mp4 confidence (0.783 -> 0.628): 19.3% of the sampled
#        detections in the reference spit clips sit above 0.75, all of them
#        perfectly measurable.
#
# Hence 0.90, and measurement says it costs nothing to be this loose. Across the
# reference clips and a correctly-placed camera, NOTHING exceeds 0.90:
#
#     sampled detections over 0.75   over 0.90
#     live camera          0.0%        0.0%
#     spit clips          19.3%        0.0%
#     fall clips           0.0%        0.0%
#     vandalism clips      0.0%        0.0%
#
# So 0.90 behaves identically to 0.75 on real footage while leaving genuine
# close-range events intact - and still catches what this exists for, someone
# filling essentially the whole frame, where the box has stopped describing a
# body at all.
#
# If real events are still being missed, raise this before touching
# REQUIRE_VISIBLE_HIPS; the log line at the end of track_segment names whichever
# rule is firing. Note that hips is the rule that actually discriminates - a
# well-placed camera shows hips on ~100% of detections, so if "hips not visible"
# is what fires, the camera placement is the thing to fix, not this number.
MAX_BOX_HEIGHT_FRACTION = 0.90


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


def pose_rejection_reason(det: dict, frame_height: int):
    """Why this person cannot be measured reliably, or None if they can.

    Returns a short string so the caller can report *what* was skipped rather
    than just how much - the difference between "the camera is too close" and
    "hips are occluded" changes what you would do about it."""
    # Hips first: it is the check that actually discriminates, so when both
    # would fire this reports the more useful reason.
    #
    # One hip is enough. Requiring both would reject anyone standing side-on
    # with a hip occluded, and a single hip still anchors the trunk and gives a
    # real torso length - which is all the arithmetic downstream needs.
    if REQUIRE_VISIBLE_HIPS:
        conf = det.get("kpts_conf")
        if conf is None or len(conf) <= R_HIP:
            return "no keypoints"
        if max(conf[L_HIP], conf[R_HIP]) < KEYPOINT_CONF_FLOOR:
            return "hips not visible"

    x1, y1, x2, y2 = det["box"]
    if frame_height and (y2 - y1) > frame_height * MAX_BOX_HEIGHT_FRACTION:
        return "too close to camera"

    return None


def whole_clip_segments(filepath: str):
    """The clip as the single segment the detectors work over, plus its fps.

    Motion gating runs once in the live engine rather than per
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
    skipped: dict = {}      # rejection reason -> how many times, for one log line at the end

    while frame_number <= segment["end_frame"]:
        ok, frame = video.read()
        if not ok:
            break

        if (frame_number - segment["start_frame"]) % frame_skip == 0:
            frame_height = frame.shape[0]
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

            # Drop anyone we cannot measure, before tracking rather than after:
            # a track built from unusable frames would still be scored, and the
            # detectors have no way to tell that its numbers are meaningless.
            usable = []
            for det in detections:
                reason = pose_rejection_reason(det, frame_height)
                if reason is None:
                    usable.append(det)
                else:
                    skipped[reason] = skipped.get(reason, 0) + 1
            detections = usable

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

    # One line, only when something was dropped. Worth surfacing rather than
    # silently discarding: a clip where every person was skipped looks exactly
    # like a clip where nothing happened, and those need telling apart.
    if skipped:
        log.info(
            "pose gate skipped %d detection(s): %s",
            sum(skipped.values()),
            ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items())),
        )

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
