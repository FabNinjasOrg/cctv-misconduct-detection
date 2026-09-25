# Detects vandalism (property damage) in a CCTV clip using two combined signals - no
# fine-tuned vandalism model, no LLM/Gemini call, pose-kinematics only. Vandalism has
# no single motion signature the way spitting or a two-person fight does, so this is a
# best-effort first pass, not a mature detector:
#
#   1. Damaging-action kinematics: repeated fast wrist/ankle strikes against something -
#      hitting, kicking, or rocking an object rarely happens just once, it's a burst of
#      several hits in a row. A single fast spike (one big move) counts for something,
#      but several separate spikes in the same tracked window count for a lot more -
#      that repetition is what actually separates "hit the bench four times" from "waved
#      an arm once". (YOLO11-pose wrist/ankle keypoint velocity, normalized per person by
#      their own box height so it's scale-invariant across camera distances.)
#   2. Object-in-hand: is a tool/weapon-like object (bat, bottle, knife, ...) near the
#      hand doing the swinging? (YOLO11 general object detector, COCO classes only -
#      there's no dedicated "spray can"/"hammer"/"brick" class, so this is a coarse
#      proxy for "holding something that could damage property".)
#
# Neither signal alone maxes out the score (see the weights below) - a bare fast swing
# could just as easily be a wave, a punch thrown in a fight, or pose-tracking noise, and
# holding an object doesn't mean it's being swung at anything. The speed/repetition
# thresholds are calibrated against the one confirmed clip available at the time of
# writing (someone repeatedly hitting a sitting bench) - they are a starting point, not
# a validated cutoff, and should be re-checked against more real footage as it shows up.
#
# A third check guards the single biggest confirmed confusion: repeated fast strikes are
# also exactly what a fist-fight looks like kinematically (checked against real fight
# footage - every clip on hand hit the same strike-score ceiling as genuine vandalism).
# The distinguishing fact pose CAN see is whether another person's body was near the
# strike location - hitting an object is solitary, hitting a person isn't - so contact
# with another tracked person during the strikes discounts the score rather than a fight
# being misread as property damage.
#
# Known accepted limitation: a solitary person repeatedly striking toward the same fixed
# point with no tool in hand and nobody else in contact (e.g. someone frantically banging
# on a door/window in distress, not damaging anything) is kinematically identical to
# bare-handed vandalism (punching/kicking an object with no tool) - both are fast, repeated,
# same-spot strikes. Checked strike cadence and peak speed as possible discriminators
# against the one confirmed real clip on hand; neither separated the two cases. Left
# unguarded rather than add an unvalidated threshold that would also blunt genuine
# bare-handed vandalism detection (e.g. fall-1.mp4's bystander banging on a glass door
# scores 0.65 via this exact path).

import os

import cv2
import numpy as np
from ultralytics import YOLO

# Pose tracking lives in pose_tracking.py, shared with the spit and fall
# detectors - all three ran the identical pass over the same frames.
from pose_tracking import TRACK_FPS, iou as _iou, track_segment, whole_clip_segments

# from motion_filter import detect_motion, group_motion_segments  # motion filter now runs once in analyze_action/action_analyzer.py

# COCO-17 keypoint indices, matching spit/kinematic_detector.py's convention
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16

# --- step 2: pose tracking settings ---
# TRACK_FPS and the box matching now come from pose_tracking.py (imported above).
CONF_FLOOR = 0.15         # below this, a keypoint is treated as missing
SMOOTH_RADIUS = 1         # light smoothing so one noisy keypoint jump isn't read as a swing

# --- damaging-action signature: repeated fast strikes, not one spike, not sustained ---
# motion like walking. Speed is in "box-heights per second" - normalized by the person's
# own bounding-box height so the same real-world hit reads the same regardless of camera
# distance. Wrist and ankle speed are combined (whichever limb is moving faster in a
# given frame) into one "limb strike speed" series, since a damaging strike could be a
# hit, a kick, or a shove - the kinematics look similar either way.
STRIKE_SPEED_MIN = 0.9        # combined wrist/ankle speed below this is never counted as a strike
STRIKE_RELEASE_RATIO = 0.6    # speed must drop back below MIN * this ratio to end one strike and
                               # allow the next one to be counted separately, instead of one long blur
STRIKE_SPEED_REF = 2.5        # a strike at/above this speed scores full marks for intensity
STRIKES_FOR_FULL_REPETITION = 3  # this many separate strikes in one tracked window = full
                                  # repetition score; fewer counts proportionally less

# --- object (tool) detection ---
# COCO classes that could plausibly be swung/thrown to damage property. Coarse by
# necessity - COCO has no "spray can", "hammer", "brick", or "rock" class.
TOOL_CLASSES = {"baseball bat", "knife", "scissors", "bottle", "sports ball", "skateboard"}
OBJECT_CONFIDENCE_MIN = 0.5
HAND_REGION_MARGIN = 0.5  # how far around the wrist to look, relative to box height

# --- contact guard: repeated fast strikes are also what a fist-fight looks like ---
CONTACT_IOU_MIN = 0.05            # another person's box overlapping this much during the strikes =
                                    # physical contact, not solitary hitting of an object
CONTACT_PROXIMITY_MARGIN = 0.5    # expand the subject's box by this fraction of its own size before
                                    # checking overlap - two people fighting are "in contact" well
                                    # before their tight bounding boxes literally intersect
CONTACT_TIME_WINDOW_SECONDS = 1.0  # look for another person within this many seconds of each strike,
                                    # not just the exact sampled frame - tracking routinely drops the
                                    # other person for a frame or two mid-fight
CONTACT_PENALTY = 0.35            # confidence multiplier applied if another person's body was near
                                    # the strikes - a fight misread as vandalism, not the reverse

# --- self-directed guard: a fast hand-to-face reach (spitting, wiping the mouth, an itch)
# produces the same burst of wrist speed as a genuine swing, but ends up at the subject's own
# nose instead of away from the body toward an external object/person. Checked against real
# footage: a spitting clip's fast hand-to-face reach and follow-through registered as two
# "strikes" (wrist ending 0.22x box-height from the subject's own nose) and crossed
# DECISION_THRESHOLD as vandalism at 0.53. Same distance/ratio spit/kinematic_detector.py
# already uses for its own hand-to-mouth check (HAND_ABS_MAX), just not previously computed here.
SELF_DIRECTED_HAND_MAX = 0.35
SELF_DIRECTED_PENALTY = 0.35      # same discount factor as CONTACT_PENALTY - this is the same
                                    # class of guard (an unrelated action's kinematics resembling
                                    # a strike), just for a different confusable cause

# --- final scoring ---
# Weights must add up to 1.0. A single fast strike alone caps out around ~0.48 (just
# under DECISION_THRESHOLD) - it takes either a repeated strike pattern or a tool in
# hand to actually cross the line, so a lone quick gesture isn't enough on its own.
W_STRIKE, W_TOOL = 0.65, 0.35
DECISION_THRESHOLD = 0.5

VANDALISM_FRAMES_DIR = "vandalism/evidence/object_action"
os.makedirs(VANDALISM_FRAMES_DIR, exist_ok=True)

object_model = YOLO("models/yolo11n.pt")  # general object detector, checked near the swinging hand

# Which keypoints connect to which, just for drawing the skeleton on evidence images
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def detect_vandalism(filepath: str, person_model, tracks_by_segment: list = None) -> dict:
    """Main entry point: detects motion, checks each tracked person's limb kinematics
    plus nearby objects in each motion window, returns a verdict.

    tracks_by_segment lets a caller that runs several detectors over one clip hand
    in the pose pass it has already done, instead of each detector repeating the
    identical work - see analyze_action/action_analyzer.py. Left out, this runs the
    pass itself, so the standalone endpoint behaves exactly as before."""
    # motion_events = detect_motion(filepath)  # motion filter now runs once in analyze_action/action_analyzer.py
    # segments = group_motion_segments(motion_events, fps, frame_count)
    fps, segments = whole_clip_segments(filepath)

    candidates = []
    for index, segment in enumerate(segments):
        tracks = (
            tracks_by_segment[index] if tracks_by_segment is not None
            else track_segment(filepath, segment, person_model, fps)
        )
        for track_id, records in tracks.items():
            features = compute_kinematics(records)
            if features is None:
                continue
            other_tracks = {tid: recs for tid, recs in tracks.items() if tid != track_id}
            confidence, signals = score_event(filepath, features, other_tracks)
            candidates.append({"track_id": track_id, "confidence": confidence, "signals": signals})

    confirmed = sorted(
        (c for c in candidates if c["confidence"] >= DECISION_THRESHOLD),
        key=lambda c: c["confidence"],
        reverse=True,
    )

    return {
        "vandalism_detected": len(confirmed) > 0,
        "confidence": round(max((c["confidence"] for c in candidates), default=0.0), 3),
        "segments_checked": len(segments),
        "person_event_count": len(candidates),
        "evidence_frames": [save_evidence_frame(filepath, c) for c in confirmed],
    }


def _expand_box(box, margin_ratio: float):
    """Grows a box outward by margin_ratio of its own width/height, so a check against it
    catches "nearby", not just "literally overlapping"."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin_ratio, (y2 - y1) * margin_ratio
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


def _smooth(values: list, radius: int = SMOOTH_RADIUS) -> list:
    """Averages each value with its nearby neighbors to reduce frame-to-frame jitter."""
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        window = values[lo:hi]
        out[i] = sum(window) / len(window)
    return out


def compute_kinematics(records: list) -> dict | None:
    """Turns a person's raw per-frame keypoints into wrist/ankle speed signals used for
    scoring. Returns None if there isn't enough data to say anything."""
    if len(records) < 4:
        return None

    n = len(records)
    ts = [r["t"] for r in records]
    box_heights = [r["box_height"] or 1.0 for r in records]

    def kp(i, idx):
        return np.asarray(records[i]["kpts_xy"][idx])

    def kconf(i, idx):
        return float(records[i]["kpts_conf"][idx])

    wrist_speed = [0.0] * n
    ankle_speed = [0.0] * n

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        bh = box_heights[i]

        wrist_speeds = []
        for idx in (L_WRIST, R_WRIST):
            if kconf(i, idx) >= CONF_FLOOR and kconf(i - 1, idx) >= CONF_FLOOR:
                dist = float(np.linalg.norm(kp(i, idx) - kp(i - 1, idx)))
                wrist_speeds.append(dist / bh / dt)
        if wrist_speeds:
            wrist_speed[i] = max(wrist_speeds)

        ankle_speeds = []
        for idx in (L_ANKLE, R_ANKLE):
            if kconf(i, idx) >= CONF_FLOOR and kconf(i - 1, idx) >= CONF_FLOOR:
                dist = float(np.linalg.norm(kp(i, idx) - kp(i - 1, idx)))
                ankle_speeds.append(dist / bh / dt)
        if ankle_speeds:
            ankle_speed[i] = max(ankle_speeds)

    return {
        "ts": ts,
        "records": records,
        "wrist_speed": _smooth(wrist_speed),
        "ankle_speed": _smooth(ankle_speed),
    }


def _read_frame_at(filepath: str, frame_number: int):
    """Reads one specific frame from a video file."""
    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = video.read()
    video.release()
    return frame if ok else None


def find_tool_near_hand(frame, keypoints, kpts_conf, box_height):
    """Runs the general object detector on a region around whichever wrist is visible,
    looking for a tool/weapon-like object (see TOOL_CLASSES) that could be used to cause
    damage. Returns the class name if found, else None."""
    height, width = frame.shape[:2]
    wrist_indices = [idx for idx in (L_WRIST, R_WRIST) if kpts_conf[idx] >= CONF_FLOOR]
    if not wrist_indices:
        return None

    margin = box_height * HAND_REGION_MARGIN
    for idx in wrist_indices:
        wx, wy = keypoints[idx]
        x1, y1 = max(0, int(wx - margin)), max(0, int(wy - margin))
        x2, y2 = min(width, int(wx + margin)), min(height, int(wy + margin))
        hand_region = frame[y1:y2, x1:x2]
        if hand_region.size == 0:
            continue

        result = object_model(hand_region, verbose=False)[0]
        for detected in result.boxes:
            class_name = object_model.names[int(detected.cls)]
            if class_name in TOOL_CLASSES and float(detected.conf) >= OBJECT_CONFIDENCE_MIN:
                return class_name
    return None


def _count_strikes(speed: list, floor: float, release_ratio: float) -> list:
    """Splits a speed series into distinct strikes: each one starts when speed rises
    above `floor` and ends once it drops back below floor*release_ratio, so a sustained
    fast stretch still counts as one strike while separate hits (speed drops back down
    to a rest in between) are counted individually. Returns a list of
    {"peak", "peak_index"} - one per strike - in the order they occurred."""
    release = floor * release_ratio
    strikes = []
    in_strike = False
    peak_value, peak_index = 0.0, 0

    for i, v in enumerate(speed):
        if not in_strike:
            if v >= floor:
                in_strike, peak_value, peak_index = True, v, i
        else:
            if v > peak_value:
                peak_value, peak_index = v, i
            if v <= release:
                strikes.append({"peak": peak_value, "peak_index": peak_index})
                in_strike = False
    if in_strike:
        strikes.append({"peak": peak_value, "peak_index": peak_index})
    return strikes


def score_event(filepath: str, features: dict, other_tracks: dict) -> tuple:
    """Scores one tracked person's limb kinematics + nearby objects and returns
    (confidence, signals) - confidence is 0-1, signals is a breakdown for debugging/evidence.
    other_tracks is every other person tracked in the same window, used only for the
    contact check below."""
    ts = features["ts"]
    wrist_speed = features["wrist_speed"]
    ankle_speed = features["ankle_speed"]
    combined_speed = [max(w, a) for w, a in zip(wrist_speed, ankle_speed)]

    # --- repeated-strike signature: how many separate fast hits, and how hard the hardest was ---
    strikes = _count_strikes(combined_speed, STRIKE_SPEED_MIN, STRIKE_RELEASE_RATIO)
    strike_count = len(strikes)

    if strikes:
        strongest = max(strikes, key=lambda s: s["peak"])
        peak_strike_speed = strongest["peak"]
        t_peak_index = strongest["peak_index"]
        intensity_score = max(0.0, min(1.0, (peak_strike_speed - STRIKE_SPEED_MIN) / (STRIKE_SPEED_REF - STRIKE_SPEED_MIN)))
        repetition_score = min(1.0, strike_count / STRIKES_FOR_FULL_REPETITION)
        strike_score = 0.6 * intensity_score + 0.4 * repetition_score
    else:
        peak_strike_speed = max(combined_speed, default=0.0)
        t_peak_index = int(np.argmax(combined_speed)) if combined_speed else 0
        strike_score = 0.0

    t_peak = ts[t_peak_index]
    action = "swing" if wrist_speed[t_peak_index] >= ankle_speed[t_peak_index] else "kick"

    # --- is there a tool/weapon-like object near the hand at that moment? ---
    peak_record = features["records"][t_peak_index]
    peak_frame_image = _read_frame_at(filepath, peak_record["frame"])
    tool_near_hand = (
        find_tool_near_hand(peak_frame_image, peak_record["kpts_xy"], peak_record["kpts_conf"], peak_record["box_height"])
        if peak_frame_image is not None else None
    )
    tool_score = 1.0 if tool_near_hand else 0.0

    # --- was another person's body near the strikes? repeated fast strikes are also
    # exactly what a fist-fight looks like kinematically - checked against a time window
    # and a margin around the subject's box, not just the exact sampled frame and a
    # literal box overlap, since tracking routinely drops the other person for a frame
    # or two mid-fight, and two people fighting are "in contact" before boxes intersect ---
    contact_with_other = False
    if strikes and other_tracks:
        other_records_flat = [r for records in other_tracks.values() for r in records]
        for strike in strikes:
            t = ts[strike["peak_index"]]
            subject_box = _expand_box(features["records"][strike["peak_index"]]["box"], CONTACT_PROXIMITY_MARGIN)
            nearby_records = [r for r in other_records_flat if abs(r["t"] - t) <= CONTACT_TIME_WINDOW_SECONDS]
            if any(_iou(subject_box, r["box"]) > CONTACT_IOU_MIN for r in nearby_records):
                contact_with_other = True
                break

    # --- was the swinging hand actually near the subject's OWN face at the strike's peak,
    # rather than away from the body toward an external target? Only checked for strikes that
    # are themselves wrist-driven (a "swing") - a leg-driven strike (a "kick") naturally swings
    # the arms for balance too, which can incidentally pass near the face without meaning
    # anything about what the LEG is doing (checked against real footage: a genuine bench-
    # kicking clip had one of its four kicks' balancing arm swing pass near the face and wrongly
    # discounted a real vandalism score). See SELF_DIRECTED_HAND_MAX above. ---
    self_directed = False
    for strike in strikes:
        i = strike["peak_index"]
        if wrist_speed[i] < ankle_speed[i]:
            continue  # this strike was leg-driven - hand position isn't relevant to it
        record = features["records"][i]
        kpts, kconf, bh = record["kpts_xy"], record["kpts_conf"], record["box_height"] or 1.0
        if kconf[NOSE] < CONF_FLOOR:
            continue
        nose = kpts[NOSE]
        hand_dists = [
            float(np.linalg.norm(kpts[idx] - nose)) / bh
            for idx in (L_WRIST, R_WRIST) if kconf[idx] >= CONF_FLOOR
        ]
        if hand_dists and min(hand_dists) <= SELF_DIRECTED_HAND_MAX:
            self_directed = True
            break

    raw_confidence = W_STRIKE * strike_score + W_TOOL * tool_score
    confidence = raw_confidence
    confidence *= CONTACT_PENALTY if contact_with_other else 1.0
    confidence *= SELF_DIRECTED_PENALTY if self_directed else 1.0
    confidence = max(0.0, min(1.0, confidence))

    signals = {
        "action": action,
        "t_peak": round(t_peak, 2),
        "strike_count": strike_count,
        "peak_strike_speed": round(peak_strike_speed, 3),
        "strike_score": round(strike_score, 3),
        "tool_near_hand": tool_near_hand,
        "tool_score": tool_score,
        "contact_penalty_applied": contact_with_other,
        "self_directed_penalty_applied": self_directed,
        "peak_frame": peak_record["frame"],
        "peak_box": peak_record["box"],
        "peak_keypoints": peak_record["kpts_xy"].tolist(),
        "peak_kpts_conf": peak_record["kpts_conf"].tolist(),
    }
    return confidence, signals


def save_evidence_frame(filepath: str, candidate: dict) -> dict:
    """Draws the person's box + skeleton + confidence label on the key frame and saves it as an image."""
    signals = candidate["signals"]
    frame = _read_frame_at(filepath, signals["peak_frame"])

    result = {
        "t": signals["t_peak"],
        "action": signals["action"],
        "confidence": round(candidate["confidence"], 3),
        "frame_path": None,
        "signals": {
            k: v for k, v in signals.items()
            if k not in ("peak_keypoints", "peak_kpts_conf")
        },
    }
    if frame is None:
        return result

    outlined = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in signals["peak_box"]]
    cv2.rectangle(outlined, (x1, y1), (x2, y2), (0, 0, 255), 2)

    keypoints = signals["peak_keypoints"]
    kpts_conf = signals["peak_kpts_conf"]
    for a, b in SKELETON_EDGES:
        if kpts_conf[a] >= CONF_FLOOR and kpts_conf[b] >= CONF_FLOOR:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(outlined, pa, pb, (0, 255, 255), 2)
    for idx, (x, y) in enumerate(keypoints):
        if kpts_conf[idx] >= CONF_FLOOR:
            cv2.circle(outlined, (int(x), int(y)), 3, (0, 255, 0), -1)

    label = f"Vandalism {candidate['confidence']:.0%} | {signals['action']} | t={signals['t_peak']}s"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(outlined, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_path = os.path.join(VANDALISM_FRAMES_DIR, f"vandalism_track{candidate['track_id']}_t{signals['t_peak']}.jpg")
    cv2.imwrite(frame_path, outlined)
    result["frame_path"] = frame_path
    return result
