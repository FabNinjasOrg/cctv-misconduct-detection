# Detects spitting in a CCTV clip using body posture only - no fine-tuned
# spitting model, no LLM call. Two steps: (1) find where motion happens in the
# clip, (2) check the person's posture in that window against a spit-like
# pattern.
#
# Scenarios checked to decide "did this person spit?" (each one below is one
# score, combined into a single confidence 0-1 at the end):
#   1. Bowed head held for a while - not just a quick bend, but leaning the
#      head down/forward and STAYING that way for at least half a second. OR
#      (see "quick lurch" below) a fast, unheld bow tightly synced with the
#      hand reaching the mouth - some real spits are a lurch-and-release
#      under a second, not a held bow.
#   2. Hand comes close to the mouth - a common spitting motion.
#   3. Shoulders/head tilted to one side - people often turn sideways to spit.
#   4. Face turned away from the camera - harder to see the face when spitting
#      into a corner or downward.
#   5. Some object appears near the head - e.g. a cup, bottle, tissue.
#   6. Person is NOT walking - if they're mid-stride, it's probably just
#      normal movement, not spitting, so the score is reduced.
#
# Three guards specifically target postures that look like a spit-bow for an
# unrelated reason:
#   - No corroboration: a bowed-and-held posture with NEITHER a hand near the
#     mouth NOR an object near the head is heavily discounted - that's also
#     what tying a shoe, picking something off the ground, or bowing as a
#     greeting looks like from posture alone.
#   - Sustained hand-at-mouth: if the hand stayed near the mouth for most of
#     the held bow (not just briefly), that's more consistent with drinking
#     from a low tap/fountain or eating than a brief spit, so it's discounted
#     rather than rewarded.
#   - Contact with another person during the bow (scenario 7, below) - e.g.
#     being pinned down in a fight also looks like a held bow.
#   - Someone lying on the floor right in front of them: stooping to check on or
#     help a collapsed person is the same bow as a spit from posture alone, and
#     what tells them apart is what the bow is aimed at.
#
# If nothing moves in the clip, none of this runs - the pose model never
# even looks at frames from a quiet stretch of footage.

import math
import os

import cv2
import numpy as np

from app.shared.pose_tracking import TRACK_FPS, iou as _iou, track_segment, whole_clip_segments
from app.actions.spit.single_frame_detector import find_object_near_head

# from app.shared.motion_filter import detect_motion, group_motion_segments  # motion filter now runs once in analyze_action/action_analyzer.py

# COCO-17 keypoint indices, matching spit_detection.py's convention
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

SEGMENT_PAD_SECONDS = 0.6    # kept for the baseline-window calc below (see "baseline_end_t") even
                             # though the motion filter that used to share this constant is now gated
                             # centrally in analyze_action/action_analyzer.py instead of here

# --- step 2: pose tracking settings ---
# TRACK_FPS, the box matching and the pass itself now live in pose_tracking.py,
# shared with the vandalism and fall detectors - all three ran the identical pass.
MIN_TRACK_RECORDS = 4        # a track shorter than this isn't trusted as a real, distinct person -
                              # too little data to say anything meaningful about them either as the
                              # subject being scored (see compute_kinematics) or as "another person"
                              # for the contact guard below (checked against real footage: a one-
                              # frame duplicate/ghost detection of the SAME person, overlapping their
                              # own box almost exactly, was wrongly counted as a second person "in
                              # contact" with them and tanked a genuine spit's score via CONTACT_PENALTY)

# --- keypoint confidence ---
CONF_FLOOR = 0.15            # below this, a keypoint is treated as missing
SMOOTH_RADIUS = 2            # smooths jitter by averaging a few neighboring frames
MAX_FILL_GAP_SECONDS = 1.0   # a reading can carry forward across a brief dropout in keypoint
                              # confidence, but not longer - otherwise a long low-confidence
                              # stretch (e.g. blurry/low-light footage) reads as a posture held
                              # steady for the whole stretch, instead of just unknown
MAX_FILL_GAP_FRAMES = round(MAX_FILL_GAP_SECONDS * TRACK_FPS)

# --- scenario 1: bowed-head signature ---
MIN_PLATEAU_SECONDS = 0.5    # the bowed posture must be held at least this long
STABILITY_MAX_DEG = 7.0      # how steady the bowed posture must be to count as "held" - a real
                              # bowed-and-holding head still sways a bit (checked against an actual
                              # clean spit clip: a genuine 4-second held bow measured ~6deg of natural
                              # wobble), so this allows for that instead of requiring a near-frozen
                              # posture that only a perfectly still mannequin would produce. (A second
                              # clip's ~9.2deg "bowed" cluster looked tempting to fold in by loosening
                              # this further, but that cluster turned out to be a walking gait's head
                              # bob, not a real bow - see scenario 1b below, redesigned to catch that
                              # clip's actual spit instead of loosening this stability check.)
SWING_REF_DEG = 12.0         # how many degrees of bow = full score
EDGE_EXCLUDE_SECONDS = 0.4   # ignore the very start/end of the event when picking the best frame
EVIDENCE_TOP_N = 2                    # how many frames to save as evidence - the ones that actually
                                       # show the spit, not an arbitrary spread. There's no mouth/lip
                                       # keypoint in COCO-17, so pose alone can't say which instant
                                       # that is, so this is picked by an actual visual signal instead
                                       # - see _spit_visibility_scores
EVIDENCE_LOCAL_WINDOW_SECONDS = 0.6   # the 2nd+ evidence frame must be within this many seconds of
                                       # the top-scoring one - a spit is one brief, localized event,
                                       # not scattered moments, so without this a high visibility
                                       # score somewhere unrelated (e.g. a hand wiping the mouth
                                       # afterwards also changes the mouth-crop a lot) could get
                                       # picked as if it were a second, unrelated sighting of it
RELEASE_WINDOW_AFTER_SECONDS = 1.5    # only look for the visible spit *after* the deepest point of
                                       # the bow, within this many seconds - people bow down and then
                                       # spit, not the other way round, and the transition into the
                                       # bow has its own big appearance change (checked against real
                                       # footage: it briefly scored even higher than the actual spit)
                                       # that would otherwise get mistaken for the spit itself
MOUTH_CROP_SIZE_RATIO = 0.16          # size of the mouth-area crop (see _mouth_crop), relative to
                                       # box height
MOUTH_CROP_SIZE = 32                  # thumbnail size (pixels) each mouth crop is resized to

# --- scenario 1b: "quick lurch" - a fast, unheld bow tightly synced with a hand-to-mouth
# reach. Real spitting is sometimes a fast lurch-and-release under a second, which the
# held-bow check above can never catch (it requires MIN_PLATEAU_SECONDS held steady).
# This doesn't need a held posture at all - instead the evidence that it was intentional
# is that the head's single deepest moment and the hand's closest-to-mouth moment happen
# within a fraction of a second of each other, which idle motion/noise won't line up. ---
QUICK_LURCH_HAND_SYNC_SECONDS = 0.3   # how close in time the hand-to-mouth minimum must land to
                                       # the neck's single deepest point to count as one event
QUICK_LURCH_SWING_REF_DEG = 20.0      # a quick, unheld lurch needs a bigger swing than a held bow
                                       # (SWING_REF_DEG=12) to count on its own - it has no held
                                       # duration to corroborate it, so the magnitude has to do more

# --- false-positive guards: a bowed-and-held posture alone is also what several
# unrelated actions look like from pose alone ---
CORROBORATION_MIN = 0.4        # hand_proximity_score or object_score must reach at least this,
                                # or the bow reading isn't trusted alone - tying a shoe, picking
                                # something off the ground, or bowing as a greeting looks the same
                                # as a spit-bow from posture alone, minus any mouth-directed motion.
                                # Set well above a coincidental hand movement (checked against real
                                # footage: someone kicking a bench raised an arm for balance, giving
                                # hand_proximity_score=0.246 - not nothing, but far below what either
                                # confirmed real spit scored, 0.80+) so only a real, deliberate
                                # hand-to-mouth reach counts as corroboration
NO_CORROBORATION_PENALTY = 0.4
SUSTAINED_HAND_CLOSE_RATIO = 0.5    # a frame counts as "close" for this check only if the hand is
                                     # within this fraction of the person's own baseline (resting)
                                     # hand-to-nose distance - relative to their own baseline, like
                                     # hand_proximity_score itself, not an absolute cutoff. A forward-
                                     # bowed posture alone already brings a hanging arm well within
                                     # HAND_ABS_MAX for most of the hold (checked against real
                                     # footage), so an absolute threshold flagged almost the entire
                                     # bow as "sustained" even for a genuine brief-touch spit
SUSTAINED_HAND_FRACTION_MAX = 0.6   # if the hand was "close" (by the ratio above) for more than this
                                     # fraction of the held bow, that reads as sustained contact
                                     # (drinking from a low tap/fountain, eating, a phone call) rather
                                     # than the brief touch a spit typically involves
SUSTAINED_HAND_PENALTY = 0.5

# --- scenario 1c: mouth-visibility (see score_event) ---
MOUTH_VISIBILITY_REF = 20.0   # how much the peak frame's mouth-crop diff must exceed the window's
                              # own typical diff to score full marks - checked against 4 real spit
                              # clips, whose peak-vs-typical delta ranged ~15.6-23 (this sits in the
                              # middle of that range, so most score close to 1.0 without all of them
                              # trivially maxing out)

# --- scenario 2-6: other signal thresholds ---
HAND_ABS_MAX = 0.35           # beyond this hand-to-mouth distance, "hand near mouth" doesn't count
SIDE_TILT_REF = 0.15          # how much side-tilt = full score
FACE_DROP_REF = 0.5           # how much face confidence must drop = full score
GAIT_DISP_PER_SEC = 0.35      # hip movement per second above this looks like walking
ZOOM_GUARD_RATIO = 0.2        # skip a hip-movement reading if the person's box size jumped more than
                              # this much between frames - that's a camera zoom/cut, not real walking
CONTACT_IOU_MIN = 0.05        # another person's box overlapping this much during the bow = physical
                              # contact (e.g. someone else grabbing/holding them down), not solitary spitting
CONTACT_PROXIMITY_MARGIN = 0.5    # expand the subject's box by this fraction of its own size before
                                    # checking overlap - two people grappling are "in contact" well
                                    # before their tight bounding boxes literally intersect
CONTACT_TIME_WINDOW_SECONDS = 1.0  # look for another person within this many seconds of each bowed
                                    # frame, not just the exact same sampled frame - person tracking
                                    # routinely drops the other person for a frame or two mid-fight
CONTACT_CROWD_MAX_OTHERS = 10       # above this many other tracked people, the scene is a busy public
                                     # crowd rather than a small-scale, private confrontation - the
                                     # expanded-box/low-threshold check below (CONTACT_PROXIMITY_MARGIN,
                                     # CONTACT_IOU_MIN) works well for a handful of people (e.g. someone
                                     # grabbing/holding the subject down, or a couple of bystanders
                                     # gathering around someone who just fell - two real falls needing
                                     # this only ever had 4 and 7 other tracked people), but in a crowd
                                     # it just catches ordinary bystander proximity instead (checked
                                     # against real footage: a genuine spit in a crowded public place,
                                     # 18 other tracked people, had a passing bystander's box overlap the
                                     # subject's expanded box the whole clip, purely from camera-angle
                                     # proximity), so past this many people STRONG_CONTACT_IOU_MIN is
                                     # used instead - see below
# --- guard: stooping over someone who is lying on the ground. One of the commonest innocent
# reasons to hold a deep forward bow on CCTV is to look at, check on or help a person who has
# collapsed, and from posture alone that is the same bow as a spit (checked against real
# footage: a man in a station concourse bending over a collapsed passenger scored 0.58 as a
# spit, clearing DECISION_THRESHOLD with no guard touching it). What gives it away is not the
# bow but what the bow is aimed at - somebody laid out flat on the floor right in front of
# them, which is the same measurement the fall detector uses to recognise a body on the ground.
GROUND_ASPECT_MAX = 1.0             # another person's box at least as wide as it is tall - i.e.
                                     # actually laid out on the floor, not merely crouching
GROUND_PERSON_MIN_SECONDS = 1.5     # ...and laid out that way this long, so a momentary clipped or
                                     # half-occluded detection (which can be any shape at all) isn't
                                     # mistaken for a body on the ground
GROUND_PERSON_IOU_MIN = 0.15        # ...and close enough to the subject to be what they're stooping
                                     # over, rather than just somewhere in the same shot (checked
                                     # against real footage: the collapsed passenger overlapped the
                                     # man stooping over him by 0.33, while the only genuine spit
                                     # clip with anyone lying down anywhere near it managed 0.05)
PERSON_DOWN_PENALTY = 0.3

STRONG_CONTACT_IOU_MIN = 0.4        # in a crowded scene (see CONTACT_CROWD_MAX_OTHERS), only a strong,
                                     # literal (unexpanded) box overlap this high is trusted as real
                                     # physical contact - checked against 5 real fight clips (14-41
                                     # other tracked people each): unexpanded overlap during the actual
                                     # grapple/exchange always reached >=0.55, while the crowded spit
                                     # clip's merely-nearby bystander never exceeded 0.27 unexpanded

# --- final scoring ---
# Weight of each scenario in the final confidence (must add up to 1.0). W_BOW and W_HAND are
# unchanged - the two most validated signals - the new W_MOUTH is carved out of W_SIDE/W_FACE/
# W_OBJ instead (which together still add up to the same 0.35 they always did).
W_BOW, W_HAND, W_MOUTH, W_SIDE, W_FACE, W_OBJ = 0.35, 0.30, 0.15, 0.05, 0.05, 0.10
DECISION_THRESHOLD = 0.5   # confidence needed to call it "spitting detected"
GAIT_PENALTY = 0.3         # confidence multiplier applied if the person looks like they're walking
CONTACT_PENALTY = 0.3      # confidence multiplier applied if another person's body overlapped theirs during the bow

ACTIVITY_FRAMES_DIR = "evidence/manual_detections/spit/kinematic"
os.makedirs(ACTIVITY_FRAMES_DIR, exist_ok=True)

# Which keypoints connect to which, just for drawing the skeleton on evidence images
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def detect_activity(filepath: str, person_model, tracks_by_segment: list = None) -> dict:
    """Main entry point: detects motion, checks posture in each motion window, returns a verdict.

    tracks_by_segment lets a caller that runs several detectors over one clip hand
    in the pose pass it has already done, instead of each detector repeating the
    identical work - see analyze_action/action_analyzer.py. Left out, this runs the
    pass itself, so the standalone endpoint behaves exactly as before."""
    # step 1: find where things move in the clip
    # motion_events = detect_motion(filepath)  # motion filter now runs once in analyze_action/action_analyzer.py
    # segments = group_motion_segments(motion_events, fps, frame_count)
    fps, segments = whole_clip_segments(filepath)

    # step 2: for each motion window, track the person and score their posture
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
            other_tracks = {
                tid: recs for tid, recs in tracks.items()
                if tid != track_id and len(recs) >= MIN_TRACK_RECORDS
            }
            confidence, signals = score_event(filepath, features, other_tracks)
            candidates.append({"track_id": track_id, "confidence": confidence, "signals": signals})

    # step 3: keep only the events confident enough to call "spitting"
    confirmed = sorted(
        (c for c in candidates if c["confidence"] >= DECISION_THRESHOLD),
        key=lambda c: c["confidence"],
        reverse=True,
    )

    return {
        "spitting_detected": len(confirmed) > 0,
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


def _pitch_angle(vec) -> float:
    """Angle of a vector from straight-up, in degrees (0 = vertical)."""
    return math.degrees(math.atan2(float(vec[0]), -float(vec[1])))


def _confidence_smooth(values: list, radius: int = SMOOTH_RADIUS) -> list:
    """Averages each value with its nearby neighbors to reduce frame-to-frame jitter."""
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        window = values[lo:hi]
        out[i] = sum(window) / len(window)
    return out


def _fill_missing(values: list, valid: list, max_gap: int = MAX_FILL_GAP_FRAMES) -> list:
    """Fills gaps (where valid=False) with the last known good value, but only across a
    short run of consecutive invalid frames (max_gap). A longer run is left as-is instead
    of assumed to still hold the last reading."""
    filled = list(values)
    last_valid = None
    gap = 0
    for i in range(len(filled)):
        if valid[i]:
            last_valid = filled[i]
            gap = 0
        elif last_valid is not None and gap < max_gap:
            filled[i] = last_valid
            gap += 1
        else:
            gap += 1
    return filled


def compute_kinematics(records: list) -> dict | None:
    """Turns a person's raw per-frame keypoints into the posture signals used for scoring.
    Returns None if there isn't enough data to say anything meaningful."""
    if len(records) < MIN_TRACK_RECORDS:
        return None

    n = len(records)
    ts = [r["t"] for r in records]
    box_heights = [r["box_height"] or 1.0 for r in records]

    def kp(i, idx):
        return np.asarray(records[i]["kpts_xy"][idx])

    def kconf(i, idx):
        return float(records[i]["kpts_conf"][idx])

    head_pitch, trunk_pitch, side_tilt, hand_to_mouth, face_conf = [], [], [], [], []
    shoulder_valid, hip_valid, nose_valid, wrist_valid = [], [], [], []

    for i in range(n):
        bh = box_heights[i]
        sh_ok = kconf(i, L_SHOULDER) >= CONF_FLOOR and kconf(i, R_SHOULDER) >= CONF_FLOOR
        hip_ok = kconf(i, L_HIP) >= CONF_FLOOR and kconf(i, R_HIP) >= CONF_FLOOR
        nose_ok = kconf(i, NOSE) >= CONF_FLOOR
        l_wr_ok, r_wr_ok = kconf(i, L_WRIST) >= CONF_FLOOR, kconf(i, R_WRIST) >= CONF_FLOOR

        shoulder_valid.append(sh_ok)
        hip_valid.append(hip_ok)
        nose_valid.append(nose_ok)
        wrist_valid.append(l_wr_ok or r_wr_ok)

        mid_shoulder = (kp(i, L_SHOULDER) + kp(i, R_SHOULDER)) / 2 if sh_ok else np.zeros(2)
        mid_hip = (kp(i, L_HIP) + kp(i, R_HIP)) / 2 if hip_ok else np.zeros(2)
        nose = kp(i, NOSE)

        # scenario 1 data: how far the head is bowed relative to the trunk
        head_pitch.append(_pitch_angle(nose - mid_shoulder) if (sh_ok and nose_ok) else 0.0)
        trunk_pitch.append(_pitch_angle(mid_shoulder - mid_hip) if (sh_ok and hip_ok) else 0.0)
        # scenario 3 data: shoulders tilted to one side
        side_tilt.append(abs(float(kp(i, L_SHOULDER)[1] - kp(i, R_SHOULDER)[1])) / bh if sh_ok else 0.0)

        # scenario 2 data: closest wrist-to-nose distance
        if (l_wr_ok or r_wr_ok) and nose_ok:
            dists = []
            if l_wr_ok:
                dists.append(float(np.linalg.norm(kp(i, L_WRIST) - nose)))
            if r_wr_ok:
                dists.append(float(np.linalg.norm(kp(i, R_WRIST) - nose)))
            hand_to_mouth.append(min(dists) / bh)
        else:
            hand_to_mouth.append(None)  # unknown for this frame - excluded from stats, not treated as "far"

        # scenario 4 data: how confidently the face is even visible
        face_conf.append((kconf(i, NOSE) + kconf(i, L_EYE) + kconf(i, R_EYE)) / 3.0)

    if not any(shoulder_valid):
        return None

    head_valid = [s and no for s, no in zip(shoulder_valid, nose_valid)]
    trunk_valid = [s and h for s, h in zip(shoulder_valid, hip_valid)]
    hand_valid = [w and no for w, no in zip(wrist_valid, nose_valid)]

    head_pitch = _confidence_smooth(_fill_missing(head_pitch, head_valid))
    trunk_pitch = _confidence_smooth(_fill_missing(trunk_pitch, trunk_valid))
    side_tilt = _confidence_smooth(_fill_missing(side_tilt, shoulder_valid))
    face_conf = _confidence_smooth(face_conf)
    # hand_to_mouth stays raw/unfilled: filling it in would let one bad frame
    # quietly set the "hand near mouth" baseline for the whole event

    neck_flex = [hp - tp for hp, tp in zip(head_pitch, trunk_pitch)]  # net head bow, trunk angle removed

    # scenario 6 data: how much the hips move frame-to-frame (walking check).
    # Skipped when the box size itself jumped a lot between frames - that's a
    # camera zoom/cut, not the person actually moving, and would otherwise be
    # misread as fast walking.
    hip_disp = []
    for i in range(1, n):
        if hip_valid[i] and hip_valid[i - 1]:
            dt = ts[i] - ts[i - 1]
            bh_prev, bh_now = box_heights[i - 1], box_heights[i]
            scale_change = abs(bh_now - bh_prev) / bh_prev if bh_prev else 1.0
            if dt > 0 and scale_change <= ZOOM_GUARD_RATIO:
                mid_hip_now = (kp(i, L_HIP) + kp(i, R_HIP)) / 2
                mid_hip_prev = (kp(i - 1, L_HIP) + kp(i - 1, R_HIP)) / 2
                hip_disp.append(float(np.linalg.norm(mid_hip_now - mid_hip_prev)) / box_heights[i] / dt)

    # "baseline" = the first bit of the event, used as each signal's normal/resting level
    baseline_end_t = ts[0] + SEGMENT_PAD_SECONDS
    baseline_indices = [i for i, t in enumerate(ts) if t <= baseline_end_t] or list(range(n))

    def _median(values, indices):
        return float(np.median([values[i] for i in indices])) if indices else 0.0

    return {
        "ts": ts,
        "records": records,
        "neck_flex": neck_flex,
        "side_tilt": side_tilt,
        "hand_to_mouth": hand_to_mouth,   # raw values; some entries may be None (unknown)
        "hand_valid": hand_valid,
        "face_conf": face_conf,
        "hip_disp": hip_disp,
        "baseline_indices": baseline_indices,
        "baseline_side_tilt": _median(side_tilt, baseline_indices),
        "baseline_face_conf": _median(face_conf, baseline_indices),
    }


def _read_frame_at(filepath: str, frame_number: int):
    """Reads one specific frame from a video file."""
    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = video.read()
    video.release()
    return frame if ok else None


def _two_cluster_split(values: list):
    """Splits a list of numbers into its two dominant levels (e.g. "bowed" vs "upright").
    Returns (assign, level_a, level_b), or None if the values don't really split into two groups."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 4:
        return None
    center_a, center_b = float(arr.min()), float(arr.max())
    if center_b - center_a < 1e-6:
        return None
    for _ in range(15):
        assign = np.abs(arr - center_b) < np.abs(arr - center_a)
        if not assign.any() or assign.all():
            return None  # everything landed in one group - no real two-level pattern here
        new_a, new_b = float(arr[~assign].mean()), float(arr[assign].mean())
        if abs(new_a - center_a) < 1e-3 and abs(new_b - center_b) < 1e-3:
            center_a, center_b = new_a, new_b
            break
        center_a, center_b = new_a, new_b
    assign = np.abs(arr - center_b) < np.abs(arr - center_a)
    return assign, center_a, center_b


def _earliest_close_hand_index(indices, hand_series: list, hand_valid: list, baseline_hand):
    """Among `indices`, the EARLIEST frame where the hand was confidently close to the mouth -
    relative to this person's own baseline hand-to-nose distance where available (same style
    as SUSTAINED_HAND_CLOSE_RATIO elsewhere), else falling back to the flat HAND_ABS_MAX cutoff
    (baseline_hand is None when the opening baseline window itself never got a confident
    nose/wrist reading - checked against real footage where that happened for a genuine spit
    clip, silently failing every baseline-relative check downstream if there's no fallback).
    Earliest, not closest overall: a real spit is delivered once, and a later, even-closer
    moment is more likely incidental aftermath (wiping the mouth, adjusting something) than a
    second spit - checked against real footage where exactly that happened, with the later
    moment's closer approach otherwise winning out over the real, earlier one. Returns None if
    nothing in `indices` ever got confidently close."""
    threshold = baseline_hand * SUSTAINED_HAND_CLOSE_RATIO if baseline_hand and baseline_hand > 1e-6 else HAND_ABS_MAX
    close = [
        i for i in indices
        if hand_valid[i] and hand_series[i] <= threshold and hand_series[i] <= HAND_ABS_MAX
    ]
    return min(close) if close else None


def _mouth_crop(frame, record: dict):
    """Crops a small area centered just below the nose keypoint - the closest available
    proxy for the mouth, since COCO-17 has no dedicated mouth/lip point - and resizes it
    to a fixed size, so it can be compared across frames regardless of exactly where or
    how big the head is in the full image. A wider head-area crop was tried first, but
    it's dominated by head/shoulder rotation (turning to leave changes far more pixels
    than a thin spit stream does), so this stays tight and mouth-specific instead."""
    if record["kpts_conf"][NOSE] < CONF_FLOOR:
        return None
    box_height = record["box_height"]
    size = box_height * MOUTH_CROP_SIZE_RATIO
    nose_x, nose_y = record["kpts_xy"][NOSE]
    cx, cy = float(nose_x), float(nose_y) + box_height * 0.06
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(cx - size / 2)), max(0, int(cy - size / 2))
    x2, y2 = min(width, int(cx + size / 2)), min(height, int(cy + size / 2))
    region = frame[y1:y2, x1:x2]
    return cv2.resize(region, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE)) if region.size else None


def _read_frames_at(filepath: str, frame_numbers: set) -> dict:
    """Reads a set of specific frames from a video file in one sequential pass - faster
    than seeking to each one individually when they're all clustered close together."""
    if not frame_numbers:
        return {}
    remaining = set(frame_numbers)
    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, min(remaining))
    images = {}
    frame_number = min(remaining)
    while remaining:
        ok, frame = video.read()
        if not ok:
            break
        if frame_number in remaining:
            images[frame_number] = frame
            remaining.discard(frame_number)
        frame_number += 1
    video.release()
    return images


def _spit_visibility_scores(filepath: str, core_indices: list, search_indices: list, features: dict) -> dict:
    """Scores how much each frame in `search_indices` differs from the *typical* mouth-
    area appearance across the whole held-bow window (`core_indices`) - a real spit
    stream/splash briefly changes that small patch of image in a way the steady bowed
    posture mostly doesn't, regardless of the saliva's color. Comparing against a median
    image (the "normal" look of this specific moment in this specific clip) rather than
    just the previous frame is more robust to ordinary frame-to-frame jitter, which was
    otherwise close enough in magnitude to the real signal to bury it."""
    frame_numbers = {i: features["records"][i]["frame"] for i in core_indices}
    images = _read_frames_at(filepath, set(frame_numbers.values()))

    thumbs = {}
    for i in core_indices:
        frame = images.get(frame_numbers[i])
        thumb = _mouth_crop(frame, features["records"][i]) if frame is not None else None
        if thumb is not None:
            thumbs[i] = thumb
    if not thumbs:
        return {}

    median_image = np.median(np.stack(list(thumbs.values())), axis=0).astype(np.uint8)
    return {
        i: float(np.mean(cv2.absdiff(thumbs[i], median_image)))
        for i in search_indices if i in thumbs
    }


def score_event(filepath: str, features: dict, other_tracks: dict) -> tuple:
    """Scores one tracked person's posture across a motion window and returns
    (confidence, signals) - confidence is 0-1, signals is a breakdown of every
    scenario check for debugging/evidence. other_tracks is every other person
    tracked in the same window, used only for the contact check below."""
    ts = features["ts"]
    neck_flex = features["neck_flex"]
    hand_series, hand_valid = features["hand_to_mouth"], features["hand_valid"]
    baseline_indices = features["baseline_indices"]
    n = len(ts)

    baseline_hand_indices = [i for i in baseline_indices if hand_valid[i]]
    baseline_hand = (
        float(np.median([hand_series[i] for i in baseline_hand_indices]))
        if baseline_hand_indices else None
    )

    # --- scenario 1: was the head bowed, and held that way for a while? ---
    cluster_result = _two_cluster_split(neck_flex)
    bow_score = 0.0
    swing_deg = 0.0
    bowed_std = None
    bowed_duration = 0.0
    bowed_indices: list = []
    reference_neck = 0.0   # whichever "upright" reference the eventual bow_peak search below
                            # should measure distance from - set once we know which of the two
                            # paths (held bow or quick lurch) actually supplied bow_score
    held_bow_ok = False    # True only for the held-bow path below, never the quick lurch one -
                            # some later checks (e.g. sustained-hand) only make sense against an
                            # actual multi-second hold, not a deliberately narrow lurch window

    if cluster_result is not None:
        assign, mean_a, mean_b = cluster_result
        mean_dt = (ts[-1] - ts[0]) / max(1, n - 1)
        # whichever of the two postures is more bowed (bigger angle) is the "bowed" one
        if abs(mean_b) >= abs(mean_a):
            candidate_mask, bowed_mean, other_mean = assign, mean_b, mean_a
        else:
            candidate_mask, bowed_mean, other_mean = ~assign, mean_a, mean_b

        candidate_indices = [i for i in range(n) if candidate_mask[i]]
        other_count = n - len(candidate_indices)
        if candidate_indices and other_count:
            bowed_values = [neck_flex[i] for i in candidate_indices]
            bowed_std = float(np.std(bowed_values))          # how steady the bowed posture was
            bowed_duration = len(candidate_indices) * mean_dt  # how long it was held
            swing_deg = abs(bowed_mean - other_mean)          # how different from the "upright" posture
            stability_ok = bowed_std <= STABILITY_MAX_DEG

            # a bow held long enough but wobblier than STABILITY_MAX_DEG can still be trusted
            # if the hand genuinely came close to the mouth WITHIN this same window - checked
            # against real footage: a genuine ~4.4s held bow (walking toward a spot to spit)
            # measured 8.8deg of wobble, just over the stability bar, but the hand dropped to
            # a confident 0.13x box-height from the nose during it; a separate clip's walking
            # GAIT alone produced an almost identical 9.2deg "bowed" cluster with no such
            # corroboration (hand never came confidently close during it). Wobble alone can't
            # tell these two apart - they're within 0.5deg of each other - hand proximity
            # inside the specific window can.
            local_hand_ok = (
                not stability_ok
                and _earliest_close_hand_index(candidate_indices, hand_series, hand_valid, baseline_hand) is not None
            )

            if bowed_duration >= MIN_PLATEAU_SECONDS and (stability_ok or local_hand_ok):
                bow_score = max(0.0, min(1.0, swing_deg / SWING_REF_DEG))
                # only trust this cluster as THE event (and search it for evidence below) once
                # it's actually passed the held-bow bar - otherwise it's just whatever the
                # 2-level split happened to split the clip into (e.g. a walking gait's head
                # bob), and scenario 1b below gets a chance to find the real event instead
                bowed_indices = candidate_indices
                reference_neck = other_mean
                # held_bow_ok (and the neck-deviation-anchored evidence search it enables
                # below) is reserved for a genuinely STABLE hold - one trusted only via hand
                # corroboration (local_hand_ok) still gets treated like a quick lurch for
                # evidence-picking purposes, anchored on the hand instead of neck extremity
                held_bow_ok = stability_ok

    # --- scenario 1b: quick lurch - didn't hold, but a fast deep bow tightly synced with
    # the hand reaching the mouth. Independent of the cluster split above (which requires two
    # stable levels). Anchored on the EARLIEST moment the hand got confidently close to the
    # mouth (relative to this person's own baseline hand-to-nose distance, same style as
    # SUSTAINED_HAND_CLOSE_RATIO elsewhere - an absolute cutoff let noise skim under it, see
    # below), then checks how much the neck swung near that same moment. Earliest, not closest
    # overall: checked against real footage where the person spat, then afterward touched/
    # wiped their mouth even MORE closely than the spit reach itself - picking the single
    # closest approach in the whole clip landed on that later wipe instead of the real, earlier
    # spit. A real spit is delivered once; a later, even-closer approach is more likely
    # incidental aftermath than a second spit. Also checked against real footage where the
    # single biggest neck swing in the whole clip was part of the person's walking gait, not
    # the spit, with the hand only barely (coincidentally, and only for one isolated frame)
    # grazing an absolute threshold there - the baseline-relative bar below doesn't count that
    # as confidently close at all. ---
    quick_lurch_swing_deg = 0.0
    quick_lurch_synced = False
    if bow_score == 0.0:
        baseline_neck = (
            float(np.median([neck_flex[i] for i in features["baseline_indices"]]))
            if features["baseline_indices"] else 0.0
        )
        lurch_peak_index = None

        hand_min_index = _earliest_close_hand_index(range(n), hand_series, hand_valid, baseline_hand)
        if hand_min_index is not None:
            nearby = [
                j for j in range(n)
                if abs(ts[j] - ts[hand_min_index]) <= QUICK_LURCH_HAND_SYNC_SECONDS
            ]
            lurch_peak_index = max(nearby, key=lambda j: abs(neck_flex[j] - baseline_neck))

        if lurch_peak_index is not None:
            quick_lurch_swing_deg = abs(neck_flex[lurch_peak_index] - baseline_neck)
            quick_lurch_synced = True
            bow_score = max(0.0, min(1.0, quick_lurch_swing_deg / QUICK_LURCH_SWING_REF_DEG))
            bowed_indices = [
                j for j in range(n) if abs(ts[j] - ts[hand_min_index]) <= QUICK_LURCH_HAND_SYNC_SECONDS
            ]
            reference_neck = baseline_neck

    # --- pick the frame(s) that actually show the spit. Two things had to be combined to
    # get this right (each tried alone, each wrong on its own on real footage):
    #   - WHEN to look: for a HELD bow (multiple seconds), search only from the deepest
    #     point forward, not the whole hold - people bow down and then spit, and the
    #     transition into the bow has its own big appearance change that isn't the spit (it
    #     briefly scored even higher than the real spit did), and picking the deepest point
    #     itself was also wrong there - the real moment landed ~0.7s after it. A quick lurch
    #     is different: it's over in a fraction of a second, so the kinematic peak (deepest
    #     bow + closest hand, already found above) effectively IS the spit moment - searching
    #     forward from it instead picks up the person snapping back upright immediately after
    #     (checked against real footage where that recovery motion's blur scored higher on raw
    #     pixel diff than the actual bow-with-visible-stream frame).
    #   - WHAT to look for: how much the mouth-area crop differs from its own typical
    #     appearance across the window (not from the previous frame - too close in magnitude
    #     to ordinary jitter to isolate the real signal - and not the whole head/shoulder
    #     area - a head turn changes far more pixels than a thin spit stream does). ---
    t_peak_index = None
    if bowed_indices:
        # ignore the very start/end of the bowed window - that's usually the
        # person walking in or straightening back up, not the spit itself
        event_start_t, event_end_t = ts[bowed_indices[0]], ts[bowed_indices[-1]]
        core_lo, core_hi = event_start_t + EDGE_EXCLUDE_SECONDS, event_end_t - EDGE_EXCLUDE_SECONDS
        if core_hi < core_lo:
            core_lo, core_hi = event_start_t, event_end_t
        core_indices = [i for i in bowed_indices if core_lo <= ts[i] <= core_hi] or bowed_indices

        if held_bow_ok:
            # a genuinely stable hold: the deepest point relative to the "upright" cluster
            # is a reliable anchor, since there's no wobble/noise fighting it
            bow_peak_index = max(core_indices, key=lambda i: abs(neck_flex[i] - reference_neck))
        else:
            # quick lurch, or a wobblier bow only trusted via hand corroboration: neck
            # extremity isn't reliable here (a wide or noisy window can have an unrelated,
            # kinematically-bigger moment elsewhere) - anchor on the hand's own earliest
            # confidently-close approach instead, same principle as scenario 1b above
            bow_peak_index = _earliest_close_hand_index(core_indices, hand_series, hand_valid, baseline_hand)
            if bow_peak_index is None:
                local_hand_valid = [i for i in core_indices if hand_valid[i]]
                bow_peak_index = (
                    min(local_hand_valid, key=lambda i: hand_series[i]) if local_hand_valid
                    else core_indices[len(core_indices) // 2]
                )
        bow_peak_t = ts[bow_peak_index]
        release_indices = [
            i for i in core_indices if bow_peak_t <= ts[i] <= bow_peak_t + RELEASE_WINDOW_AFTER_SECONDS
        ] or core_indices

        visibility = _spit_visibility_scores(filepath, core_indices, release_indices, features)

        if held_bow_ok and visibility:
            t_peak_index = max(visibility, key=visibility.get)
            nearby = [i for i in release_indices if abs(ts[i] - ts[t_peak_index]) <= EVIDENCE_LOCAL_WINDOW_SECONDS]
            nearby_ranked = sorted(nearby, key=lambda i: visibility.get(i, 0.0), reverse=True)
            burst_indices = sorted(nearby_ranked[:EVIDENCE_TOP_N])
        else:
            # quick lurch (or a hand-corroborated wobbly bow, or the nose was never confident
            # enough to crop a mouth area at all): anchor on the kinematic peak itself - it's
            # always included in what gets saved - only using visibility to pick any additional
            # nearby frame(s), never to replace it (checked against real footage where nearby
            # recovery-motion blur outscored the actual bow-with-visible-stream frame on raw
            # pixel diff alone)
            t_peak_index = bow_peak_index
            nearby = [i for i in core_indices if abs(ts[i] - ts[t_peak_index]) <= EVIDENCE_LOCAL_WINDOW_SECONDS]
            nearby_ranked = sorted(nearby, key=lambda i: visibility.get(i, 0.0), reverse=True) if visibility else []
            burst_candidates = [t_peak_index] + [i for i in nearby_ranked if i != t_peak_index]
            burst_indices = sorted(burst_candidates[:EVIDENCE_TOP_N])
    else:
        event_start_t, event_end_t = ts[0], ts[-1]
        t_peak_index = n // 2
        burst_indices = [t_peak_index]
        visibility = {}
    t_peak = ts[t_peak_index]

    # --- scenario 1c: did something actually appear at the mouth? Reuses the same mouth-crop
    # diffing already computed above to pick the evidence frame, but now also scores it: how
    # much the PEAK frame's mouth-area differs from the WINDOW's own typical (mean) diff - a
    # real spit stream/splash should stand out from the window's ordinary jitter level, not
    # just be the frame with the single highest raw diff (lighting/motion blur can produce a
    # high raw diff on their own). Checked against 4 real spit clips: peak-vs-typical delta
    # ranged ~15.6-23 (see MOUTH_VISIBILITY_REF). ---
    mouth_visibility_score = 0.0
    if visibility:
        vis_values = list(visibility.values())
        peak_visibility = max(vis_values)
        typical_visibility = float(np.mean(vis_values))
        mouth_visibility_score = max(0.0, min(1.0, (peak_visibility - typical_visibility) / MOUTH_VISIBILITY_REF))

    # --- scenario 2: did a hand come close to the mouth? ---
    valid_indices = [i for i in range(n) if hand_valid[i]]
    baseline_valid_indices = [i for i in baseline_indices if hand_valid[i]] or valid_indices

    if valid_indices and baseline_valid_indices:
        min_hand = min(hand_series[i] for i in valid_indices)
        baseline_hand = float(np.median([hand_series[i] for i in baseline_valid_indices]))
    else:
        min_hand, baseline_hand = None, None

    if min_hand is None or min_hand > HAND_ABS_MAX or baseline_hand <= 1e-6:
        hand_proximity_score = 0.0
    else:
        hand_proximity_score = max(0.0, min(1.0, 1 - min_hand / baseline_hand))

    # --- guard: was the hand at the mouth for most of the held bow, not just briefly?
    # sustained contact reads as drinking/eating rather than a spit (checked only over the
    # held-bow window, so a quick-lurch event - which has no held window - is unaffected).
    # "Close" here is relative to this person's own baseline hand-to-nose distance, same as
    # hand_proximity_score - a forward-bowed posture alone already brings a hanging arm
    # within an absolute distance cutoff for most of the hold, so only a baseline-relative
    # threshold actually isolates real sustained contact from an ordinary bowed arm. ---
    sustained_hand = False
    if held_bow_ok and bowed_indices and baseline_hand and baseline_hand > 1e-6:
        bowed_hand_valid = [i for i in bowed_indices if hand_valid[i]]
        if bowed_hand_valid:
            close_threshold = baseline_hand * SUSTAINED_HAND_CLOSE_RATIO
            close_count = sum(1 for i in bowed_hand_valid if hand_series[i] <= close_threshold)
            sustained_hand = (close_count / len(bowed_hand_valid)) > SUSTAINED_HAND_FRACTION_MAX

    # --- scenario 3: did the shoulders/head tilt to one side? ---
    side_tilt_delta = max(features["side_tilt"]) - features["baseline_side_tilt"]
    side_tilt_score = max(0.0, min(1.0, side_tilt_delta / SIDE_TILT_REF))

    # --- scenario 4: did the face turn away from the camera at the key moment? ---
    window_lo = max(0, t_peak_index - SMOOTH_RADIUS)
    window_hi = min(n, t_peak_index + SMOOTH_RADIUS + 1)
    face_conf_near_bow = min(features["face_conf"][window_lo:window_hi])
    face_drop = features["baseline_face_conf"] - face_conf_near_bow
    face_occlusion_score = max(0.0, min(1.0, face_drop / FACE_DROP_REF))

    # --- scenario 5: is there some object near the head at that moment? ---
    peak_record = features["records"][t_peak_index]
    peak_frame_image = _read_frame_at(filepath, peak_record["frame"])
    object_near_head = (
        find_object_near_head(peak_frame_image, peak_record["box"], peak_record["box_height"])
        if peak_frame_image is not None else None
    )
    object_score = 1.0 if object_near_head else 0.0

    # --- guard: a bowed-and-held posture with no hand near the mouth, no object near the
    # head, AND no visible change at the mouth-crop isn't trusted alone - that's also what
    # tying a shoe, picking something off the ground, or bowing as a greeting looks like from
    # posture alone ---
    no_corroboration = (
        hand_proximity_score < CORROBORATION_MIN
        and object_score < CORROBORATION_MIN
        and mouth_visibility_score < CORROBORATION_MIN
    )

    # --- scenario 6: is the person walking? (if so, reduce confidence) ---
    hip_disp = features["hip_disp"]
    walking = bool(hip_disp) and (float(np.mean(hip_disp)) > GAIT_DISP_PER_SEC)

    # --- scenario 7: was another person's body near theirs during the bow?
    # (e.g. being grabbed/held down in a fight looks like a held bow too.) Checked against a
    # time window and a margin around the subject's box, not just the exact sampled frame and
    # a literal box overlap - person tracking routinely drops the other person for a frame or
    # two mid-fight, and two people grappling are "in contact" before tight boxes intersect. ---
    contact_with_other = False
    if bowed_indices and other_tracks:
        other_records_flat = [r for records in other_tracks.values() for r in records]
        # a crowded scene makes the expanded-box/low-threshold check below fire on ordinary
        # bystander proximity, not real contact (checked against real footage: a genuine spit in
        # a crowded public place, 18 other tracked people, had a passing bystander's expanded box
        # overlap the subject's for the whole clip purely from camera-angle proximity) - once the
        # scene is this busy, only a strong, literal (unexpanded) body overlap is trusted instead,
        # since real physical contact still produces that even with many bystanders also in frame
        # (checked against 5 real fight clips, 14-41 other tracked people each: unexpanded overlap
        # during the grapple/exchange always reached >=0.55, while the crowded spit clip's merely-
        # nearby bystander never exceeded 0.27 unexpanded)
        crowded_scene = len(other_tracks) > CONTACT_CROWD_MAX_OTHERS
        for i in bowed_indices:
            t = ts[i]
            subject_box_raw = features["records"][i]["box"]
            nearby_records = [r for r in other_records_flat if abs(r["t"] - t) <= CONTACT_TIME_WINDOW_SECONDS]
            if crowded_scene:
                if any(_iou(subject_box_raw, r["box"]) > STRONG_CONTACT_IOU_MIN for r in nearby_records):
                    contact_with_other = True
                    break
            else:
                subject_box = _expand_box(subject_box_raw, CONTACT_PROXIMITY_MARGIN)
                if any(_iou(subject_box, r["box"]) > CONTACT_IOU_MIN for r in nearby_records):
                    contact_with_other = True
                    break

    # --- guard: was somebody lying on the floor right in front of them while they bowed?
    # Then the bow is aimed at that person - checking on or helping them - not at the ground.
    # See GROUND_ASPECT_MAX above. ---
    attending_person_down = False
    if bowed_indices and other_tracks:
        for other_records in other_tracks.values():
            flat_records = [
                r for r in other_records
                if (r["box"][3] - r["box"][1]) / max(1e-6, r["box"][2] - r["box"][0]) <= GROUND_ASPECT_MAX
            ]
            if len(flat_records) < 2:
                continue
            if flat_records[-1]["t"] - flat_records[0]["t"] < GROUND_PERSON_MIN_SECONDS:
                continue
            for i in bowed_indices:
                t = ts[i]
                subject_box = _expand_box(features["records"][i]["box"], CONTACT_PROXIMITY_MARGIN)
                if any(
                    abs(r["t"] - t) <= CONTACT_TIME_WINDOW_SECONDS
                    and _iou(subject_box, r["box"]) > GROUND_PERSON_IOU_MIN
                    for r in flat_records
                ):
                    attending_person_down = True
                    break
            if attending_person_down:
                break

    # --- combine every scenario into one final confidence ---
    raw_confidence = (
        W_BOW * bow_score
        + W_HAND * hand_proximity_score
        + W_MOUTH * mouth_visibility_score
        + W_SIDE * side_tilt_score
        + W_FACE * face_occlusion_score
        + W_OBJ * object_score
    )
    confidence = raw_confidence
    confidence *= GAIT_PENALTY if walking else 1.0
    confidence *= CONTACT_PENALTY if contact_with_other else 1.0
    confidence *= PERSON_DOWN_PENALTY if attending_person_down else 1.0
    confidence *= NO_CORROBORATION_PENALTY if no_corroboration else 1.0
    confidence *= SUSTAINED_HAND_PENALTY if sustained_hand else 1.0
    confidence = max(0.0, min(1.0, confidence))

    signals = {
        "t_peak": round(t_peak, 2),
        "swing_deg": round(swing_deg, 2),
        "bowed_std_deg": round(bowed_std, 2) if bowed_std is not None else None,
        "bowed_duration_s": round(bowed_duration, 2),
        "bow_score": round(bow_score, 3),
        "quick_lurch_swing_deg": round(quick_lurch_swing_deg, 2),
        "quick_lurch_synced": quick_lurch_synced,
        "mouth_visibility_score": round(mouth_visibility_score, 3),
        "min_hand_to_mouth_ratio": round(min_hand, 3) if min_hand is not None else None,
        "hand_proximity_score": round(hand_proximity_score, 3),
        "side_tilt_delta": round(side_tilt_delta, 3),
        "side_tilt_score": round(side_tilt_score, 3),
        "face_occlusion_score": round(face_occlusion_score, 3),
        "object_near_head": object_near_head,
        "object_score": object_score,
        "hip_disp_available": bool(hip_disp),
        "walking_penalty_applied": walking,
        "contact_penalty_applied": contact_with_other,
        "person_down_penalty_applied": attending_person_down,
        "no_corroboration_penalty_applied": no_corroboration,
        "sustained_hand_penalty_applied": sustained_hand,
        "peak_frame": peak_record["frame"],
        "peak_box": peak_record["box"],
        "peak_keypoints": peak_record["kpts_xy"].tolist(),
        "peak_kpts_conf": peak_record["kpts_conf"].tolist(),
        "burst_frames": [
            {
                "t": round(ts[i], 2),
                "frame": features["records"][i]["frame"],
                "box": features["records"][i]["box"],
                "kpts_xy": features["records"][i]["kpts_xy"].tolist(),
                "kpts_conf": features["records"][i]["kpts_conf"].tolist(),
            }
            for i in burst_indices
        ],
        "event_start_t": round(event_start_t, 2),
        "event_end_t": round(event_end_t, 2),
    }
    return confidence, signals


def save_evidence_frame(filepath: str, candidate: dict) -> dict:
    """Draws the person's box + skeleton + confidence label on a spread of frames across
    the held-bow window (not just one guessed frame) and saves each as its own image.
    There's no mouth/lip keypoint to pinpoint the exact instant something becomes visible
    at the mouth, so this saves enough of the window that the real moment is reliably
    included among the saved frames, instead of betting everything on a single guess."""
    signals = candidate["signals"]

    result = {
        "t_peak": signals["t_peak"],
        "confidence": round(candidate["confidence"], 3),
        "frames": [],
        "signals": {
            k: v for k, v in signals.items()
            if k not in ("peak_keypoints", "peak_kpts_conf", "burst_frames")
        },
    }

    for burst in signals["burst_frames"]:
        frame = _read_frame_at(filepath, burst["frame"])
        if frame is None:
            continue

        outlined = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in burst["box"]]
        cv2.rectangle(outlined, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # draw the skeleton (only where keypoints are confident enough to trust)
        keypoints, kpts_conf = burst["kpts_xy"], burst["kpts_conf"]
        for a, b in SKELETON_EDGES:
            if kpts_conf[a] >= CONF_FLOOR and kpts_conf[b] >= CONF_FLOOR:
                pa = (int(keypoints[a][0]), int(keypoints[a][1]))
                pb = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(outlined, pa, pb, (0, 255, 255), 2)
        for idx, (x, y) in enumerate(keypoints):
            if kpts_conf[idx] >= CONF_FLOOR:
                cv2.circle(outlined, (int(x), int(y)), 3, (0, 255, 0), -1)

        label = f"Spit {candidate['confidence']:.0%} | t={burst['t']}s"
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(outlined, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_path = os.path.join(ACTIVITY_FRAMES_DIR, f"spit_track{candidate['track_id']}_t{burst['t']}.jpg")
        cv2.imwrite(frame_path, outlined)
        result["frames"].append({"t": burst["t"], "frame_path": frame_path})

    return result
