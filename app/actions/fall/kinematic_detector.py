# Detects a person falling in a CCTV clip using body posture only - no fine-tuned fall
# model, no LLM/Gemini call, pose-kinematics only. Two different fall signatures are
# checked, since a fall doesn't always end with the person visibly lying on the ground:
#
#   1. Falls onto a visible floor: the person's trunk angle and bounding-box shape flip
#      from upright (tall/narrow box, near-vertical trunk) to down (wide/short box,
#      near-horizontal trunk) within well under a second - not a gradual change like
#      someone slowly sitting down on purpose - and then stays down for a bit, not just
#      a brief dip (a quick duck, dodge, or stumble-and-recover isn't a fall).
#   2. Falls out of view - into a gap, down a stairwell/shaft, off an edge: the person
#      never ends up visibly lying down at all, they just accelerate downward and
#      vanish from the camera's view entirely. This looks nothing like signature 1 (no
#      held horizontal posture to see), so it's checked separately: a fast downward
#      drop in the moments right before the person's track is lost mid-clip, away from
#      the frame edges (so it isn't confused with someone just walking out of shot).
#   3. Falls along the camera's line of sight - collapsing while walking toward or away
#      from the lens, ending up laid out head-away/feet-toward it. Signature 1 measures
#      the trunk angle in the flat image, and a body laid out along that axis still
#      projects as near-vertical, so signature 1 is blind to this by construction (a real
#      collapse measured 1-7deg of trunk angle for its entire clip). What gives it away
#      instead is the body's outline: it goes from a tall standing box to a squat
#      foreshortened one within about a second and never straightens back up. How squat a
#      lying body looks depends entirely on the camera angle, so it's judged against the
#      same person's own outline while they were still upright, not a fixed ratio.
#
# This deliberately doesn't try to guess WHY someone fell (tripped, pushed, fainted,
# slipped, ...) - "due to any reason" per the brief - it only looks at the motion
# signature of going down (visibly or out of view) unexpectedly fast.
#
# Three guards target postures that look like signature 1 for an unrelated reason
# (checked against real footage - all three were confirmed false triggers, not hypothetical):
#   - A bystander crouching down to help someone who already fell also flips from
#     upright to bent-over fast and holds it - the guard checks whether another
#     person was ALREADY down/bent-over nearby at the moment THIS person went down;
#     if so, they're more likely responding to an earlier fall than having one.
#   - Repeated cycling between down and up within the same tracked window (exercise
#     like push-ups/burpees, playing, rolling around) is discounted - a real fall
#     transitions one way and stays, it doesn't cycle.
#   - A deep, fast, held forward bow at the waist (e.g. spitting) swings the trunk
#     angle just as far and fast as a real fall, and holds it just as steadily - the
#     one thing pose alone can still tell them apart is the body's box shape, which a
#     standing bow keeps tall/narrow the whole time but a real fall almost always
#     flips wide/short. A trunk-angle "fall" with no box-shape corroboration at all
#     isn't trusted alone.

import math
import os

import cv2
import numpy as np

# Pose tracking lives in pose_tracking.py, shared with the spit and vandalism
# detectors - all three ran the identical pass over the same frames.
from app.shared.pose_tracking import TRACK_FPS, iou as _iou, track_segment, whole_clip_segments

# note: fall used a more generous SEGMENT_PAD_SECONDS (1.5s vs spit/vandalism's ~0.6s) when this
# ran its own gating - a person who faints/is knocked out stops generating any motion at all right
# after impact, so the padding after the last motion tick had to be long enough on its own to
# capture them staying down, or MIN_DOWN_SECONDS below could never be satisfied for exactly the
# cases that matter most. Worth knowing if motion gating is ever reintroduced.

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# --- step 2: pose tracking settings ---
# TRACK_FPS and the box matching now come from pose_tracking.py (imported above).
CONF_FLOOR = 0.15
SMOOTH_RADIUS = 2
MAX_FILL_GAP_SECONDS = 1.0    # a reading can carry forward across a brief dropout in keypoint
                               # confidence, but not longer - hitting the ground often briefly
                               # scrambles hip/shoulder keypoints, but a long low-confidence
                               # stretch shouldn't read as a posture held steady the whole time
MAX_FILL_GAP_FRAMES = round(MAX_FILL_GAP_SECONDS * TRACK_FPS)

# --- fall signature: a sudden posture flip, held ---
MIN_DOWN_SECONDS = 0.6         # the down posture must be held at least this long
STABILITY_MAX_DEG = 20.0       # how steady the down posture must be to count as "held" - lying on
                                # the ground is noisier for a pose model than standing (it's trained
                                # mostly on upright people), and a person may still be shifting/
                                # struggling, so this is far more forgiving than a standing check
MIN_DOWN_SWING_DEG = 30.0      # below this much trunk-angle swing, the posture change isn't scored as
                                # a fall at all. Leaning, ducking, stumbling and throwing a punch all
                                # tip the trunk some of the way over without anyone going to the
                                # ground, and scoring those proportionally handed out half marks for
                                # a lean (checked against real footage: a fighter tipping 21.9deg, a
                                # bench-kicker 23.6deg and several bowing/spitting subjects 24-30deg
                                # all scored ~0.5 on the old straight-proportional scale, leaving them
                                # one weak corroboration away from crossing DECISION_THRESHOLD). Both
                                # real falls on hand swing 55deg+, so a genuine collapse clears this
                                # with room to spare
DOWN_ANGLE_REF_DEG = 45.0      # trunk-angle swing (upright -> down) at/above this = full score
MAX_TRANSITION_SECONDS = 1.2   # the switch from upright to down must happen within this long to
                                # count as sudden - a real fall (trip, slip, faint) is fast; someone
                                # gradually lying down (stretching, sitting then lying back) takes
                                # noticeably longer than this
SAMPLE_GAP_MAX_SECONDS = 0.4   # two consecutive sightings further apart than this aren't one
                                # continuously-tracked stretch - the person was lost and re-detected
                                # in between. Durations are measured over continuously-tracked
                                # stretches only (see _longest_contiguous_run): counting every "down"
                                # sighting instead, times the track's average sampling interval, let
                                # scattered re-detections across a long clip masquerade as one held
                                # posture (checked against real footage: an intermittently-tracked
                                # fighter, 11 sightings spread over 90s, was credited with a 37.7s
                                # "held" down posture)

# --- aspect-ratio corroboration: does the box shape itself look like a standing or lying body? ---
STANDING_ASPECT_MIN = 1.4     # box height/width at/above this looks like a standing/upright body
LYING_ASPECT_MAX = 1.2        # box height/width at/below this looks like a body lying flat

# --- guard: a deep, fast, held forward bow at the waist (checked against real footage: 4
# separate spitting clips) swings the trunk-pitch angle just as far and fast as a real fall,
# and holds it just as steadily - fall_score alone can't tell a standing bow from a collapse,
# only the box shape can. A person genuinely down on the ground almost always also goes wide/
# short in box shape (checked against both real falls on hand: aspect_score reached 0.47-1.0
# whenever fall_score was real), while a standing bow keeps the body's box tall/narrow the
# whole time (aspect_score pinned at 0.0 in all 4 spitting clips, even though one swung the
# trunk pitch 33.8deg in 0.08s and held it - a fall_score of 0.751 on its own, well past
# DECISION_THRESHOLD). So a fall_score with no box-shape corroboration at all isn't trusted.
NO_ASPECT_CORROBORATION_MIN = 0.15   # aspect_score below this = no real corroboration
NO_ASPECT_CORROBORATION_PENALTY = 0.3

# --- guard: a bystander crouching down to help someone already on the ground also flips
# from upright to bent-over fast and holds it - checked against real footage (a subway
# clip where a passerby crouched to help a fallen man), this was a confirmed false
# trigger. Only counts if the OTHER person was already down/bent-over nearby AT the
# moment this person's own transition happens, not any time during the whole window -
# otherwise it would also suppress the real, earlier faller once a helper arrives. ---
HELPER_PROXIMITY_MARGIN = 0.6
HELPER_TIME_WINDOW_SECONDS = 1.0
HELPER_ASPECT_MAX = 1.0        # the other person's box must be at least as wide as it is tall - i.e.
                                 # actually laid out on the ground - to count as "already down". This
                                 # used to accept anyone merely bent-over-ish (1.6), which on a busy
                                 # or overhead-angled camera is just what ordinary standing people
                                 # measure, so any crowd supplied a fake "someone was already down
                                 # here" and the real faller got written off as a bystander helping
                                 # (checked against real footage: a genuine collapse in a crowded
                                 # concourse, scoring a perfect 1.0 on every fall signal, was
                                 # discounted to 0.35 by passers-by measuring 1.08-1.53). The people
                                 # who really were already down, in the two clips that need this
                                 # guard, all measure 0.51-0.66
HELPER_IOU_MIN = 0.02
HELPER_PENALTY = 0.35

# --- guard: repeated down/up cycling within one window (push-ups, burpees, playing,
# rolling around) isn't a fall - a real fall transitions one way and stays down ---
OSCILLATION_PENALTY = 0.5

# --- fall signature 3: collapsed along the camera's line of sight ---
# Signature 1 measures the trunk angle in the flat image, so it only sees a fall that happens
# ACROSS the camera's view. Someone who collapses while walking toward or away from the lens
# ends up lying head-away/feet-toward the camera, and a body laid out along that axis still
# projects as near-vertical - checked against real footage of a man collapsing in a courtyard,
# whose trunk angle never left 1-7deg for the entire clip (a fall_score of 0.07 for an
# unmistakable collapse). What does change is the body's outline: it goes from a tall, narrow
# standing box to a squat, foreshortened one, and stays that way. That can't be judged against
# a fixed aspect ratio, because how squat a lying body looks depends entirely on the camera
# angle, so it's measured against the SAME person's own outline while they were still upright.
# These thresholds gate hard rather than scoring softly (see COLLAPSE_BASE) - a crouch, a
# kneel and a bow all compact the outline too, and what separates them from a collapse is not
# how far it compacts but that a collapse happens fast, holds, and doesn't get back up.
COLLAPSE_BASELINE_SECONDS = 1.0   # how far back from the moment they go down to look for "this
                                   # person, still upright". Measured immediately before the event
                                   # rather than at the start of the track: a fall is a transition
                                   # between two adjacent moments, and over a long track the person
                                   # walks nearer and further, turns, and is occluded, so their
                                   # outline at the start says nothing about the moment they went down
COLLAPSE_SCALE_CHANGE_MAX = 2.0   # the person's box may not change area by more than this factor
                                   # across the event. The same body going down roughly where it
                                   # stood stays roughly the same size on screen (1.35x for the real
                                   # collapse on hand); a much bigger jump means the box stopped
                                   # describing the same thing - the tracker drifted or merged with
                                   # something else - so its change of shape isn't evidence of
                                   # anything (checked against real footage: a track that slid off a
                                   # distant figure onto a man bent over a weights machine right in
                                   # front of the lens grew 7.3x and read as a textbook collapse)
COLLAPSE_UPRIGHT_FRACTION = 0.9   # within the baseline window, samples at least this much of the
                                   # person's most-upright outline still count as "not yet going
                                   # down" - the clock on how fast they went down starts at the last
                                   # of those, not at the last sample before the threshold was
                                   # crossed (a gradual crouch crosses any threshold in one sampled
                                   # frame too, so measuring from the crossing makes every descent
                                   # look instant)
COLLAPSE_RATIO_MAX = 0.65         # the outline must compact to at most this fraction of that
                                   # upright baseline to count as down at all
COLLAPSE_ASPECT_MAX = 1.5         # ...and be at least this squat in absolute terms, so a very tall
                                   # narrow detection shrinking to merely normal proportions doesn't
                                   # qualify (the two people who rushed in to kneel beside the
                                   # collapsed man measure 1.74 and 1.95 here, against his 1.41)
COLLAPSE_REF_RATIO = 0.30         # compacting to this fraction or below scores full marks
COLLAPSE_HOLD_SECONDS = 1.5       # the compacted outline must hold this long, continuously tracked -
                                   # the single most useful threshold here (checked against real
                                   # footage: people crouching, kneeling, bowing and being knocked
                                   # about in fights compact their outline just as far, but hold it
                                   # 0.25-1.0s before moving again, against 3.5s for a real collapse)
COLLAPSE_TAIL_MAX_SECONDS = 0.5   # ...and must still be compacted when the FOOTAGE ends, not merely
                                   # when this person's track does. Someone who gets back up during
                                   # the clip got up, so they didn't stay down (the fight clips'
                                   # knocked-about fighters recover 3.7-12.3s before their tracks
                                   # end; both real collapses never recover). Measuring it against
                                   # the end of the track would make losing someone count as them
                                   # staying down, which is exactly backwards - checked against real
                                   # footage, a passer-by whose box shrank as they walked off into a
                                   # crowd, and whose track then died with 3.8s of clip left to run
COLLAPSE_BASE = 0.6               # clearing every gate above is itself the detection, so a collapse
                                   # starts here and only the depth of the compaction moves it up -
                                   # a foreshortened collapse can't compact as far as a sideways one
                                   # (0.56 of baseline, against 0.24-0.26 for the two sideways falls),
                                   # so scoring purely on depth would rank the clearest real collapse
                                   # on hand below a fighter briefly going down

# --- fall signature 2: dropped out of view entirely (a gap, a stairwell, an edge) ---
DISAPPEAR_GAP_MIN_SECONDS = 1.5     # a gap at least this long between two consecutive sightings of
                                      # the same track means real tracking was lost (fell out of view
                                      # or was occluded), not just an ordinary brief dropout
DISAPPEAR_LOOKBACK_SECONDS = 0.6    # how far back before the vanish point to measure the drop speed
DISAPPEAR_DROP_MIN_SPEED = 0.5      # downward box-center speed (box-heights/sec) floor, in that
                                      # lookback window, needed to call the disappearance a "dropped
                                      # out of view" rather than just walking away/out of frame at an
                                      # ordinary pace
DISAPPEAR_DROP_REF_SPEED = 2.0      # speed at/above this = full vanish score
DISAPPEAR_EDGE_MARGIN_RATIO = 0.08  # if they vanished within this fraction of a frame edge, treat it
                                      # as an ordinary exit (walked off camera) rather than a fall - a
                                      # fall into a gap/hole typically happens in the frame interior

# --- final scoring ---
# fall_score (the sudden-collapse-and-held signal) requires suddenness, duration, AND
# stability all at once - see score_event - so aspect_score alone can never cross
# DECISION_THRESHOLD; it can only corroborate a fall_score that's already real. The
# vanish_score (signature 2) is scored separately and combined with max(), not summed -
# a single event is either "ended up visibly down" or "vanished out of view", not both.
W_FALL, W_ASPECT = 0.7, 0.3
DECISION_THRESHOLD = 0.5

FALL_FRAMES_DIR = "debug/fall/kinematic"

# Which keypoints connect to which, just for drawing the skeleton on evidence images
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def detect_fall(filepath: str, person_model, tracks_by_segment: list = None) -> dict:
    """Main entry point: detects motion, checks posture change in each motion window,
    returns a verdict.

    tracks_by_segment lets a caller that runs several detectors over one clip hand
    in the pose pass it has already done, instead of each detector repeating the
    identical work - see the live engine. Left out, this runs the
    pass itself, so the standalone endpoint behaves exactly as before."""
    video = cv2.VideoCapture(filepath)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    video.release()

    fps, segments = whole_clip_segments(filepath)

    candidates = []
    for index, segment in enumerate(segments):
        tracks = (
            tracks_by_segment[index] if tracks_by_segment is not None
            else track_segment(filepath, segment, person_model, fps)
        )
        for track_id, records in tracks.items():
            features = compute_kinematics(records, segment["end_t"])
            if features is None:
                continue
            other_tracks = {tid: recs for tid, recs in tracks.items() if tid != track_id}
            confidence, signals = score_event(features, frame_width, frame_height, other_tracks)
            candidates.append({"track_id": track_id, "confidence": confidence, "signals": signals})

    confirmed = sorted(
        (c for c in candidates if c["confidence"] >= DECISION_THRESHOLD),
        key=lambda c: c["confidence"],
        reverse=True,
    )

    return {
        "fall_detected": len(confirmed) > 0,
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
    """Angle of a vector from straight-up, in degrees (0 = vertical, 90 = horizontal)."""
    return math.degrees(math.atan2(float(vec[0]), -float(vec[1])))


def _smooth(values: list, radius: int = SMOOTH_RADIUS) -> list:
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
    short run of consecutive invalid frames (max_gap)."""
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


# --- the bend-over guard -----------------------------------------------------
#
# A trunk that flips from upright to horizontal is the fall signal. The trouble
# is that bending over a desk does exactly that, and nothing else in signature 1
# disagrees: the bend is sudden (well under MAX_TRANSITION_SECONDS), it is held
# (you stay bent), and the swing is 60-80 degrees. All three checks pass, and
# the detector calls it a fall at 80-90%.
#
# What it never asks is whether the person actually went DOWN. Bending is a
# rotation ABOUT the hip, so the hip barely moves; a fall puts the hip on the
# floor. That difference is invisible to trunk angle and to box aspect, but it
# is trivial to measure directly.
#
# Measured over 180 seconds of desk footage against the four known fall clips,
# hip descent in units of the person's own standing box height:
#
#     real falls    0.61  0.75  0.77  2.29      (min 0.61)
#     desk footage  median 0.06, max 0.42       (0.42 = sitting down)
#
# A clean gap, so 0.50 sits in the middle of it with room on both sides. Two
# deliberate choices:
#
#   - Normalised by the person's own standing height, so it means the same at
#     any distance from the camera and needs no per-camera tuning.
#   - Applied ONLY to signature 1 (the trunk flip). The collapse and vanish
#     signatures already require real evidence of going down, and gating them
#     too would reject a fall along the camera's line of sight - the exact case
#     they exist to catch.
#
# Two earlier attempts at this failed on measurement and are worth not
# repeating: ankle height (bending reads 1.14, a real fall 1.10) and leg
# direction (desk 21% of frames horizontal, real falls only 6.7% - sitting puts
# the legs horizontal, so it separates sitting from standing, not fallen from
# upright).
MIN_HIP_DESCENT = 0.50

# The same idea for the collapse signature, which needs its own - and lower -
# bar. Blocking the trunk flip alone was not enough: a desk segment still scored
# 0.818 through "collapsed", because standing up and sitting down compacts the
# body's outline exactly the way going down does (ratio 0.46), and the old test
# asked only about the outline.
#
# It is lower because this signature already demands a great deal before it will
# fire - clearly upright beforehand, compacted within MAX_TRANSITION_SECONDS,
# held COLLAPSE_HOLD_SECONDS, still down when the footage ends - so the descent
# check only has to rule out the one case those miss: compacting without
# actually going anywhere. Measured over the collapse window itself:
#
#     real falls    0.41  0.60  0.62
#     desk segment  0.03
#
# fall-3 is why this cannot simply reuse 0.50 - it descends 0.41 here and would
# have been thrown away. 0.20 sits an order of magnitude above the desk case and
# comfortably under every real one.
MIN_HIP_DESCENT_COLLAPSE = 0.20

# How far back to look for "how high were their hips before this". Long enough
# to span a fall, short enough that it means the current posture rather than
# where they were standing a minute ago.
HIP_BASELINE_SECONDS = 2.0


def _hip_descent(features: dict, indices: list) -> float:
    """How far the hip dropped, in the person's own standing box heights.

    For each candidate frame, looks back HIP_BASELINE_SECONDS for that person's
    most upright moment - smallest hip y, i.e. highest on screen - and measures
    the drop from there, scaled by how tall their box was at that moment.
    Returns the largest drop found, or 0.0 if it cannot be measured."""
    ts = features["ts"]
    hip_y = features["hip_y"]
    records = features["records"]

    worst = 0.0
    for i in indices:
        if hip_y[i] is None:
            continue
        earlier = [
            j for j in range(i)
            if hip_y[j] is not None and ts[i] - ts[j] <= HIP_BASELINE_SECONDS
        ]
        if not earlier:
            continue

        highest = min(earlier, key=lambda j: hip_y[j])
        standing_height = max(records[highest]["box_height"], 1e-6)
        worst = max(worst, (hip_y[i] - hip_y[highest]) / standing_height)

    return worst


def compute_kinematics(records: list, segment_end_t: float) -> dict | None:
    """Turns a person's raw per-frame keypoints + box shape into the posture signals
    used for scoring. Returns None if there isn't enough data to say anything."""
    if len(records) < 4:
        return None

    n = len(records)
    ts = [r["t"] for r in records]

    def kp(i, idx):
        return np.asarray(records[i]["kpts_xy"][idx])

    def kconf(i, idx):
        return float(records[i]["kpts_conf"][idx])

    trunk_pitch = [0.0] * n
    aspect = [0.0] * n
    trunk_valid = [False] * n
    hip_y = [None] * n          # for the descent check - see _hip_descent below

    for i in range(n):
        box_height = max(1e-6, records[i]["box_height"])
        box_width = max(1e-6, records[i]["box_width"])
        aspect[i] = box_height / box_width  # tall/narrow (standing) = big, wide/short (lying) = small

        sh_ok = kconf(i, L_SHOULDER) >= CONF_FLOOR and kconf(i, R_SHOULDER) >= CONF_FLOOR
        hip_ok = kconf(i, L_HIP) >= CONF_FLOOR and kconf(i, R_HIP) >= CONF_FLOOR
        trunk_valid[i] = sh_ok and hip_ok
        if trunk_valid[i]:
            mid_shoulder = (kp(i, L_SHOULDER) + kp(i, R_SHOULDER)) / 2
            mid_hip = (kp(i, L_HIP) + kp(i, R_HIP)) / 2
            trunk_pitch[i] = abs(_pitch_angle(mid_shoulder - mid_hip))  # 0 = upright, ~90 = horizontal
            hip_y[i] = float(mid_hip[1])

    if not any(trunk_valid):
        return None

    trunk_pitch = _smooth(_fill_missing(trunk_pitch, trunk_valid))
    aspect = _smooth(aspect)

    # --- did the track vanish partway through, right after tracking was lost? (e.g.
    # they fell out of the camera's view entirely, rather than ending up visibly down
    # on the floor) - either a large gap appears between two sightings, or the track's
    # last sighting ends notably before the tracked segment itself does ---
    vanish_at = None
    for i in range(n - 1):
        if ts[i + 1] - ts[i] >= DISAPPEAR_GAP_MIN_SECONDS:
            vanish_at = i
            break
    if vanish_at is None and segment_end_t - ts[-1] >= DISAPPEAR_GAP_MIN_SECONDS:
        vanish_at = n - 1

    return {
        "ts": ts,
        "records": records,
        "trunk_pitch": trunk_pitch,
        "aspect": aspect,
        "hip_y": hip_y,
        "vanish_at": vanish_at,
        "segment_end_t": segment_end_t,
    }


def _two_cluster_split(values: list):
    """Splits a list of numbers into its two dominant levels (e.g. "standing" vs "down").
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


def _read_frame_at(filepath: str, frame_number: int):
    """Reads one specific frame from a video file."""
    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = video.read()
    video.release()
    return frame if ok else None


def _longest_contiguous_run(indices: list, ts: list) -> list:
    """The longest stretch of `indices` the person was actually tracked through without
    being lost - consecutive sightings no further apart than SAMPLE_GAP_MAX_SECONDS.
    Everything about a fall is a claim over time ("flipped within a second", "stayed down
    for two"), and those claims only mean something across sightings that are genuinely
    consecutive - see SAMPLE_GAP_MAX_SECONDS."""
    best: list = []
    current: list = []

    def duration(run):
        return ts[run[-1]] - ts[run[0]] if len(run) > 1 else -1.0

    for i in indices:
        if current and ts[i] - ts[current[-1]] > SAMPLE_GAP_MAX_SECONDS:
            if duration(current) > duration(best):
                best = current
            current = []
        current.append(i)
    return current if duration(current) > duration(best) else best


def score_event(features: dict, frame_width: float, frame_height: float, other_tracks: dict) -> tuple:
    """Scores one tracked person's posture change and returns (confidence, signals) -
    confidence is 0-1, signals is a breakdown of every check for debugging/evidence.
    other_tracks is every other person tracked in the same window, used only for the
    helper-response guard below."""
    ts = features["ts"]
    trunk_pitch = features["trunk_pitch"]
    aspect = features["aspect"]
    n = len(ts)

    cluster_result = _two_cluster_split(trunk_pitch)
    fall_score = 0.0
    hip_descent = 0.0
    swing_deg = 0.0
    down_std = None
    down_duration = 0.0
    transition_seconds = None
    down_indices: list = []
    oscillating = False
    helper_response = False

    if cluster_result is not None:
        assign, mean_a, mean_b = cluster_result
        # whichever posture is more horizontal (bigger pitch angle) is the "down" one
        if mean_b >= mean_a:
            down_mask, down_mean, upright_mean = assign, mean_b, mean_a
        else:
            down_mask, down_mean, upright_mean = ~assign, mean_a, mean_b

        all_down_indices = [i for i in range(n) if down_mask[i]]
        upright_indices = [i for i in range(n) if not down_mask[i]]
        # only the stretch they were continuously tracked through can support a claim about
        # how long the posture was held, or how fast they got into it - see
        # _longest_contiguous_run
        down_indices = _longest_contiguous_run(all_down_indices, ts)

        # how many separate times did they go down? (down, up, down again, ...) - a real
        # fall transitions once and stays; repeated cycling is exercise/play, not a fall
        down_runs = 0
        prev_down = False
        for is_down in down_mask:
            if is_down and not prev_down:
                down_runs += 1
            prev_down = bool(is_down)
        oscillating = down_runs >= 2

        if len(down_indices) > 1 and upright_indices:
            down_values = [trunk_pitch[i] for i in down_indices]
            down_std = float(np.std(down_values))
            down_duration = ts[down_indices[-1]] - ts[down_indices[0]]
            swing_deg = down_mean - upright_mean

            # how fast was the switch from upright to down?
            upright_before = [i for i in upright_indices if i < down_indices[0]]
            if upright_before:
                transition_seconds = ts[down_indices[0]] - ts[max(upright_before)]

            # fall_score requires ALL THREE at once: it happened fast, it flipped the
            # posture by a real amount, and it was then held steady - any one of these
            # alone (e.g. just "wide box" or just "held still") isn't enough on its own
            sudden = transition_seconds is not None and transition_seconds <= MAX_TRANSITION_SECONDS
            held = down_duration >= MIN_DOWN_SECONDS and down_std <= STABILITY_MAX_DEG

            # ...and they have to have actually gone DOWN, not just leaned over.
            # Without this, bending over a desk clears every other check here -
            # see MIN_HIP_DESCENT.
            hip_descent = _hip_descent(features, down_indices)
            descended = hip_descent >= MIN_HIP_DESCENT

            if sudden and held and descended:
                fall_score = max(0.0, min(1.0,
                    (swing_deg - MIN_DOWN_SWING_DEG) / (DOWN_ANGLE_REF_DEG - MIN_DOWN_SWING_DEG)
                ))

    # --- fall signature 3: the body's outline collapsed and stayed collapsed, measured
    # against this same person while they were still upright. Catches the fall that
    # signature 1 is blind to by construction - one along the camera's line of sight, where
    # the trunk angle in the image never flips. See COLLAPSE_RATIO_MAX above. ---
    collapse_score = 0.0
    collapse_ratio = None
    collapse_hold = 0.0
    collapse_indices: list = []

    compact = [i for i in range(n) if aspect[i] <= COLLAPSE_ASPECT_MAX]
    run = _longest_contiguous_run(compact, ts)
    if len(run) > 1:
        compact_set = set(compact)
        # how they looked in the second before going down - their most upright moment there
        baseline_window = [
            i for i in range(run[0])
            if ts[run[0]] - ts[i] <= COLLAPSE_BASELINE_SECONDS and i not in compact_set
        ]
        if len(baseline_window) >= 3:
            baseline_aspect = max(aspect[i] for i in baseline_window)
            still_upright = [
                i for i in baseline_window
                if aspect[i] >= baseline_aspect * COLLAPSE_UPRIGHT_FRACTION
            ]
            collapse_transition = ts[run[0]] - ts[max(still_upright)] if still_upright else None
            collapse_hold = ts[run[-1]] - ts[run[0]]
            still_down_at_end = features["segment_end_t"] - ts[run[-1]] <= COLLAPSE_TAIL_MAX_SECONDS

            def _area(i):
                record = features["records"][i]
                return max(1e-6, record["box_height"] * record["box_width"])

            baseline_area = float(np.median([_area(i) for i in baseline_window]))
            collapsed_area = float(np.median([_area(i) for i in run]))
            scale_change = max(collapsed_area / baseline_area, baseline_area / collapsed_area)

            # only a person seen clearly upright just beforehand can be seen to collapse
            if (
                baseline_aspect > STANDING_ASPECT_MIN
                and collapse_hold >= COLLAPSE_HOLD_SECONDS
                and collapse_transition is not None
                and collapse_transition <= MAX_TRANSITION_SECONDS
                and still_down_at_end
                and scale_change <= COLLAPSE_SCALE_CHANGE_MAX
            ):
                ratio = float(np.median([aspect[i] for i in run])) / baseline_aspect
                # ...and the hip has to have gone down with the outline. Sitting
                # down compacts the box just as well as falling does - see
                # MIN_HIP_DESCENT_COLLAPSE.
                if (ratio <= COLLAPSE_RATIO_MAX
                        and _hip_descent(features, run) >= MIN_HIP_DESCENT_COLLAPSE):
                    collapse_indices = run
                    collapse_ratio = ratio
                    hip_descent = max(hip_descent, _hip_descent(features, run))
                    depth = (COLLAPSE_RATIO_MAX - ratio) / (COLLAPSE_RATIO_MAX - COLLAPSE_REF_RATIO)
                    collapse_score = max(0.0, min(1.0, COLLAPSE_BASE + (1.0 - COLLAPSE_BASE) * depth))

    # --- guard: was another person already down nearby at the moment THIS person went
    # down? More consistent with crouching to help someone who had already fallen than
    # falling themselves. Only checked against the other person's state at/before this
    # person's own transition (not any time during the whole window), so it doesn't also
    # suppress the real, earlier faller once a helper arrives afterward. ---
    onset_indices = down_indices or collapse_indices
    if other_tracks and onset_indices:
        onset_t = ts[onset_indices[0]]
        subject_box = _expand_box(features["records"][onset_indices[0]]["box"], HELPER_PROXIMITY_MARGIN)
        for other_records in other_tracks.values():
            nearby = [
                r for r in other_records
                if onset_t - HELPER_TIME_WINDOW_SECONDS <= r["t"] <= onset_t
            ]
            for r in nearby:
                other_aspect = r["box_height"] / max(1e-6, r["box_width"])
                if other_aspect <= HELPER_ASPECT_MAX and _iou(subject_box, r["box"]) > HELPER_IOU_MIN:
                    helper_response = True
                    break
            if helper_response:
                break

    # --- aspect-ratio corroboration: does the box shape itself look like a lying body? ---
    aspect_score = 0.0
    if down_indices:
        down_aspect = float(np.median([aspect[i] for i in down_indices]))
        if down_aspect <= LYING_ASPECT_MAX:
            aspect_score = 1.0
        elif down_aspect < STANDING_ASPECT_MIN:
            aspect_score = 1.0 - (down_aspect - LYING_ASPECT_MAX) / (STANDING_ASPECT_MIN - LYING_ASPECT_MAX)

    # --- guard: fall_score alone (a fast, held trunk-angle swing) is also exactly what a deep
    # standing bow looks like - only trusted at full weight once the box shape itself also
    # corroborates a body actually going down, not just the trunk pitching forward. See
    # NO_ASPECT_CORROBORATION_MIN above. ---
    no_aspect_corroboration = fall_score > 0.0 and aspect_score < NO_ASPECT_CORROBORATION_MIN

    visible_confidence = max(0.0, min(1.0, W_FALL * fall_score + W_ASPECT * aspect_score))
    visible_confidence *= OSCILLATION_PENALTY if oscillating else 1.0
    visible_confidence *= HELPER_PENALTY if helper_response else 1.0
    visible_confidence *= NO_ASPECT_CORROBORATION_PENALTY if no_aspect_corroboration else 1.0
    visible_confidence = max(0.0, min(1.0, visible_confidence))

    # --- fall signature 2: dropped out of view right after a fast downward drop, e.g.
    # falling into a gap/hole/stairwell rather than landing visibly on the floor ---
    vanish_score = 0.0
    vanish_speed = None
    vanish_at = features["vanish_at"]
    if vanish_at is not None:
        lookback = [i for i in range(vanish_at + 1) if ts[vanish_at] - ts[i] <= DISAPPEAR_LOOKBACK_SECONDS]
        if len(lookback) >= 2:
            i0, i1 = lookback[0], lookback[-1]
            box0, box1 = features["records"][i0]["box"], features["records"][i1]["box"]
            box_h = max(1e-6, features["records"][i1]["box_height"])
            y0, y1 = (box0[1] + box0[3]) / 2, (box1[1] + box1[3]) / 2
            dt = ts[i1] - ts[i0]
            if dt > 0:
                vanish_speed = (y1 - y0) / box_h / dt  # positive = moving down

            last_box = features["records"][vanish_at]["box"]
            cx, cy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
            near_edge = (
                cx <= frame_width * DISAPPEAR_EDGE_MARGIN_RATIO
                or cx >= frame_width * (1 - DISAPPEAR_EDGE_MARGIN_RATIO)
                or cy <= frame_height * DISAPPEAR_EDGE_MARGIN_RATIO
                or cy >= frame_height * (1 - DISAPPEAR_EDGE_MARGIN_RATIO)
            )
            if vanish_speed is not None and vanish_speed >= DISAPPEAR_DROP_MIN_SPEED and not near_edge:
                vanish_score = max(0.0, min(1.0,
                    (vanish_speed - DISAPPEAR_DROP_MIN_SPEED) / (DISAPPEAR_DROP_REF_SPEED - DISAPPEAR_DROP_MIN_SPEED)
                ))

    # the collapse signature is gated on the outline itself (see COLLAPSE_RATIO_MAX), so
    # the aspect-corroboration guard doesn't apply to it, and neither does the trunk-angle
    # oscillation count - it has its own "one continuous stretch, still down at the end"
    # requirement. A helper crouching over someone already on the ground compacts their
    # outline just like a collapse does, so that guard does apply.
    collapse_confidence = collapse_score * (HELPER_PENALTY if helper_response else 1.0)

    # --- whichever signature is stronger wins, and sets the single evidence frame. One
    # event is either "ended up visibly down", "collapsed where they stood" or "vanished
    # out of view" - never a sum of all three. ---
    paths = [
        (visible_confidence, "visible" if down_indices else "none",
         down_indices[len(down_indices) // 2] if down_indices else n // 2),
        (collapse_confidence, "collapsed",
         collapse_indices[len(collapse_indices) // 2] if collapse_indices else n // 2),
        (vanish_score, "vanished", vanish_at if vanish_at is not None else n // 2),
    ]
    confidence, detection_path, evidence_index = max(paths, key=lambda p: p[0])

    evidence_record = features["records"][evidence_index]
    signals = {
        "detection_path": detection_path,
        "swing_deg": round(swing_deg, 2),
        "down_std_deg": round(down_std, 2) if down_std is not None else None,
        "down_duration_s": round(down_duration, 2),
        "transition_seconds": round(transition_seconds, 2) if transition_seconds is not None else None,
        "fall_score": round(fall_score, 3),
        # How far the hip actually dropped, in the person's own standing box
        # heights. Surfaced because it is the check that separates a fall from a
        # bend, so when a verdict looks wrong this is the first number to read.
        "hip_descent": round(hip_descent, 3),
        "aspect_score": round(aspect_score, 3),
        "collapse_score": round(collapse_score, 3),
        "collapse_ratio": round(collapse_ratio, 3) if collapse_ratio is not None else None,
        "collapse_hold_s": round(collapse_hold, 2),
        "oscillation_penalty_applied": oscillating,
        "helper_penalty_applied": helper_response,
        "no_aspect_corroboration_penalty_applied": no_aspect_corroboration,
        "vanish_speed": round(vanish_speed, 3) if vanish_speed is not None else None,
        "vanish_score": round(vanish_score, 3),
        "t": round(ts[evidence_index], 2),
        "frame": evidence_record["frame"],
        "box": evidence_record["box"],
        "keypoints": evidence_record["kpts_xy"].tolist(),
        "kpts_conf": evidence_record["kpts_conf"].tolist(),
    }
    return confidence, signals


def save_evidence_frame(filepath: str, candidate: dict) -> dict:
    """Draws the person's box + skeleton + confidence label on the single frame that
    best shows the fall - either partway through them lying down, or (if they vanished
    into a gap/hole instead) the last moment they were visible before that - and saves
    it as one image."""
    signals = candidate["signals"]
    frame = _read_frame_at(filepath, signals["frame"])

    result = {
        "t": signals["t"],
        "confidence": round(candidate["confidence"], 3),
        "frame_path": None,
        "signals": {k: v for k, v in signals.items() if k not in ("keypoints", "kpts_conf")},
    }
    if frame is None:
        return result

    outlined = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in signals["box"]]
    cv2.rectangle(outlined, (x1, y1), (x2, y2), (0, 0, 255), 2)

    keypoints, kpts_conf = signals["keypoints"], signals["kpts_conf"]
    for a, b in SKELETON_EDGES:
        if kpts_conf[a] >= CONF_FLOOR and kpts_conf[b] >= CONF_FLOOR:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(outlined, pa, pb, (0, 255, 255), 2)
    for idx, (x, y) in enumerate(keypoints):
        if kpts_conf[idx] >= CONF_FLOOR:
            cv2.circle(outlined, (int(x), int(y)), 3, (0, 255, 0), -1)

    label = f"Fall {candidate['confidence']:.0%} | t={signals['t']}s"
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(outlined, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    os.makedirs(FALL_FRAMES_DIR, exist_ok=True)  # only when there is something to save
    frame_path = os.path.join(FALL_FRAMES_DIR, f"fall_track{candidate['track_id']}_t{signals['t']}.jpg")
    cv2.imwrite(frame_path, outlined)
    result["frame_path"] = frame_path
    return result
