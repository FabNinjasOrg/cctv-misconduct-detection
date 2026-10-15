# Notices vandalism of the camera itself: covered, blurred or sprayed, or turned to
# look somewhere else. The most common thing done to a CCTV camera before anything
# else is done in front of it, and the one kind of vandalism no pose detector can
# see - once the lens is covered there is nobody left in the picture to measure.
#
# No model and no training. It compares the picture with a learned picture of what
# this camera normally sees:
#
#   covered   almost no contrast left - a hand, a cloth or tape over the lens, or the
#             room gone dark. Measured as the spread of brightness in the picture.
#   blurred   contrast is still there but the detail isn't - sprayed, smeared or
#             knocked out of focus. Measured as how many edges are left compared with
#             the normal view, after stretching the contrast so a dimmer room
#             doesn't count as "fewer edges".
#   moved     plenty of detail, but not the usual detail - the camera has been turned.
#             Measured as how well the picture's edges line up with the normal
#             view's: the correlation of their gradient strength, which a change of
#             lighting doesn't move (it scales every gradient alike) but a new view
#             does.
#
# Each must hold for HOLD_SECONDS before it is reported, so a person walking right
# past the lens, or the lights flickering, is never enough on its own. People found
# by the pose model are masked out of the comparison: someone standing in front of
# the usual view is not the camera having moved.
#
# PROVISIONAL - the thresholds were checked against this camera's own recordings and
# against covered / blurred / moved versions of them made by hand, not against a real
# attack on the camera. Measured (lowest 1% of checks for real footage):
#
#                                     contrast   detail   similarity
#     this camera, 4 min, 4 days        51+       0.73+      0.85+
#     ... lights dimmed / IR / sun      18+       0.90+      0.82+
#     hand or cloth over the lens       ~1          -          ~0
#     sprayed / knocked out of focus    44        0.08-0.34  0.38-0.56
#     turned / pointed elsewhere        44-51     0.62+      0.22-0.49
#
# Every snapshot's .json records the measured values, so they can be confirmed on the
# first real one. A camera that is meant to move (handheld, pan-tilt) will read as
# "moved" - this assumes a fixed camera.

import cv2
import numpy as np

WIDTH = 160                 # everything is judged on a small grey copy - plenty for "is the view there"
CHECK_SECONDS = 0.5         # how often to look
LEARN_SECONDS = 3.0         # the first few seconds after start-up build the normal view
LEARN_TIME_CONSTANT = 120.0 # the normal view then follows slow changes (daylight) over about this long
HOLD_SECONDS = 5.0          # a tamper must hold this long before it is reported
CLEAR_SECONDS = 2.0         # ...and the view be back to normal this long before it is cleared

# A moved camera is reported, and if it stays moved this long the new view becomes
# the normal one - someone re-aiming the camera on purpose shouldn't alarm forever.
# Covered and blurred never become normal: there is nothing to watch in them.
ACCEPT_MOVED_SECONDS = 300.0

COVERED_CONTRAST_MAX = 12.0     # spread of brightness (std, 0-255) below this = covered or dark
BLURRED_DETAIL_MAX = 0.40       # fewer than this fraction of the normal view's edges left = blurred
MOVED_SIMILARITY_MAX = 0.60     # edges line up with the normal view worse than this = moved

# If the people found cover more than this of the picture, there is too little view
# left to judge, and nothing is decided until they move.
MAX_PEOPLE_COVER = 0.5

CANNY_LOW, CANNY_HIGH = 60, 150


def _small_grey(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _edges(grey):
    """Edges after stretching the contrast to full range, so a dimmer or brighter
    room keeps the same edges and only a real loss of detail removes them."""
    low, high = np.percentile(grey, (2, 98))
    stretched = np.clip((grey - low) * (255.0 / max(high - low, 1.0)), 0, 255).astype(np.uint8)
    return cv2.Canny(cv2.GaussianBlur(stretched, (3, 3), 0), CANNY_LOW, CANNY_HIGH) > 0


def _gradient(grey):
    return cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))


def _correlation(a, b) -> float:
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def _people_mask(shape, boxes, scale):
    """True where nobody is - person boxes, grown a little, are left out."""
    mask = np.ones(shape, bool)
    for x1, y1, x2, y2 in boxes:
        mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        a, b = int(max(0, (x1 - mx) * scale)), int(max(0, (y1 - my) * scale))
        c, d = int(min(shape[1], (x2 + mx) * scale + 1)), int(min(shape[0], (y2 + my) * scale + 1))
        mask[b:d, a:c] = False
    return mask


class TamperDetector:
    """Feed it every frame with the person boxes found in it; read .state()."""

    def __init__(self):
        self._reference = None      # the normal view, as a float grey image
        self._learned = 0           # how many checks have gone into it so far
        self._last_check = None
        self._kind = None           # what the current check sees: None, covered, blurred, moved
        self._since = None          # when the current kind started
        self._reported = None       # the tamper being reported, or None
        self._normal_since = None
        self._signals = {}

    def update(self, frame, person_boxes, now: float) -> None:
        if self._last_check is not None and now - self._last_check < CHECK_SECONDS:
            return
        self._last_check = now
        grey = _small_grey(frame)

        if self._reference is None or self._learned < LEARN_SECONDS / CHECK_SECONDS:
            # Still learning the normal view: a plain average of the first few seconds.
            self._learned += 1
            self._reference = grey if self._reference is None else (
                self._reference + (grey - self._reference) / self._learned)
            return

        kind, signals = self._judge(grey, person_boxes, frame.shape[1] and WIDTH / frame.shape[1])
        self._signals = signals
        if kind == "unsure":
            return                  # too many people in the way - decide nothing, learn nothing

        if kind != self._kind:
            self._kind, self._since = kind, now

        if kind is None:
            self._normal_since = self._normal_since or now
            if self._reported and now - self._normal_since >= CLEAR_SECONDS:
                self._reported = None
            if not self._reported:
                rate = CHECK_SECONDS / LEARN_TIME_CONSTANT
                self._reference += (grey - self._reference) * rate
            return

        self._normal_since = None
        held = now - self._since
        if held >= HOLD_SECONDS:
            self._reported = kind
        if kind == "moved" and held >= ACCEPT_MOVED_SECONDS:
            self._reference = grey          # the new view is the normal one now
            self._kind, self._reported = None, None

    def _judge(self, grey, person_boxes, scale):
        contrast = float(grey.std())
        free = _people_mask(grey.shape, person_boxes, scale)
        if free.mean() < 1 - MAX_PEOPLE_COVER:
            return "unsure", {"people_cover": round(1 - float(free.mean()), 2)}

        reference_edges = _edges(self._reference) & free
        current_edges = _edges(grey) & free
        detail = float(current_edges.sum()) / max(int(reference_edges.sum()), 1)
        similarity = _correlation(_gradient(self._reference)[free], _gradient(grey)[free])

        signals = {"contrast": round(contrast, 1), "detail": round(detail, 2),
                   "similarity": round(similarity, 2), "people_cover": round(1 - float(free.mean()), 2)}
        if contrast < COVERED_CONTRAST_MAX:
            return "covered", signals
        if detail < BLURRED_DETAIL_MAX:
            return "blurred", signals
        if similarity < MOVED_SIMILARITY_MAX:
            return "moved", signals
        return None, signals

    def state(self) -> dict:
        """{kind: covered/blurred/moved or None, confidence 0..1, signals}."""
        if not self._reported:
            return {"kind": None, "confidence": 0.0, "signals": dict(self._signals)}
        s = self._signals
        confidence = {
            "covered": 1.0 - s.get("contrast", 0.0) / COVERED_CONTRAST_MAX,
            "blurred": 1.0 - s.get("detail", 0.0) / BLURRED_DETAIL_MAX,
            "moved": 1.0 - s.get("similarity", 0.0) / MOVED_SIMILARITY_MAX,
        }.get(self._reported, 1.0)
        # Once reported the view is plainly not normal; how far past the line it is
        # matters less than that it held for HOLD_SECONDS, so never show it as weak.
        return {"kind": self._reported, "confidence": float(min(1.0, max(0.6, confidence))),
                "signals": dict(s)}
