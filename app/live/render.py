"""Draws the overlay: who is in shot, their skeleton, and what it looks like
they are doing.

Kept apart from engine.py so that how something is decided and how it is shown
stay separable - the engine has no idea what colour anything is, and this file
does no thinking about behaviour.

Colour is the whole language of the view, so it is deliberately minimal:

    grey    tracked, still gathering history - too early to say anything
    green   watched, nothing above the threshold
    red     something crossed the threshold

Scores below the threshold are still printed, small, under the label. A live
view is most useful while you are trying to trust it, and "fall 0.31" tells you
far more about why something nearly fired than a box that silently stays green.
"""
import cv2

# COCO pose skeleton - the same joint pairs the detectors' own playback scripts
# draw, so a live skeleton and a recorded one look like the same picture.
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 6), (5, 11), (6, 12), (11, 12),     # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
]

KEYPOINT_CONF_FLOOR = 0.15      # matches the detectors - below this a joint is missing

# BGR, because OpenCV.
GREY = (150, 150, 150)
GREEN = (80, 200, 80)
RED = (60, 60, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# A track needs at least this much history before a score means anything - below
# it the person is drawn grey and unlabelled rather than green, so "we are not
# judging yet" never looks like "we have judged and it is fine".
WARMING_UP_RECORDS = 8

# Matches the fight detector's own CONFIDENCE_MIN, so the live view calls a
# fight at exactly the point the detector does.
FIGHT_THRESHOLD = 0.5


def draw(frame, people: list, status: dict, fight: float = 0.0):
    """Returns the frame with everyone boxed, labelled and skeletoned.

    Draws onto a copy: the caller's frame may still be needed unmarked, and a
    detector looking back at a buffered frame must not find our overlay burned
    into it."""
    canvas = frame.copy()

    for person in people:
        warming = person["history"] < WARMING_UP_RECORDS
        colour = GREY if warming else (RED if person["action"] else GREEN)

        _draw_skeleton(canvas, person, colour)
        _draw_box(canvas, person, colour)
        _draw_label(canvas, person, colour, warming)

    _draw_fight(canvas, fight)
    _draw_status(canvas, status)
    return canvas


def _draw_fight(canvas, fight: float) -> None:
    """A border and banner across the whole frame when a fight is detected.

    Drawn differently from the other three on purpose. They belong to a person
    and get a box round that person; fight is a judgement about the entire
    scene, with no one body to point at. Framing the whole picture says that."""
    if fight < FIGHT_THRESHOLD:
        return

    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), RED, 6)
    _text_with_backing(
        canvas, f"FIGHT {fight * 100:.0f}%", (width // 2 - 60, 32),
        RED, scale=0.8, thickness=2,
    )


def _draw_box(canvas, person: dict, colour) -> None:
    x1, y1, x2, y2 = (int(v) for v in person["box"])
    thickness = 3 if person["action"] else 2
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)


def _draw_skeleton(canvas, person: dict, colour) -> None:
    points, confidences = person["kpts_xy"], person["kpts_conf"]

    for a, b in SKELETON_EDGES:
        if confidences[a] < KEYPOINT_CONF_FLOOR or confidences[b] < KEYPOINT_CONF_FLOOR:
            continue
        pa = (int(points[a][0]), int(points[a][1]))
        pb = (int(points[b][0]), int(points[b][1]))
        cv2.line(canvas, pa, pb, colour, 1, cv2.LINE_AA)

    for index, (x, y) in enumerate(points):
        if confidences[index] >= KEYPOINT_CONF_FLOOR:
            cv2.circle(canvas, (int(x), int(y)), 2, colour, -1, cv2.LINE_AA)


def _draw_label(canvas, person: dict, colour, warming: bool) -> None:
    """The headline above the box, plus the raw scores underneath it."""
    x1, y1, _, _ = (int(v) for v in person["box"])

    if warming:
        headline = f"#{person['track_id']}  reading..."
    elif person["action"]:
        headline = f"#{person['track_id']}  {person['action'].upper()} {person['confidence'] * 100:.0f}%"
    else:
        headline = f"#{person['track_id']}  clear"

    _text_with_backing(canvas, headline, (x1, max(y1 - 8, 14)), colour, scale=0.5, thickness=1)

    # The near-misses. Only worth showing once there is something to show, and
    # only the non-zero ones - three zeroes under every person is noise.
    scores = {name: value for name, value in (person["scores"] or {}).items() if value > 0.01}
    if not scores and not person["action"]:
        return

    detail = "  ".join(
        f"{name} {value:.2f}"
        for name, value in sorted(scores.items(), key=lambda item: -item[1])
    )
    if detail:
        _text_with_backing(canvas, detail, (x1, max(y1 - 24, 28)), GREY, scale=0.38, thickness=1)


def _draw_status(canvas, status: dict) -> None:
    """One line along the bottom: is it connected, and how fast is it going."""
    height = canvas.shape[0]

    connected = status.get("connected")
    parts = [
        "LIVE" if connected else "RECONNECTING",
        f"{status.get('fps', 0):.1f} fps",
        f"{status.get('people', 0)} tracked",
    ]
    if status.get("note"):
        parts.append(status["note"])

    _text_with_backing(
        canvas, "   ".join(parts), (8, height - 10),
        GREEN if connected else RED, scale=0.45, thickness=1,
    )


def _text_with_backing(canvas, text: str, origin, colour, scale: float, thickness: int) -> None:
    """Draws text twice - black underneath, colour on top.

    Cheaper and more legible than a filled background box, and it stays readable
    over both a bright wall and a dark doorway, which a single colour does not."""
    cv2.putText(canvas, text, origin, FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, FONT, scale, colour, thickness, cv2.LINE_AA)
