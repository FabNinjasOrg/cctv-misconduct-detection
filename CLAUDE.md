# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

One program that watches a live RTSP camera and says, in real time, whether
someone is **falling**, **fighting**, **spitting** or **vandalising**. It draws
boxes and labels on the feed, serves that to a browser, and saves one annotated
still per detection.

There is no API, no Docker, no recorder and no database. It used to have all of
those; they were removed deliberately. Don't reintroduce a service layer without
being asked.

`README.md` is the user-facing guide — keep it accurate when paths change.

## Layout

```
camera.yaml          the camera (the user edits this)
.env                 the camera password (the user edits this)

app/
  live/              the program itself
    __main__.py      entry point, argument parsing, the frame loop and its pacing
    capture.py       RTSP thread - always hands back the newest frame, drops the rest
    engine.py        pose -> tracking -> scoring, plus the fight classifier
    render.py        boxes, skeletons, labels, the fight banner
    evidence.py      one snapshot per event, with a cooldown
    server.py        MJPEG over HTTP
    config.py        reads camera.yaml
  actions/           one package per action
    spit/  fight/  vandalism/  fall/
  shared/
    pose_tracking.py the pose pass and the usable-pose gate, shared by three

evidence/live_view/  results - one folder per camera, annotated stills only
models/              YOLO11-pose, YOLO11 (objects, for vandalism), movinet/
uploaded_clips/      test clips with known answers, for regression checking
docs/                design notes (general.md)
```

**Every model and evidence path in the code is relative to the project root**, so
commands must be run from there.

## Commands

```bash
pip install -r requirements.txt

set -a && source .env && set +a      # camera password into the environment
python -m app.live                   # then http://localhost:8088

python -m app.live --no-save         # watch only, write nothing
python -m app.live --width 480       # faster on a slow machine
python -m app.live --port 9000       # if 8088 is taken
```

Python is `/Users/harshprajapati/miniconda3/bin/python3` (3.13). No test suite,
linter or build step is configured.

## Architecture

    Capture (thread) -> latest frame -> LiveEngine -> render -> MJPEG -> browser
    always newest,                      pose+track+score      overlay
    drops the rest                      + fight classifier             + snapshot

Four threads plus the HTTP server's own: capture, the main frame loop, pose
scoring, and fight classification. They are separate so that a slow score costs
label latency rather than tracking fidelity.

### Three things that are load-bearing

**The detectors are reused, not reimplemented.** Spit, vandalism and fall each
split into `compute_kinematics(records)` (pure maths over one person's history)
and `score_event(...)` (features -> confidence). Both take a list of per-frame
records, which is exactly what a rolling window is, so `engine.py` builds the
window and calls their real functions. `_install_frame_adapters` swaps
`_read_frame_at`/`_read_frames_at` in the spit and vandalism modules for versions
reading an in-memory ring buffer, which is what lets those run with no file on
disk. **Don't reimplement scoring in app/live/** — the two would drift apart.

**15 fps is a correctness requirement, not a smoothness one.** The detectors
count history in records (`MAX_FILL_GAP_FRAMES = MAX_FILL_GAP_SECONDS *
TRACK_FPS`), so a different rate silently rescales their tolerances. At 8 fps
`fall-1.mp4` scores *spit* 0.70 instead of *fall* 1.00. The main loop paces to
`TARGET_FPS`; `--threads` exists because torch defaulting to all cores made it
slower, not faster.

**A still person is not scored at all.** The recorded pipeline only ever scored
stretches where something moved, so a window existing already meant something
had happened. Live there is no such filter and everyone is scored every 0.3s —
about 12,000 attempts per person per hour, against guards calibrated for dozens.
`_pose_change()` restores it per-person: average joint wander across the window,
in box heights, and below `MIN_POSE_CHANGE` the person is skipped and reported
as zero. All four actions are transitions, so nothing real is lost by ignoring a
posture held unchanged. That threshold is **reasoned, not yet measured** — every
snapshot's `.json` records the value as `pose_change` so it can be confirmed
against real events.

**`SCORE_INTERVAL_SECONDS = 0.3` is measured, not chosen for comfort.** At 0.7
the scorer stepped straight over spit-1's peak (0.23 against 0.77). Widening the
window instead changed nothing — 6s and 12s scored identically.

### The four detectors

- `spit/kinematic_detector.py` — tracks each person and scores keypoint
  kinematics over time: a held bowed-head plateau, walking vs stationary,
  contact with another person.
- `vandalism/object_action_detector.py` — scores wrist/ankle velocity for a fast
  swing or kick, boosted if a YOLO11-detected tool (bat, bottle, knife) is near
  the swinging hand. Deliberately conservative; vandalism has no single motion
  signature the way spitting and fighting do.
- `fall/kinematic_detector.py` — flags a trunk angle and box shape flipping from
  upright to horizontal in well under a second, and staying down. Doesn't try to
  guess *why* they fell.
- `fight/movinet_detector.py` — **the odd one out.** A MoViNet clip classifier
  scoring the whole picture over 8 frames sampled at 8 fps. Scene-level, not
  per-person, so it gets its own buffer, thread and banner rather than a label on
  somebody. `violence_classifier([frames])` is called directly by the live
  engine; `detect_fighting_mil()` is the file-based version, kept for checking
  against a recording.

## Accuracy work already done — read before tuning

Four hypotheses were tested against real footage and **three of them failed**.
The constants that survived carry their measurements in comments; read those
before changing a number.

| Idea | Outcome |
|---|---|
| Usable-pose gate (hips visible, not too close) | **Kept** — `pose_tracking.py` |
| Fall: require the hip to actually descend | **Kept** — two thresholds, one per signature |
| Spit: bound the bow's size and duration | **Kept** — 65°, 6.5s |
| Spit: reject a bow filling the whole window | **Kept** — the 6.5s bar was unreachable live |
| Fall: ankle height | Failed — bending reads 1.14, a real fall 1.10 |
| Fall: leg direction | Failed *backwards* — sitting makes legs horizontal |
| Spit: require legs visible | Failed *backwards* — 100% of desk footage vs 81% of real spits |
| Vandalism: a desk bottle being read as a tool | Failed — YOLO found no tool near the hand in either false vandalism; the score was pure limb speed |

**Always check both directions.** A gate tuned only against false positives will
silently delete real events — the pose gate at 0.60 destroyed a genuine spit
detection, caught only by re-running the known clips.

### What counts as evidence (changed 2026-09-25)

**`evidence/live_view/` is the test set now, not `uploaded_clips/`.** The user's
instruction: the snapshots are real data from the real camera in its real
position, and that is what to judge against. Each snapshot has a `.json` beside
it with every score and the winning detector's full signals — that pair is the
unit of evidence. `uploaded_clips/` stays on disk but is no longer the bar.

That swap has one sharp edge worth stating plainly, because it is the trap this
whole section exists to prevent: **a folder of false positives contains no true
positives.** Tuning against it alone can only ever make the detectors quieter,
and the quietest detector is one that never fires. Before changing any
threshold, get the other half — ask the user to perform each action in front of
the camera so there are real events from the same room to protect.

## Constraints

- **Spit, vandalism and fall are pose-only** — no fine-tuned models, no LLM
  calls. Fight is the exception (MoViNet).
- Licensing, unresolved: `models/movinet/` is CC BY-NC 4.0 (cannot ship
  commercially); Ultralytics YOLO is AGPL-3.0.
- Camera placement beats tuning. The detectors assume a whole person at a
  distance; a close-up camera breaks the arithmetic rather than degrading it.
  Moving one camera took 6 false alarms in 13 minutes to 0, with no code change.
