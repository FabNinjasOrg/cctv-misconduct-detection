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
    replay.py        runs recorded clips through the same engine, into evidence/<date>/
  actions/           one package per action
    spit/  fight/  vandalism/  fall/
  shared/
    pose_tracking.py the pose pass and the usable-pose gate, shared by three

evidence/live_view/  results - one folder per camera, annotated stills only
evidence/<date>/     replay results - one folder per clip, plus summary.json
models/              YOLO11-pose, YOLO11 (objects, for vandalism), and our own
                     fall_stgcn.pt and fight_movinet.pt. movinet/ is the old
                     fight model (CC BY-NC) - kept, no longer loaded
training/            the Kaggle notebooks that trained fall_stgcn.pt and fight_movinet.pt
uploaded_clips/      test clips with known answers, for regression checking;
                     real-clips/ is footage from the real camera
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

python -m app.live.replay uploaded_clips/real-clips   # recorded clips -> evidence/<today>/
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

**A score crossing the line is a candidate, not an alarm.** The detectors look
back but never forward, so they call the start of something before seeing how it
ends. `_update_events` in `engine.py` watches each candidate for
`CONFIRM_SECONDS` (1.0) and confirms it only if the score never falls below
`HOLD_THRESHOLD` (0.5) while the person is in view. Someone who leaves the
picture is judged on what was seen (provisional, not measured). Measured on
every clip with its last frame held 4s: false alarms 26 → 8, and all 11 real
spits and falls still alarm. At 1.5s, spit-6 is lost. fall-1 was never really
detected: its old fall alarms were a phantom box over an empty pit. `action` from
`people()` is a confirmed alarm only, and saved pictures come from
`take_events()` at the candidate's best moment, not at confirmation. Fight is
not wait-and-confirm — MoViNet already judges a clip, and your acted fight only
crosses the line for single seconds (the moments of contact), which a wait would
throw away.

### The four detectors

- `spit/kinematic_detector.py` — **turned off** (commented out in `engine.py`).
  Tracks each person and scores keypoint kinematics over time: a held
  bowed-head plateau, walking vs stationary, contact with another person.
- `vandalism/object_action_detector.py` — scores wrist/ankle velocity for a fast
  swing or kick, boosted if a YOLO11-detected tool (bat, bottle, knife) is near
  the swinging hand. Deliberately conservative; vandalism has no single motion
  signature the way spitting and fighting do.
- `vandalism/tamper_detector.py` — vandalism of the camera itself: covered,
  blurred/sprayed, or turned. Plain image maths against a learned normal view,
  held 5s. Scene-level, with its own banner and one snapshot per tamper.
- **Fall is two judges, and the score is the lower of them** (see "fall: two
  judges" in `engine.py`). `fall/kinematic_detector.py` flags a trunk angle and
  box shape flipping from upright to horizontal in well under a second, and
  staying down. `fall/stgcn_detector.py` is an ST-GCN trained on Le2i skeletons,
  reading each person's last 3 seconds. Alone, the rules called every sitter on
  the live camera a fall, and the model called crowds and brawls falls; together
  they kept the real falls and dropped both.
- `fight/movinet_detector.py` — **the odd one out.** MoViNet-A0 fine-tuned on
  RWF-2000 + RLVS (`fight/clip_classifier.py`), scoring a square crop around
  everyone in shot over 8 frames sampled at 8 fps — the crop is what it was
  trained on, so each sampled frame keeps its person boxes. Scene-level, not
  per-person, so it gets its own buffer, thread and banner rather than a label on
  somebody. `fight_classifier(frames, boxes)` is called directly by the live
  engine; `detect_fighting_mil()` is the file-based version, kept for checking
  against a recording.

**The trained models must be fed exactly as in training** — 640 wide, 15 fps
skeletons from yolo11n-pose for fall; 8 fps, the people crop, 172x172 for fight.
Both check their checkpoint's config at load. Changing the frame width, rate or
pose model means retraining, not just a constant.

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

- **Spit and vandalism are pose-only** (plus the tamper check, plain image
  maths) — no trained models, no LLM calls. Fall and fight use models trained by
  the notebooks in `training/`.
- Licensing, unresolved: the fight model starts from MoViNet's Apache-2.0
  Kinetics weights, but RWF-2000, RLVS and Le2i are research-only datasets, so
  both trained models need retraining on your own footage before anything ships.
  `models/movinet/` (old, unused) is CC BY-NC 4.0; Ultralytics YOLO is AGPL-3.0.
- Camera placement beats tuning. The detectors assume a whole person at a
  distance; a close-up camera breaks the arithmetic rather than degrading it.
  Moving one camera took 6 false alarms in 13 minutes to 0, with no code change.
