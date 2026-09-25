# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A FastAPI service that accepts a CCTV video clip upload and analyzes it for
misconduct / suspicious activity — currently spitting, physical fights,
vandalism, and falls, each with more than one detector (vandalism and falls
have one so far).

## Commands

```bash
pip install -r requirements.txt        # install deps
uvicorn main:app --reload              # dev server on http://127.0.0.1:8000
open http://127.0.0.1:8000/docs        # Swagger UI — easiest way to test the upload

# upload a clip from the CLI
curl -F "file=@uploaded_clips/20260831_135400_cctv-footage.mp4" \
     http://127.0.0.1:8000/upload-clip

# play a clip back with the spit-detection skeleton + live confidence overlaid
python3 -m spit.playback uploaded_clips/<clip>.mp4

# play a clip back with the MoViNet fight-detection confidence overlaid
python3 -m fight.playback uploaded_clips/<clip>.mp4

# play a clip back with the vandalism-detection skeleton + live confidence overlaid
python3 -m vandalism.playback uploaded_clips/<clip>.mp4

# play a clip back with the fall-detection skeleton + live confidence overlaid
python3 -m fall.playback uploaded_clips/<clip>.mp4
```

Python is `/Users/harshprajapati/miniconda3/bin/python3` (3.13). All commands
run from the project root — the code relies on that for its relative model and
evidence-frame paths. No test suite, linter, or build step is configured yet.

## Architecture

`main.py` is a thin FastAPI layer — it wires up routes and a shared YOLO11-pose
`person_model`, and delegates all real work to the `spit/`, `fight/`,
`vandalism/`, and `fall/` packages:

- `GET /` — health check.
- `POST /upload-clip` — streams the uploaded file to disk in 1 MB chunks
  (never buffers the whole video in memory) under `uploaded_clips/` with a
  `YYYYMMDD_HHMMSS_<original-name>` prefix, then returns the saved path.
- `POST /spit/single-frame` → `spit/single_frame_detector.py` — per-motion-event
  heuristic: YOLO11-pose posture (head bow, side tilt, hand-near-mouth) plus a
  YOLO11 object check near the head, scored one frame at a time.
- `POST /spit/kinematic` → `spit/kinematic_detector.py` — the more accurate
  spit detector: tracks each person across a motion segment and scores
  keypoint *kinematics* over time (a held bowed-head plateau, walking vs.
  stationary, contact with another person, ...), not just a single frame.
- `POST /fight/yolo` → `fight/yolo_detector.py` — a fine-tuned YOLOv8
  (`fight/fight_yolov8n.pt`) detecting a `violence` class per frame.
- `POST /fight/movinet` → `fight/movinet_detector.py` — a MoViNet clip
  classifier (`fight/vd_mil/`, Multiple Instance Learning, CC BY-NC 4.0 /
  internal-research-use-only weights — see `fight/vd_mil/NOTICE.md`) scoring a
  sliding window of frames instead of single frames.
- `POST /vandalism/object-action` → `vandalism/object_action_detector.py` — a
  first-pass heuristic: tracks each person across a motion segment and scores
  YOLO11-pose wrist/ankle keypoint *velocity* for a fast swing or kick, boosted
  if a YOLO11-detected tool/weapon-like object (bat, bottle, knife, ...) is
  near the swinging hand. No fine-tuned vandalism model exists yet, and
  vandalism has no single motion signature the way spitting/fighting do, so
  this is deliberately conservative — see the file's header comment.
- `POST /fall/kinematic` → `fall/kinematic_detector.py` — flags a person
  whose trunk angle and bounding-box shape flip from upright to horizontal
  within well under a second (a sudden collapse, not a gradual sit/lie-down)
  and who then stays down. Doesn't try to guess *why* they fell (tripped,
  pushed, fainted, slipped, ...) - just the motion signature of going down
  and staying there.

Supporting/shared pieces at the root:

- `models/` — shared YOLO weights (`yolo11n-pose.pt`, `yolo11n.pt`) used by
  `spit/`, `fight/`, `vandalism/`, and `fall/`. `fight/fight_yolov8n.pt` and
  `fight/vd_mil/` are fight-specific, so they live under `fight/` instead.
- `gemini_review.py` — an LLM (Gemini) sanity-check call, given a short clip
  and a yes/no question. Currently only wired into (and commented out in)
  `fight/yolo_detector.py`; nothing calls it from the spit path yet.
- `spit/playback.py` — replays an already-analyzed clip with the pose
  skeleton and a live confidence label drawn frame-by-frame, synced to
  playback. It reuses `spit/kinematic_detector.py`'s pipeline as-is; the score
  shown per person is the one score computed for their whole tracked
  appearance, not recalculated frame-by-frame.
- `fight/playback.py` — the fight-detection counterpart: replays a clip with
  a live `Fight: X%` label, reusing `fight/movinet_detector.py`'s
  sliding-window classifier as-is. Unlike spit's playback, this score
  genuinely updates over time, since the underlying classifier already scores
  in windows rather than once per whole clip.
- `vandalism/playback.py` — same pattern as `spit/playback.py`: reuses
  `vandalism/object_action_detector.py`'s pipeline as-is, one score per
  tracked person for their whole appearance (not recalculated frame-by-frame).
- `fall/playback.py` — same pattern again, reusing `fall/kinematic_detector.py`'s
  pipeline as-is, one static score per tracked person for their whole appearance.
- `uploaded_clips/` — the on-disk store for received videos, read by every
  detector above. Not cleaned up automatically.
- `spit/evidence/`, `fight/evidence/`, `vandalism/evidence/`, and
  `fall/evidence/` — evidence frames each detector saves out (one subfolder
  per detector), not cleaned up automatically.

## Intended pipeline (see `general.md`)

The detection flow is planned as a layered pipeline, each layer added on top of
the previous one rather than replacing it:

1. **opencv-python** — decode the clip, sample/skip frames, crop, draw overlays.
   This is also the first-pass motion detector (background subtraction) so the
   expensive stages only see frames that contain activity.
2. **ultralytics YOLO11-pose** — per-frame person detection plus skeleton
   keypoints. It locates people and body joints; it has no notion of "actions",
   so behaviour has to be derived from keypoint motion over time.
3. **An LLM call** (Gemini today, see `gemini_review.py`) — receives the
   extracted evidence (a short clip around the candidate moment) and makes the
   final judgement about whether misconduct occurred. Not practical to run on
   every frame of 24x7 footage, so it's a final check on top of stage 1-2
   candidates, not a replacement for them.

An optional React/Vue front end may be added later to consume the REST API.

Keep the per-stage boundary clean: OpenCV work should be usable without
ultralytics or an API key installed, so the project stays runnable while later
stages are still being built.
