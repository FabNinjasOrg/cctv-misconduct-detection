# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A FastAPI service that analyzes CCTV video for misconduct / suspicious activity
— currently spitting, physical fights, vandalism, and falls, each with one or
more detectors (vandalism and falls have one so far). It runs two ways: clips
uploaded by hand, or a live RTSP camera recorded and analyzed continuously.

`README.md` is the user-facing setup guide — keep it accurate when paths change.

## Layout

```
main.py              the FastAPI app - the only .py at the root
stream.yaml          camera config (the user edits this)
compose.yaml         stays at the root so `docker compose up -d` is a one-liner

app/
  actions/           one package per action, plus the combined analyzer
    analyzer.py      runs all four detectors, returns one verdict
    spit/  fight/  vandalism/  fall/
  stream/            RTSP recorder + analysis worker
  shared/            motion_filter, pose_tracking, gemini_review

evidence/            all detector output, kept at the top level so it's visible
  detections/        combined verdict frames (the ones that matter)
  live/              per-camera output from the stream service + detections.jsonl
  spit/  fight/  vandalism/  fall/      per-detector working frames

uploaded_clips/      received videos
spool/               recorded segments awaiting analysis (self-emptying queue)
models/              all weights: YOLO11, YOLO11-pose, fight YOLOv8, movinet/
docker/              Dockerfile, pinned requirements.txt, DOCKER.md
docs/                design notes (general.md)
```

**Every model and evidence path in the code is relative to the project root.**
All commands must run from the root, and `WORKDIR /app` in the container
reproduces that layout so no path needed changing to containerise.

## Commands

```bash
pip install -r requirements.txt        # install deps (unpinned, local dev)
uvicorn main:app --reload              # dev server on http://127.0.0.1:8000
open http://127.0.0.1:8000/docs        # Swagger UI - easiest way to test the upload

docker compose up -d --build           # the whole stack: detector + camera recorder
python -m app.stream                   # just the camera recorder, on the host

# upload a clip from the CLI
curl -F "file=@uploaded_clips/20260831_135400_cctv-footage.mp4" \
     http://127.0.0.1:8000/upload-clip

# play a clip back with that detector's skeleton + live confidence overlaid
python3 -m app.actions.spit.playback      uploaded_clips/<clip>.mp4
python3 -m app.actions.fight.playback     uploaded_clips/<clip>.mp4
python3 -m app.actions.vandalism.playback uploaded_clips/<clip>.mp4
python3 -m app.actions.fall.playback      uploaded_clips/<clip>.mp4
```

Python is `/Users/harshprajapati/miniconda3/bin/python3` (3.13). No test suite,
linter, or build step is configured yet.

## Architecture

`main.py` is a thin FastAPI layer — it wires up routes and a shared YOLO11-pose
`person_model`, and delegates all real work to `app/actions/`:

- `GET /` — health check.
- `POST /upload-clip` — streams the uploaded file to disk in 1 MB chunks
  (never buffers the whole video in memory) under `uploaded_clips/` with a
  `YYYYMMDD_HHMMSS_<original-name>` prefix, then returns the saved path.
- `POST /analyze-action` → `app/actions/analyzer.py` — the combined endpoint:
  motion-gates the clip once, runs the primary detector for each action, and
  reduces the four results to a single verdict (highest confidence wins). Reuses
  the winning detector's own evidence jpg and adds one banner, so no detector
  file needs to change. This is what the stream worker calls.
- `POST /spit/single-frame` → `app/actions/spit/single_frame_detector.py` —
  per-motion-event heuristic: YOLO11-pose posture (head bow, side tilt,
  hand-near-mouth) plus a YOLO11 object check near the head, scored one frame
  at a time.
- `POST /spit/kinematic` → `app/actions/spit/kinematic_detector.py` — the more
  accurate spit detector: tracks each person across a motion segment and scores
  keypoint *kinematics* over time (a held bowed-head plateau, walking vs.
  stationary, contact with another person, ...), not just a single frame.
- `POST /fight/yolo` → `app/actions/fight/yolo_detector.py` — a fine-tuned
  YOLOv8 (`models/fight_yolov8n.pt`) detecting a `violence` class per frame.
- `POST /fight/movinet` → `app/actions/fight/movinet_detector.py` — a MoViNet
  clip classifier (weights in `models/movinet/`, Multiple Instance Learning,
  CC BY-NC 4.0 / internal-research-use-only — see `models/movinet/NOTICE.md`)
  scoring a sliding window of frames instead of single frames.
- `POST /vandalism/object-action` → `app/actions/vandalism/object_action_detector.py`
  — a first-pass heuristic: tracks each person across a motion segment and
  scores YOLO11-pose wrist/ankle keypoint *velocity* for a fast swing or kick,
  boosted if a YOLO11-detected tool/weapon-like object (bat, bottle, knife, ...)
  is near the swinging hand. No fine-tuned vandalism model exists yet, and
  vandalism has no single motion signature the way spitting/fighting do, so
  this is deliberately conservative — see the file's header comment.
- `POST /fall/kinematic` → `app/actions/fall/kinematic_detector.py` — flags a
  person whose trunk angle and bounding-box shape flip from upright to
  horizontal within well under a second (a sudden collapse, not a gradual
  sit/lie-down) and who then stays down. Doesn't try to guess *why* they fell.

### `app/shared/`

- `motion_filter.py` — background-subtraction motion gate, stage 1 of the
  pipeline. The analyzer runs it once for all four detectors.
- `pose_tracking.py` — runs YOLO11-pose across a motion window once and follows
  each person frame to frame. Spit, vandalism and fall all need exactly this,
  so the pass happens once and the result is shared. The detectors still work
  standalone: called without a pre-computed track set, each runs it itself.
- `gemini_review.py` — an LLM (Gemini) sanity-check call. Currently only wired
  into (and commented out in) `app/actions/fight/yolo_detector.py`. It reads
  `os.environ["GEMINI_KEY"]` at module level, so that variable must be set for
  the app to import at all, even though nothing calls Gemini.

### `app/stream/`

`python -m app.stream` starts one `Recorder` per camera plus one `Worker`:

    Recorder (1/camera)  ->  spool/  ->  Worker (1 total)
    keeps ffmpeg running     finished .mp4   POSTs /analyze-action, saves evidence

Recorders and the worker are deliberately independent — recording must never
pause because analysis is slow. They share only the spool folder on disk.
`config.py` reads `stream.yaml` and percent-encodes camera credentials, which is
why passwords go into `.env` raw.

### Playback scripts

Each action has a `playback.py` that replays an analyzed clip with the pose
skeleton and a live confidence label drawn frame-by-frame. Spit, vandalism and
fall show one score per tracked person for their whole appearance (not
recalculated per frame); fight/movinet genuinely updates over time, since that
classifier already scores in windows.

## Constraints

- **Spit, vandalism and fall are pose-only** — no fine-tuned models, no LLM /
  Gemini calls. Fight is the exception (fine-tuned YOLOv8 + MoViNet MIL).
- Keep the per-stage boundary clean: OpenCV work should be usable without
  ultralytics or an API key installed, so the project stays runnable while
  later stages are still being built.
- Licensing, unresolved: `models/movinet/` is CC BY-NC 4.0 (cannot ship
  commercially); Ultralytics YOLO is AGPL-3.0.

## Intended pipeline (see `docs/general.md`)

A layered pipeline, each layer added on top of the previous rather than
replacing it:

1. **opencv-python** — decode, sample/skip frames, crop, draw overlays. Also the
   first-pass motion detector, so the expensive stages only see frames with
   activity.
2. **ultralytics YOLO11-pose** — per-frame person detection plus skeleton
   keypoints. It has no notion of "actions", so behaviour has to be derived from
   keypoint motion over time.
3. **An LLM call** (Gemini) — receives a short clip around the candidate moment
   and makes the final judgement. Not practical on every frame of 24x7 footage,
   so it's a final check on stage 1-2 candidates, not a replacement.

An optional React/Vue front end may be added later to consume the REST API.
