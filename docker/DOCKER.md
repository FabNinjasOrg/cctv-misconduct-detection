# Running in Docker

> For plain setup instructions, read `README.md` at the project root first.
> This file is the deeper reference: how the pieces fit, what to tune, and what
> tends to go wrong.

Containerised deployment. Two services from one image:

- **`detector`** — the FastAPI app, same as `uvicorn main:app`, no code changes.
- **`stream`** — records the cameras and feeds each finished chunk to the detector.

## Run it

```bash
cp .env.example .env          # first time only, if you have no .env yet
# set your camera URL in stream.yaml
docker compose up -d --build
```

Then:

```bash
curl http://localhost:8000/           # -> {"status":"ok"}
open http://localhost:8000/docs       # Swagger UI
docker compose logs -f stream         # watch it record and detect
```

## Live camera capture

**`stream.yaml` is the one place a camera is configured.** Set the URL there and
`docker compose up -d` does the rest.

```yaml
cameras:
  - name: cam-1
    url: rtsp://admin:${CAM1_PASSWORD}@192.168.1.64:554/Streaming/Channels/101
```

`${VAR}` is substituted from the environment, so put passwords in `.env` and
`stream.yaml` stays safe to commit. A `${VAR}` that is not set is a startup
error, not a confusing authentication failure later.

### How it works

```
RTSP camera
    |  ffmpeg, -c copy (no re-encode, so recording costs almost no CPU)
    v
spool/<camera>/<camera>_<timestamp>.mp4      10-second chunks
    |  worker takes each finished chunk
    v
POST /analyze-action  ->  detector service
    |  anything at or above min_confidence
    v
evidence/live_detections/<camera>/   frame + clip
evidence/live_detections/detections.jsonl
```

Each detection appends one line to `detections.jsonl`:

```json
{"detected_at":"2026-09-24T07:20:30+00:00","camera":"cam-1",
 "segment":"cam-1_20260924-071830.mp4","action":"fall","confidence":0.704,
 "scores":{"spit":0.185,"vandalism":0.207,"fall":0.704,"fight":0.0},
 "evidence_frame":"evidence/live_detections/cam-1/..._fall_70pct.jpg",
 "evidence_clip":"evidence/live_detections/cam-1/..._fall_70pct.mp4"}
```

### Keeping up

Detection is slower than real time on modest hardware — a 10-second chunk took
~50-70s to analyze on an 8-core Mac. Two settings in `stream.yaml` handle that:

- **`analyze_one_in`** — analyze a sample instead of everything. `3` analyzes
  roughly 20 seconds out of every minute. This is the main lever.
- **`max_backlog_segments`** — when analysis falls this far behind, the oldest
  waiting chunks are dropped rather than queued. Without it the spool grows until
  the disk fills, and by the time those chunks were analyzed the alert would be
  minutes stale anyway.

The worker logs a warning whenever a chunk takes longer to analyze than it took
to record, so the problem is visible rather than silent.

### Testing without a camera

A local RTSP server plus a looped test clip stands in for a real one:

```bash
NET=cctv-detection_default
docker run -d --name mediamtx --network $NET bluenviron/mediamtx:latest
docker run -d --name cam-publisher --network $NET \
  -v "$PWD/uploaded_clips:/clips:ro" --entrypoint ffmpeg cctv-detection:latest \
  -re -stream_loop -1 -i /clips/fall-4.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -an \
  -f rtsp -rtsp_transport tcp rtsp://mediamtx:8554/live
```

Then set `url: rtsp://mediamtx:8554/live` in `stream.yaml` and
`docker compose up -d stream`. Clean up with
`docker rm -f mediamtx cam-publisher`.

## Analyze a clip

Paths are the project's own paths — no translation:

```bash
curl -F "filepath=uploaded_clips/spit-1.mp4" \
     http://localhost:8000/analyze-action
```

Every clip already in `uploaded_clips/` is visible to the container, and evidence
is written straight back into `evidence/manual_detections/`, `evidence/manual_detections/spit/` and so
on, exactly where a native run puts it.

## Endpoints

| Method | Path | What |
|---|---|---|
| GET | `/` | health check |
| POST | `/analyze-action` | combined verdict across all four detectors |
| POST | `/upload-clip` | upload a file into `uploaded_clips/` |
| POST | `/spit/kinematic` | spit |
| POST | `/spit/single-frame` | spit, per-frame variant |
| POST | `/fight/movinet` | fight |
| POST | `/fight/yolo` | fight, YOLOv8 variant |
| POST | `/vandalism/object-action` | vandalism |
| POST | `/fall/kinematic` | fall |

All take a form field `filepath`, except `/upload-clip` which takes `file`.

## Everyday commands

```bash
docker compose logs -f        # follow logs
docker compose ps             # status
docker compose down           # stop
docker compose up -d --build  # rebuild after changing code
```

## Files

| File | Purpose |
|---|---|
| `docker/Dockerfile` | The image. App copied to `/app`. |
| `.dockerignore` | Keeps clips, evidence, caches and `.env` out of the image. |
| `compose.yaml` | Ports, env, volumes, restart policy, log rotation. |
| `docker/requirements.txt` | Pinned dependency set for reproducible builds. |
| `.env.example` | Template for `.env`. |

## Things worth knowing

**`GEMINI_KEY` must be set or the app will not start.** `main.py` imports
`app/actions/fight/yolo_detector.py`, which imports `app/shared/gemini_review.py`, which reads
`os.environ["GEMINI_KEY"]` at module level. An unset value is a `KeyError` during
import. A placeholder is fine — nothing the container runs calls Gemini.

**Code is copied into the image at build time, not mounted.** Editing a detector
does not affect a running container until `docker compose up -d --build`. That is
what makes a client install self-contained, but it surprises people
mid-development — keep using native `uvicorn main:app --reload` for day-to-day
work and Docker for deployment.

**Pinned vs unpinned deps.** `requirements.txt` stays unpinned for local
development; `docker/requirements.txt` pins everything so the image builds
identically later. They must be kept in sync by hand.

**Architecture.** An image built on an Apple Silicon Mac is `arm64` and will not
run on an x86 client PC. For an Intel/AMD target:

```bash
docker buildx build --platform linux/amd64 -t cctv-detection:amd64 .
```

**Size.** ~2.8GB — torch, ultralytics and the MoViNet stack. The CPU-only torch
index in the `docker/Dockerfile` already saves about 2.5GB over the default CUDA wheels.
For GPU, swap that index URL for the matching `cu___` channel.

**Speed.** A 6-second clip took ~25s to analyze in this container on an 8-core
M-series Mac — roughly 4x slower than real time. Docker on macOS runs in a VM and
is pessimistic, so measure natively too, but plan for workload dials (frame rate,
duty cycle, detector selection) rather than assuming real-time analysis.

## Before this ships to a client

- **`models/movinet/` is CC BY-NC 4.0, internal/research use only** (see
  `models/movinet/NOTICE.md`). Those weights are baked into the image, so it cannot
  currently be distributed commercially.
- **Ultralytics YOLO is AGPL-3.0.** Worth a proper look before selling a service
  built on it.
- The container runs as **root**. Fine for an isolated appliance, worth hardening
  before wider deployment.
- Evidence is **never cleaned up**. On 24/7 operation the evidence folders grow
  until the disk fills.

## Next

1. Config file for stream URLs
2. ffmpeg recorder writing fixed-length segments
3. Worker that analyzes each segment and saves evidence
4. Retention, alerting, health reporting
