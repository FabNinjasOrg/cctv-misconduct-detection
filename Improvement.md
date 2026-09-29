```text
                    CCTV
                      │
               YOLO + ByteTrack
                      │
              Per-person tracks
                      │
            ┌─────────┼─────────┐
            │         │         │
            ▼         ▼         ▼
          FALL       FIGHT    SPITTING
            │         │         │
        Pose + RGB  RGB +      Head/
        + motion    interaction mouth
            │         │         │
            └─────────┼─────────┘
                      ▼
              Temporal Model
             X3D / VideoMAE
                      │
                      ▼
               State Machine
                      │
                      ▼
                  CONFIRMED
                      │
                      ▼
                    ALERT
```

### The biggest difference

**Fall:**
Your current pose information is very useful.

**Fight:**
You need **interaction between people**, not just individual pose. For example:

```text
Person A movement
        +
Person B movement
        +
distance between A/B
        +
rapid movement
        ↓
fight candidate
```

**Spitting:**
This is the difficult one. If the person is far away or the CCTV resolution is low, even a very good temporal model may not have enough visual information. You may need a **head/mouth crop + higher-resolution stream** for this particular event.

### If I were building it

I'd aim for:

```text
Phase 1:
Fall + Fight
        ↓
prove temporal pipeline


Phase 2:
Spitting
        ↓
specialized head/mouth pipeline


Phase 3:
Collect false positives
        ↓
fine-tune on your actual CCTV


Phase 4:
Measure separately:
Fall precision/recall
Fight precision/recall
Spitting precision/recall
False alerts/hour
Detection latency
```

So **8/10 architecture** is reasonable, but I would *not* interpret that as "8/10 accuracy." Your eventual accuracy could be substantially different depending on your camera footage and training data.


============

Build a **Mac M2 (Apple Silicon) 8 GB RAM–friendly CCTV AI action-recognition MVP** in Python.

The goal is to create a live CCTV-style demo where a video/RTSP stream is displayed in real time with person bounding boxes, tracking IDs, current activity labels, and event detection. When a suspicious activity such as a fall or fight is detected, the system should recognize the event using a temporal action-recognition model and automatically save the relevant event clip.

## 1. MVP goal

Build this pipeline:

LIVE VIDEO
↓
OpenCV / FFmpeg
↓
YOLO person detection
↓
ByteTrack tracking
↓
Live overlay
↓
Per-person rolling frame buffer
↓
Temporal action recognition (X3D-S initially)
↓
State machine / confidence filtering
↓
Event confirmation
↓
Save event clip

The system must NOT block the live video stream while X3D is performing inference.

The live stream must continue consuming/displaying frames independently of temporal-model inference.

---

# 2. Target platform

Primary development environment:

* macOS
* Apple Silicon M2
* 8 GB unified memory
* Python 3.11
* PyTorch with Apple MPS support

Use:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Do NOT require CUDA or NVIDIA-specific libraries for the MVP.

Do not require TensorRT, Docker, Kubernetes, or cloud APIs.

The architecture should later be portable to NVIDIA/CUDA hardware.

---

# 3. Input sources

Support two input modes:

### Mode A — MP4

Example:

```bash
python main.py --source videos/test.mp4
```

Treat the MP4 like a live camera feed rather than processing it as a batch file.

The UI should play the video continuously and the AI pipeline should process it asynchronously.

### Mode B — RTSP

Support:

```bash
python main.py --source "rtsp://..."
```

Use OpenCV/FFmpeg for the video source.

Create a clean `VideoSource` abstraction so MP4 and RTSP use the same downstream pipeline.

---

# 4. Person detection

Use Ultralytics YOLO.

Initially use a lightweight model suitable for the M2, such as a nano/small model.

Only detect people initially.

The detector should output:

```text
person_id
bounding_box
confidence
frame_number
timestamp
```

Do not detect unnecessary object classes in the first MVP.

Allow the YOLO model path and inference size to be configured from a YAML/config file.

---

# 5. Tracking

Use ByteTrack.

Each detected person should receive a stable tracking ID.

Example:

```text
Person #1
Person #2
Person #3
```

The tracking ID must remain stable while the person remains visible.

Create a clean interface:

```python
class PersonTracker:
    def update(self, detections, frame):
        ...
```

Do not tightly couple tracking code to the UI.

---

# 6. Live overlay

Display the video using OpenCV.

For each tracked person draw:

* bounding box
* person ID
* current action/activity
* confidence
* optional state

Example:

```text
┌──────────────────────┐
│ Person #7             │
│ WALKING 94%           │
└──────────────────────┘
```

For a suspicious event:

```text
Person #7
FIGHT 91%
VERIFYING
```

For confirmed:

```text
Person #7 + #9
FIGHT DETECTED
91%
```

Also display global information:

```text
CAMERA: CAM-01
FPS: 14.8
AI STATUS: MONITORING
```

The overlay must not significantly reduce the video FPS.

---

# 7. Rolling frame buffer

Do NOT continuously save the entire CCTV stream to disk.

Instead create an in-memory rolling buffer.

Use something similar to:

```python
collections.deque
```

Maintain approximately 2–4 seconds of recent frames.

The buffer should be configurable:

```yaml
buffer:
  seconds: 4
  fps: 15
```

The system should keep the most recent frames in RAM.

Do not write every frame to disk.

---

# 8. Per-person temporal buffer

Because X3D is an action-recognition model, maintain temporal frame sequences for tracked people.

Example:

```text
Person #7

Frame 101
Frame 102
Frame 103
...
Frame 132
```

The temporal model should receive a short sequence rather than a single frame.

The implementation must handle:

* person entering the scene
* person leaving the scene
* temporary tracking loss
* tracker ID changes
* multiple people
* limited memory

Clean old buffers when a person disappears for a configurable timeout.

---

# 9. Temporal model

Initially use:

**X3D-S**

Run it locally using PyTorch.

The model should be loaded once at startup.

Do NOT load/reload the model for every prediction.

Create an abstraction:

```python
class ActionRecognizer:

    def predict(self, frames):
        ...
```

Then implement:

```python
class X3DRecognizer(ActionRecognizer):
    ...
```

This is important because later we may replace X3D with:

* X3D-M
* MoViNet
* VideoMAE
* another temporal model
* a local VLM

without changing the rest of the application.

---

# 10. Asynchronous architecture

This is VERY important.

Do NOT implement:

```text
read frame
↓
YOLO
↓
X3D
↓
wait
↓
read next frame
```

This causes the live stream to fall behind.

Instead implement an asynchronous pipeline:

```text
                 VIDEO THREAD
                      │
                      ▼
                Frame Queue
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Detection/Tracking       Rolling Buffer
          │                       │
          ▼                       ▼
    Live Overlay             X3D Worker
                                  │
                                  ▼
                           Prediction Queue
                                  │
                                  ▼
                            Event Manager
```

The live video producer must continue even if X3D is busy.

Use bounded queues.

Do not allow unlimited frames to accumulate.

If the temporal model falls behind, discard stale temporal windows and process the latest available window.

The system must prefer **freshness over processing every historical frame**.

---

# 11. X3D inference frequency

Do NOT necessarily run X3D on every video frame.

Make inference frequency configurable.

For example:

```yaml
action_recognition:
  clip_length: 32
  inference_interval: 0.75
```

This means approximately one temporal prediction every 0.75 seconds.

The exact values should be easy to change.

Log:

```text
X3D inference time
X3D FPS
queue size
latency
```

---

# 12. Initial action classes

For the first MVP implement:

```text
NORMAL
FALL
FIGHT
```

Do NOT implement spitting yet.

Structure the code so additional actions can later be added:

```text
SPITTING
VANDALISM
RUNNING
PUSHING
KICKING
```

without rewriting the pipeline.

---

# 13. State machine

Do not trigger an alert from one X3D prediction.

Implement temporal smoothing/state confirmation.

Example:

```text
NORMAL
   ↓
SUSPICIOUS
   ↓
VERIFYING
   ↓
CONFIRMED
```

For example:

```text
Prediction 1:
FIGHT 0.72

Prediction 2:
FIGHT 0.81

Prediction 3:
FIGHT 0.91

→ CONFIRMED
```

Make thresholds configurable.

Example:

```yaml
events:
  fight:
    confirmation_threshold: 0.85
    required_predictions: 2
    cooldown_seconds: 10

  fall:
    confirmation_threshold: 0.85
    required_predictions: 2
    cooldown_seconds: 10
```

These values are starting defaults, not claims of optimal accuracy.

---

# 14. Event recording

When an event is confirmed:

DO NOT start the video recording from zero.

Use the existing rolling buffer.

Save:

```text
3 seconds BEFORE event
+
event
+
3–5 seconds AFTER event
```

Example:

```text
event_2026-09-28_10-32-18_fight.mp4
```

The event recorder should continue capturing frames after the event is detected, then combine the pre-event and post-event frames into a single video file.

Use OpenCV VideoWriter or FFmpeg.

The event recording must happen asynchronously so saving the MP4 does not freeze the live stream.

---

# 15. Event metadata

For every confirmed event save a JSON record:

```json
{
  "event_id": "uuid",
  "camera_id": "CAM-01",
  "event_type": "fight",
  "confidence": 0.91,
  "timestamp": "2026-09-28T10:32:18",
  "track_ids": [7, 9],
  "clip_path": "events/event_...mp4"
}
```

Store event metadata initially in SQLite.

---

# 16. Event dashboard

Create a simple local web UI.

Use:

**FastAPI + simple HTML/JavaScript**

Do not build a complicated frontend framework for the MVP.

The page should show:

```text
┌────────────────────────────────────────────────┐
│ CCTV AI MONITOR                                │
├────────────────────────────────────────────────┤
│                                                │
│              LIVE VIDEO                        │
│                                                │
│       bounding boxes + activity labels         │
│                                                │
├────────────────────────────────────────────────┤
│ EVENTS                                         │
│                                                │
│ 10:32:18  FIGHT  91%   [View Clip]             │
│ 10:35:42  FALL   94%   [View Clip]             │
│                                                │
└────────────────────────────────────────────────┘
```

The MVP may initially display the OpenCV video window instead of browser streaming if implementing browser video streaming would add unnecessary complexity.

However, keep the UI layer separated so browser streaming can be added later.

---

# 17. Performance monitoring

Display/log:

```text
Camera FPS
Detection FPS
X3D inference time
X3D prediction frequency
Queue size
End-to-end event latency
GPU/MPS utilization where available
RAM usage
```

The most important metrics are:

```text
LIVE FPS
AI LATENCY
EVENT DETECTION LATENCY
```

The live stream must not continuously fall further behind the source.

---

# 18. Important anti-lag behavior

Implement a "latest frame/window wins" strategy.

If X3D is busy:

```text
Window 1
Window 2
Window 3
Window 4
Window 5
```

and the model is still processing Window 1, do not blindly process all five afterward.

Prefer the newest valid window.

The goal is:

```text
LOW LATENCY
```

rather than:

```text
PROCESS EVERY FRAME NO MATTER HOW OLD
```

This is critical for real-time CCTV.

---

# 19. M2 memory constraints

The machine has only 8 GB unified memory.

Therefore:

* use lightweight YOLO
* start with X3D-S
* use small frame buffers
* avoid loading multiple large models unnecessarily
* avoid storing full-resolution frames indefinitely
* resize frames before inference
* release unused tensors
* avoid unnecessary copies between CPU and MPS
* keep queues bounded

Make all major memory-related settings configurable.

---

# 20. Project structure

Use:

```text
cctv-ai-mvp/
│
├── app/
│   ├── video/
│   │   ├── source.py
│   │   ├── buffer.py
│   │   └── recorder.py
│   │
│   ├── detection/
│   │   └── yolo_detector.py
│   │
│   ├── tracking/
│   │   └── bytetrack.py
│   │
│   ├── action/
│   │   ├── base.py
│   │   ├── x3d.py
│   │   └── preprocessing.py
│   │
│   ├── events/
│   │   ├── state_machine.py
│   │   └── manager.py
│   │
│   ├── pipeline/
│   │   └── workers.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   └── config.py
│
├── models/
│
├── videos/
│
├── events/
│
├── web/
│   └── templates/
│
├── tests/
│
├── config.yaml
├── requirements.txt
├── main.py
└── README.md
```

---

# 21. Configuration

Create:

```yaml
camera:
  id: CAM-01
  source: "videos/test.mp4"

video:
  processing_fps: 10
  display_fps: 15
  inference_width: 640

buffer:
  seconds: 4

detection:
  model: "yolo_light.pt"
  confidence: 0.40

action_recognition:
  model: "x3d_s"
  clip_length: 32
  inference_interval: 0.75
  device: "mps"

events:
  fall:
    threshold: 0.85
    required_predictions: 2
    cooldown_seconds: 10

  fight:
    threshold: 0.85
    required_predictions: 2
    cooldown_seconds: 10

recording:
  pre_event_seconds: 3
  post_event_seconds: 5
  output_dir: "events"
```

---

# 22. Development stages

Do not attempt to build everything simultaneously.

Build in this order:

### Stage 1

```text
MP4
 ↓
OpenCV
 ↓
Live display
```

### Stage 2

```text
OpenCV
 ↓
YOLO
 ↓
Bounding boxes
```

### Stage 3

```text
YOLO
 ↓
ByteTrack
 ↓
Person IDs
```

### Stage 4

```text
Rolling buffer
 ↓
visualize buffer status
```

### Stage 5

```text
X3D-S
 ↓
single-person action prediction
```

### Stage 6

```text
X3D
 ↓
state machine
 ↓
event confirmation
```

### Stage 7

```text
event confirmation
 ↓
save pre-event + post-event clip
```

### Stage 8

Add:

```text
FastAPI
 ↓
event dashboard
```

### Stage 9

Replace:

```text
MP4
```

with:

```text
RTSP
```

---

# 23. Acceptance criteria

The MVP is considered successful when:

1. A video plays continuously as a live stream.
2. Person bounding boxes appear.
3. ByteTrack IDs remain stable.
4. The overlay shows the current activity.
5. X3D receives temporal clips directly from the live frame buffer.
6. X3D inference does not block frame capture/display.
7. If X3D is slower than the stream, old temporal windows are dropped rather than accumulating indefinitely.
8. A confirmed fall/fight changes the overlay to an event state.
9. The system saves approximately 3 seconds before the event and 3–5 seconds after it.
10. Saving the event clip does not freeze the live stream.
11. Event metadata is stored in SQLite.
12. The system reports inference latency and FPS.
13. The system runs on an M2 Mac with 8 GB RAM using MPS where supported.
14. No cloud VLM/API is required.

---

# 24. Important engineering principle

The system must prioritize:

```text
LIVE STREAM CONTINUITY
        >
PROCESSING EVERY FRAME
```

If the AI cannot process every frame fast enough, it should intelligently skip/drop stale inference work rather than allowing latency to grow indefinitely.

The live video should remain close to real time while the temporal model analyzes the most recent available temporal window.

---

# 25. Do NOT over-engineer

This is an MVP.

Do NOT add:

* Kubernetes
* Docker
* cloud VLM
* microservices
* Redis
* Kafka
* distributed inference
* TensorRT
* CUDA
* multi-GPU support
* complex authentication
* production deployment infrastructure

Keep everything local and simple.

The purpose of this MVP is to validate:

1. Can we process live CCTV?
2. Can we maintain person tracking?
3. Can X3D recognize the action?
4. Can we maintain near-real-time display?
5. Can we detect an event with acceptable latency?
6. Can we automatically save the relevant event clip?

After these are proven, we can optimize for NVIDIA edge hardware and production deployment.

---

# 26. First implementation request

Start by implementing **Stage 1–3 only**:

```text
MP4
 ↓
OpenCV
 ↓
YOLO
 ↓
ByteTrack
 ↓
Live overlay
```

Make this stable first.

Then add the temporal/action-recognition worker separately.

Do not generate a huge monolithic script.

Use clean modules/classes and explain how to run each stage.

Provide:

* `requirements.txt`
* `config.yaml`
* project files
* setup instructions for macOS M2
* MPS detection
* commands to run
* troubleshooting for common Apple Silicon/PyTorch issues

After Stage 1–3 are working, continue with the X3D rolling-buffer implementation.

========

Fine-tune a pretrained model — recommended first.
Train a model from scratch — usually unnecessary and needs much more data.
Use multiple models together — YOLO + ByteTrack + pose + temporal model + rules can significantly reduce false positives.
Use a VLM as a secondary verifier — useful, but adds latency/cost.
Improve the dataset and hard negatives — often one of the biggest accuracy improvements.