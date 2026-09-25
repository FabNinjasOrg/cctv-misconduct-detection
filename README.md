# CCTV Detection

Watches CCTV video and tells you when something happens — someone **spitting**, a **fight**, **vandalism**, or a **fall**.

When it finds something it saves a picture of the moment, with a box drawn around the person and a label like `Fight 99%`. That picture is the point of the whole thing; everything else here exists to produce it.

There are two ways to use it, and most people end up using both:

| Way | How it works | Results land in |
|---|---|---|
| **Live camera** | Point it at a camera and leave it running. It records and checks around the clock, on its own. | `evidence/live_detections/` |
| **Upload a video** | Hand it a file and ask what's in it. Good for checking an incident someone told you about. | `evidence/manual_detections/` |

---

## Quick start

Three steps. You need your camera's **username** and its **IP address** — both are in the camera's phone app, usually under *Advanced Settings → Camera Account*.

> **The username trips up almost everyone.** On TP-Link Tapo cameras it's a name *you* chose when setting the camera up, and it's usually **not** `admin`. If the logs later say `401 Unauthorized`, this is nearly always why.

**1. Create your settings file**

```bash
cp .env.example .env
```

**2. Put your camera password in `.env`**

```
CAM1_PASSWORD=your-camera-password-here
```

No quotes. If the password contains awkward characters like `@` or `:`, leave them exactly as they are — the code escapes them for you.

**3. Put your camera's address in `stream.yaml`** (line 11)

```yaml
cameras:
  - name: cam-1
    url: rtsp://USERNAME:${CAM1_PASSWORD}@CAMERA-IP:554/stream1
```

Swap in your username and IP address (something like `192.168.1.64`). Leave `${CAM1_PASSWORD}` alone — it reaches into `.env` and fetches the password, which is what keeps the real password out of this file.

**Then start it:**

```bash
docker compose up -d --build
```

That's everything. It's now recording and checking. The first run takes a few minutes to build; after that it's quick. Results appear in **`evidence/live_detections/`**.

---

## Everyday commands

**Starting and stopping the camera:**

| Command | What it does |
|---|---|
| `docker compose start stream` | **Start recording** |
| `docker compose stop stream` | **Stop recording** |
| `docker compose restart stream` | **Apply a change** you made in `stream.yaml` |
| `docker compose up -d --build stream` | **Apply a code change** (rebuilds first) |

These four only touch the camera. The API keeps running throughout, so <http://localhost:8000/docs> stays available for uploading videos by hand.

> If you've run `docker compose down`, the containers no longer exist and `start` won't find them. Use `docker compose up -d stream` instead — that works whether the container exists or not, so it's the safe one to remember.

**Everything else:**

| Command | What it does |
|---|---|
| `docker compose logs -f stream` | Watch what the camera is doing right now (Ctrl-C to exit) |
| `docker compose ps` | Check what's running |
| `docker compose up -d` | Start everything |
| `docker compose down` | Stop everything |

Your evidence is safe through all of these — it lives in folders on your machine, not inside the containers.

---

## Checking a video manually

Open **<http://localhost:8000/docs>**. You get a page listing everything the system can do, and you can run any of it there without touching a terminal.

It's a two-step job:

1. **`POST /upload-clip`** → *Try it out* → choose your video → *Execute*. It hands back a path like `uploaded_clips/20260925_141500_myvideo.mp4`.
2. **`POST /analyze-action`** → *Try it out* → paste that path into the `filepath` box → *Execute*.

After a minute or so you get the verdict:

```json
{
  "activity_detected": true,
  "action": "vandalism",
  "confidence": 0.881,
  "final_frame": "evidence/manual_detections/vandalism-1_vandalism_88pct.jpg"
}
```

Open that `final_frame` file to see the moment it caught, boxed and labelled.

`/analyze-action` runs all four detectors and returns the single strongest answer, which is almost always what you want. The other endpoints on that page run one detector on its own — useful when you're investigating why something was or wasn't flagged, but not the everyday route.

---

## Where everything lives

```
cctv-detection/
│
├── evidence/              ← THE RESULTS
│   ├── live_detections/       from the cameras
│   └── manual_detections/     from videos you uploaded
│
├── uploaded_clips/        ← videos you've given it to check
├── stream.yaml            ← your camera settings      (you edit)
├── .env                   ← your camera password      (you edit)
│
├── main.py                the web API
├── app/                   all the code
├── models/                the trained AI files
├── docker/                build and packaging
├── debug/                 detector working-out, safe to delete
├── spool/                 scratch space, empties itself
└── docs/                  design notes
```

Three things here are yours: **`evidence/`**, **`stream.yaml`** and **`.env`**. The rest is machinery you can leave alone.

**Inside `evidence/`** there are exactly two folders:

| Folder | What's in it |
|---|---|
| **`live_detections/`** | What the cameras caught. A folder per camera holding the picture and the video clip it came from, plus **`detections.json`** — the running log of everything found. |
| **`manual_detections/`** | What the videos you uploaded caught. One labelled picture per video. |

**Inside `app/`**, if you ever need it: `actions/` holds the four detectors, `stream/` does the camera recording, `shared/` is the common code between them.

---

## The detection log

`evidence/live_detections/detections.json` is the timestamped record of everything the cameras found. It's a plain JSON array, newest entry last, so you can open it in any editor or load it straight into a script, spreadsheet or dashboard.

```json
[
  {
    "recorded_at": "2026-09-24T07:18:30+05:30",
    "detected_at": "2026-09-24T07:20:30+05:30",
    "camera": "cam-1",
    "action": "fall",
    "confidence": 0.704,
    "confidence_percent": 70,
    "evidence_image": "evidence/live_detections/cam-1/cam-1_20260924-071830_fall_70pct.jpg",
    "evidence_clip": "evidence/live_detections/cam-1/cam-1_20260924-071830_fall_70pct.mp4",
    "scores": { "spit": 0.185, "vandalism": 0.207, "fall": 0.704, "fight": 0.0 },
    "source_segment": "cam-1_20260924-071830.mp4"
  }
]
```

| Field | What it tells you |
|---|---|
| **`recorded_at`** | **When it actually happened.** This is the one you want. |
| `detected_at` | When the computer finished checking. Can be a minute or more later — that gap is analysis time, not a second incident. |
| `evidence_image` | The labelled picture. Open this to see what it caught. |
| `evidence_clip` | The ten seconds of video it came from. A still often isn't enough to judge. |
| `confidence` | How sure it is, 0 to 1. Anything below `min_confidence` in `stream.yaml` never gets here. |
| `scores` | What *all four* detectors thought, not just the winner. Useful when a result surprises you. |

**Set `TZ` in `.env`** (e.g. `TZ=Asia/Kolkata`) or these times come out in UTC. It applies to the recorded filenames too, so everything stays consistent.

The file is never appended to blindly: it's rewritten through a temporary file and renamed into place, so pulling the plug mid-write leaves the last good version rather than a broken one. If it ever does become unreadable it's moved aside as `detections.json.corrupt-<timestamp>` instead of being overwritten — it's the only record you have.

Uploaded videos don't write to this log; their results come back in the API response and land in `manual_detections/`.

---

## When something's wrong

| What you see | What it usually means |
|---|---|
| **`401 Unauthorized`** | Wrong username or password. Start with the username — it's usually not `admin`. |
| **`Connection refused`** or a timeout | Either the IP is wrong, or the camera is on a different network from this computer. A camera at another site isn't reachable just because you know its address; the two networks have to be genuinely connected. |
| **Nothing in `live_detections/`** | Normal if nothing has happened. Run `docker compose logs -f` — if it's working you'll see it recording and checking each clip. |
| **"not keeping up in real time"** | This computer can't check every clip. Set `analyze_one_in: 3` in `stream.yaml` and it'll check a sample instead — roughly twenty seconds out of every minute, usually plenty. |

---

## Running without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --reload     # the API, at http://127.0.0.1:8000/docs
python -m app.stream          # the camera recorder
```

Run these from this folder — the code looks for `models/` and `evidence/` relative to where you started it.

You can also watch a video play back with the detection drawn on top as it goes, which is the best way to get a feel for what the detectors actually see:

```bash
python3 -m app.actions.spit.playback      uploaded_clips/your-clip.mp4
python3 -m app.actions.fight.playback     uploaded_clips/your-clip.mp4
python3 -m app.actions.vandalism.playback uploaded_clips/your-clip.mp4
python3 -m app.actions.fall.playback      uploaded_clips/your-clip.mp4
```

---

## Before you sell this to anyone

Two licensing problems worth sorting out early — both are far easier to fix now than after a customer is using it:

- The fight detector's weights in `models/movinet/` are **CC BY-NC 4.0 — non-commercial use only**. They're baked into the Docker image, so as things stand that image can't be sold. See `models/movinet/NOTICE.md`.
- **Ultralytics YOLO is AGPL-3.0**, which carries real obligations for anything built on it. Worth a proper look.

One operational thing too: **nothing cleans up old evidence**. Running a camera around the clock, those folders grow until the disk fills.

---

For deployment details and tuning, see [`docker/DOCKER.md`](docker/DOCKER.md). For how the detection pipeline is designed, see [`docs/general.md`](docs/general.md).
