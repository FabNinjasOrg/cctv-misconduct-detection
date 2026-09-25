# CCTV Detection

Watches CCTV video and flags four things: **spitting, fighting, vandalism, falls.**

You can use it two ways:

- **Live** — point it at a camera, it records and checks around the clock.
- **By hand** — upload a video file and get an answer back.

---

## 1. What's in each folder

Only three of these are yours to open day to day. They're the first three.

| Folder / file | What's inside |
|---|---|
| **`evidence/`** | **Everything it caught.** A picture of the moment, plus the video clip. Start here. |
| **`uploaded_clips/`** | Videos you gave it to check. |
| **`stream.yaml`** | **Your camera settings** — address, how often to check. You edit this. |
| `.env` | Your camera password. You edit this once. |
| `main.py` | The web API. |
| `app/` | All the code. |
| `models/` | The AI brain files. Don't touch. |
| `docker/` | Build/packaging files. Don't touch. |
| `spool/` | Scratch space while recording. Empties itself. |
| `docs/` | Design notes. |

### Inside `evidence/`

| Folder | What's inside |
|---|---|
| **`evidence/detections/`** | **The final answer for each video.** One picture, labelled `Fight 99%`. This is the one you want. |
| **`evidence/live/`** | What the cameras caught, one folder per camera. Picture + video clip + `detections.jsonl` (the running list). |
| `evidence/spit/` `fight/` `vandalism/` `fall/` | Each detector's own rough working-out. For digging into *why* something was flagged. |

### Inside `app/`

| Folder | What's inside |
|---|---|
| `app/actions/` | The four detectors: `spit/`, `fight/`, `vandalism/`, `fall/` — plus `analyzer.py`, which runs all four and picks the winner. |
| `app/stream/` | Talks to the cameras and records them. |
| `app/shared/` | Helper code used by all the detectors. |

---

## 2. Set it up (once)

**Step 1 — make your settings file**

```bash
cp .env.example .env
```

**Step 2 — put your camera password in `.env`**

Open `.env`, find this line, type the password after the `=`:

```
CAM1_PASSWORD=your-camera-password-here
```

No quotes. Special characters like `@` are fine — leave them as they are.

**Step 3 — put your camera address in `stream.yaml`**

Open `stream.yaml`, line 11:

```yaml
cameras:
  - name: cam-1
    url: rtsp://USERNAME:${CAM1_PASSWORD}@CAMERA-IP:554/stream1
```

Change two things:

- `USERNAME` → the camera's login name
- `CAMERA-IP` → the camera's address on your network, e.g. `192.168.1.64`

Leave `${CAM1_PASSWORD}` alone — it picks the password up from `.env` for you.

> **Where do I find these?** In your camera's phone app, usually under
> Advanced Settings → Camera Account. On TP-Link Tapo the username is one you
> chose yourself — it is usually **not** `admin`.

---

## 3. Start it

```bash
docker compose up -d --build
```

That's it. It is now recording and checking.

Anything it finds appears in **`evidence/live/`**.

### Other commands

```bash
docker compose logs -f          # watch what it's doing right now
docker compose down             # stop it
docker compose restart stream   # reload after editing stream.yaml
```

---

## 4. Check one video by hand

Open **<http://localhost:8000/docs>** in your browser.

1. `POST /upload-clip` → **Try it out** → pick your video → **Execute**.
   It replies with a path like `uploaded_clips/20260925_141500_myvideo.mp4`.
2. `POST /analyze-action` → **Try it out** → paste that path in → **Execute**.

You get back the verdict:

```json
{
  "activity_detected": true,
  "action": "vandalism",
  "confidence": 0.881,
  "final_frame": "evidence/detections/vandalism-1_vandalism_88pct.jpg"
}
```

Open that `final_frame` file to see the moment, boxed and labelled.

---

## 5. If it's not working

| What you see | What it usually means |
|---|---|
| `401 Unauthorized` in the logs | Wrong username or password. The username is the tricky one — check the camera app. |
| `Connection refused` / `timed out` | Wrong IP, or the camera is on a different network from this computer. |
| Nothing in `evidence/live/` | Normal if nothing happened. Run `docker compose logs -f` to confirm it's recording. |
| "not keeping up in real time" warning | This computer is too slow for every-segment checking. In `stream.yaml`, set `analyze_one_in: 3`. |

---

## Running without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --reload           # http://127.0.0.1:8000/docs
python -m app.stream                # the camera recorder
```

Run these **from this folder** — the code looks for `models/` and `evidence/`
relative to it.

Watch one video play back with the detection drawn on top:

```bash
python3 -m app.actions.spit.playback      uploaded_clips/your-clip.mp4
python3 -m app.actions.fight.playback     uploaded_clips/your-clip.mp4
python3 -m app.actions.vandalism.playback uploaded_clips/your-clip.mp4
python3 -m app.actions.fall.playback      uploaded_clips/your-clip.mp4
```

---

## Two things to know before selling this

- The fight model's weights (`models/movinet/`) are **CC BY-NC 4.0 —
  non-commercial only.** See `models/movinet/NOTICE.md`.
- Ultralytics YOLO is **AGPL-3.0**, which has its own conditions.

More detail: `docker/DOCKER.md` (deployment), `docs/general.md` (design notes).
