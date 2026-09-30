# CCTV Live Detection

Watch a CCTV camera and see, as it happens, when someone **falls**, **fights**, **spits** or **vandalises** something.

---

## What this is

- A program that connects to **one CCTV camera** on your network.
- It watches every person in the picture and works out what they are doing.
- It shows the live video **in your web browser**, with a box and a label on each person.
- When it spots one of the four actions, it **saves a picture** of that moment as proof.
- It records no video and keeps no database. Only the pictures are kept.

---

## How it works

1. **Watch.** It reads the camera's live video, 15 pictures every second.
2. **Find people.** An AI model finds each person and marks their joints: head, shoulders, hips, knees and so on.
3. **Follow them.** Each person gets a number and is followed from picture to picture, so it builds up a few seconds of how they moved.
4. **Judge.** That movement is checked against each action, and every action gets a score from 0 to 100%:
   - **Fall:** the body goes from upright to lying down in under a second, and stays down.
   - **Spit:** the head dips sharply forward and holds there.
   - **Vandalism:** a very fast swing of an arm or kick of a leg. It scores higher if a bat, bottle or knife is in the hand.
   - **Fight:** judged on the **whole picture**, not on one person. A second AI model watches one second of video and says how much it looks like a fight.
5. **Ignore stillness.** Someone standing or sitting still isn't judged at all. All four actions involve movement, so skipping still people prevents false alarms.
6. **Show.** Boxes and labels are drawn on the video, and it goes to your browser.
7. **Wait and check.** When a score reaches 60%, it isn't called yet. The box turns amber and it watches for 1 more second. If the score stays above 50% the whole time, it's confirmed. If it drops, it was something else, such as a head dipped at a phone, and nothing happens.
8. **Save.** Once confirmed, the box turns red and a picture of the moment it peaked is saved.

### For developers

```
Camera ─► capture.py ─► engine.py ──────────────────────────► render.py ─► server.py ─► browser
(RTSP)    keeps only    pose (YOLO11) → tracking →              boxes,      MJPEG on
          the newest    spit / fall / vandalism scores          labels      port 8088
          frame         + fight (MoViNet, whole scene)            │
                                                                  └─► evidence.py ─► evidence/live_view/
```

- **Entry point:** `python -m app.live`, in `app/live/__main__.py`.
- **Detectors:** `app/actions/`, one folder per action. The live engine calls their real scoring functions rather than keeping its own copy.
- **Threads:** four of them, for capture, the frame loop, pose scoring and fight scoring. A slow score only delays a label; it never makes tracking skip frames.
- **Design notes:** `CLAUDE.md` and `docs/general.md`. Read them before changing any threshold.

---

## What you need

| | |
|---|---|
| **Docker** | Docker Desktop on Mac or Windows, or Docker Engine with Compose on Linux |
| **A camera** | Any RTSP camera on the same network as this computer |
| **Processor** | 4 cores minimum, **8 recommended** |
| **Memory** | 4 GB minimum, **8 GB recommended** |
| **Disk** | About 10 GB free. The Docker image is big because of the AI libraries. |
| **Graphics card** | Not needed. The Docker setup runs on the processor only. |

---

## Set it up with Docker

Run every command from the project folder.

### 1. Install Docker

- **Mac or Windows:** install Docker Desktop and open it.
- **Linux:** install Docker Engine and the Compose plugin.
- **Check it works:**

  ```bash
  docker compose version
  ```

### 2. Create your settings file

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
CAM1_PASSWORD=your-camera-password
```

- **`CAM1_PASSWORD`:** the camera password exactly as it is. Don't add quotes. Characters like `@` or `:` are fine.

### 3. Add your camera's address

Open `camera.yaml` and change the `url` line:

```yaml
cameras:
  - name: cam-1
    url: rtsp://USERNAME:${CAM1_PASSWORD}@192.168.1.64:554/stream1
```

- **What to change:** `USERNAME`, the IP address and the ending (`/stream1`), to match your camera.
- **What to leave alone:** `${CAM1_PASSWORD}`. It fills in the password from `.env`.
- **The username is usually not `admin`.** On Tapo, Reolink and most home cameras, you create a separate camera account in the phone app, often under *Advanced Settings → Camera Account*. Creating that account is often what turns the video feed on in the first place.
- **No camera yet?** Skip this step. The page still opens, but the picture stays blank.

### 4. Build and start

```bash
docker compose up -d --build
```

- **First time:** it takes several minutes, because it downloads a few GB of AI libraries.
- **After that:** it starts in seconds.

### 5. Open it in your browser

- **On this computer:** **<http://localhost:9000>**
- **Allow about a minute after starting** while the AI models load. It is ready when `docker compose logs` shows `live view on http://localhost:9000`.

---

## Everyday commands

| To… | Run |
|---|---|
| Start it | `docker compose up -d` |
| Stop it | `docker compose down` |
| Restart it | `docker compose restart` |
| See what it's doing | `docker compose logs -f` (Ctrl-C stops watching, not the program) |
| Check it's running | `docker compose ps` |

- **It runs in the background.** It also starts again by itself whenever Docker restarts, until you run `docker compose down`.
- **Stopping it never deletes your pictures.** They are saved in the `evidence/` folder on your computer, not inside Docker.

---

## After you change something

| What you changed | Run | Why |
|---|---|---|
| `camera.yaml` | `docker compose restart` | The camera settings are only read at startup |
| `.env` | `docker compose up -d` | Docker picks up the new values and starts a fresh container |
| `compose.yaml` | `docker compose up -d` | Same as above |
| Code in `app/`, `models/`, `requirements.txt` or `Dockerfile` | `docker compose up -d --build` | The code is copied into the image, so the image has to be rebuilt |

**Not sure which one?** `docker compose up -d --build` is always safe. It just takes a little longer.

---

## What you'll see

| | |
|---|---|
| **Grey** box | Just spotted someone. It is still gathering their movement, so it's too early to judge. |
| **Green** box | Watching, and nothing detected |
| **Amber** box | Something looks likely, e.g. `spit? checking...`. It is waiting 1 second to see if it holds. Nothing is saved yet. |
| **Red** box | Something detected and confirmed. The label says what, e.g. `FALL 79%`. |
| **Red border** round the whole picture | A **fight** |

- **Why fight looks different:** it is judged on the whole scene, not on one person. There's no single body to point at, so the whole picture gets the border.
- **Raw scores:** under each label are the scores for every action, e.g. `vandalism 0.31  fall 0.29`. They show what *nearly* fired, which is useful while you are still learning to trust it.
- **Frame rate:** shown bottom-left. It should read close to 15.

---

## Where the pictures go

```
evidence/live_view/cam-1/cam-1_20260925-161424_fall_79pct.jpg
                         cam-1_20260925-161424_fall_79pct.json
                         └ camera   └ when       └ what └ how sure
```

- **One picture per event:** each confirmed detection saves one picture, with a small `.json` file beside it.
- **The picture:** it shows the moment the action peaked, not the second later when it was confirmed. It has the box, skeleton and label drawn on, so it shows what the system thought, not just what the camera saw.
- **The `.json` file:** it holds every action's score, the reasons behind it, and how the 1-second check went. That's how you tell a wrong alarm from a right one.

  ```json
  { "action": "spit", "confidence": 0.6,
    "scores": { "spit": 0.6, "vandalism": 0.17, "fall": 0.0 },
    "signals": { "swing_deg": 41.8, "bowed_duration_s": 3.9, "hand_proximity_score": 0.81 },
    "confirmation": { "held_for_s": 1.2, "lowest_score_while_held": 0.58 } }
  ```

---

## Running without Docker

You need Python 3.11 or newer. then run:

```bash
pip install -r requirements.txt
set -a && source .env && set +a     # load the camera password
python -m app.live                  # then open http://localhost:8088
```

Press **Ctrl-C** to stop it.

---

## Where everything lives

```
cctv-detection/
│
├── evidence/live_view/    ← THE RESULTS, one folder per camera
├── camera.yaml            ← your camera address     (you edit)
├── .env                   ← your camera password    (you edit)
│
├── Dockerfile             how the Docker image is built
├── compose.yaml           how Docker runs it: port, settings, folders
│
├── app/
│   ├── live/              the live view itself
│   ├── actions/           the four detectors
│   └── shared/            pose tracking, shared by three of them
│
├── models/                the trained AI files
├── uploaded_clips/        test clips
└── docs/                  design notes
```

Three things here are yours: **`evidence/`**, **`camera.yaml`** and **`.env`**. The rest is machinery you can leave alone.
