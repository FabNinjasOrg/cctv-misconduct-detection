# CCTV Misconduct Detection

Watch an ordinary CCTV camera and spot **misconduct as it happens**: someone **fighting**, **vandalising** property or the camera itself, **falling** and needing help, or **spitting** anywhere.

> **Early stage.** It runs, but it is not reliable yet. Treat every alert as "go and look", never as proof. See [How well it works today](#how-well-it-works-today).

---

## The goal

Most CCTV is only watched after something has gone wrong. The aim here is a system that watches for misconduct quickly after it happens, so maybe someone can step in while it is still happening.

- **Real time.** Spot it as it happens, not hours later in a recording.
- **Works with ordinary CCTV cameras.** Any camera that can stream video over your network (RTSP) will do. No special hardware.
- **Runs on a normal computer.** No graphics card and no cloud service. The video never leaves your building.
- **Saves proof, not hours of video.** For each event it saves one picture, plus a note of how sure it was and why.
- **Few false alarms.** An alarm nobody trusts is worse than none, so it double-checks each detection before raising an alert.
- **Open, so it can get better.** Recognising behaviour well needs lots of real examples and many people testing. That is why this is open source.

---

## What it detects today

| Behaviour | Status | How it decides |
|---|---|---|
| **Fighting** | On | An AI model looks at the last second of video, focused on the area where the people are, and decides whether it looks like a fight. It only alerts when at least 2 people are in view. |
| **Vandalism** | On | Looks for a sudden, very fast arm swing or kick, like someone hitting or kicking something. If they are holding something like a bat, bottle or knife, it is more sure. Fixed rules, no AI training. |
| **Vandalism of the camera** | On | It learns what the camera normally sees. If the view goes dark (covered), goes blurry (sprayed or smeared) or changes (camera turned) for 5 seconds or more, it raises an alert. |
| **Falling** | On | Two checks, and both must agree. An AI model that learned from videos of people falling, and fixed rules that check the person went from standing to lying down, quickly, and stayed down. |
| **Spitting** | **Off** | It gave too many false alarms: someone looking down at their phone looks a lot like someone spitting. The code is still in `app/actions/spit/`; it is switched off in `app/live/engine.py`. |

---

## How well it works today

**Not reliable yet.** The first test on footage from our own camera (5 clips, about 6 minutes, one person acting out the events):

- **Caught 1 of 4 events** (a kick, as vandalism).
- **5 of its 6 alarms were wrong.**

**Why:** the two AI models were trained on other people's videos, filmed in other places. The vandalism check only measures how fast an arm or leg moves. None of it has been trained on footage from a real CCTV camera stream. Fixing that is the main job now.

---

## What's next

The plan, in order:

1. **Record real footage.** Several people act out each behaviour in front of a real CCTV camera, plus long stretches of ordinary activity, with everyone's consent. This is the biggest need.
2. **Label it and measure.** Write down when each event happens in each clip (clip, start, end, behaviour). Add a script that runs the clips through the system and counts what it caught, what it missed and its false alarms. Then every change can be measured instead of guessed.
3. **Retrain the fall and fight models** on that footage, and test them on clips they haven't seen. Training on our own footage also means the models are no longer limited to research use (see [What it's built with](#what-its-built-with)).
4. **Fix the known weak spots:**
   - **Fall:** both checks do notice the fall, but at slightly different moments, so they never agree at the same time and the fall is missed.
   - **Fight:** hardly reacts to the fights we acted out in front of our own camera.
   - **Vandalism:** goes off on any fast arm swing, even when nothing gets hit.
   - **Spit:** needs a new approach before it can be switched back on.
5. **More misconduct behaviours**, once these are reliable.

**Want to help with any of these?** Open an issue and say which one.

---

## What this is

- A program that connects to **one CCTV camera** on your network.
- It watches every person in the picture and works out what they are doing.
- It shows the live video **in your web browser**, with a box and a label on each person.
- When it spots one of the actions, it **saves a picture** of that moment as proof.
- It records no video and keeps no database. Only the pictures are kept.

---

## How it works

1. **Watch.** It reads the camera's live video, 15 pictures every second.
2. **Find people.** An AI model finds each person and marks their joints: head, shoulders, hips, knees and so on.
3. **Follow them.** Each person gets a number and is followed from picture to picture, so it builds up a few seconds of how they moved.
4. **Judge.** That movement is checked against each action, and every action gets a score from 0 to 100%:
   - **Fall:** two checks must agree. An AI model that learned from videos of people falling looks at how the person moved over the last 3 seconds. Fixed rules check that they went from standing to lying down, quickly, and stayed down. Either check alone gave too many false alarms.
   - **Vandalism:** a very fast swing of an arm or kick of a leg. It scores higher if a bat, bottle or knife is in the hand.
   - **Vandalism of the camera:** the camera is covered, blurred or sprayed, or turned to look somewhere else, for 5 seconds or more.
   - **Fight:** judged for the **group**, not for one person. An AI model that learned from fight videos looks at the last second of video, focused on the area where the people are, and scores how much it looks like a fight.
   - **Spit:** turned off for now.
5. **Ignore stillness.** Someone standing or sitting still isn't judged at all. All four actions involve movement, so skipping still people prevents false alarms.
6. **Show.** Boxes and labels are drawn on the video, and it goes to your browser.
7. **Wait and check.** When a score reaches 60%, it isn't called yet. The box turns amber and it watches for 1 more second. If the score stays above 50% the whole time, it's confirmed. If it drops, it was something else, such as a head dipped at a phone, and nothing happens.
8. **Save.** Once confirmed, the box turns red and a picture of the moment it peaked is saved.

### For developers

```
Camera ─► capture.py ─► engine.py ──────────────────────────► render.py ─► server.py ─► browser
(RTSP)    keeps only    pose (YOLO11) → tracking →              boxes,      MJPEG on
          the newest    fall (rules + ST-GCN) / vandalism       labels      port 8088
          frame         + fight (MoViNet, people crop)            │
                        + camera tamper                           │
                                                                  └─► evidence.py ─► evidence/live_view/
```

- **Entry point:** `python -m app.live`, in `app/live/__main__.py`.
- **Detectors:** `app/actions/`, one folder per action. The live engine calls their real scoring functions rather than keeping its own copy.
- **Threads:** four of them, for capture, the frame loop, pose scoring and fight scoring. A slow score only delays a label; it never makes tracking skip frames.
- **Design notes:** `CLAUDE.md` and `docs/general.md`. Read them before changing any threshold.

---

## What it's built with

### Ready-made AI models (used as they are)

Both are made by Ultralytics. We didn't train them.

| Model | What it does here | Learned from |
|---|---|---|
| **YOLO11-pose** (`yolo11n-pose.pt`) | Finds each person and marks 17 points on their body: head, shoulders, elbows, hips, knees and so on | **COCO-Pose**: photos of people with their joints marked |
| **YOLO11** (`yolo11n.pt`) | Spots things someone could hit with (bat, bottle, knife…), for the vandalism check | **COCO**: everyday photos with the objects in them marked |

### Models we trained

Both were trained with the notebooks in `training/`, on Kaggle's free GPUs.

| Model | Detects | What kind | Trained on |
|---|---|---|---|
| `fall_stgcn.pt` | Falling | **ST-GCN**, a model that learns from body movement. It watches how a person's joints move over 3 seconds. | **Le2i Fall Dataset**: people falling and doing everyday things, filmed in a home, office, coffee room and lecture room |
| `fight_movinet.pt` | Fighting | **MoViNet-A0**, a video model. It watches a 1-second clip. | **RWF-2000**: 2,000 CCTV clips from YouTube, half of them fights. **RLVS** (Real Life Violence Situations): 2,000 YouTube clips, half of them violent. Before this, the model had already learned general human movement from **Kinetics-600** (videos of 600 everyday actions), and we taught it fighting on top. |

### No training

- **Vandalism** and **spit** are fixed rules that look at how a person's joints move, using YOLO11-pose.
- **Camera vandalism** compares what the camera sees now with what it normally sees.

### Tools

| Tool | Used for |
|---|---|
| **Python** | Everything |
| **PyTorch** + torchvision | Running and training the fall and fight models |
| **Ultralytics** | Running YOLO11 |
| **MoViNet-pytorch** | Provides the MoViNet model we trained for fights |
| **OpenCV** | Reading the camera, drawing boxes, saving pictures |
| **NumPy** | The maths in every detector |
| **Kaggle notebooks** | Training, on free GPUs |

### Licences to know about

- **The training datasets are for research.** RWF-2000, for example, does not allow commercial use without its authors' approval. So `fall_stgcn.pt` and `fight_movinet.pt` are for research and testing only, until they are retrained on footage we have the rights to.
- **YOLO11 is AGPL-3.0** (Ultralytics).

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
| **Amber** box | Something looks likely, e.g. `fall? checking...`. It is waiting 1 second to see if it holds. Nothing is saved yet. |
| **Red** box | Something detected and confirmed. The label says what, e.g. `FALL 79%`. |
| **Red border** round the whole picture | A **fight**. A thin red square marks the part of the picture the fight model checked. |
| **`VANDALISM: CAMERA COVERED`** banner | The camera has been covered, blurred or turned (`COVERED`, `BLURRED` or `MOVED`) for 5 seconds or more |

- **Why fight looks different:** it is judged for the group, not for one person. There's no single body to point at, so the whole picture gets the border.
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
- **Camera tamper:** one picture when it starts, named like `cam-1_..._tamper-covered_100pct.jpg`. No more until the view is back to normal.
- **The picture:** it shows the moment the action peaked, not the second later when it was confirmed. It has the box, skeleton and label drawn on, so it shows what the system thought, not just what the camera saw.
- **The `.json` file:** it holds every action's score, the reasons behind it, and how the 1-second check went. That's how you tell a wrong alarm from a right one.

  ```json
  { "action": "fall", "confidence": 0.79,
    "scores": { "vandalism": 0.05, "fall": 0.79 },
    "signals": { "rules_score": 0.91, "model_score": 0.79, "swing_deg": 62.4, "hip_descent": 0.31 },
    "confirmation": { "held_for_s": 1.2, "lowest_score_while_held": 0.72 } }
  ```

### Checking recorded clips

The same detectors can be run over clips the camera already recorded:

```bash
python -m app.live.replay uploaded_clips/real-clips
```

The pictures go to `evidence/<today's date>/<clip name>/`, and `summary.json` in the date folder lists what was found in every clip, including the ones where nothing was. Each picture's `.json` also says which clip it came from and how many seconds in (`clip`, `clip_time_s`).

---

## Test videos

Test video clips are in `uploaded_clips/`. Feel free to use them.

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
├── evidence/live_view/    ← THE RESULTS, one folder per camera (never uploaded to git)
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
├── training/              the notebooks that trained the fall and fight models
├── uploaded_clips/        test videos, free to use
└── docs/                  design notes
```

Three things here are yours: **`evidence/`**, **`camera.yaml`** and **`.env`**. The rest is machinery you can leave alone.
