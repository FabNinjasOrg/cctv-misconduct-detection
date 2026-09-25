# CCTV Detection

This watches CCTV video and tells you when something happens — someone
**spitting**, a **fight**, **vandalism**, or a **fall**.

There are two ways to use it. You can point it at a camera and leave it running,
and it will keep watching on its own. Or you can hand it a video file and ask
what's in it. Most people end up doing both: the camera for everyday monitoring,
the file upload for checking an incident someone told you about.

When it finds something, it saves a picture of the moment with a box drawn
around the person and a label like `Fight 99%`. That picture is the point of the
whole thing — everything else here exists to produce it.

---

## Getting set up

You need two pieces of information from your camera: its **username** and its
**address on your network**. Both are in the camera's phone app, usually under
something like *Advanced Settings → Camera Account*.

A word of warning about the username, because this is where almost everyone gets
stuck: on TP-Link Tapo cameras it is a name **you** chose when you set the camera
up, and it is usually *not* `admin`. If the logs later say `401 Unauthorized`,
this is nearly always why.

**First, make your settings file.** There's a template ready to copy:

```bash
cp .env.example .env
```

**Then put your camera password in `.env`.** Open it, find the `CAM1_PASSWORD`
line, and type the password after the `=`:

```
CAM1_PASSWORD=your-camera-password-here
```

Don't put quotes around it. If your password has awkward characters like `@` or
`:` in it, leave them exactly as they are — the code handles the escaping for
you.

**Then put your camera's address in `stream.yaml`.** Line 11 is the one you
want:

```yaml
cameras:
  - name: cam-1
    url: rtsp://USERNAME:${CAM1_PASSWORD}@CAMERA-IP:554/stream1
```

Replace `USERNAME` with the camera's login name and `CAMERA-IP` with its address
on your network — something like `192.168.1.64`. Leave `${CAM1_PASSWORD}` alone;
it reaches into `.env` and fetches the password for you, which is what keeps the
real password out of this file.

---

## Starting it

```bash
docker compose up -d --build
```

That's the whole thing. It's now recording your camera and checking every clip
as it goes. The first run takes a few minutes because it has to build; after
that it's quick.

Anything it finds shows up in **`evidence/live_detections/`**.

A few other commands worth knowing:

```bash
docker compose logs -f          # see what it's doing right now
docker compose down             # stop it
docker compose restart stream   # pick up changes after editing stream.yaml
```

---

## Checking a video yourself

Open **<http://localhost:8000/docs>** in your browser. You'll get a page listing
everything the system can do, and you can run any of it right there without
touching a terminal.

It's a two-step job. First, `POST /upload-clip` → **Try it out** → choose your
video → **Execute**. It hands back a path, something like
`uploaded_clips/20260925_141500_myvideo.mp4`.

Then `POST /analyze-action` → **Try it out** → paste that path into the
`filepath` box → **Execute**. After a minute or so you get the verdict:

```json
{
  "activity_detected": true,
  "action": "vandalism",
  "confidence": 0.881,
  "final_frame": "evidence/manual_detections/vandalism-1_vandalism_88pct.jpg"
}
```

Open that `final_frame` file and you'll see the moment it caught, boxed and
labelled.

`/analyze-action` runs all four detectors and gives you the single strongest
answer, which is almost always what you want. The other endpoints on that page
run one detector on its own — handy if you're investigating why something was or
wasn't flagged, but not the everyday route.

---

## Where everything lives

Three things here are yours. The rest you can leave alone.

**`evidence/`** is where the results go, and it has two folders:

- **`live_detections/`** — what the cameras caught. One folder per camera, each
  holding the picture and the video clip it came from, plus `detections.jsonl`,
  a running list of every detection with its time and score.
- **`manual_detections/`** — what the videos you uploaded caught. One labelled
  picture per video.

**`uploaded_clips/`** holds the videos you've given it to check.

**`stream.yaml`** is your camera settings, and **`.env`** is your camera
password. These two are the only files you normally edit.

Everything else is machinery:

| | |
|---|---|
| `main.py` | The web API — the thing your browser talks to. |
| `app/` | All the code. `actions/` holds the four detectors, `stream/` does the camera recording, `shared/` is the common bits. |
| `models/` | The trained AI files the detectors use. |
| `docker/` | Build and packaging files. |
| `debug/` | Each detector's rough working-out. Appears on its own when there's something to save; safe to delete any time. |
| `spool/` | Scratch space while recording. Empties itself. |
| `docs/` | Design notes. |

---

## When something's wrong

**`401 Unauthorized` in the logs.** The username or password is wrong. Start with
the username — it's usually not `admin`, whatever you might expect.

**`Connection refused`, or it just times out.** Either the IP address is wrong,
or the camera is on a different network from this computer. A camera at another
site won't be reachable just because you know its address; the two networks have
to be genuinely connected.

**Nothing is appearing in `evidence/live_detections/`.** That's normal if nothing
has actually happened. Run `docker compose logs -f` and you should see it
recording and checking each clip. If it's working, you'll see lines going past.

**A warning about "not keeping up in real time".** This computer isn't fast
enough to check every single clip. Open `stream.yaml` and set `analyze_one_in: 3`
— it'll check a sample instead, roughly twenty seconds out of every minute,
which is usually plenty.

---

## Running it without Docker

If you'd rather run it directly:

```bash
pip install -r requirements.txt
uvicorn main:app --reload     # the API, at http://127.0.0.1:8000/docs
python -m app.stream          # the camera recorder
```

Run these from this folder — the code looks for `models/` and `evidence/`
relative to where you started it.

You can also watch a video play back with the detection drawn on top as it goes,
which is the best way to get a feel for what the detectors actually see:

```bash
python3 -m app.actions.spit.playback      uploaded_clips/your-clip.mp4
python3 -m app.actions.fight.playback     uploaded_clips/your-clip.mp4
python3 -m app.actions.vandalism.playback uploaded_clips/your-clip.mp4
python3 -m app.actions.fall.playback      uploaded_clips/your-clip.mp4
```

---

## Before you sell this to anyone

Two licensing problems worth sorting out early, because both are easier to fix
now than after a customer is using it:

- The fight detector's trained weights in `models/movinet/` are **CC BY-NC 4.0**,
  which means **non-commercial use only**. They're baked into the Docker image,
  so as things stand the image can't be sold. `models/movinet/NOTICE.md` has the
  details.
- **Ultralytics YOLO is AGPL-3.0**, which comes with real obligations if you
  build a product on it. Worth a proper look.

One operational thing too: nothing cleans up old evidence. Running a camera
around the clock, those folders will grow until the disk fills.

---

For deployment details and the tuning knobs, see `docker/DOCKER.md`. For notes
on how the detection pipeline is designed, see `docs/general.md`.
