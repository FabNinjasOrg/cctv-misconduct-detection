"""Replays recorded clips through the live engine and saves what it finds.

    python -m app.live.replay uploaded_clips/real-clips

WHAT: feeds each clip, frame by frame, through the same LiveEngine, renderer and
      Snapshotter the live view uses, and writes the snapshots to
      evidence/<today>/<clip name>/ - one folder per run date, one per clip.
WHY:  the live view only ever sees what happens while it is running. A clip the
      camera already recorded has to be judged by the same code, or a result
      here says nothing about what the live view would have done.
HOW:  the engine is not started, so it scores in step with each frame instead
      of on background threads - and it is told each frame's time in the clip
      rather than reading the wall clock. The detectors reason in seconds, so
      the clip is judged at its own speed however fast this machine replays it,
      and the same clip always gives the same answer.

Frames are kept at up to TARGET_FPS (15) a second, as the live loop paces to.
This camera records at about 14.3, so every frame of its clips is used.

Each clip's last frame is held for TAIL_SECONDS after it ends. Live footage does
not stop the moment something happens, and without the tail an event in the
clip's last second never gets the CONFIRM_SECONDS it needs to be called.

Each snapshot's .json gets two extra keys: `clip` and `clip_time_s`, the second
into the clip of the picture. A summary of every clip, including the ones where
nothing was found, goes in summary.json beside the clip folders.
"""
import argparse
import json
import logging
import os
import re
import time
from datetime import date

import cv2

from app.live import render
from app.live.__main__ import DEFAULT_THREADS, DEFAULT_WIDTH, TARGET_FPS, _resize
from app.live.engine import LiveEngine
from app.live.evidence import Snapshotter

log = logging.getLogger("replay")

TAIL_SECONDS = 4.0

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def _clip_start(path: str):
    """When the clip begins, as a Unix time, if its name says so - this camera
    names its exports by the millisecond they start (1790683829129_0.mp4).
    Snapshots are then named by when it really happened. Otherwise None, and
    they are named by when the replay ran."""
    match = re.match(r"(\d{13})(?!\d)", os.path.basename(path))
    return int(match.group(1)) / 1000 if match else None


def _clips(paths: list) -> list:
    """Every video named, with folders expanded to the videos inside them."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            found += sorted(
                os.path.join(path, name) for name in os.listdir(path)
                if name.lower().endswith(VIDEO_EXTENSIONS)
            )
        else:
            found.append(path)
    return found


def _label_with_clip(path: str, clip: str, events: list, now: float) -> dict:
    """Adds where in the clip a snapshot came from to its .json, and returns the
    line for the summary.

    A person's picture is their best moment, from before it was confirmed, so
    its time is the event's own; fight and tamper are the frame just seen."""
    sidecar = os.path.splitext(path)[0] + ".json"
    clip_time = now
    try:
        with open(sidecar) as handle:
            record = json.load(handle)
        clip_time = next((event["t"] for event in events
                          if event["track_id"] == record.get("track_id")
                          and event["action"] == record.get("action")), now)
        record["clip"] = clip
        record["clip_time_s"] = round(clip_time, 2)
        with open(sidecar, "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
    except Exception as e:
        log.warning("could not label %s with its clip: %s", sidecar, e)
        record = {}

    return {
        "clip_time_s": round(clip_time, 2),
        "action": record.get("action"),
        "confidence": record.get("confidence"),
        "track_id": record.get("track_id"),
        "image": os.path.relpath(path),
    }


def replay(path: str, person_model, out_dir: str, width: int, rate: float,
           tail_seconds: float) -> dict:
    """Runs one clip through a fresh engine. Returns its summary."""
    clip = os.path.basename(path)
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        log.warning("could not open %s - skipped", path)
        return {"clip": clip, "error": "could not open"}
    fps = video.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    start = _clip_start(path)

    # A fresh engine and snapshotter per clip: tracks, cooldowns and the tamper
    # check's learned view all belong to one camera position, one recording.
    engine = LiveEngine(person_model)       # not started - see HOW above
    snapshotter = Snapshotter(os.path.splitext(clip)[0], out_dir)
    found = []

    def step(frame, now: float) -> None:
        people = engine.process(frame, now)
        fight = engine.fight()
        fight_detail = engine.fight_detail()
        tamper = engine.tamper()
        annotated = render.draw(frame, people, fight=fight,
                                fight_crop=fight_detail.get("crop"), tamper=tamper)

        events = engine.take_events()
        for event in events:
            event["seen_at"] = start + event["t"] if start is not None else None

        saved = snapshotter.update(annotated, people, now, fight=fight, events=events,
                                   fight_detail=fight_detail, tamper=tamper,
                                   when=start + now if start is not None else None)
        found.extend(_label_with_clip(saved_path, clip, events, now) for saved_path in saved)

    began = time.monotonic()
    index, used, frame, next_due = 0, 0, None, 0.0
    while True:
        ok, raw = video.read()
        if not ok:
            break
        now = index / fps
        index += 1
        # Pace to `rate` on the clip's clock, as the live loop does on the wall's.
        if now + 1e-6 < next_due:
            continue
        next_due = max(next_due + 1.0 / rate, now)
        frame = _resize(raw, width)
        step(frame, now)
        used += 1
    video.release()

    duration = index / fps
    if frame is not None:
        step_seconds = 1.0 / min(fps, rate)
        for k in range(1, round(tail_seconds / step_seconds) + 1):
            step(frame, duration + k * step_seconds)

    log.info("%s: %.1fs at %.1f fps, %d frames used, %d saved, took %.0fs",
             clip, duration, fps, used, len(found), time.monotonic() - began)
    return {"clip": clip, "seconds": round(duration, 1), "fps": round(fps, 2),
            "frames_used": used, "folder": os.path.relpath(snapshotter.directory),
            "found": found}


def main():
    parser = argparse.ArgumentParser(
        prog="python -m app.live.replay",
        description="Run recorded clips through the live detectors and save what they find.",
    )
    parser.add_argument("clips", nargs="+", help="video files, or folders of them")
    parser.add_argument("--out", default=os.path.join("evidence", date.today().isoformat()),
                        help="where snapshots go (default: evidence/<today>)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"working frame width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--fps", type=float, default=TARGET_FPS,
                        help=f"most frames per second to use (default: {TARGET_FPS})")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"CPU threads per inference (default: {DEFAULT_THREADS})")
    parser.add_argument("--tail", type=float, default=TAIL_SECONDS,
                        help=f"seconds to hold each clip's last frame (default: {TAIL_SECONDS})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    clips = _clips(args.clips)
    if not clips:
        raise SystemExit("no clips found")

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))
    person_model = YOLO("models/yolo11n-pose.pt")

    log.info("replaying %d clip(s) into %s/", len(clips), args.out)
    results = [replay(path, person_model, args.out, args.width, args.fps, args.tail)
               for path in clips]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump({"clips": results}, handle, indent=2)

    for result in results:
        found = result.get("found", [])
        print(f"\n{result['clip']}  -  {len(found)} saved")
        for item in found:
            print(f"   {item['clip_time_s']:7.2f}s  {item['action']:<16} "
                  f"{(item['confidence'] or 0) * 100:3.0f}%  {item['image']}")


if __name__ == "__main__":
    main()
