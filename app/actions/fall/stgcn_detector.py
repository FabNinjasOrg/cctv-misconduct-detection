# Detects a person falling with a trained skeleton model (ST-GCN), not hand-written
# rules. It reads the last 3 seconds of one person's YOLO11-pose skeleton and says how
# likely it is that a fall happened inside them.
#
# Trained by training/fall_stgcn_kaggle.ipynb on the Le2i fall dataset. That notebook
# turned every video into skeletons exactly the way the live app sees a camera -
# yolo11n-pose, frames scaled to 640 wide, 15 per second - and this file must keep
# doing exactly the same, or the model is asked about a picture it never learned:
#
#   - 45 samples, 15 a second: 3 seconds. Built on a time grid, so a frame the
#     tracker missed is an empty sample (all zeros), the way a frame with nobody in
#     it was during training - not silently skipped, which would squeeze 3.5 seconds
#     into 3.
#   - normalise(): centred on the hips, scaled by the body, one centre per window so
#     the hips dropping is kept. Copied from the notebook, unchanged.
#   - A window where the person is missing for most of it is not scored. Training
#     skipped those too, so the model has no idea what they mean.
#
# What the model learned to call a fall: a window with the WHOLE fall inside it -
# standing, going down, down. Lying still afterwards is "not a fall" in Le2i's labels,
# so the score is high for the couple of seconds after a fall and then drops back on
# its own. That is the right shape for wait-and-confirm in the live engine (the
# score has to hold for CONFIRM_SECONDS), and it is why someone who has been lying
# down for a minute isn't re-reported every scoring pass.
#
# Known limits, stated plainly:
#   - Trained on 127 Le2i videos only - a handful of actors in a few rooms. Tested on
#     videos it never saw (15 of 15 falls caught, no false alarm in 4.4 minutes), but
#     those are the same actors and rooms. It has not learned your camera yet - that
#     comes from fine-tuning on your own footage.
#   - Le2i's camera is high in a corner. A camera at head height, close to people,
#     sees a different shape of fall.

import numpy as np
import torch
from torch import nn

MODEL_PATH = "models/fall_stgcn.pt"

# These have to match what the model was trained on - see the checkpoint's config,
# checked in _load() below.
RATE = 15.0          # samples per second
WINDOW = 45          # samples per window: 3 seconds
CONF_MIN = 0.3       # a keypoint below this confidence counts as not seen
MIN_JOINTS = 5       # a sample needs this many seen joints to count as "the person is there"
MIN_VISIBLE = 0.5    # ...and at least this fraction of the window's samples must have them

# A record counts for a grid slot if it is within this of the slot's time. One
# sample period: the live loop paces to 15 fps but never exactly, so asking for
# half a period would turn ordinary timing jitter into empty samples.
SLOT_TOLERANCE = 1.0 / RATE

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]   # COCO-17, as YOLO gives them


def _skeleton_graph():
    a = np.eye(17, dtype=np.float32)
    for i, j in EDGES:
        a[i, j] = a[j, i] = 1
    d = a.sum(axis=1)
    return a / np.sqrt(np.outer(d, d))


class Block(nn.Module):
    """One ST-GCN layer: mix each joint with its neighbours, then look across time.
    Must stay identical to the notebook's, or the saved weights won't load."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.graph = nn.Parameter(torch.tensor(_skeleton_graph()))
        self.spatial = nn.Conv2d(c_in, c_out, 1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(0.2))
        self.shortcut = (nn.Identity() if c_in == c_out and stride == 1 else
                         nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                                  # batch, channels, time, joints
        y = torch.einsum("nctv,vw->nctw", self.spatial(x), self.graph)
        return self.relu(self.temporal(y) + self.shortcut(x))


class FallSTGCN(nn.Module):
    """Input: batch x 45 samples x 17 joints x (x, y, confidence), from normalise().
    Output: two scores, [not fall, fall]."""

    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(17 * 3)
        self.blocks = nn.Sequential(
            Block(3, 64), Block(64, 64), Block(64, 128, stride=2),
            Block(128, 128), Block(128, 256, stride=2), Block(256, 256))
        self.head = nn.Linear(256, 2)

    def forward(self, x):                                  # batch, time, joints, 3
        n, t, v, c = x.shape
        x = self.input_norm(x.permute(0, 2, 3, 1).reshape(n, v * c, t))
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1)      # batch, 3, time, joints
        return self.head(self.blocks(x).mean(dim=(2, 3)))


def _load():
    saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    config = saved["config"]
    expected = {"rate": RATE, "window": WINDOW, "conf_min": CONF_MIN}
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"{MODEL_PATH} was trained with {key}={config.get(key)}, this code feeds it {value}")
    model = FallSTGCN()
    model.load_state_dict(saved["state_dict"])
    return model.eval(), config


model, model_config = _load()


def normalise(window):
    """Pixels -> centred on the hips and scaled to body size, so where someone
    stands and how far they are from the camera don't matter. One centre for the
    whole window, not one per frame, so the hips dropping in a fall is kept.
    Identical to the training notebook's normalise()."""
    xy = window[..., :2].copy()
    conf = window[..., 2].copy()
    seen = conf >= CONF_MIN
    hips = seen[:, 11] & seen[:, 12]
    if hips.any():
        centre = np.median((xy[hips, 11] + xy[hips, 12]) / 2, axis=0)
    elif seen.any():
        centre = xy[seen].mean(axis=0)
    else:
        return None
    size = 0.0
    for t in range(len(window)):
        points = xy[t][seen[t]]
        if len(points) >= 4:
            size = max(size, float((points.max(axis=0) - points.min(axis=0)).max()))
    if size < 1:
        return None
    xy = (xy - centre) / size
    xy[~seen] = 0
    conf[~seen] = 0
    return np.concatenate([xy, conf[..., None]], axis=-1).astype(np.float32)


def skeleton_window(records: list, end_t: float):
    """The 3 seconds of one person's history up to end_t, as 45 x 17 x 3.

    Laid out on a 15-a-second time grid: each slot takes the record closest to its
    time, or stays zeros if the tracker had nobody there - see the note at the top."""
    window = np.zeros((WINDOW, 17, 3), np.float32)
    if not records:
        return window
    times = np.array([r["t"] for r in records])
    for k in range(WINDOW):
        slot_t = end_t - (WINDOW - 1 - k) / RATE
        nearest = int(np.abs(times - slot_t).argmin())
        if abs(times[nearest] - slot_t) <= SLOT_TOLERANCE:
            record = records[nearest]
            window[k, :, :2] = record["kpts_xy"]
            window[k, :, 2] = record["kpts_conf"]
    return window


def score_people(windows: dict) -> dict:
    """Fall probability for each person, from their own history.

    windows is {track_id: records}. Returns {track_id: (confidence, signals)}. Everyone
    is scored in one batch - the model costs about the same for one person as for ten.

    Each person's 3 seconds end at their own last sighting, not at "now": scoring runs
    on its own clock, a little behind the frames, and ending at now would fill the
    newest slots with nothing - which reads as the person vanishing."""
    ready, samples, results = [], [], {}
    for track_id, records in windows.items():
        if not records:
            continue
        window = skeleton_window(records, records[-1]["t"])
        visible = float(((window[..., 2] >= CONF_MIN).sum(axis=1) >= MIN_JOINTS).mean())
        sample = normalise(window) if visible >= MIN_VISIBLE else None
        if sample is None:
            results[track_id] = (0.0, {"model": "stgcn", "visible": round(visible, 2),
                                       "scored": False})
            continue
        ready.append((track_id, visible))
        samples.append(sample)

    if samples:
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(np.stack(samples))), dim=1)[:, 1].numpy()
        for (track_id, visible), prob in zip(ready, probs):
            results[track_id] = (float(prob), {"model": "stgcn", "visible": round(visible, 2),
                                               "scored": True, "fall_probability": round(float(prob), 3)})
    return results
