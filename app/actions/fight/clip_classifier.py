"""The fight model: MoViNet-A0 fine-tuned on fights, looking at a crop of the people.

Trained by training/fight_movinet_kaggle.ipynb on RWF-2000 and Real Life Violence
Situations. It starts from MoViNet's own Kinetics-600 weights (Apache 2.0) rather
than the violence-detection-mil checkpoint (CC BY-NC 4.0), and the WHOLE network was
trained, not just a last layer - see models/movinet/NOTICE.md for the old one.

What it is fed has to match training exactly, or it is judging a picture it never
learned:

  - 8 frames sampled at 8 fps - one second of video - scaled to 640 wide first.
  - Cropped to the people: one square holding every yolo11n-pose person box seen
    across the 8 frames, grown by 1.25, at least 96 px, at most the frame's shorter
    side. Nobody seen: the whole frame. See people_crop().
  - Resized to 172 x 172 with INTER_AREA, RGB, 0..1.

The crop is the point. The old model saw the whole picture squeezed to 172 x 172,
where two people fighting across the room are a few dozen pixels - your acted
fight scored 0-3% that way.
"""
import cv2
import numpy as np
import torch
from torch import nn
from movinets import MoViNet
from movinets.config import _C

MODEL_PATH = "models/fight_movinet.pt"

# Must match the checkpoint's config - checked in __init__.
RATE = 8.0
WINDOW = 8
SIZE = 172
CROP_GROW = 1.25
CROP_MIN_SIDE = 96


class FightMoViNet(nn.Module):
    """MoViNet-A0 stream backbone + a new last layer: [normal, fight]. Must stay
    identical to the notebook's, or the saved weights won't load."""

    def __init__(self):
        super().__init__()
        # pretrained=False: the weights come from our own checkpoint, so nothing is
        # downloaded. The other settings are the ones pretrained=True forces.
        self.backbone = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=False,
                                num_classes=600, conv_type="2plus1d", tf_like=True)
        self.backbone.classifier[3] = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 2))

    def forward(self, x):
        # The stream version carries memory from one call to the next. Each window
        # is judged on its own, as in training, so start clean every time.
        self.backbone.clean_activation_buffers()
        return self.head(self.backbone(x))


def people_crop(shape, boxes, grow=CROP_GROW, min_side=CROP_MIN_SIDE):
    """The square of the picture holding everyone seen during the window, with a
    margin, as (x1, y1, x2, y2). Nobody seen: the whole picture. Identical to the
    training notebook's people_crop()."""
    h, w = shape[:2]
    if not boxes:
        return 0, 0, w, h
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(max(x2 - x1, y2 - y1) * grow, min_side), h, w)
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    s = int(round(side))
    return x0, y0, x0 + s, y0 + s


class FightClassifier:
    """Scores one-second windows for fighting. Call with the window's frames (BGR,
    already 640 wide) and, for each frame, the person boxes found in it. Returns
    (fight probability 0..1, the crop it looked at)."""

    def __init__(self, path: str = MODEL_PATH, device: str = "cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        config = saved["config"]
        for key, value in {"rate": RATE, "window": WINDOW, "size": SIZE}.items():
            if config.get(key) != value:
                raise RuntimeError(f"{path} was trained with {key}={config.get(key)}, this code feeds it {value}")
        self.fight_index = config["classes"].index("fight")
        self.device = device
        self.model = FightMoViNet()
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, frames: list, boxes: list) -> tuple:
        crop = people_crop(frames[0].shape, [box for frame_boxes in boxes for box in frame_boxes])
        x1, y1, x2, y2 = crop
        clip = np.stack([
            cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB)
            for frame in frames
        ])
        x = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].float().div(255.0).to(self.device)
        probability = torch.softmax(self.model(x), dim=1)[0, self.fight_index]
        return float(probability), crop
