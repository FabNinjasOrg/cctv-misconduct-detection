"""
Reimplementation of the classifier from https://github.com/roggerfq/violence-detection-mil
(Roger Figueroa Quintero). Architecture matches upstream's `Net` + `MovinetClassifier`
exactly, so it loads the same trained weights - but the head is loaded from a plain
state_dict (`Net.load_state_dict`) instead of unpickling a whole third-party nn.Module
object (`torch.load(..., weights_only=False)`), to avoid running arbitrary code from
the checkpoint file. See NOTICE.md for the license on the weights this loads.
"""

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from movinets import MoViNet
from movinets.config import _C

CLIP_SIZE = (172, 172)
BACKBONE_FEATURES = 2048


class ViolenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(BACKBONE_FEATURES, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class MovinetViolenceClassifier:
    """Scores fixed-length clips for violence. Call with a list of clips, each a
    list of same-length BGR frames (any frame size - they're resized internally);
    returns one score in [0, 1] per clip."""

    def __init__(self, backbone_weights_dir: str, head_state_dict_path: str, device: str = "cpu"):
        self.device = device

        self.backbone = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True, model_dir=backbone_weights_dir)
        self.backbone.classifier[3] = nn.Identity()
        self.backbone.to(device)
        self.backbone.eval()

        self.head = ViolenceHead()
        self.head.load_state_dict(torch.load(head_state_dict_path, map_location=device, weights_only=True))
        self.head.to(device)
        self.head.eval()

    @torch.no_grad()
    def __call__(self, clips: list) -> np.ndarray:
        tensors = []
        for clip in clips:
            resized = [cv2.resize(frame, CLIP_SIZE, interpolation=cv2.INTER_AREA) for frame in clip]
            rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized]
            clip_tensor = torch.from_numpy(np.stack(rgb)).permute(3, 0, 1, 2)[None].float() / 255.0
            tensors.append(clip_tensor.to(self.device))

        batch = torch.cat(tensors, dim=0)

        self.backbone.clean_activation_buffers()
        features = F.normalize(self.backbone(batch), dim=1, p=2)
        scores = self.head(features).squeeze(1)
        return scores.cpu().numpy()
