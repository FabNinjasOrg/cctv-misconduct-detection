# Third-party model: VD-MIL (MoViNet-based violence detection)

Source: https://github.com/roggerfq/violence-detection-mil
Author: Roger Figueroa Quintero

## What lives in this folder

- `movinet_weights/modelA0_stream_statedict_v3.zip` — MoViNet-A0 backbone weights,
  pulled in via the MIT-licensed fork https://github.com/roggerfq/MoViNet-pytorch.
  Commercial use is fine.
- `model_20_state_dict.pt` — the fight/violence classifier head trained by Figueroa
  Quintero on RWF-2000 + SCVD. **Licensed CC BY-NC 4.0 (non-commercial only):**
  https://creativecommons.org/licenses/by-nc/4.0/legalcode
- `movinet_classifier.py` — architecture reimplementation (see file header) needed
  to load the above head; not a byte-for-byte copy of the upstream file.

## Restriction

`model_20_state_dict.pt` and any detection results it produces are for internal
testing / research use only, per upstream's license. Do not ship this specific
checkpoint as part of a commercial product without written permission from the
author (contact via https://www.linkedin.com/in/roger-figueroa-quintero/).

`fight_detection_mil.py` (the FastAPI-facing wrapper, project root) is this
project's own code and carries no such restriction — only the checkpoint file
in this folder does.
