from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from datetime import datetime
import os
from dotenv import load_dotenv
from ultralytics import YOLO

from app.actions.spit.single_frame_detector import detect_spitting
from app.actions.fight.yolo_detector import detect_fighting
from app.actions.fight.movinet_detector import detect_fighting_mil
from app.actions.spit.kinematic_detector import detect_activity
from app.actions.vandalism.object_action_detector import detect_vandalism
from app.actions.fall.kinematic_detector import detect_fall
from app.actions.analyzer import analyze_action

load_dotenv()

app = FastAPI()

person_model = YOLO("models/yolo11n-pose.pt")  # detects people and their body keypoints, shared across endpoints

UPLOAD_DIR = "uploaded_clips"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _require_file(filepath: str):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"file not found: {filepath}")


@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/upload-clip")
async def upload_clip(file: UploadFile = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB at a time
            f.write(chunk)

    return {
        "status": "received",
        "filename": filename,
        "saved_to": filepath,
    }

@app.post("/spit/single-frame")
async def detect_spit_single_frame(filepath: str = Form(...)):
    """Per-motion-event spit detector: YOLO11-pose posture (head bow, side tilt,
    hand-near-mouth) plus a YOLO11 object check near the head, scored one frame at a time."""
    _require_file(filepath)
    return detect_spitting(filepath, person_model)

@app.post("/spit/kinematic")
async def detect_spit_kinematic(filepath: str = Form(...)):
    """Posture/kinematics-based spit detection: motion-gates the clip, then verifies
    each candidate purely from YOLO11-pose keypoint dynamics (bow-then-snap head
    motion, hand-to-mouth proximity, ...) - no fine-tuned model, no LLM call."""
    _require_file(filepath)
    return detect_activity(filepath, person_model)

@app.post("/fight/yolo")
async def detect_fight_yolo(filepath: str = Form(...)):
    """Fine-tuned YOLOv8 fight detector - flags a "violence" class per frame."""
    _require_file(filepath)
    return detect_fighting(filepath, person_model)

@app.post("/fight/movinet")
async def detect_fight_movinet(filepath: str = Form(...)):
    """MoViNet-based fight detector (app/actions/fight/vd_mil/), local/CV-only - no Gemini call.
    Internal/research use only, see models/movinet/NOTICE.md."""
    _require_file(filepath)
    return detect_fighting_mil(filepath, person_model)

@app.post("/vandalism/object-action")
async def detect_vandalism_object_action(filepath: str = Form(...)):
    """Object + pose based vandalism detector: flags a fast arm-swing or kick (a
    striking/damaging motion) via YOLO11-pose keypoint velocity, boosted if a
    tool/weapon-like object (bat, bottle, knife, ...) is near the swinging hand."""
    _require_file(filepath)
    return detect_vandalism(filepath, person_model)

@app.post("/fall/kinematic")
async def detect_fall_kinematic(filepath: str = Form(...)):
    """Posture-based fall detector: flags a person whose trunk angle and bounding-box
    shape flip from upright to horizontal within well under a second (a sudden
    collapse, not a gradual sit/lie-down) and who then stays down - no fine-tuned
    model, no LLM call, doesn't try to guess why they fell."""
    _require_file(filepath)
    return detect_fall(filepath, person_model)

@app.post("/analyze-action")
async def analyze_action_clip(filepath: str = Form(...)):
    """Runs the primary detector for each action (spit/kinematic, fight/movinet,
    vandalism/object-action, fall/kinematic) and returns the single highest-
    confidence detection as one final frame with a box + activity label."""
    _require_file(filepath)
    return analyze_action(filepath, person_model)
