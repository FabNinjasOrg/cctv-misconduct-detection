import base64
import os
import tempfile

import cv2
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"
REVIEW_CLIP_SECONDS = 6.0  # footage around the candidate frame - longer window gives more motion context to judge from
REVIEW_TEMPERATURE = 0.0   # make the yes/no verdict deterministic instead of varying call to call

gemini_client = genai.Client(api_key=os.environ["GEMINI_KEY"])


def review_clip(filepath: str, center_frame: int, fps: float, question: str) -> tuple:
    """Sends a short clip centered on center_frame to Gemini and asks it `question`.
    Returns (confirmed, reason) - confirmed is True only if the answer starts with 'yes'."""
    clip_bytes = _extract_clip(filepath, center_frame, fps)
    if clip_bytes is None:
        return False, "could not extract review clip"

    try:
        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=[
                {"type": "text", "text": question},
                {"type": "video", "data": base64.b64encode(clip_bytes).decode("utf-8"), "mime_type": "video/mp4"},
            ],
            generation_config={"temperature": REVIEW_TEMPERATURE},
        )
        answer = (interaction.output_text or "").strip()
    except Exception as exc:
        answer = f"prediction failed: {exc}"

    return answer.lower().startswith("yes"), answer


def _extract_clip(filepath: str, center_frame: int, fps: float):
    """Writes REVIEW_CLIP_SECONDS of footage centered on center_frame to a temp mp4, returns its bytes."""
    half_window = round(REVIEW_CLIP_SECONDS / 2 * fps)
    start_frame = max(0, center_frame - half_window)

    video = cv2.VideoCapture(filepath)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = video.read()
    if not ok:
        video.release()
        return None
    height, width = frame.shape[:2]

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    writer.write(frame)

    target_frames = half_window * 2
    frames_written = 1
    while frames_written < target_frames:
        ok, frame = video.read()
        if not ok:
            break
        writer.write(frame)
        frames_written += 1

    writer.release()
    video.release()

    with open(temp_path, "rb") as f:
        clip_bytes = f.read()
    os.remove(temp_path)
    return clip_bytes
