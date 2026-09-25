# syntax=docker/dockerfile:1
#
# Packages the detection service so it runs the same way on any machine.
#
# The application is copied to /app, which reproduces this project's own layout:
# every model and evidence path in the code is relative to the project root (see
# README.md), so WORKDIR /app makes them all resolve unchanged. That is why no
# detector needed editing to be containerised.

FROM python:3.13-slim

# ffmpeg        - video decode, and the segment recorder the stream worker will use next.
# libgl1        - opencv-python links against libGL at import time; without it, `import cv2` fails.
# libglib2.0-0  - same, for libgthread.
# git           - requirements install the MoViNet backbone straight from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/opt/ultralytics

# Ultralytics writes a settings.json on first import and does not create its
# parent, so without this it warns and silently falls back to a different path.
RUN mkdir -p /opt/ultralytics

WORKDIR /app

COPY docker/requirements.txt /tmp/requirements.txt

# torch first, and only from PyTorch's CPU index. The default PyPI wheel bundles
# the CUDA runtime and adds roughly 2.5GB to the image for GPU support a client's
# office PC will not have. Doing it in its own layer also keeps the slowest part
# of the build cached when application code changes.
#
# For an NVIDIA deployment, change the index URL to the matching cu___ channel.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code last, so editing a detector rebuilds only this layer and not
# the multi-gigabyte dependency install above.
COPY . /app

EXPOSE 8000

# Checks that uvicorn is listening, rather than round-tripping a request.
#
# The detection routes are `async def` but call synchronous, CPU-bound code, so
# they block the event loop for as long as an analysis takes - a minute or more
# on modest hardware. An HTTP healthcheck therefore fails during every single
# analysis and reports a perfectly working service as unhealthy. A TCP connect
# still succeeds, because the kernel accepts it into the listen backlog.
#
# The trade-off is a weaker check: this catches a crashed or unbound process,
# not a hung one. Worth revisiting if the routes are ever made non-blocking
# (changing them from `async def` to `def` hands them to FastAPI's threadpool).
#
# Generous start period: importing the app loads YOLO11-pose, the fight YOLOv8
# and the MoViNet classifier before uvicorn serves anything.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import socket,sys; socket.create_connection(('localhost',8000),timeout=8).close()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
