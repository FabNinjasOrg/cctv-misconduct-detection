FROM python:3.13-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0t64 git tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ultralytics pinned to the version the detectors were tuned on.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ultralytics==8.4.136

COPY . .

EXPOSE 8088
CMD ["python", "-m", "app.live"]
