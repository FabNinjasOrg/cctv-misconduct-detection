Tools we are using:

1. opencv-python -> reads/writes/crops/draws on video frames

2. ultralytics with YOLO11-pose -> finds people + gives skeleton keypoints, no understanding of "actions"

3. openai api -> llm call for the judgement

Framework we will use:

1. Fast API -> we'll create the rest api that process the clip

2. Optional web view with react or view simply

=================

24th Sept EOD:

HARSH-73: CCTV Footage Misconducting Detection
- Converting this whole setup into installable package into any local PC
    - Created the docker setup
    - Setup the CCTV stream pipeline
    - Implemented the ffmpeg for stream segment and chunking
    - Validation through our action detection scripts
- Review the ffmpeg media framework and it's usage in our application for convert the RTSP stream into video chunks

- Pending: a live CCTV connection to test against what we have develop

==========

Fight Detection:

1. fight_yolov8n
2. violence-detection-mil (Multiple Instance Learning) | model: MoViNet | dataset: RWF-2000 | SCVD
3. gemini judgement call

Spit detection:

1. yolo11n-pose (person detection) -> YOLO11 (object detection) -> object near head judgement
2. gemini judgement call

- gemini usage is not practical to use which takes footage as input and we cant use it for 24x7 footage coverage

==========

* Testing

spit : python -m spit.playback uploaded_clips/spit-1.mp4
fight : python -m fight.playback uploaded_clips/fight-7.mp4
vandalism : python -m vandalism.playback uploaded_clips/vandalism-1.mp4
fall : python -m fall.playback uploaded_clips/fall-1.mp4

Challenges:

    1. If we use AI models VLMs for the action detection then it is the expensive and slow process.
    as VLM has to process the 4 frames per 1 second of the CCTV feed
    2. That's why we have one cheap filter that detects the action for us and that can be the posture based action detection or the CV model based detection
    3. For more accuracy we can fine tune the CV model but the issue in that is the raw footages are not there to train on the specific actions

==========

If the scene is busy (people moving)

Processed:  20 seconds
Skipped:    40 seconds
Checks 20s, skips 40s — because a busy 10-second chunk takes ~28 seconds to analyze.

If the scene is quiet (nothing happening)

Processed:  60 seconds
Skipped:     0 seconds
Checks all 60s, skips nothing — an empty chunk is rejected by the motion check in ~1.5 seconds, so it keeps up easily.

==========

What next today ?

-> testing with camera
-> make it end-to-end package any one can use
-> document for use
-> What device can we use for setting it up on site

** call pankaj after lunch