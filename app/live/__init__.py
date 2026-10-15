"""Live view: watch a camera and see what it thinks is happening, right now.

    python -m app.live            then open http://localhost:8088

Three things are watched for - a **fall**, a **fight** and **vandalism**
(spitting is turned off for now). Each person in shot gets a box, a skeleton and
a label; when something crosses the threshold the box turns red and one
annotated still is saved to evidence/live_view/<camera>/. No video is ever recorded, and `--no-save`
turns off even the stills.

How it fits together:

    capture.py   holds the RTSP connection, always hands back the newest frame
    engine.py    pose -> tracking -> scoring, plus the fight classifier
    render.py    boxes, skeletons, labels, the fight banner
    evidence.py  one snapshot per event, with a cooldown so it stays readable
    server.py    serves the annotated feed to a browser as MJPEG
    config.py    reads camera.yaml
    __main__.py  wires the above together and paces the loop

Two things about it are load-bearing and easy to break by accident:

**The detectors are reused, not reimplemented.** Vandalism and fall (and spit,
when it is on) each split cleanly into compute_kinematics (pure maths over one
person's history) and score_event (features -> confidence). Both take a list of per-frame records,
which is exactly what a rolling window is, so engine.py builds the window and
calls their real functions. A number on screen is therefore the number those
detectors give, not a separate live approximation that drifts as one side gets
tuned and the other does not.

**Fight is different in kind.** The others score a PERSON from pose over
time; MoViNet scores the PEOPLE TOGETHER - one square crop around everyone in
shot - over one second of video and returns a single number for the scene, so it
gets a frame buffer, a thread and a banner of its own rather than a per-person
label. Fall adds a trained model too (ST-GCN), alongside its rules. Both models
are our own (models/fight_movinet.pt, models/fall_stgcn.pt, trained by the
notebooks in training/), but on research-only datasets - retrain on your own
footage before anything ships.
"""
