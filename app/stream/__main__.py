"""Entry point for the stream service: `python -m app.stream`.

WHAT: starts one recorder per camera plus one analysis worker, then waits.
WHY:  this is the process the `stream` container runs; everything else in the
      package is a piece it wires together.
HOW:  recorders and the worker each get a thread, the main thread does nothing
      but wait for Ctrl-C or a shutdown signal from Docker.

The shape of it:

    Recorder (1 per camera)  ->  spool/  ->  Worker (1 total)
    keeps ffmpeg running         finished .mp4      analyzes and saves evidence

Recorders and the worker are deliberately independent. Recording must never
pause because analysis is slow, and analysis must never stop the camera being
recorded - so they share nothing except the spool folder on disk.
"""
import logging
import signal
import threading

from app.stream import config as config_module
from app.stream.recorder import Recorder
from app.stream.worker import Worker


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("stream")

    config = config_module.load()
    log.info(
        "%d camera(s), %ds segments, analyzing 1 in %d",
        len(config.cameras), config.segment_seconds, config.analyze_one_in,
    )

    # Set when it is time to shut down. Every thread watches this, which is how
    # they all stop together.
    stop = threading.Event()

    def shutdown(signum, _frame):
        log.info("signal %s received, shutting down", signal.Signals(signum).name)
        stop.set()

    # SIGTERM is what `docker compose down` sends; SIGINT is Ctrl-C.
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    recorders = [
        Recorder(camera, config.spool_dir, config.segment_seconds)
        for camera in config.cameras
    ]
    for recorder in recorders:
        recorder.start()

    worker = Worker(config)
    worker_thread = threading.Thread(target=worker.run, args=(stop,), name="worker")
    worker_thread.start()

    try:
        # The worker runs in its own thread so this one stays free to receive
        # signals. Python only delivers them to the main thread, and a thread
        # sitting inside a ten-minute HTTP call cannot react to Ctrl-C.
        while not stop.wait(1.0):
            pass
    finally:
        # Recorders first: stop producing new work before waiting for the worker
        # to finish what it has.
        for recorder in recorders:
            recorder.stop()

        # The worker may be part-way through an analysis that cannot be
        # interrupted, so allow it the full API timeout to finish and tidy up.
        worker_thread.join(timeout=config.timeout_seconds + 10)
        log.info("stopped")


if __name__ == "__main__":
    main()
