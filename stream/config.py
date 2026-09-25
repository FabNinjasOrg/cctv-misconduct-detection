"""Reads stream.yaml - the one file where cameras and capture behaviour are set.

WHAT: turns the YAML file into a Config object the rest of the package uses.
WHY:  so a deployment is configured by editing one file, not by touching code.
HOW:  parse -> fill in ${VARS} from the environment -> validate -> return.

Validation is deliberately strict and fails at startup. A typo in a camera URL
should stop the program immediately with a clear message, not silently record
nothing for twenty minutes before anyone notices.
"""
import os
import re
from dataclasses import dataclass
from urllib.parse import quote

import yaml

CONFIG_PATH = "stream.yaml"

# A camera's name becomes a folder name (stream/spool/<name>/...), so it is
# restricted to characters that are safe in a path. This blocks a name like
# "../../etc" from escaping the spool directory, and avoids spaces that would
# need quoting everywhere downstream.
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Matches a ${VAR} that os.path.expandvars left behind because the variable was
# not set - see _resolve_url below for why that has to be an error.
UNRESOLVED_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class Camera:
    """One camera to record. `url` is ready to hand to ffmpeg as-is."""
    name: str
    url: str


@dataclass
class Config:
    """Everything stream.yaml controls, already validated."""
    cameras: list

    # capture: how the recorder chops the stream into files
    segment_seconds: int
    spool_dir: str

    # analysis: how those files get sent to the detection API
    api_url: str
    analyze_one_in: int
    max_backlog_segments: int
    min_confidence: float
    timeout_seconds: int

    # output: where detections end up
    evidence_dir: str
    detections_log: str
    keep_segment_on_detection: bool


def load(path: str = CONFIG_PATH) -> Config:
    """Reads and validates stream.yaml. Exits with a readable message if anything
    is wrong, because there is no sensible way to carry on with a bad config."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No {path} found. Copy the example, set your camera URL in it, and try again."
        )

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Each top-level section is optional in the file; missing ones fall back to
    # the defaults spelled out below.
    capture = raw.get("capture") or {}
    analysis = raw.get("analysis") or {}
    output = raw.get("output") or {}

    config = Config(
        cameras=_read_cameras(raw.get("cameras") or [], path),

        segment_seconds=int(capture.get("segment_seconds", 10)),
        spool_dir=str(capture.get("spool_dir", "stream/spool")),

        api_url=os.path.expandvars(
            str(analysis.get("api_url", "http://detector:8000/analyze-action"))
        ),
        analyze_one_in=int(analysis.get("analyze_one_in", 1)),
        max_backlog_segments=int(analysis.get("max_backlog_segments", 10)),
        min_confidence=float(analysis.get("min_confidence", 0.5)),
        timeout_seconds=int(analysis.get("timeout_seconds", 600)),

        evidence_dir=str(output.get("evidence_dir", "stream/evidence")),
        detections_log=str(output.get("detections_log", "stream/evidence/detections.jsonl")),
        keep_segment_on_detection=bool(output.get("keep_segment_on_detection", True)),
    )

    _check_values(config, path)
    return config


def _read_cameras(entries: list, path: str) -> list:
    """Turns the `cameras:` list from the YAML into validated Camera objects."""
    cameras = []
    seen_names = set()

    for entry in entries:
        name = str(entry.get("name", "")).strip()
        url = str(entry.get("url", "")).strip()

        if not name or not url:
            raise SystemExit(f"{path}: every camera needs both a name and a url.")
        if not SAFE_NAME.match(name):
            raise SystemExit(
                f"{path}: camera name {name!r} may only contain letters, digits, dot, "
                "dash and underscore - it is used as a folder name."
            )
        if name in seen_names:
            raise SystemExit(f"{path}: camera name {name!r} is used more than once.")

        seen_names.add(name)
        cameras.append(Camera(name=name, url=_resolve_url(url, name, path)))

    if not cameras:
        raise SystemExit(f"{path}: no cameras configured - nothing to record.")
    return cameras


def _resolve_url(url: str, name: str, path: str) -> str:
    """Fills in ${VARS} from the environment and makes the result safe to use.

    Why the ${VAR} support: it keeps camera passwords in .env, so stream.yaml can
    be committed and shared without leaking credentials."""
    url = os.path.expandvars(url)

    # expandvars silently leaves an unknown ${VAR} exactly as written. ffmpeg
    # would then try to log in with the literal text "${CAM1_PASSWORD}" and the
    # camera would answer "401 Unauthorized" - sending someone hunting for a
    # wrong password when the real problem is a missing .env entry. Catch it here.
    missing = UNRESOLVED_VAR.findall(url)
    if missing:
        raise SystemExit(
            f"{path}: camera {name!r} uses {', '.join(missing)}, which is not set. "
            "Add it to .env (or remove it from the URL if the camera needs no password)."
        )

    return _encode_credentials(url)


def _encode_credentials(url: str) -> str:
    """Percent-encodes the username and password inside a stream URL.

    WHY: camera passwords often contain '@', and '@' is the character a URL uses
    to separate the credentials from the host:

        rtsp://admin:pa@ss@192.168.1.5:554/stream1
                        ^ a parser splits here, at the FIRST @

    The host is then read as garbage and the camera replies 401 - a "wrong
    password" error for a password that is completely correct. Encoding the '@'
    as %40 removes the ambiguity.

    HOW: split on the LAST '@' to find the real host boundary, and on the FIRST
    ':' of the credentials to find the real user/password boundary. That way each
    part can safely contain the other's delimiter. Passwords are written raw in
    .env - the encoding happens here so nobody has to remember to do it by hand.
    """
    if "://" not in url:
        return url

    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url          # no credentials in this URL, nothing to encode

    credentials, host = rest.rsplit("@", 1)
    if ":" in credentials:
        user, password = credentials.split(":", 1)
        credentials = f"{quote(user, safe='')}:{quote(password, safe='')}"
    else:
        credentials = quote(credentials, safe="")   # username only, no password

    return f"{scheme}://{credentials}@{host}"


def _check_values(config: Config, path: str) -> None:
    """Rejects numbers that would make the rest of the package behave strangely
    rather than fail outright - easier to debug as a startup error."""
    if config.segment_seconds < 2:
        raise SystemExit(f"{path}: segment_seconds must be at least 2.")
    if config.analyze_one_in < 1:
        raise SystemExit(f"{path}: analyze_one_in must be 1 or more (1 = analyze every segment).")
    if config.max_backlog_segments < 1:
        raise SystemExit(f"{path}: max_backlog_segments must be 1 or more.")
