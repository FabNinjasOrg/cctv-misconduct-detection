"""Reads camera.yaml - the one file where cameras are configured.

WHAT: turns the YAML file into Camera objects the live view can connect with.
WHY:  so a deployment is set up by editing one file, not by touching code.
HOW:  parse -> fill in ${VARS} from the environment -> validate -> return.

Validation is deliberately strict and fails at startup. A typo in a camera URL
should stop the program immediately with a message that says what to fix, rather
than open a black window and leave someone guessing.
"""
import os
import re
from dataclasses import dataclass
from urllib.parse import quote

import yaml

CONFIG_PATH = "camera.yaml"

# A camera's name becomes a folder name under evidence/live_view/, so it is
# restricted to characters that are safe in a path. This blocks a name like
# "../../etc" from escaping the evidence directory, and avoids spaces that would
# need quoting everywhere downstream.
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Matches a ${VAR} that os.path.expandvars left behind because the variable was
# not set - see _resolve_url below for why that has to be an error.
UNRESOLVED_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class Camera:
    """One camera to watch. `url` is ready to open as-is."""
    name: str
    url: str


def load(path: str = CONFIG_PATH) -> list:
    """Reads and validates camera.yaml, returning a list of Camera.

    Exits with a readable message if anything is wrong, because there is no
    sensible way to carry on without a camera to watch."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No {path} found. Copy camera.example.yaml to {path}, put your "
            "camera's address in it, and try again."
        )

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cameras = []
    seen_names = set()

    for entry in raw.get("cameras") or []:
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
        raise SystemExit(f"{path}: no cameras configured - nothing to watch.")
    return cameras


def _resolve_url(url: str, name: str, path: str) -> str:
    """Fills in ${VARS} from the environment and makes the result safe to use.

    Why the ${VAR} support: it keeps camera passwords in .env, so camera.yaml can
    be committed and shared without leaking credentials."""
    url = os.path.expandvars(url)

    # expandvars silently leaves an unknown ${VAR} exactly as written. The camera
    # would then be sent the literal text "${CAM1_PASSWORD}" as a password and
    # answer "401 Unauthorized" - sending someone hunting for a wrong password
    # when the real problem is a missing .env entry. Catch it here instead.
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
