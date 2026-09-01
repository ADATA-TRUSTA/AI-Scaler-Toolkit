"""Launcher state, kept in a small JSON file.

Only settings and facts the launcher owns live here. Anything the backend or the WebUI
already stores stays where it is -- two homes for one setting is how they disagree.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import config

_LOCK = threading.Lock()

# Bumped whenever the meaning of a key changes. A file written by an older, different
# launcher must not be partially adopted: an earlier prototype stored `provisioned:
# true` alongside keys that no longer exist, and honouring that flag would skip setup
# on a machine that has none of what this version installs.
SCHEMA = 1

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    # Set once first-run provisioning finishes. The acceptance run in CI polls this,
    # so it must be written only after setup_env actually succeeded.
    "provisioned": False,
    "service_port": config.DEFAULT_SERVICE_PORT,
    # On by default: the point of the tray app is that the service is simply there.
    "autostart": True,
    # Which accelerator setup_env settled on, for display only.
    "accelerator": None,
    "installed_version": None,
    "last_update_check": None,
    # How many packages a complete install has on this machine. Used as the
    # denominator for the setup progress bar; until a run records one, a shipped
    # estimate stands in.
    "package_count": None,
}


def load() -> dict[str, Any]:
    path = config.state_file()
    if not path.is_file():
        return dict(DEFAULTS)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt state file must not stop the app from starting; defaults are all
        # recoverable and provisioning detects what is already installed.
        return dict(DEFAULTS)
    if stored.get("schema") != SCHEMA:
        # Not an error: an older layout is simply unusable, and a fresh first run is
        # the correct response to one.
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return merged


def save(values: dict[str, Any]) -> None:
    path = config.state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        current = load()
        current.update(values)
        current["schema"] = SCHEMA
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


def get(key: str) -> Any:
    return load().get(key, DEFAULTS.get(key))


def set(key: str, value: Any) -> None:  # noqa: A001 - reads naturally at the call site
    save({key: value})


def service_port() -> int:
    try:
        return int(get("service_port"))
    except (TypeError, ValueError):
        return config.DEFAULT_SERVICE_PORT
