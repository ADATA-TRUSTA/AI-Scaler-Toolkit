"""Launcher state, kept in a small JSON file.

Only settings and facts the launcher owns live here. Anything the backend or the WebUI
already stores stays where it is -- two homes for one setting is how they disagree.
"""

from __future__ import annotations

import json
import threading
import time
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
    "last_update_check": None,
    # True from the moment setup starts until it finishes. Finding it still set on a
    # later start means the previous attempt was interrupted -- a quit, a reboot, a
    # crash -- so the next one can say it is carrying on rather than looking like a
    # fresh install of something the user thought was already installed.
    "setup_started": False,
    # How many packages a complete install has on this machine. Used as the
    # denominator for the setup progress bar; until a run records one, a shipped
    # estimate stands in.
    "package_count": None,
    # How long setup took and on which accelerator. The completion screen no longer
    # waits to be dismissed, so this is where that information stays available.
    "last_setup": None,
    # [width, height] the user last dragged the window to, kept per shape because the
    # window has two: the settings column on its own, and that column with the chat
    # panel beside it. One remembered size for both is what made showing the chat
    # panel appear to do nothing -- it applied to whichever shape you switched to, so
    # the window never changed and the button looked broken. Null until dragged, and
    # still trimmed to whatever screen the window is next opened on.
    "compact_size": None,
    "expanded_size": None,
    # Whether the service accepts connections from other machines on the network.
    # Off, and deliberately: the API has no authentication, so binding it beyond
    # loopback hands everyone on the network the model, the settings and the ability
    # to load and unload. That is a reasonable thing to want and not a reasonable
    # default, so it is a decision the user makes once, knowing what it means.
    "lan_access": False,
    # The label of the starter model once it is registered here -- see
    # starter_model.py. Set once and never cleared, so a model the user deletes stays
    # deleted instead of coming back on the next start.
    "starter_model": None,
    # The application version that started the running service. A service outlives
    # the launcher and is adopted by the next one; after an update that would be the
    # old backend going on under a new application, so a mismatch restarts it.
    "service_version": None,
    # The device ComfyUI last ran on ("cuda:0", "xpu:0", "cpu"), so its card opens on
    # the choice that was made rather than on the detected default every time.
    "comfyui_device": None,
}


def load() -> dict[str, Any]:
    path = config.state_file()
    if not path.is_file():
        return dict(DEFAULTS)
    stored = None
    for attempt in range(3):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            break
        except json.JSONDecodeError:
            # Almost certainly a read that landed mid-replace. Falling straight back to
            # defaults here reports "not provisioned", which sends a fully installed
            # machine to the setup screen -- so retry before believing it.
            time.sleep(0.05 * (attempt + 1))
        except OSError:
            break
    if stored is None:
        # A genuinely unreadable state file must not stop the app from starting;
        # defaults are recoverable and provisioning detects what is already installed.
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
