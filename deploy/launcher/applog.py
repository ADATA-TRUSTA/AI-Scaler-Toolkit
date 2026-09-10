"""A log the launcher writes about itself, from its first line.

Without this there is nothing to look at when the application stops responding. The
provisioning log only exists once provisioning has started, and in a windowed build
there is no console, so a hang anywhere before that leaves no trace at all -- which is
exactly the situation this was added to fix.

Everything before and around provisioning is recorded here: what preflight found,
which port was chosen and why, which keys were written to .env, and the exact command
and working directory handed to setup_env. Enough to say where it stopped without
having to reproduce it.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

from . import config

LOG_NAME = "launcher.log"
_CONFIGURED = False


def log_path() -> Path:
    return config.log_dir() / LOG_NAME


def setup(level: int = logging.DEBUG) -> logging.Logger:
    """Attach handlers once. Safe to call from anywhere; later calls are no-ops."""
    global _CONFIGURED
    root = logging.getLogger("launcher")
    if _CONFIGURED:
        return root

    root.setLevel(level)
    config.log_dir().mkdir(parents=True, exist_ok=True)

    # Rotating rather than truncating: the interesting run is often the previous one,
    # and this application restarts at every login.
    handler = logging.handlers.RotatingFileHandler(
        log_path(), maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(handler)

    # A windowed build has no stderr at all -- sys.stderr is None, and a StreamHandler
    # on it raises during logging rather than at setup, which would turn a diagnostic
    # into a second failure.
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(console)

    _CONFIGURED = True
    return root


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"launcher.{name}")


def banner() -> None:
    """First thing written, so a log always identifies what produced it."""
    log = get("start")
    log.info("=" * 68)
    log.info("%s starting", config.APP_TITLE)
    log.info("frozen        : %s", bool(getattr(sys, "frozen", False)))
    log.info("executable    : %s", sys.executable)
    log.info("python        : %s", sys.version.split()[0])
    log.info("platform      : %s %s", platform.system(), platform.release())
    log.info("app dir       : %s", config.app_dir())
    log.info("payload dir   : %s", config.payload_dir())
    log.info("backend dir   : %s", config.backend_dir())
    log.info("resource root : %s", config.resource_root())
    log.info("data dir      : %s", config.data_dir())
    log.info("venv dir      : %s", config.venv_dir())
    log.info("state file    : %s", config.state_file())
    log.info("log file      : %s", log_path())
    log.info("LOCALAPPDATA  : %s", os.environ.get("LOCALAPPDATA", "<unset>"))
    log.info("=" * 68)


def flush() -> None:
    """Push every handler's buffer to disk.

    Needed because the launcher ends with os._exit on Linux, which skips logging's
    own atexit flush -- and the last lines before an exit are exactly the ones worth
    having.
    """
    for handler in logging.getLogger().handlers:
        with contextlib.suppress(Exception):
            handler.flush()
