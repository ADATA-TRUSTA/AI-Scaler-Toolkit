"""Install the prerequisites this application needs, when they are missing.

One entry is left, and the line drawn is not "can we install it" but "whose problem is
it". WebView2 is needed by *setup itself* -- without it there is no window to show --
and its bootstrapper installs per-user when it is not elevated. Leaving that to the
user means an application that cannot start and cannot say usefully why.

Optional third-party tools are the opposite case: not needed by setup, not this
application's to manage, and no longer listed anywhere in it.

Two entries were removed rather than fixed, because each was the wrong answer to a
question that had a better one, and each raised a UAC prompt in an installer built
from end to end to need none:

  git, installed through winget, which has no per-user Git package and elevates
  whatever --scope it is given. setup_env fetches the llama.cpp sources as an archive
  now and does not need git at all.

  The Visual C++ runtime, which is a machine-wide component with no per-user form, so
  installing it always prompts. The payload carries the three DLLs instead and
  setup_env places them beside the binaries that import them -- the same app-local
  deployment the frozen launcher already relies on. See deploy/package/vcruntime.py.

So this installation now raises no administrator prompt at all, which is what it was
always supposed to do.
"""

from __future__ import annotations

import subprocess
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import applog, config

log = applog.get("remedy")

# Microsoft's evergreen bootstrapper: small, and it installs the current runtime rather
# than pinning one that will be stale by the time anyone runs it.
WEBVIEW2_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

_TIMEOUT = 900


def _run(command: list[str], what: str) -> bool:
    log.info("%s: %s", what, " ".join(command))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            creationflags=config.subprocess_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("%s failed to run: %s", what, exc)
        return False
    if result.returncode != 0:
        log.warning("%s exited %s: %s", what, result.returncode,
                    (result.stderr or result.stdout or "").strip()[:400])
        return False
    log.info("%s completed", what)
    return True


def _download(url: str, name: str) -> Path | None:
    target = Path(tempfile.gettempdir()) / name
    log.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
            while chunk := response.read(1024 * 256):
                out.write(chunk)
    except (OSError, ValueError) as exc:
        log.warning("could not download %s: %s", url, exc)
        return None
    log.info("downloaded %s (%d bytes)", target.name, target.stat().st_size)
    return target


def install_webview2() -> bool:
    """Runs per-user when not elevated, which is what we want."""
    installer = _download(WEBVIEW2_URL, "MicrosoftEdgeWebview2Setup.exe")
    if installer is None:
        return False
    return _run([str(installer), "/silent", "/install"], "WebView2 runtime")


# Keyed by the fixer name each check declares.
FIXERS: dict[str, Callable[[], bool]] = {
    "webview2": install_webview2,
}


def can_fix(fixer: str) -> bool:
    return fixer in FIXERS


def apply(fixer: str) -> bool:
    handler = FIXERS.get(fixer)
    if handler is None:
        return False
    try:
        return handler()
    except Exception:
        log.exception("installing %s failed unexpectedly", fixer)
        return False
