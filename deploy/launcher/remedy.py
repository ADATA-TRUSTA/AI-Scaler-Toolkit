"""Install the prerequisites this application needs, when they are missing.

The line drawn here is not "can we install it" but "whose problem is it". Both of these
are needed by *setup itself*: without WebView2 there is no window to show, and without
the Visual C++ runtime the GPU packages will not import. Leaving those to the user
means an application that cannot start and cannot say usefully why.

Optional third-party tools are the opposite case: not needed by setup, not this
application's to manage, and no longer listed anywhere in it.

git used to be a third entry, installed through winget. It is gone because setup_env
no longer needs it, which is the better fix: winget's Git.Git has no per-user
installer, so that one line put a UAC prompt in front of a user who had just run an
installer built entirely to avoid one.

WebView2's bootstrapper installs per-user when it is not elevated. The Visual C++
runtime does not and cannot -- it is a machine-wide component, there is no per-user
form of it, and installing it prompts. That is the one prompt this installation can
raise, and only on a machine that does not already have a runtime most desktop
software ships with.
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
VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

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


def install_vcredist() -> bool:
    installer = _download(VCREDIST_URL, "vc_redist.x64.exe")
    if installer is None:
        return False
    return _run([str(installer), "/install", "/quiet", "/norestart"], "Visual C++ runtime")


# Keyed by the fixer name each check declares.
FIXERS: dict[str, Callable[[], bool]] = {
    "webview2": install_webview2,
    "vcredist": install_vcredist,
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
