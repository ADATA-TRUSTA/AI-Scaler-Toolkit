"""Install the prerequisites this application needs, when they are missing.

The line drawn here is not "can we install it" but "whose problem is it". These three
are needed by *setup itself*: without WebView2 there is no window to show, without the
Visual C++ runtime the GPU packages will not import, and without git the provisioning
script fails part-way through fetching the conversion tooling. Leaving those to the
user means an application that cannot start and cannot say usefully why.

Optional third-party tools are the opposite case: not needed by setup, not this
application's to manage, and no longer listed anywhere in it.

On Windows everything below is per-user and needs no elevation, which is what keeps
that installation free of an administrator prompt. Linux has no per-user equivalent
for git, so that one goes through the package manager and polkit's password dialog --
the only prompt in the whole installation, and one a desktop user recognises.
"""

from __future__ import annotations

import os
import shutil
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

# winget is the only one of these that can install git without elevation and without us
# hosting an installer ourselves.
GIT_PACKAGE = "Git.Git"

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


def _as_root() -> list[str] | None:
    """How to run one command as root without a terminal, or None if we cannot.

    pkexec draws polkit's own password dialog, which is the only way to ask on a
    desktop: the application was double-clicked, so there is no terminal for sudo to
    prompt in. Passwordless sudo comes first because it asks nothing at all.
    """
    if config.IS_WINDOWS:  # also keeps geteuid, which Windows has no equivalent of, unreached
        return None
    if os.geteuid() == 0:
        return []
    try:
        if subprocess.run(["sudo", "-n", "true"], capture_output=True,
                          timeout=10).returncode == 0:
            return ["sudo", "-n"]
    except (OSError, subprocess.TimeoutExpired):
        pass
    if shutil.which("pkexec") and (os.environ.get("DISPLAY")
                                   or os.environ.get("WAYLAND_DISPLAY")):
        return ["pkexec"]
    return None


def install_git_apt() -> bool:
    if shutil.which("apt-get") is None:
        log.warning("not an apt-based system, so git cannot be installed automatically")
        return False
    as_root = _as_root()
    if as_root is None:
        log.warning("no way to gain root without a terminal, so git cannot be installed")
        return False
    # env, not a bare assignment: both sudo and pkexec drop the caller's environment.
    return _run(
        [*as_root, "env", "DEBIAN_FRONTEND=noninteractive",
         "apt-get", "install", "-y", "git"],
        "git",
    )


def install_git() -> bool:
    if shutil.which("winget") is None:
        log.warning("winget is not available, so git cannot be installed automatically")
        return False
    return _run(
        ["winget", "install", "--exact", "--id", GIT_PACKAGE, "--scope", "user",
         "--silent", "--accept-package-agreements", "--accept-source-agreements"],
        "git",
    )


# Keyed by the fixer name each check declares.
FIXERS: dict[str, Callable[[], bool]] = {
    "webview2": install_webview2,
    "vcredist": install_vcredist,
    "winget": install_git,
    "apt": install_git_apt,
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
