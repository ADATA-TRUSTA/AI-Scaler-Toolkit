"""Check for a newer release, and hand the user the installer if they ask for one.

Nothing downloads or runs without an explicit click. The check is read-only and
silent about failure -- an update check with no network found nothing, which is not an
error worth interrupting anyone over.
"""

from __future__ import annotations

import subprocess
import tempfile
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from . import config, github, state


@dataclass(frozen=True)
class UpdateStatus:
    current: str | None
    latest: str | None
    available: bool
    # Why there is no newer version, when there is none. "none published yet" and
    # "could not ask" look the same to the caller otherwise, and only one of them is
    # something the user should do anything about.
    reason: str = "ok"
    release_url: str = ""
    asset_url: str | None = None
    asset_name: str | None = None


def asset_suffix() -> str:
    return ".exe" if config.IS_WINDOWS else ".AppImage"


def check(repo: str = github.DEFAULT_REPO) -> UpdateStatus:
    current = state.get("installed_version")
    try:
        release = github.latest(repo=repo, asset_suffix=asset_suffix())
    except github.NoReleases:
        return UpdateStatus(current=current, latest=None, available=False, reason="none-published")
    if release is None:
        return UpdateStatus(current=current, latest=None, available=False, reason="unreachable")

    state.set("last_update_check", release.tag)
    return UpdateStatus(
        current=current,
        latest=release.tag,
        available=github.is_newer(release.tag, current),
        release_url=release.url,
        asset_url=release.asset_url,
        asset_name=release.asset_name,
    )


def open_release_page(status: UpdateStatus) -> None:
    if status.release_url:
        webbrowser.open(status.release_url)


def download_and_launch(status: UpdateStatus) -> Path:
    """Fetch the installer and start it. Windows installs over the top of itself;
    on Linux the AppImage is simply replaced by the user."""
    if not status.asset_url or not status.asset_name:
        raise RuntimeError("that release has no installer attached")

    target = Path(tempfile.gettempdir()) / status.asset_name
    with urllib.request.urlopen(status.asset_url, timeout=600) as response, target.open("wb") as out:
        while chunk := response.read(1024 * 256):
            out.write(chunk)

    if config.IS_WINDOWS:
        subprocess.Popen([str(target)], close_fds=True)
    else:
        target.chmod(0o755)
    return target
