"""Check for a newer release, and hand the user the installer if they ask for one.

Nothing downloads or runs without an explicit click. The check is read-only and
silent about failure -- an update check with no network found nothing, which is not an
error worth interrupting anyone over.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import applog, browser, config, github, state

log = applog.get("updater")


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
    asset_sha256: str | None = None


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
        asset_sha256=release.asset_sha256,
    )


def open_release_page(status: UpdateStatus) -> None:
    if status.release_url:
        browser.open_url(status.release_url)


def download_and_launch(status: UpdateStatus) -> Path:
    """Fetch the installer, check it, and start it.

    Windows installs over the top of itself; on Linux the AppImage is replaced by the
    user. Either way this is the one download in the product that is then *executed*,
    so it is the one that must not be trusted to having transferred without error.

    Checked against the SHA-256 GitHub publishes for the asset. That hash comes from
    api.github.com and the file from objects.githubusercontent.com, so it catches a
    truncated transfer, a stale mirror and a proxy rewriting one host and not the
    other; it is not a defence against a compromised GitHub, and code signing is what
    would be. A release with no published digest is refused rather than run on trust.
    """
    if not status.asset_url or not status.asset_name:
        raise RuntimeError("that release has no installer attached")
    if not status.asset_sha256:
        raise RuntimeError(
            "that release publishes no SHA-256 for its installer, so it cannot be "
            "verified. Download it from the release page instead."
        )

    # Written under a temporary name and only renamed once it matches, so an
    # interrupted or rejected download cannot be left sitting at the name the next
    # attempt -- or anything else -- would pick up and run.
    target = Path(tempfile.gettempdir()) / status.asset_name
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(status.asset_url, timeout=600) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 256):
            out.write(chunk)
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual != status.asset_sha256:
        partial.unlink(missing_ok=True)
        log.warning("%s does not match its published hash (expected %s, got %s)",
                    status.asset_name, status.asset_sha256, actual)
        raise RuntimeError(
            f"the downloaded installer does not match the hash published for it.\n"
            f"  expected {status.asset_sha256}\n  got      {actual}\n"
            "It was not run. Try again, and if it happens twice, report it."
        )
    partial.replace(target)
    log.info("%s matches its published SHA-256", status.asset_name)

    if config.IS_WINDOWS:
        subprocess.Popen([str(target)], close_fds=True)
    else:
        target.chmod(0o755)
    return target
