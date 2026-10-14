"""Check for a newer release, and install it when the user asks.

Nothing downloads or runs without an explicit click. The check is read-only and
silent about failure -- an update check with no network found nothing, which is not an
error worth interrupting anyone over.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
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
    current = config.app_version()
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


def download_verified(status: UpdateStatus, into: Path,
                      on_bytes: Callable[[int, int], None] | None = None) -> Path:
    """Fetch the release's file into `into` and check it against its published SHA-256.

    This is the one download in the product that is then *executed*, so it is the one
    that must not be trusted to having transferred without error. The hash comes from
    api.github.com and the file from objects.githubusercontent.com, so it catches a
    truncated transfer, a stale mirror and a proxy rewriting one host and not the
    other; it is not a defence against a compromised GitHub -- on Windows the
    installer's signature is checked too. A release with no published digest is
    refused rather than run on trust.

    `on_bytes(done, total)` reports progress; total is 0 when the server sends none.
    """
    if not status.asset_url or not status.asset_name:
        raise RuntimeError("this release has no installer for this system.")
    if not status.asset_sha256:
        raise RuntimeError("this release cannot be verified, so it was not installed.")

    # Written under a temporary name and only renamed once it matches, so an
    # interrupted or rejected download cannot be left sitting at the name the next
    # attempt -- or anything else -- would pick up and run.
    target = into / status.asset_name
    partial = target.with_name(target.name + ".part")
    # A connection that ends early can look like a finished one. On a laptop whose
    # network reset connections, the first try came back short and failed the hash
    # check with an alarming message about a mismatch; the transfer, not the file, was
    # the problem. So: a short download is called that, and one more try is made.
    for attempt in (1, 2):
        digest = hashlib.sha256()
        done = total = 0
        try:
            with (urllib.request.urlopen(status.asset_url, timeout=600) as response,
                  partial.open("wb") as out):
                total = int(response.headers.get("Content-Length") or 0)
                while chunk := response.read(1024 * 256):
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_bytes:
                        on_bytes(done, total)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            log.warning("download of %s broke off (try %d): %s", status.asset_name, attempt, exc)
            if attempt == 2:
                raise RuntimeError("the download was interrupted. Please try again.") from exc
            continue
        if total and done != total:
            partial.unlink(missing_ok=True)
            log.warning("%s arrived incomplete (try %d): %d of %d bytes",
                        status.asset_name, attempt, done, total)
            if attempt == 2:
                raise RuntimeError("the download was incomplete. Please try again.")
            continue
        actual = digest.hexdigest()
        if actual == status.asset_sha256:
            partial.replace(target)
            log.info("%s matches its published SHA-256", status.asset_name)
            return target
        partial.unlink(missing_ok=True)
        log.warning("%s does not match its published hash (try %d; expected %s, got %s)",
                    status.asset_name, attempt, status.asset_sha256, actual)
    # The hashes are in the log above. The person at the window needs to know only that
    # nothing was installed and what to do.
    raise RuntimeError("the download could not be verified, so nothing was installed. "
                       "Please try again.")


# Who signs the Windows installer. A file that matches GitHub's hash but is not
# signed by this publisher is not run: the hash only proves the transfer.
SIGNER = "ADATA Technology Co., Ltd."


def check_signature(installer: Path) -> None:
    """Refuse an installer that is not validly signed by SIGNER (Windows)."""
    script = (f"$s = Get-AuthenticodeSignature -LiteralPath '{installer}'; "
              "$s.Status.ToString() + '|' + $s.SignerCertificate.Subject")
    proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=60,
                          creationflags=config.subprocess_flags())
    status, _, subject = proc.stdout.strip().partition("|")
    if status != "Valid" or SIGNER not in subject:
        log.warning("%s signature: %s (%s)", installer.name, status or "none", subject or "-")
        raise RuntimeError("the installer's signature could not be verified, so nothing "
                           "was installed.")
    log.info("%s is signed by %s", installer.name, SIGNER)


def install_dir() -> Path:
    """Where the download goes: beside the AppImage it replaces, or the temp directory."""
    appimage = os.environ.get("APPIMAGE")
    if not config.IS_WINDOWS and appimage:
        return Path(appimage).resolve().parent
    return Path(tempfile.gettempdir())


def apply_and_relaunch(downloaded: Path, launcher_pid: int) -> None:
    """Hand the update to a helper that waits for this process to exit, then applies it.

    It waits because the new copy cannot start while this one runs: the single-instance
    guard would send it to the old window and it would exit. The caller quits right
    after this returns.

    Windows: the helper runs the installer silently -- an upgrade is allowed silently
    -- and then opens the application through Explorer, the way a double-click does
    (the installer's own "start" step is skipped when silent, and a child of Setup
    inherits a mitigation that breaks uv; see installer.iss).

    Linux: the AppImage is replaced in place now, which is safe while it runs (the
    running copy keeps the old inode), so the path the desktop entry and autostart
    point at is the new version. The helper then starts it, in an environment with
    this AppImage's own variables taken out.
    """
    if config.IS_WINDOWS:
        check_signature(downloaded)
        app = Path(sys.executable).resolve()
        # The helper keeps its own log: once this process is gone, it is the only
        # record of whether the installer ran and the application was reopened.
        helper_log = config.log_dir() / "update.log"
        setup_log = config.log_dir() / "update-setup.log"
        script = (
            f"$log = '{helper_log}'; "
            "function Note($m) { "
            "Add-Content -LiteralPath $log -Value ((Get-Date).ToString('s') + ' ' + $m) }; "
            f"Note 'waiting for the launcher (pid {launcher_pid}) to exit'; "
            f"Wait-Process -Id {launcher_pid} -ErrorAction SilentlyContinue; "
            f"Note 'running {downloaded.name}'; "
            f"$p = Start-Process -FilePath '{downloaded}' -PassThru -Wait "
            f"-ArgumentList '/SILENT','/SUPPRESSMSGBOXES','/NORESTART','/LOG=\"{setup_log}\"'; "
            "Note ('installer exited ' + $p.ExitCode); "
            f"Start-Process -FilePath \"$env:WINDIR\\explorer.exe\" -ArgumentList '\"{app}\"'; "
            "Note 'reopened the application'"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        command = ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                   "-EncodedCommand", encoded]
        # A hidden console, not none: under DETACHED_PROCESS powershell exited 0 at once
        # without running a line, so the application quit and nothing was installed.
        detached = 0x00000200 | 0x08000000  # NEW_PROCESS_GROUP, NO_WINDOW
        # Out of this process's job, if it is in one. A launcher started from a
        # scheduled task is, and the job took the helper down with the launcher: the
        # download and both checks passed, the application quit, and nothing was
        # installed. A job that forbids breaking away refuses the flag; then the
        # helper is started without it, which is all an ordinary start ever needed.
        try:
            subprocess.Popen(command, close_fds=True, creationflags=detached | 0x01000000)
        except OSError:
            subprocess.Popen(command, close_fds=True, creationflags=detached)
        log.info("update handed to the installer: %s (helper log: %s)", downloaded.name, helper_log)
        return

    appimage = os.environ.get("APPIMAGE")
    if not appimage:
        log.warning("not running from an AppImage; the new one is at %s", downloaded)
        raise RuntimeError(f"the new version was saved to {downloaded}; start it from there.")
    current = Path(appimage).resolve()
    downloaded.chmod(0o755)
    # Same directory, so this is a rename: atomic, and the running copy is untouched.
    os.replace(downloaded, current)
    log.info("replaced %s with the new version", current)
    env = {k: v for k, v in os.environ.items()
           if k not in {"APPDIR", "APPIMAGE", "ARGV0", "OWD", "PYTHONPATH", "PYTHONHOME"}}
    appdir = os.environ.get("APPDIR", "")
    env["PATH"] = os.pathsep.join(
        part for part in env.get("PATH", "").split(os.pathsep)
        if part and not (appdir and part.startswith(appdir)))
    helper = ["sh", "-c", 'while kill -0 "$1" 2>/dev/null; do sleep 0.5; done; exec "$2"',
              "sh", str(launcher_pid), str(current)]
    # In a unit of its own. A new session is not enough: systemd stops every process
    # in the launcher's unit when the launcher exits, and took the helper with it --
    # the AppImage was replaced, the application quit, and nothing reopened it. A
    # desktop without a user systemd gets the plain session instead.
    if shutil.which("systemd-run"):
        command = ["systemd-run", "--user", "--collect", "--quiet"] + [
            f"--setenv={k}={env[k]}" for k in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY",
                                               "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
            if env.get(k)] + helper
        if subprocess.run(command, env=env, capture_output=True, timeout=30).returncode == 0:
            log.info("reopening %s from its own unit once this one exits", current.name)
            return
        log.warning("systemd-run could not start the helper; using a plain session")
    subprocess.Popen(helper, env=env, start_new_session=True, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
