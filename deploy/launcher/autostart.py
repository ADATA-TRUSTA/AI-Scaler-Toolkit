"""Start with the machine, per user, without administrator rights.

On by default. The app is a tray-resident service manager, and one that has to be
launched by hand before anything works is a worse version of running the scripts
directly.

Both mechanisms here are per-user on purpose: HKCU rather than HKLM, and the user's
own autostart directory rather than a system one. Neither needs elevation, which is
what lets the whole installer stay per-user.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = config.APP_NAME


# Passed by the autostart entry and by nothing else, so the application can tell a
# login from someone choosing to open it. Without that distinction, logging in to
# Windows could begin an unattended eight-minute setup: nothing in the launcher knew
# it had not been asked for.
LOGIN_FLAG = "--autostart"


def _launch_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).resolve()}" {LOGIN_FLAG}'
    return f'"{sys.executable}" -m launcher {LOGIN_FLAG}'


def is_enabled() -> bool:
    if config.IS_WINDOWS:
        proc = subprocess.run(
            ["reg", "query", rf"HKCU\{RUN_KEY}", "/v", RUN_VALUE],
            capture_output=True, text=True, creationflags=config.subprocess_flags(),
        )
        return proc.returncode == 0
    return config.autostart_target().is_file()


def recorded_command() -> str | None:
    """What the autostart entry launches today, or None if there is no entry."""
    if config.IS_WINDOWS:
        proc = subprocess.run(
            ["reg", "query", rf"HKCU\{RUN_KEY}", "/v", RUN_VALUE],
            capture_output=True, text=True, creationflags=config.subprocess_flags(),
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            if RUN_VALUE in line and "REG_SZ" in line:
                return line.split("REG_SZ", 1)[1].strip()
        return None

    target = config.autostart_target()
    if not target.is_file():
        return None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith("Exec="):
            return line[len("Exec="):].strip()
    return None


def _executable_in(command: str) -> Path | None:
    """The program a recorded autostart command runs, quoted or not."""
    command = command.strip()
    if command.startswith('"'):
        end = command.find('"', 1)
        if end == -1:
            return None
        return Path(command[1:end])
    first = command.split(" ", 1)[0]
    return Path(first) if first else None


def refresh() -> bool:
    """Repoint the autostart entry when what it names is gone.

    The entry records an absolute path, and the application moves: a reinstall into a
    different directory, an upgrade, a developer build replaced by a real install. The
    old entry then names a path that no longer exists, and the failure mode is the
    quietest one there is -- nothing starts at login, forever, with no error anywhere.
    Found on a machine where it had already happened, naming a build directory that
    had since been renamed.

    Only when the recorded program is missing, never merely when it differs from this
    one. Two working copies can exist -- a release install and a checkout being worked
    on -- and whichever ran last should not get to claim the login entry from the other.

    Returns True if the entry was rewritten.
    """
    if not is_enabled():
        return False
    current = recorded_command()
    if current is None:
        return False
    program = _executable_in(current)
    if program is not None and program.exists():
        return False
    enable()
    return True


def enable() -> None:
    if config.IS_WINDOWS:
        subprocess.run(
            ["reg", "add", rf"HKCU\{RUN_KEY}", "/v", RUN_VALUE, "/t", "REG_SZ",
             "/d", _launch_command(), "/f"],
            check=True, capture_output=True, creationflags=config.subprocess_flags(),
        )
        return

    target = config.autostart_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={config.APP_TITLE}\n"
        f"Exec={_launch_command()}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n",
        encoding="utf-8",
    )


def disable() -> None:
    """Also called by the uninstaller.

    An autostart entry left pointing at a removed executable is the classic uninstall
    leftover: nothing visibly breaks, the user just gets a silent failure at every
    login forever.
    """
    if config.IS_WINDOWS:
        subprocess.run(
            ["reg", "delete", rf"HKCU\{RUN_KEY}", "/v", RUN_VALUE, "/f"],
            capture_output=True, text=True, creationflags=config.subprocess_flags(),
        )
        return
    config.autostart_target().unlink(missing_ok=True)


def set_enabled(enabled: bool) -> None:
    enable() if enabled else disable()
