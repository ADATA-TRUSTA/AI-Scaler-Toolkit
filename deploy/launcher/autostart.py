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


def _launch_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).resolve()}"'
    return f'"{sys.executable}" -m launcher'


def is_enabled() -> bool:
    if config.IS_WINDOWS:
        proc = subprocess.run(
            ["reg", "query", rf"HKCU\{RUN_KEY}", "/v", RUN_VALUE],
            capture_output=True, text=True, creationflags=config.subprocess_flags(),
        )
        return proc.returncode == 0
    return config.autostart_target().is_file()


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
