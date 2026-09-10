"""Put the application in the Linux applications menu and on the desktop.

The Windows installer creates a Start Menu entry and offers a desktop icon. Linux had
neither: only the autostart entry was written, so between logins the only way to start
the program was to find the .AppImage file in whatever directory it was downloaded to.
This closes that gap.

An AppImage is normally left to integrate itself, and that is what this is -- the
files are per-user, under the user's own data directory, and the uninstaller removes
them.

Everything here is best effort. A menu entry that could not be written is a smaller
problem than a launcher that refuses to start because of one, so failures are logged
and swallowed.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

from . import applog, autostart, config

log = applog.get("shortcuts")


def _entry_text() -> str:
    # Categories so the entry lands somewhere sensible in a menu that has sections.
    # StartupWMClass matches what the toolkit reports for the window, which is what
    # lets a running window associate with this icon in a dock instead of appearing
    # as a second, unnamed one.
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={config.APP_TITLE}\n"
        "Comment=Fine-tune and serve local language models\n"
        f"Exec={autostart.launch_command()}\n"
        f"Icon={config.APP_ID}\n"
        "Terminal=false\n"
        "Categories=Development;Science;\n"
        f"StartupWMClass={config.APP_ID}\n"
    )


def _write_icon() -> bool:
    source = config.payload_dir() / "app-mark.png"
    if not source.is_file():
        log.info("no app-mark.png in the payload; the menu entry gets no icon")
        return False
    target = config.menu_icon_target()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as exc:
        log.warning("could not write the icon to %s: %s", target, exc)
        return False
    return True


def _trust(path: Path) -> None:
    """Make a desktop-directory launcher runnable rather than suspicious.

    GNOME will not run a .desktop file dropped on the desktop unless it is executable
    *and* marked trusted; without both it shows as "Untrusted application launcher"
    and does nothing when double-clicked. gio is how that mark is set, and it is
    absent on desktops that do not need it, so a failure here is not an error.
    """
    with contextlib.suppress(OSError):
        path.chmod(0o755)
    if not shutil.which("gio"):
        return
    subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"],
                   capture_output=True, check=False)


def ensure() -> None:
    """Create the menu entry and the desktop icon if they are not already there."""
    if config.IS_WINDOWS:
        return

    text = _entry_text()
    _write_icon()

    menu = config.menu_entry_target()
    try:
        menu.parent.mkdir(parents=True, exist_ok=True)
        if _read(menu) != text:
            menu.write_text(text, encoding="utf-8")
            log.info("wrote the menu entry to %s", menu)
            _refresh_menu()
    except OSError as exc:
        log.warning("could not write the menu entry to %s: %s", menu, exc)

    desktop = config.desktop_shortcut_target()
    if not desktop.parent.is_dir():
        # No desktop directory: a session without one, or a user who removed it.
        # Nothing to put an icon on, and creating the directory would be presumptuous.
        return
    try:
        if _read(desktop) != text:
            desktop.write_text(text, encoding="utf-8")
            log.info("wrote the desktop shortcut to %s", desktop)
        _trust(desktop)
    except OSError as exc:
        log.warning("could not write the desktop shortcut to %s: %s", desktop, exc)


def remove() -> None:
    """Called by the uninstaller."""
    if config.IS_WINDOWS:
        return
    for path in (config.menu_entry_target(), config.desktop_shortcut_target(),
                 config.menu_icon_target()):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s: %s", path, exc)
    _refresh_menu()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _refresh_menu() -> None:
    """Some desktops read the cache rather than the directory, and show nothing until
    it is rebuilt. Absent on the ones that watch the directory instead."""
    tool = shutil.which("update-desktop-database")
    if not tool:
        return
    directory = config.menu_entry_target().parent
    subprocess.run([tool, str(directory)], capture_output=True, check=False)
