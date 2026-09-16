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
import tempfile
from pathlib import Path

from . import applog, autostart, config

log = applog.get("shortcuts")


def _entry_text(icon: Path | None) -> str:
    # Icon= is the installed file's path, not the theme name it was installed under.
    #
    # A themed name is the tidier form and it does not work on a first run: the icon
    # goes into ~/.local/share/icons/hicolor, that directory has no index.theme of its
    # own, and GNOME's desktop-icons extension does not resolve a name there until its
    # caches are rebuilt. The shortcut therefore showed a blank document until the user
    # logged out and back in -- which is what hid this, since the icon was fine on every
    # machine that had been restarted since installing.
    #
    # A path is resolved by the loader directly, so it is right immediately. The file is
    # still installed into hicolor, which is where it belongs and where anything that
    # does look up by name will find it.
    #
    # Categories so the entry lands somewhere sensible in a menu that has sections.
    # StartupWMClass matches what the toolkit reports for the window, which is what lets
    # a running window associate with this icon in a dock instead of appearing as a
    # second, unnamed one.
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={config.APP_TITLE}\n"
        "Comment=Fine-tune and serve local language models\n"
        f"Exec={autostart.launch_command()}\n"
        f"Icon={icon if icon else config.APP_ID}\n"
        "Terminal=false\n"
        "Categories=Development;Science;\n"
        f"StartupWMClass={config.APP_ID}\n"
    )


def _write_icon() -> Path | None:
    """Install the icon and return where it landed, or None if it could not be."""
    source = config.payload_dir() / "app-mark.png"
    if not source.is_file():
        log.info("no app-mark.png in the payload; the menu entry gets no icon")
        return None
    target = config.menu_icon_target()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as exc:
        log.warning("could not write the icon to %s: %s", target, exc)
        return None
    return target


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


def _should_write(entry: Path, text: str) -> bool:
    """Whether to write over an entry that is already there.

    An absent or unreadable one is written. An existing one is left alone unless the
    program it names is gone, which is the rule autostart.refresh() already applies to
    the login entry and for the same reason: two working copies can exist -- an install
    and a newer build someone downloaded to try -- and whichever ran last should not get
    to claim the other's shortcuts. Rewriting on every difference did exactly that, and
    the copy from Downloads is the one that then gets deleted, leaving both entries
    pointing at nothing.

    A stale entry is still repointed, because an entry naming a path that no longer
    exists is the quiet failure this is here to avoid.
    """
    current = _read(entry)
    if current is None:
        return True
    if current == text:
        return False
    program = _program_in(current)
    return program is None or not program.exists()


def _program_in(entry_text: str) -> Path | None:
    """The executable an entry's Exec= line runs, quoted or not."""
    for line in entry_text.splitlines():
        if not line.startswith("Exec="):
            continue
        command = line[len("Exec="):].strip()
        if command.startswith('"'):
            end = command.find('"', 1)
            return Path(command[1:end]) if end != -1 else None
        first = command.split(" ", 1)[0]
        return Path(first) if first else None
    return None


def ensure() -> None:
    """Create the menu entry and the desktop icon if they are not already there."""
    if config.IS_WINDOWS:
        return

    text = _entry_text(_write_icon())

    menu = config.menu_entry_target()
    try:
        menu.parent.mkdir(parents=True, exist_ok=True)
        if _should_write(menu, text):
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
        if _should_write(desktop, text):
            desktop.write_text(text, encoding="utf-8")
            log.info("wrote the desktop shortcut to %s", desktop)
        _trust(desktop)
    except OSError as exc:
        log.warning("could not write the desktop shortcut to %s: %s", desktop, exc)


def ensure_component(component_id: str, program: Path, fallback_name: str,
                     *, terminal: bool = False, comment: str = "") -> bool:
    """Put a component we installed into the applications menu.

    A program dropped in a directory integrates with nothing by itself: it is invisible
    to the menu, the dock and the launcher's search, and the only way to start it is to
    find the file. On Windows the components' own installers write Start Menu entries,
    so this is the Linux half of what they already do.

    For an AppImage the entry is built from the one inside it, so the name, the icon and
    the categories are the vendor's own -- only Exec is rewritten, to the path we
    installed it at, since theirs assumes it is running from inside the mounted image.

    terminal=True for a command-line tool. The desktop then opens it in whatever
    terminal that desktop uses, which is the only portable answer: guessing between
    gnome-terminal, konsole and the rest gets it wrong on the desktops we did not think
    of, and getting it wrong means a menu entry that does nothing when clicked.
    """
    if config.IS_WINDOWS or not program.is_file():
        return False

    details = _read_appimage_entry(program) if program.suffix.lower() == ".appimage" else {}
    name = details.get("Name") or fallback_name
    icon = _extract_appimage_icon(program, component_id, details.get("Icon", "")) if details else None
    # %U and friends are placeholders the desktop fills in; an Exec that keeps them
    # while naming our path is still correct, and dropping them would lose the ability
    # to open a file with the component.
    arguments = details.get("_exec_args", "")

    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={name}\n"
        f"Comment={details.get('Comment') or comment}\n"
        f"Exec=\"{program}\"{(' ' + arguments) if arguments else ''}\n"
        f"Icon={icon if icon else 'utilities-terminal' if terminal else 'application-x-executable'}\n"
        f"Terminal={'true' if terminal else 'false'}\n"
        f"Categories={details.get('Categories', 'Development;Utility;')}\n"
        # Ours, so the uninstaller can tell an entry it wrote from one it did not.
        f"X-TrustaComponent={component_id}\n"
    )
    target = config.component_entry_target(component_id)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry, encoding="utf-8")
    except OSError as exc:
        log.warning("could not write the menu entry for %s: %s", component_id, exc)
        return False
    log.info("wrote the menu entry for %s to %s", component_id, target)
    _refresh_menu()
    return True


def _read_appimage_entry(program: Path) -> dict[str, str]:
    """The .desktop an AppImage carries, as a dict. Empty when it cannot be read.

    Every AppImage has one at its root -- that is how the format works -- so this is
    reading the vendor's own answer rather than guessing at a name and an icon.
    """
    with tempfile.TemporaryDirectory() as work:
        try:
            subprocess.run([str(program), "--appimage-extract", "*.desktop"],
                           cwd=work, capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            log.info("could not read the desktop entry inside %s: %s", program.name, exc)
            return {}
        found = sorted(Path(work).rglob("*.desktop"))
        if not found:
            return {}
        values: dict[str, str] = {}
        for line in found[0].read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if _ and key not in values:
                values[key.strip()] = value.strip()
    # Exec is the vendor's own command inside the image; only its arguments survive.
    exec_line = values.get("Exec", "")
    values["_exec_args"] = exec_line.split(" ", 1)[1] if " " in exec_line else ""
    return values


def _extract_appimage_icon(program: Path, component_id: str, name: str) -> Path | None:
    """Copy the icon out of the AppImage, and return where it landed.

    By path, not by theme name, for the reason recorded in _entry_text: a name in
    hicolor does not resolve until the icon caches are rebuilt, which on a first run
    means a blank square until the next login.

    From the theme directory inside the image, not from its root. The root carries the
    icon and .DirIcon as symlinks into usr/share/icons, so extracting either of those
    names on its own produces a dangling link -- a path that exists, is not a file, and
    silently yields no icon at all. Measured on ComfyUI: the root name extracted
    cleanly and gave nothing, while the theme directory gave eight real sizes.
    """
    patterns = [
        f"usr/share/icons/hicolor/*/apps/{name}.png" if name else "usr/share/icons/*",
        "usr/share/icons/*",
        f"{name}.png" if name else "*.png",
    ]
    with tempfile.TemporaryDirectory() as work:
        for pattern in patterns:
            try:
                subprocess.run([str(program), "--appimage-extract", pattern],
                               cwd=work, capture_output=True, timeout=120, check=False)
            except (OSError, subprocess.SubprocessError):
                return None
            # The largest, because an AppImage carries several sizes and the menu wants
            # the one that still looks right on a high-density display. is_file() also
            # rejects the dangling symlinks described above.
            icons = [p for p in Path(work).rglob("*")
                     if p.is_file() and not p.is_symlink() and p.stat().st_size > 0]
            if not icons:
                continue
            best = max(icons, key=lambda p: p.stat().st_size)
            target = config.component_icon_target(component_id)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(best, target)
            except OSError as exc:
                log.info("could not install the icon for %s: %s", component_id, exc)
                return None
            return target
    return None


def remove() -> None:
    """Called by the uninstaller."""
    if config.IS_WINDOWS:
        return
    for path in _component_entries():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s: %s", path, exc)
    for path in (config.menu_entry_target(), config.desktop_shortcut_target(),
                 config.menu_icon_target()):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s: %s", path, exc)
    _refresh_menu()


def _component_entries() -> list[Path]:
    """Every menu entry this application wrote for a component, and their icons.

    Found by the prefix rather than by a list of component ids, so an entry left by a
    component that has since been removed from the catalogue is still cleaned up.
    """
    found: list[Path] = []
    for directory, pattern in (
        (config.menu_entry_target().parent, f"{config.APP_ID}-*.desktop"),
        (config.menu_icon_target().parent, f"{config.APP_ID}-*.png"),
    ):
        with contextlib.suppress(OSError):
            found.extend(p for p in directory.glob(pattern)
                         if p != config.menu_entry_target() and p != config.menu_icon_target())
    return found


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
