"""Remove what this application installed, on the platform with no uninstaller.

Windows has one: installer.iss clears the program directory, asks whether to keep the
data Strata prepared and the model cache, and takes the autostart entry and the
shortcuts with it. An AppImage has none at all -- deleting the file leaves the
environment, ComfyUI, the menu entries and the autostart entry behind, several
gigabytes of them, and nothing says so.

The two questions are the ones the Windows uninstaller asks, and they default the same
way: both are expensive to rebuild, so keeping them is what a bare Enter does, and so
is what a run with no terminal to answer on does.

The AppImage itself is not touched. It is the file the user is holding, it may be
running this code, and removing a file is their job and one command.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import autostart, config, shortcuts, state

# What Strata derives from a model. config.strata_data_dir keeps it out of the install,
# whose directory goes wholesale; the name is needed here to carve it out of the sweep.
STRATA_DATA = "Strata-data"


def _ask(question: str, detail: str) -> bool:
    """True to remove. Anything short of an explicit yes keeps it."""
    print()
    print(question)
    print(detail)
    try:
        return input("Remove it? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("N (nothing here to answer on, so it is kept)")
        return False


def _remove(path: Path, what: str) -> bool:
    """Printed rather than logged: the log directory is one of the things going."""
    if not path.exists():
        return True
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as exc:
        print(f"  could not remove {what}: {exc}")
        return False
    print(f"  removed {what}")
    return True


def _remove_all_but(directory: Path, keep: str) -> None:
    """Everything in the data directory except the one entry being kept."""
    if not directory.is_dir():
        return
    for child in sorted(directory.iterdir()):
        if child.name != keep:
            _remove(child, child.name)


COMFY_USER_DATA = ("models", "output")


def _comfy_dir(data: Path) -> Path:
    return data / "components" / "comfyui" / "ComfyUI"


def _rescue_comfy_data(data: Path, keep: list[str]) -> Path | None:
    """Move ComfyUI's models/output out of the data directory before it is swept.

    Beside the data directory, so it is a rename on the same volume and not a copy of
    what can be a hundred gigabytes. Returns the folder they went to.
    """
    target = data.parent / f"{config.APP_ID}-kept"
    moved = False
    for name in keep:
        src = _comfy_dir(data) / name
        if not src.is_dir():
            continue
        dest = target / f"ComfyUI-{name}"
        suffix = 1
        while dest.exists():
            suffix += 1
            dest = target / f"ComfyUI-{name}-{suffix}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        except OSError as exc:
            print(f"  could not keep {src}: {exc}")
            continue
        print(f"  kept ComfyUI {name} at {dest}")
        moved = True
    return target if moved else None


def _stop_the_service() -> None:
    from .service_manager import listening_pid, probe, terminate_pid

    port = state.service_port()
    pid = listening_pid(port)
    if pid is None:
        return
    if probe(port) != "ours":
        print(f"  port {port} is held by pid {pid}, which is not this service; leaving it")
        return
    print(f"  stopping the service on port {port} (pid {pid})")
    terminate_pid(pid)


def _application_file() -> Path:
    """The AppImage the user would delete. $APPIMAGE is set by the type-2 runtime and
    is the path to the file itself; sys.argv[0] inside one is the mount point."""
    return Path(os.environ.get("APPIMAGE") or sys.argv[0]).resolve()


def run(assume_yes: bool = False) -> int:
    if config.IS_WINDOWS:
        print("On Windows the installer owns this: use Settings > Apps, or "
              f"Uninstall {config.APP_TITLE} in the Start menu.")
        return 1

    data = config.data_dir()
    settings = config.config_dir()
    strata_data = config.strata_data_dir()
    models = config.hf_home()

    print(f"Removing {config.APP_TITLE}.")
    print(f"  components and data   {data}")
    print(f"  settings              {settings}")

    remove_strata_data = False
    if strata_data.is_dir() and not assume_yes:
        remove_strata_data = _ask(
            "Also remove the data Strata prepared from your models?",
            f"  {strata_data}\n"
            "  This can be several gigabytes. Strata's setup made it from the "
            "downloaded models,\n"
            "  and makes it again the next time a model is loaded with Strata.")

    comfy_keep = []
    for name in COMFY_USER_DATA:
        folder = _comfy_dir(data) / name
        if not folder.is_dir():
            continue
        if assume_yes or not _ask(
                f"Also remove ComfyUI's {name}?",
                f"  {folder}\n"
                "  These can be tens of gigabytes and were put there by you or by "
                "ComfyUI.\n"
                "  Keeping them moves them next to the data directory."):
            comfy_keep.append(name)

    remove_models = False
    if models.is_dir() and not assume_yes:
        remove_models = _ask(
            "Also remove the downloaded models?",
            f"  {models}\n"
            "  These can be several tens of gigabytes. Other tools on this machine "
            "may use the\n"
            "  same cache, and keeping them means a reinstall does not download them "
            "again.")

    keep_strata_data = strata_data.is_dir() and not remove_strata_data

    print()
    _stop_the_service()
    _rescue_comfy_data(data, comfy_keep)
    autostart.disable()
    shortcuts.remove()
    print("  removed the autostart entry, the menu entry and the desktop shortcut")

    if keep_strata_data:
        _remove_all_but(data, STRATA_DATA)
    else:
        _remove(data, "the components and data")
    _remove(settings, "the settings")
    if remove_models:
        _remove(models, "the model cache")

    print()
    if keep_strata_data:
        print(f"Kept the data Strata prepared, at {strata_data}")
    if models.is_dir():
        print(f"Kept the model cache, at {models}")
    print("The application file itself is still here; remove it when you want to:")
    print(f"  rm {_application_file()}")
    return 0
