"""Remove what provisioning installed.

Used from three places that must agree: the settings window's repair action, the
developer reset script, and the acceptance run in CI. They had one implementation
between them already -- this is where it lives now, so that the frozen application can
use it too rather than shelling out to a script that is not shipped with it.

The model cache is treated separately everywhere. It does not affect which provisioning
path runs, it is shared with any other Hugging Face tooling on the machine, and it
reaches tens of gigabytes -- so removing it is always a deliberate, separate decision.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import applog, autostart, config, shortcuts

log = applog.get("maintenance")


@dataclass
class Removal:
    removed: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        if self.failed:
            return f"removed {len(self.removed)}, could not remove {', '.join(self.failed)}"
        return f"removed {len(self.removed)} of {len(self.removed) + len(self.absent)} items"


def _remove(path: Path, label: str, result: Removal, *, dry_run: bool) -> None:
    if not path.exists():
        result.absent.append(label)
        return
    if dry_run:
        result.removed.append(f"{label} (would remove {path})")
        return
    try:
        if path.is_dir():
            remove_tree(path)
        else:
            path.unlink()
    except OSError as exc:
        log.warning("could not remove %s (%s): %s", label, path, exc)
        result.failed.append(label)
        return
    log.info("removed %s: %s", label, path)
    result.removed.append(label)


def _clear_readonly(func, name, _exc):
    import os
    import stat

    os.chmod(name, stat.S_IWRITE)
    func(name)


def _extended(path: Path) -> str:
    r"""The path in the form Windows does not cut off at 260 characters.

    The backend's environment holds files deeper than that: torch ships licence texts
    at ...\torch-*.dist-info\licenses\third_party\..., 254 characters plus the length
    of the account's name under the default install location -- past the limit for
    any name of six or more. uv writes them without trouble, and then nothing that goes
    through the ordinary path form can delete them -- shutil.rmtree stops with "the
    system cannot find the path specified" and leaves the tree half there. The \\?\
    form has no such limit and needs no machine-wide setting.
    """
    if not config.IS_WINDOWS:
        return str(path)
    full = str(path.resolve())
    if full.startswith("\\\\?\\"):
        return full
    if full.startswith("\\\\"):
        return "\\\\?\\UNC\\" + full[2:]
    return "\\\\?\\" + full


def remove_tree(path: Path) -> None:
    """Remove a directory completely, or raise OSError.

    Two things stop an ordinary removal part-way on Windows, and a tree removed
    part-way is worse than one left alone: read-only files (provisioning leaves a git
    checkout behind, and git marks its pack files read-only) and paths past 260
    characters (see _extended). Checked afterwards as well, because a removal that
    reports nothing and leaves the directory behind is how a broken environment gets
    taken for an installed one.
    """
    shutil.rmtree(_extended(path), onerror=_clear_readonly)
    if path.exists():
        raise OSError(f"{path} is still there after removing it")


def old_window_environment() -> None:
    """Remove the environment earlier AppImages built to draw their window with.

    They made one on the first start, in the data directory, and ran the launcher on
    it. The AppImage carries its interpreter now (deploy/linux/build_appimage.sh), so
    what an earlier version left is a hundred megabytes nothing will use again.

    Called once this copy holds the single-instance lock, and not from AppRun: an
    older copy still running is running *on* that environment, and a newer AppImage
    started beside it only asks it to come forward and exits. Removing it there would
    pull the packages out from under a process that is still using them.
    """
    if config.IS_WINDOWS:
        return
    here = Path(sys.prefix).resolve()
    for name in ("launcher-venv", "launcher-venv.new"):
        path = config.data_dir() / name
        if not path.exists():
            continue
        # A copy started on it -- an older AppRun, or by hand -- is this very process.
        if here == path.resolve() or path.resolve() in here.parents:
            continue
        shutil.rmtree(path, ignore_errors=True)
        log.info("removed the window environment an earlier version built: %s", path)


def provisioning_output(*, dry_run: bool = False) -> Removal:
    """Everything setup_env put on this machine, and the launcher's own state.

    Deliberately not the model cache, and deliberately not the program directory --
    the installer owns that one.
    """
    result = Removal()
    _remove(config.venv_dir(), "virtual environment", result, dry_run=dry_run)
    _remove(config.llama_dir(), "llama runtime", result, dry_run=dry_run)
    _remove(config.llama_build_dir(), "llama sources and build tree", result, dry_run=dry_run)
    _remove(config.legacy_llama_binary(), "llama binary (legacy location)", result, dry_run=dry_run)
    _remove(config.gguf_tools_dir(), "GGUF conversion tooling", result, dry_run=dry_run)
    _remove(config.llama_staging_dir(), "llama installer staging", result, dry_run=dry_run)
    _remove(config.strata_dir(), "Strata engine", result, dry_run=dry_run)
    _remove(config.state_file(), "launcher state", result, dry_run=dry_run)
    # Regenerated on the next start. Leaving a stale port behind is the kind of
    # half-reset that makes the next run behave unlike a real first run.
    _remove(config.backend_dir() / ".env", "backend .env", result, dry_run=dry_run)
    return result


def model_cache(*, dry_run: bool = False) -> Removal:
    result = Removal()
    _remove(config.hf_home(), "model cache", result, dry_run=dry_run)
    return result


def clear_autostart(*, dry_run: bool = False) -> bool:
    """Cleared on uninstall as well as on reset.

    An autostart entry pointing at a removed executable fails silently at every login,
    forever, and nothing ever surfaces it.
    """
    if dry_run:
        return autostart.is_enabled()
    autostart.disable()
    # The menu entry and desktop icon go with it. They name the same absolute path,
    # so leaving them behind gives the same silent failure with a visible icon on it.
    shortcuts.remove()
    log.info("autostart entry cleared")
    return True
