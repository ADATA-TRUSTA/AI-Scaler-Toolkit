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
from dataclasses import dataclass, field
from pathlib import Path

from . import applog, autostart, config

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
            # Provisioning leaves a git checkout behind and git marks its pack files
            # read-only, which stops an ordinary tree removal part-way on Windows.
            shutil.rmtree(path, onerror=_clear_readonly)
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


def provisioning_output(*, dry_run: bool = False) -> Removal:
    """Everything setup_env put on this machine, and the launcher's own state.

    Deliberately not the model cache, and deliberately not the program directory --
    the installer owns that one.
    """
    result = Removal()
    _remove(config.venv_dir(), "virtual environment", result, dry_run=dry_run)
    _remove(config.llama_dir(), "llama runtime", result, dry_run=dry_run)
    _remove(config.legacy_llama_binary(), "llama binary (legacy location)", result, dry_run=dry_run)
    _remove(config.gguf_tools_dir(), "GGUF conversion tooling", result, dry_run=dry_run)
    _remove(config.llama_staging_dir(), "llama installer staging", result, dry_run=dry_run)
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
    log.info("autostart entry cleared")
    return True
