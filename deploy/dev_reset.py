"""Reset this machine to a pre-install state, re-provision, or just launch the GUI.

Used for two things, and the sharing is the point:

- Development: get back to a clean first run without reinstalling.
- CI acceptance: the installer only owns the program directory, so provisioning output
  survives an uninstall. Left in place, the *next* acceptance run finds a virtual
  environment and an llama binary already there, skips the work it exists to test, and
  passes. One purge implementation for both keeps them from drifting.

The model cache is separate from the rest. It does not affect which provisioning path
runs and it reaches tens of gigabytes, so --keep-model-cache exists and CI uses it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launcher import autostart, config, state  # noqa: E402


def _remove(path: Path, label: str, *, dry_run: bool) -> bool:
    if not path.exists():
        print(f"  - {label}: not present")
        return False
    if dry_run:
        print(f"  - {label}: would remove {path}")
        return True
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as exc:
        print(f"  ! {label}: could not remove {path} ({exc})")
        return False
    print(f"  - {label}: removed {path}")
    return True


def purge(*, keep_model_cache: bool, dry_run: bool) -> None:
    print("Removing provisioning output:")
    _remove(config.venv_dir(), "virtual environment", dry_run=dry_run)
    _remove(config.llama_binary(), "llama binary", dry_run=dry_run)
    _remove(config.gguf_tools_dir(), "GGUF conversion tooling", dry_run=dry_run)
    _remove(config.llama_staging_dir(), "llama installer staging", dry_run=dry_run)

    print("Removing launcher state:")
    _remove(config.state_file(), "state.json", dry_run=dry_run)
    _remove(config.log_dir(), "logs", dry_run=dry_run)

    # The backend's .env is written by the launcher on first run and appended to by
    # setup_env. Both are regenerated, and leaving a stale port behind is exactly the
    # kind of half-reset that makes the next run behave unlike a real first run.
    env_path = config.backend_dir() / ".env"
    _remove(env_path, "backend .env", dry_run=dry_run)

    print("Removing autostart entry:")
    if dry_run:
        print(f"  - autostart: currently {'enabled' if autostart.is_enabled() else 'disabled'}")
    else:
        autostart.disable()
        print("  - autostart: cleared")

    if keep_model_cache:
        print(f"Keeping the model cache at {config.hf_home()}")
    else:
        print("Removing the model cache:")
        _remove(config.hf_home(), "HF_HOME", dry_run=dry_run)


def reprovision() -> int:
    from launcher import provision

    try:
        accelerator = provision.run(
            on_line=lambda line: print(line, flush=True),
            on_phase=lambda _key, label: print(f"== {label}", flush=True),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"provisioned, accelerator={accelerator or 'unknown'}")
    return 0


def show_status() -> None:
    stored = state.load()
    print(f"provisioned      : {stored['provisioned']}")
    print(f"accelerator      : {stored['accelerator'] or '-'}")
    print(f"service port     : {stored['service_port']}")
    print(f"autostart        : {'on' if autostart.is_enabled() else 'off'}")
    print(f"venv             : {config.venv_dir()} {'(present)' if config.venv_dir().is_dir() else '(absent)'}")
    print(f"llama            : {config.llama_binary()} "
          f"{'(present)' if config.llama_binary().exists() else '(absent)'}")
    print(f"convert tooling  : {config.gguf_tools_dir()} "
          f"{'(present)' if config.gguf_tools_dir().is_dir() else '(absent)'}")
    print(f"model cache      : {config.hf_home()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--purge", action="store_true", help="remove provisioning output and launcher state")
    parser.add_argument("--keep-model-cache", action="store_true",
                        help="leave HF_HOME alone (tens of GB, and irrelevant to which setup path runs)")
    parser.add_argument("--reprovision", action="store_true", help="run setup_env again after purging")
    parser.add_argument("--gui", action="store_true", help="launch the tray application")
    parser.add_argument("--status", action="store_true", help="report what is currently installed")
    parser.add_argument("--dry-run", action="store_true", help="say what would be removed, remove nothing")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="actually delete. Without it --purge only reports, because the paths it "
                             "clears are shared with a hand-installed setup on a developer machine")
    args = parser.parse_args()

    # A purge removes the llama binary and the model cache at their conventional
    # locations, which on a development machine are the ones already in use for
    # everything else. Reporting by default and deleting only when asked keeps this
    # from being a one-typo mistake; CI passes --yes explicitly.
    if args.purge and not (args.yes or args.dry_run):
        print("--purge needs -y/--yes to delete anything. Showing what it would remove:\n")
        args.dry_run = True

    if not any((args.purge, args.reprovision, args.gui, args.status)):
        parser.print_help()
        return 2

    if args.status:
        show_status()

    if args.purge:
        purge(keep_model_cache=args.keep_model_cache, dry_run=args.dry_run)

    if args.reprovision:
        if args.dry_run:
            print("--dry-run: skipping re-provisioning")
        else:
            code = reprovision()
            if code != 0:
                return code

    if args.gui:
        return subprocess.call([sys.executable, "-m", "launcher"], cwd=str(Path(__file__).resolve().parent))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
