"""Turn the launcher into something the installer can create a shortcut to.

A shortcut has to point at an executable with an icon and no console window, so the
launcher is frozen rather than shipped as scripts. Only the launcher is frozen -- the
backend stays a normal uv project, because its environment is built on the user's
machine against their accelerator and cannot be prebuilt here.

What went in is recorded by deploy/uv.lock, which is committed and is what uv builds
the environment from. A requirements.txt used to be exported beside the launcher
sources as well -- from that same lock, ignored by git, never shipped, and read by
nothing. It was a copy of the record rather than the record, kept next to source code
where it looked like one more thing to maintain.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402
import make_icon  # noqa: E402

DEPLOY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_DIR.parent
EXE_NAME = "AIScalerToolkit"

# Bundled as data because they are read at runtime by path, and PyInstaller's import
# analysis cannot see either of them.
DATA = (
    ("launcher/ui", "launcher/ui"),
)


def run(*args: str, cwd: Path) -> None:
    print(f"[launcher] $ {' '.join(args)}")
    proc = subprocess.run(args, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit {proc.returncode}: {' '.join(args)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEPLOY_DIR / "build" / "launcher"),
                        help="where to put the built executable")
    parser.add_argument("--icon", default="", help="override the icon file")
    parser.add_argument("--source", default=str(REPO_ROOT),
                        help="assembled layout to take the icon from (default: this repository)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The same icon the payload ships, so the executable, its shortcut and the tray
    # cannot disagree about which mark this build wears.
    icon = Path(args.icon).resolve() if args.icon else make_icon.build(
        layout.ICON_BRAND, out_dir / f"{layout.ICON_BRAND}.ico"
    )
    if not icon.is_file():
        raise SystemExit(f"icon not found: {icon}")

    separator = ";" if sys.platform == "win32" else ":"
    command = [
        "uv", "run", "--", "pyinstaller",
        "--name", EXE_NAME,
        # onefile unpacks to a temp directory on every launch, and this one starts at login
        "--onedir",
        "--noconsole",
        "--noconfirm",
        "--clean",
        "--distpath", str(out_dir),
        "--workpath", str(work_dir),
        "--specpath", str(work_dir),
    ]
    if icon:
        command += ["--icon", str(icon)]
    for source, dest in DATA:
        command += ["--add-data", f"{DEPLOY_DIR / source}{separator}{dest}"]
    # launcher_main.py, not launcher/__main__.py: freezing the latter runs it outside
    # its package and breaks every relative import.
    command += [str(DEPLOY_DIR / "launcher_main.py")]

    run(*command, cwd=DEPLOY_DIR)

    built = out_dir / EXE_NAME
    exe = built / (f"{EXE_NAME}.exe" if sys.platform == "win32" else EXE_NAME)
    if not exe.is_file():
        raise SystemExit(f"PyInstaller reported success but {exe} is missing")

    print(f"[launcher] built {exe} ({exe.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[launcher] pass --launcher-dir {built} to make_payload.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
