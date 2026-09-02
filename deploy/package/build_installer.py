"""Compile the Windows installer from the payload.

Locating the compiler is the fiddly part. Inno Setup does not put ISCC on PATH, so
the existing build machine invokes it by absolute path -- which means a PATH-only
check reports it missing on the one machine that definitely has it. This looks in
PATH, in an override variable, and in the places the installer actually puts it.

Nothing here installs anything. A build that provisions its own toolchain stops
being reproducible: whichever version the package manager happened to serve that day
ends up baked into the output, with nothing in the log to say so. When the compiler
is absent the right outcome is a message naming the one command to run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_DIR.parent

# Where the Inno Setup installer puts the compiler, per-machine then per-user. Neither
# is on PATH.
CANDIDATES = (
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
)

INSTALL_HINT = """Inno Setup's compiler (ISCC) was not found.

  winget install JRSoftware.InnoSetup --scope user

Or point ISCC at it if it is installed somewhere else:

  ISCC=D:\\tools\\InnoSetup\\ISCC.exe

Searched: PATH, the ISCC variable, and
""" + "\n".join(f"  {path}" for path in CANDIDATES)

# It fails intermittently on the build machine; this is not defensive padding.
ATTEMPTS = 3
RETRY_SECONDS = 5


def find_iscc() -> Path | None:
    """The Inno Setup compiler, or None.

    Order matters: an explicit ISCC beats a discovered one, so a machine with two
    versions installed can be told which to use without editing anything.
    """
    override = os.environ.get("ISCC", "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None

    on_path = shutil.which("iscc") or shutil.which("ISCC")
    if on_path:
        return Path(on_path)

    for candidate in CANDIDATES:
        path = Path(os.path.expandvars(candidate))
        if path.is_file():
            return path

    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe"
    return local if local.is_file() else None


def compile_installer(iscc: Path, script: Path, payload: Path, version: str, out_dir: Path) -> Path:
    command = [
        str(iscc),
        f"/DPayloadDir={payload}",
        f"/DAppVersion={version}",
        f"/O{out_dir}",
        str(script),
    ]
    print(f"[installer] {' '.join(command)}")

    for attempt in range(1, ATTEMPTS + 1):
        result = subprocess.run(command, cwd=script.parent)
        if result.returncode == 0:
            break
        print(f"[installer] attempt {attempt} failed with exit {result.returncode}")
        if attempt == ATTEMPTS:
            raise SystemExit(f"iscc failed {ATTEMPTS} times")
        time.sleep(RETRY_SECONDS)

    built = sorted(out_dir.glob("*.exe"), key=lambda p: p.stat().st_mtime)
    if not built:
        # iscc has been known to report success and produce nothing.
        raise SystemExit(f"iscc reported success but produced no installer in {out_dir}")
    return built[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", default=str(DEPLOY_DIR / "build" / "payload-windows"),
                        help="the payload directory to package")
    parser.add_argument("--version", default="0.0.0", help="version stamped into the installer")
    parser.add_argument("--out", default=str(DEPLOY_DIR / "build" / "installer"),
                        help="where to put the built installer")
    parser.add_argument("--script", default=str(DEPLOY_DIR / "windows" / "installer.iss"))
    args = parser.parse_args()

    iscc = find_iscc()
    if iscc is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    print(f"[installer] compiler: {iscc}")

    payload = Path(args.payload).resolve()
    if not (payload / "AIScalerToolkit.exe").is_file():
        raise SystemExit(
            f"{payload} has no AIScalerToolkit.exe -- run build_launcher.py and "
            "make_payload.py --launcher-dir first"
        )

    script = Path(args.script).resolve()
    if not script.is_file():
        raise SystemExit(f"installer script not found: {script}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    built = compile_installer(iscc, script, payload, args.version, out_dir)
    print(f"[installer] built {built} ({built.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
