"""Build the installer payload for one platform.

The brand is decided here, by picking one of the two frontend packages and copying it
to the single path the backend looks for. The backend serves whatever sits at
`frontend/dist`, so once the payload holds exactly one package there is nothing left
to pick wrong at runtime.

Doing it with an environment variable instead was rejected: the variable only applies
when the launcher spawns the service itself. Every other way the service can start --
run_service.bat, a restart by the service manager, a hand-run uvicorn -- would quietly
fall back to the other brand, and nothing would report it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8571

# platform -> (source package in the assembled layout, brand it carries)
BRAND_BY_PLATFORM = {
    "windows": ("dist_client", "ADATA"),
    "linux": ("dist", "TRUSTA"),
}

PAYLOAD_DIRS = ("backend", "console")


def copy_filtered(src: Path, dest: Path) -> int:
    count = 0
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        parts = item.relative_to(src).parts
        if any(p in {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache"} for p in parts):
            continue
        target = dest / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        count += 1
    return count


def rewrite_config(config_path: Path, port: int) -> None:
    """Two keys, one write.

    apiBaseUrl has to agree with SERVICE_PORT because the frontend reads it at runtime
    and hands it straight to axios as the base URL. hideChatPanel is what turns the
    full WebUI into the settings-only view the desktop window shows.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["apiBaseUrl"] = f"http://127.0.0.1:{port}"
    config["hideChatPanel"] = True
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[payload] config.json: apiBaseUrl=http://127.0.0.1:{port}, hideChatPanel=true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(BRAND_BY_PLATFORM), required=True)
    parser.add_argument("--out", default="payload", help="payload directory to create")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--source", default=str(REPO_ROOT), help="assembled layout to read from")
    parser.add_argument("--launcher-dir", default="",
                        help="frozen launcher directory from build_launcher.py. Without it the "
                             "payload ships launcher sources, which is fine for development but "
                             "leaves the installer with no executable to shortcut to")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    payload = Path(args.out).resolve()
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True)

    package, brand = BRAND_BY_PLATFORM[args.platform]
    src_dist = source / "frontend" / package
    if not src_dist.is_dir():
        raise SystemExit(f"{src_dist} not found; the assembled layout must carry both frontend packages")

    dest_dist = payload / "frontend" / "dist"
    copied = copy_filtered(src_dist, dest_dist)
    print(f"[payload] frontend: {package} -> frontend/dist ({copied} files, {brand})")

    config_path = dest_dist / "config.json"
    if not config_path.is_file():
        raise SystemExit("config.json missing from the frontend package")
    rewrite_config(config_path, args.port)

    # Support can tell which package a payload carries without unpacking the bundle;
    # the directory is called dist either way.
    (dest_dist / "BRAND.txt").write_text(f"{brand}\nsource: frontend/{package}\n", encoding="utf-8")

    for name in PAYLOAD_DIRS:
        src = source / name
        if not src.is_dir():
            raise SystemExit(f"{src} not found in the assembled layout")
        count = copy_filtered(src, payload / name)
        print(f"[payload] {name}: {count} files")

    if args.launcher_dir:
        launcher_build = Path(args.launcher_dir).resolve()
        if not launcher_build.is_dir():
            raise SystemExit(f"--launcher-dir {launcher_build} is not a directory")
        # The frozen build goes to the payload root so the installer's shortcut target
        # is simply {app}/AIScalerToolkit.exe.
        count = copy_filtered(launcher_build, payload)
        print(f"[payload] launcher (frozen): {count} files")
    else:
        count = copy_filtered(REPO_ROOT / "deploy" / "launcher", payload / "launcher")
        print(f"[payload] launcher (sources, no executable): {count} files")

    print(f"[payload] built {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
