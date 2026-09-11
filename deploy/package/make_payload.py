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
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402
import make_icon  # noqa: E402
import vcruntime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8571

# platform -> (source package in the assembled layout, brand it carries)
BRAND_BY_PLATFORM = {
    "windows": ("dist_client", "ADATA"),
    "linux": ("dist", "TRUSTA"),
}

# The console stays in the public layout but not in an installation: nothing here
# runs it, and only the backend gets an environment built for it, so shipping it put
# 67 files a user could not run into a directory they might try to run them from.
PAYLOAD_DIRS = ("backend",)


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


def running_from(payload: Path) -> str:
    """Name a process running out of this payload, or "" if none is obvious.

    Only for the error message. Deciding whether the payload can be replaced is the
    rename's job below, because this cannot see every holder: the backend service runs
    from an interpreter installed elsewhere and merely has its working directory inside
    the payload, which is enough to lock it and invisible to a scan of executable paths.
    """
    if sys.platform != "win32":
        return ""
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath } | "
             "Select-Object -ExpandProperty ExecutablePath"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    target = str(payload).lower()
    for line in result.stdout.splitlines():
        path = line.strip()
        if path and path.lower().startswith(target):
            return path
    return ""


def clear_payload(payload: Path) -> None:
    """Move the old payload aside, then delete it.

    Deleting in place is what went wrong twice: a locked file stops the walk partway
    and leaves a directory that still looks like a payload, passes a glance, and fails
    every assertion afterwards. A rename either succeeds completely or changes nothing,
    so a payload that cannot be replaced is left exactly as it was.

    It also catches holders a process scan cannot see -- anything with a working
    directory or an open file inside, not just something running the executable.
    """
    parked = payload.with_name(f"{payload.name}.old-{int(time.time())}")
    try:
        payload.rename(parked)
    except OSError as exc:
        culprit = running_from(payload)
        raise SystemExit(
            f"[payload] {payload} cannot be replaced: {exc}\n"
            + (f"[payload] {culprit} is running from it.\n" if culprit else "")
            + "[payload] Close the application and stop its backend service, then try "
              "again. Nothing has been deleted."
        ) from None
    layout.force_rmtree(parked)


def check_layout_matches_sources(source: Path, sources: dict[str, Path]) -> None:
    """Refuse a layout that no longer matches the repositories it came from.

    The default layout is the mirror in this repository, which is only current after
    assemble.py has been run into it. Packaging a stale one produces an installer that
    is wrong in a way nothing announces: it builds, it installs, it runs, and it ships
    a backend from whenever the mirror was last refreshed. It happened here -- the
    mirror was 63 files behind the backend -- and what caught it was assert_payload
    noticing the health endpoint had lost keys, which is luck rather than a check.

    Only possible when the source repositories are reachable, so the release pipeline,
    which clones them somewhere temporary and assembles in the same run, is unaffected.
    """
    for key, repo in sources.items():
        if not repo.is_dir():
            continue
        if key == "frontend":
            expected = frontend_expectation(repo)
        else:
            expected = {rel: path.stat().st_size
                        for rel, path in layout.source_files(repo, key).items()}
        actual = layout.assembled_files(source / key)
        if expected == actual:
            continue

        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(rel for rel in set(expected) & set(actual)
                         if expected[rel] != actual[rel])
        detail = []
        for label, items in (("missing", missing), ("changed", changed), ("no longer in the source", extra)):
            if items:
                shown = ", ".join(items[:4])
                more = f" and {len(items) - 4} more" if len(items) > 4 else ""
                detail.append(f"    {len(items)} {label}: {shown}{more}")
        raise SystemExit(
            "\n".join(
                [f"[payload] {source / key} does not match {repo}:"]
                + detail
                + [f"[payload] Run: python package/assemble.py --into {source}",
                   "[payload] Or pass --skip-source-check to package the layout as it is."]
            )
        )


def frontend_expectation(frontend_repo: Path) -> dict[str, int]:
    """The frontend arrives as build output, so both packages are compared verbatim.

    A stale frontend is the same failure wearing different clothes: it builds, it
    installs, and it serves an interface from whenever the mirror was last refreshed.
    """
    out: dict[str, int] = {}
    for package in layout.FRONTEND_PACKAGES:
        for rel, size in layout.assembled_files(frontend_repo / package).items():
            full = f"{package}/{rel}"
            if not layout.is_excluded(full, "frontend"):
                out[full] = size
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(BRAND_BY_PLATFORM), required=True)
    # Where build_installer.py reads from. It used to default to ./payload, which that
    # script does not look at: a default-flag build wrote a payload nobody packaged and
    # then built an installer from whatever the previous run had left behind.
    parser.add_argument("--out", default="",
                        help="payload directory to create (default: build/payload-<platform>)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--source", default=str(REPO_ROOT), help="assembled layout to read from")
    parser.add_argument("--launcher-dir", default="",
                        help="frozen launcher directory from build_launcher.py. Without it the "
                             "payload ships launcher sources, which is fine for development but "
                             "leaves the installer with no executable to shortcut to")
    parser.add_argument("--skip-source-check", action="store_true",
                        help="package the layout even if it no longer matches the source "
                             "repositories beside it")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    payload = Path(args.out).resolve() if args.out else (
        REPO_ROOT / "deploy" / "build" / f"payload-{args.platform}")

    if not args.skip_source_check:
        siblings = REPO_ROOT.parent
        # console is not checked: it is part of the layout, but nothing here copies it
        # into a payload, so a layout whose console has drifted cannot affect the build.
        check_layout_matches_sources(source, {
            "backend": siblings / "trusta-ast-backend",
            "frontend": siblings / "trusta-ast-frontend",
        })
        print(f"[payload] layout at {source} matches the sources beside it")

    if payload.exists():
        clear_payload(payload)
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

    # Built from the brand artwork rather than scavenged from the frontend package.
    # The favicons that ship there carry one or two sizes; Windows draws this at nine.
    make_icon.build(layout.ICON_BRAND, payload / layout.PAYLOAD_ICON)
    make_icon.build_png(layout.ICON_BRAND, payload / layout.PAYLOAD_MARK)

    for name in PAYLOAD_DIRS:
        src = source / name
        if not src.is_dir():
            raise SystemExit(f"{src} not found in the assembled layout")
        count = copy_filtered(src, payload / name)
        print(f"[payload] {name}: {count} files")

    # Windows only: torch and llama.cpp both import MSVCP140 / VCRUNTIME140 /
    # VCRUNTIME140_1, and Windows does not carry them. Shipping them beside the
    # binaries is what removes the administrator prompt that installing Microsoft's
    # redistributable at first run used to raise. See vcruntime.py.
    if args.platform == "windows":
        vcruntime.collect(payload, os.environ.get("VCRUNTIME_DIR"))

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
