"""Verify the assembled tree before anything downstream trusts it.

The negative half matters more than the positive half. If a target directory was
merged into rather than replaced, every positive check still passes -- the tree just
quietly carries both the new package and the old modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files without which provisioning has nothing to sync, so a release built from this
# tree would fail on the user's machine rather than in CI.
SOURCE_MUST_EXIST: tuple[str, ...] = (
    "backend/pyproject.toml",
    "backend/uv.lock",
    "backend/.python-version",
    "backend/.env.example",
    "backend/scripts/windows/setup_env.ps1",
    "backend/scripts/linux/setup_env.sh",
    "backend/service/app.py",
    "console/pyproject.toml",
    "console/uv.lock",
    "frontend/dist/index.html",
    "frontend/dist_client/index.html",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", default=str(REPO_ROOT), help="layout to check")
    args = parser.parse_args()
    root = Path(args.into).resolve()

    failures: list[str] = []

    for rel in layout.ROOT_MUST_EXIST:
        if not (root / rel).exists():
            failures.append(f"root content went missing: {rel}")

    for rel in SOURCE_MUST_EXIST:
        if not (root / rel).exists():
            failures.append(f"assembled tree is incomplete: {rel}")

    for rel in layout.MUST_NOT_EXIST:
        if (root / rel).exists():
            failures.append(
                f"stale layout survived assembly: {rel} -- the target was merged into, not replaced"
            )

    # A secret reaching the public mirror is unrecoverable once pushed.
    for key in ("backend", "console"):
        leaked = root / key / ".env"
        if leaked.exists():
            failures.append(f"{key}/.env must never ship (only .env.example does)")

    internal = root / "console" / "internal"
    if internal.exists():
        failures.append("console/internal must not reach the public layout")

    if failures:
        print("[assert] FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1

    print(f"[assert] ok ({len(layout.ROOT_MUST_EXIST)} root, {len(SOURCE_MUST_EXIST)} source, "
          f"{len(layout.MUST_NOT_EXIST)} negative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
