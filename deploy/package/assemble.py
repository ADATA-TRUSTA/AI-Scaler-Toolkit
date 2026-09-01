"""Build the public layout from the three sources.

One engine, two callers. The release pipeline clones each source at its recorded
commit and points this at the result; a developer points it at their working copies.
That is deliberate: the alternative is a convenience script that drifts from the
pipeline, and the first time anyone notices is when something the local build excluded
turns up in a release -- or the other way round.

Each target directory is *replaced*, never merged into. See layout.MUST_NOT_EXIST for
what a merge leaves behind and why no positive check catches it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def replace(dest_root: Path) -> None:
    """Wipe and recreate. Done explicitly so the destructive step is visible rather
    than implied by a copy flag."""
    if dest_root.exists():
        layout.force_rmtree(dest_root)
    dest_root.mkdir(parents=True)


def copy_tree(src_root: Path, dest_root: Path, key: str) -> int:
    """Copy the INCLUDE list out of src_root, applying the exclusion rules."""
    copied = 0
    for pattern in layout.INCLUDE[key]:
        for src in sorted(src_root.glob(pattern)):
            rel_top = src.relative_to(src_root).as_posix()
            if layout.is_excluded(rel_top, key):
                continue
            if src.is_file():
                target = dest_root / rel_top
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied += 1
                continue
            for item in sorted(src.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(src_root).as_posix()
                if layout.is_excluded(rel, key):
                    continue
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                copied += 1
    return copied


def assemble_repo(key: str, src_root: Path, dest_root: Path) -> int:
    if not src_root.is_dir():
        raise SystemExit(f"{key}: {src_root} is not a directory")
    replace(dest_root)
    count = copy_tree(src_root, dest_root, key)
    print(f"[assemble] {key}: {count} files from {src_root}")
    return count


def assemble_frontend(src_root: Path, dest_root: Path) -> int:
    """Both packages must be present.

    The Windows installer takes dist_client and the Linux one takes dist, so a layout
    carrying only one of them cannot be packaged for both platforms -- and the failure
    would land in the build stage, long after assembly looked fine.
    """
    missing = [name for name in layout.FRONTEND_PACKAGES if not (src_root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"frontend: {src_root} is missing {', '.join(missing)}. "
            "Run `bun run build:all` in the frontend repo, or point --frontend at a "
            "directory holding both packages."
        )

    replace(dest_root)
    total = 0
    for name in layout.FRONTEND_PACKAGES:
        for item in sorted((src_root / name).rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(src_root).as_posix()
            if layout.is_excluded(rel, "frontend"):
                continue
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            total += 1
    print(f"[assemble] frontend: {total} files ({', '.join(layout.FRONTEND_PACKAGES)}) from {src_root}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # from sibling working copies, for local iteration
  python deploy/package/assemble.py \\
      --backend ../trusta-ast-backend \\
      --console ../trusta-ast-console \\
      --frontend ../trusta-ast-frontend

  # the pipeline does the same, against clones checked out at recorded commits
""",
    )
    # Defaults are the sibling working copies, which is how the repositories are laid
    # out during development. The pipeline always passes explicit paths.
    siblings = REPO_ROOT.parent
    parser.add_argument("--backend", default=str(siblings / "trusta-ast-backend"),
                        help="backend repository root (default: sibling working copy)")
    parser.add_argument("--console", default=str(siblings / "trusta-ast-console"),
                        help="console repository root (default: sibling working copy)")
    parser.add_argument("--frontend", default=str(siblings / "trusta-ast-frontend"),
                        help="directory holding both built frontend packages, dist/ and dist_client/ "
                             "(default: sibling working copy)")
    parser.add_argument("--into", default=str(REPO_ROOT), help="layout to write into")
    parser.add_argument("--skip-assert", action="store_true",
                        help="skip the layout check afterwards (the pipeline runs it as its own step)")
    args = parser.parse_args()

    into = Path(args.into).resolve()
    assemble_repo("backend", Path(args.backend).resolve(), into / "backend")
    assemble_repo("console", Path(args.console).resolve(), into / "console")
    assemble_frontend(Path(args.frontend).resolve(), into / "frontend")

    if args.skip_assert:
        print("[assemble] done (layout not checked)")
        return 0

    sys.argv = [sys.argv[0], "--into", str(into)]
    import assert_layout

    return assert_layout.main()


if __name__ == "__main__":
    raise SystemExit(main())
