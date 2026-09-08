#!/usr/bin/env python
"""
Close the descriptors DeepSpeed's async NVMe path opens and never releases.

``deepspeed_io_handle_t::wait()`` ends each completed operation with:

    if (!completed_op->_filename.empty()) { (completed_op->_fd); }

which reads the descriptor and discards the value. It is a statement with no
effect, so an asynchronous ``pread``/``pwrite`` that opened its own file leaks
one descriptor per operation -- measured at exactly 1:1 on deepspeed 0.19.2.
The synchronous paths in the same file close theirs, and the guard on
``_filename`` is precisely "this operation owns a descriptor we opened from a
name", so the intended statement is ``close(...)``.

It matters because NVMe parameter offload issues these per swapped tensor, on
every rank. A single-GPU job walks into ``EMFILE`` on a long run; a multi-GPU
one gets there sooner, with every rank spending from the same per-process
budget.

Reapply this after anything that reinstalls deepspeed -- ``uv sync`` replaces
site-packages wholesale. ``tests/unit/training/test_deepspeed_aio_patch.py``
fails loudly when the installed copy is unpatched, so a fresh environment does
not quietly go back to leaking.

Editing the source is enough to trigger a rebuild: the aio op is JIT-compiled
through ninja, which rebuilds on mtime. The stale build directory is removed
anyway so the next load cannot serve an old object.

    python scripts/patch_deepspeed_aio.py [--check]

Exit codes: 0 patched (or already patched), 1 the expected line was not found.
"""

import argparse
import shutil
import sys
from pathlib import Path

RELATIVE = Path("ops/csrc/aio/py_lib/deepspeed_py_io_handle.cpp")
BROKEN = "if (!completed_op->_filename.empty()) { (completed_op->_fd); }"
FIXED = "if (!completed_op->_filename.empty()) { close(completed_op->_fd); }"


def source_path() -> Path:
    """Locate the aio handle source inside the installed deepspeed."""
    import deepspeed  # pyright: ignore[reportMissingImports]  # linux-only optional dep

    # Only a namespace package has no __file__, which a real deepspeed install never is.
    origin = deepspeed.__file__
    if origin is None:
        raise RuntimeError("deepspeed has no __file__, so its source tree cannot be located")
    return Path(origin).parent / RELATIVE


def build_dirs() -> list[Path]:
    """JIT build directories that may hold an object built from the old source."""
    root = Path.home() / ".cache" / "torch_extensions"
    if not root.is_dir():
        return []
    return [p for p in root.glob("*/async_io") if p.is_dir()]


def main() -> int:
    """Apply the close(), or report what is in the way, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the state and exit without writing",
    )
    args = parser.parse_args()

    path = source_path()
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")

    if FIXED in text:
        print(f"already patched: {path}")
        return 0

    if BROKEN not in text:
        # Upstream may have fixed this, or moved the line. Either way, guessing
        # would be worse than stopping: say what was expected and let a human look.
        print(
            f"expected statement not found in {path}\n"
            f"  looked for: {BROKEN}\n"
            "The installed deepspeed differs from the version this patch was written "
            "against (0.19.2). Check whether upstream has fixed it before forcing.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(f"UNPATCHED: {path}")
        return 1

    path.write_text(text.replace(BROKEN, FIXED), encoding="utf-8")
    print(f"patched: {path}")

    for d in build_dirs():
        shutil.rmtree(d, ignore_errors=True)
        print(f"cleared stale build: {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
