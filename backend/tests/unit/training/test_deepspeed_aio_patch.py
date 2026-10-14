"""
The installed DeepSpeed must close the descriptors its async NVMe path opens.

`deepspeed_io_handle_t::wait()` ends each completed operation with a statement
that reads the descriptor and throws the value away:

    if (!completed_op->_filename.empty()) { (completed_op->_fd); }

It compiles, it warns about nothing, and it leaks one descriptor per
asynchronous `pread`/`pwrite` that opened its own file. Measured on 0.19.2 by
issuing 40 async writes through `aio_handle` and counting /proc/self/fd:
exactly 40 descriptors left behind, one per operation. With the statement
changed to `close(...)`, the same 40 operations leak none.

That rate matters because NVMe parameter offload issues one of these per swapped
tensor per rank, so a long run walks into EMFILE and a multi-GPU run gets there
sooner -- every rank spends from the same per-process budget.

This test is a tripwire, not the fix. `scripts/patch_deepspeed_aio.py` is the
fix, `scripts/linux/setup_env.sh` applies it, and `uv sync` undoes it by
replacing site-packages. Without something failing here, the next environment
rebuild would go back to leaking with nothing to say so.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytestmark = pytest.mark.finetune

BROKEN = "if (!completed_op->_filename.empty()) { (completed_op->_fd); }"
FIXED = "if (!completed_op->_filename.empty()) { close(completed_op->_fd); }"
RELATIVE = Path("ops/csrc/aio/py_lib/deepspeed_py_io_handle.cpp")


@pytest.fixture(scope="module")
def aio_source() -> str:
    deepspeed = pytest.importorskip("deepspeed", reason="deepspeed is not installed")
    path = Path(deepspeed.__file__).parent / RELATIVE
    if not path.is_file():
        pytest.skip(f"aio source not present at {path}")
    return path.read_text(encoding="utf-8")


def test_the_async_path_closes_its_descriptor(aio_source):
    """
    Fails after any reinstall that drops the patch, which is the whole point:
    the leak is invisible until a run exhausts its descriptors hours in.
    """
    assert BROKEN not in aio_source, (
        "the installed deepspeed still discards the descriptor instead of closing it; "
        "run `python scripts/patch_deepspeed_aio.py` (uv sync replaces site-packages, "
        "so this comes back every time the environment is rebuilt)"
    )
    assert FIXED in aio_source, (
        "neither the broken statement nor the patched one is present -- the installed "
        "deepspeed differs from the version this was written against (0.19.2). Check "
        "whether upstream fixed it before reapplying the patch."
    )


def test_the_synchronous_paths_were_already_correct(aio_source):
    """
    Pins the asymmetry the patch relies on: the same file closes descriptors on
    its synchronous paths. If those ever stop closing, this patch is too narrow
    and the leak is wider than one statement.
    """
    assert aio_source.count("close(fd);") >= 2, (
        "the synchronous read/write paths no longer close their descriptors; "
        "the aio fd handling has changed more than this patch accounts for"
    )
