"""
A failed load keeps its traceback in /inference/error_details, OOM included.

An OOM or CUDA error makes the worker report a second error and exit to give the VRAM
back. The manager keeps only the last error it receives, and that second message used
to carry no traceback, so error_details had none for exactly the failures where it
matters (measured: a Qwen3-14B bf16 load on a 16 GB card came back with null).
"""

import queue
from types import SimpleNamespace

import pytest

from service.inference.model_inference_process import (
    ModelInferenceProcess,
    _load_failure_statuses,
)

pytestmark = pytest.mark.inference

TRACEBACK = 'Traceback (most recent call last):\n  File "transformers_engine.py"\n'


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError, which the API process does not import."""


def _error_details_after(statuses):
    mip = ModelInferenceProcess()
    mip.status_queue = queue.Queue()
    for status in statuses:
        mip.status_queue.put(status)
    mip.process = SimpleNamespace(is_alive=lambda: False)
    return mip.get_error_details()


def test_an_oom_reports_twice_and_both_carry_the_traceback():
    statuses = _load_failure_statuses(
        OutOfMemoryError("CUDA out of memory. Tried to allocate 14.27 GiB"), TRACEBACK
    )

    assert len(statuses) == 2
    assert all(s["error_traceback"] == TRACEBACK for s in statuses)
    assert statuses[-1]["is_oom"] is True


def test_error_details_keep_the_traceback_of_an_oom():
    statuses = _load_failure_statuses(
        OutOfMemoryError("CUDA out of memory. Tried to allocate 14.27 GiB"), TRACEBACK
    )

    details = _error_details_after(statuses)

    assert details["is_oom"] is True
    assert details["error_type"] == "OutOfMemoryError"
    assert details["error"].startswith("OOM Error: CUDA out of memory")
    assert details["error_traceback"] == TRACEBACK


def test_any_other_failure_reports_once():
    statuses = _load_failure_statuses(OSError("no such model"), TRACEBACK)

    assert len(statuses) == 1
    details = _error_details_after(statuses)
    assert details["is_oom"] is False
    assert details["error_type"] == "OSError"
    assert details["error_traceback"] == TRACEBACK
