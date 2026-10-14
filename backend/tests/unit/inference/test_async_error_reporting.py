"""
Getting a server-engine failure to the UI intact.

/inference/error_details is what the frontend reads to decide whether to say "reduce
the context length" or "retry". Its source is the worker process, which records its
own crashes — but a server engine (llama-server / vLLM) generates over direct HTTP
and never goes through the worker, so a failure there left the worker with nothing to
report and the classification AsyncServerClient had already computed went nowhere.

Two halves are covered here: the classification itself has to be right, and it has to
reach the endpoint.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.inference.async_server_client import (  # noqa: E402
    _as_runtime_error,
    classify_server_error,
)

pytestmark = pytest.mark.inference


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory",
        "ggml_backend_alloc_buffer: failed to allocate buffer",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "server reported OOM while processing the request",
    ],
)
def test_a_real_oom_is_named(message):
    payload = classify_server_error(RuntimeError(message))

    assert payload["is_oom"] is True
    assert payload["recoverable"] is False


@pytest.mark.parametrize(
    "message",
    [
        # "oom" used to be matched as a bare substring, so each of these was reported
        # as an out-of-memory failure and marked unrecoverable.
        "Connection to zoom.us timed out",
        "no room left in the queue",
        "failed to load TheBloke/Zoomer-7B-GGUF",
    ],
)
def test_the_letters_oom_inside_a_word_are_not_an_oom(message):
    """
    A misfire is not cosmetic.

    recoverable=False tells the caller to stop retrying and tell the user to shrink
    the context — the wrong advice, and an unrecoverable verdict, for what is really
    a transient network fault.
    """
    payload = classify_server_error(RuntimeError(message))

    assert payload["is_oom"] is False
    assert payload["recoverable"] is True


def test_a_dropped_socket_is_recoverable():
    payload = classify_server_error(RuntimeError("Server disconnected without sending a response"))

    assert payload["error_type"] == "LlamaServerDisconnected"
    assert payload["is_oom"] is False
    assert payload["recoverable"] is True


# --------------------------------------------------------------------------
# Delivery
# --------------------------------------------------------------------------


class _StubWorker:
    """Stands in for the worker process, which knows nothing about the async path."""

    def __init__(self, details=None):
        self._details = details

    def get_error_details(self):
        return self._details


def _manager():
    from service.model_manager import ModelManager

    manager = ModelManager()
    manager._last_async_error = None
    return manager


def test_an_async_failure_reaches_the_endpoint(monkeypatch):
    """Without this the UI showed the opaque string the classification exists to replace."""
    manager = _manager()
    monkeypatch.setattr(manager, "inference_process", _StubWorker(None))

    error = _as_runtime_error(RuntimeError("CUDA error: out of memory"))
    manager._record_async_error(error)

    details = manager.get_error_details()
    assert details is not None
    assert details["is_oom"] is True
    assert details["error_type"] == "LlamaServerOOM"


def test_the_worker_still_wins_when_it_has_something(monkeypatch):
    """
    The worker's payload is the richer one — exit code, stderr, a live port probe — so
    a transformers-engine crash must not be masked by a stale async classification.
    """
    manager = _manager()
    worker_details = {"error": "worker crashed", "error_type": "ProcessExited"}
    monkeypatch.setattr(manager, "inference_process", _StubWorker(worker_details))
    manager._record_async_error(_as_runtime_error(RuntimeError("out of memory")))

    assert manager.get_error_details() == worker_details


def test_an_unclassified_exception_records_nothing(monkeypatch):
    """Only AsyncServerClient attaches a payload; anything else must not invent one."""
    manager = _manager()
    monkeypatch.setattr(manager, "inference_process", _StubWorker(None))

    manager._record_async_error(RuntimeError("plain error with no payload"))

    assert manager.get_error_details() is None
