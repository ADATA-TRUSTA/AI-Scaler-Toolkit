"""
Unloading a model must not create a CUDA context in the API process.

The API process never allocates on the GPU; the model lives in the worker. Its unload
path still clears "the main process's" CUDA cache, and when that was gated on
is_available() the synchronize() call initialized CUDA, leaving a 128 MiB context on
the card for the rest of the service's life. A vLLM restart in the KV-offload e2e then
missed its gpu_memory_utilization=0.9 budget by 10 MB.

A fake torch module stands in for the real one, so no GPU is needed and the test sees
exactly which CUDA calls were made.
"""

import sys
from types import SimpleNamespace

import pytest

from service.inference import model_inference_process as mip

pytestmark = pytest.mark.inference


def _fake_torch(*, initialized: bool) -> tuple[SimpleNamespace, list[str]]:
    """Return a torch stand-in whose CUDA calls are recorded, and the call log."""
    calls: list[str] = []

    def record(name: str, result: object = None):  # noqa: ANN202 - test helper
        def call(*_args: object) -> object:
            calls.append(name)
            return result

        return call

    cuda = SimpleNamespace(
        is_available=record("is_available", True),
        is_initialized=record("is_initialized", initialized),
        empty_cache=record("empty_cache"),
        synchronize=record("synchronize"),
        memory_allocated=record("memory_allocated", 0),
    )
    return SimpleNamespace(cuda=cuda), calls


def test_uninitialized_cuda_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CUDA never initialized here, nothing that could initialize it is called."""
    torch, calls = _fake_torch(initialized=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    mip._clear_main_process_cuda_cache()

    assert "synchronize" not in calls
    assert "empty_cache" not in calls


def test_initialized_cuda_is_still_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    """If something in this process did use CUDA, its cache is cleared as before."""
    torch, calls = _fake_torch(initialized=True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    mip._clear_main_process_cuda_cache()

    assert calls[-3:] == ["empty_cache", "synchronize", "memory_allocated"]


def test_torch_is_not_imported_to_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without torch already loaded, the check neither imports nor touches it."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    mip._clear_main_process_cuda_cache()

    assert "torch" not in sys.modules
