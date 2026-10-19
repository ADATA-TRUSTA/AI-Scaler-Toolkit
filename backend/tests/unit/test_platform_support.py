"""
Platform gating for the two Linux-only features.

Both refusals are asserted through the endpoints rather than only against the
helper, because the point of the change is *where* the refusal happens: before
the training worker is spawned and before the vLLM engine is constructed. A test
that only exercised platform_support directly would still pass if the guard were
dropped from app.py.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import service.app as app_module
from service import platform_support
from service.config_models import InferenceEngine

# As this machine: see test_remote_access.py.
client = TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


def _training_payload(**overrides):
    payload = {
        "model_name": "test-model",
        "method": "lora",
        "dataset_path": "data.jsonl",
        "output_dir": "out",
    }
    payload.update(overrides)
    return payload


def _inference_payload(**overrides):
    payload = {
        "model_name": "test-model",
        "quantization": "none",
        "device_map": "cpu",
    }
    payload.update(overrides)
    return payload


def _pretend(monkeypatch, system: str) -> None:
    """Report a given OS from platform.system(), as platform_support reads it."""
    monkeypatch.setattr(platform_support.platform, "system", lambda: system)


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_training_is_refused_off_linux(monkeypatch, system):
    _pretend(monkeypatch, system)
    support = platform_support.training_support()

    assert not support.supported
    # The message has to name both the requirement and this host, or the reader
    # cannot tell what to change.
    assert "Linux" in (support.reason or "")
    assert system in (support.reason or "")


def test_training_is_allowed_on_linux(monkeypatch):
    _pretend(monkeypatch, "Linux")

    assert platform_support.training_support().supported
    assert platform_support.training_support().reason is None


def _vllm(monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    """Say whether vLLM is installed, so a test does not depend on the machine running it."""
    monkeypatch.setattr(platform_support, "vllm_installed", lambda: installed)


def test_wsl_counts_as_linux(monkeypatch):
    """
    WSL is not special-cased anywhere, and must not need to be.

    Under WSL platform.system() is "Linux", and both features do work there once
    the GPU runtime is present — so the generic Linux branch is the correct answer.
    """
    _pretend(monkeypatch, "Linux")
    _vllm(monkeypatch, installed=True)

    assert platform_support.training_support().supported
    assert InferenceEngine.VLLM in platform_support.supported_engines()


def test_vllm_is_refused_on_linux_when_it_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Linux is necessary and not sufficient.

    A host set up without vLLM (TRUSTA_SETUP_VLLM=0) runs Linux and has none.
    Offering it there gave an engine picker with an entry whose every load failed.
    """
    _pretend(monkeypatch, "Linux")
    _vllm(monkeypatch, installed=False)

    support = platform_support.engine_support(InferenceEngine.VLLM)
    assert not support.supported
    assert "not installed" in (support.reason or "")
    assert InferenceEngine.VLLM not in platform_support.supported_engines()
    # The others are untouched by it.
    assert InferenceEngine.LLAMA_SERVER in platform_support.supported_engines()


def test_windows_refusal_names_the_platform_whether_or_not_vllm_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off Linux the platform is the reason, and a stray install does not change that."""
    _pretend(monkeypatch, "Windows")
    _vllm(monkeypatch, installed=True)

    support = platform_support.engine_support(InferenceEngine.VLLM)
    assert not support.supported
    assert "Linux" in (support.reason or "")


@pytest.mark.parametrize("engine", [InferenceEngine.TRANSFORMERS, InferenceEngine.LLAMA_SERVER])
def test_cross_platform_engines_are_never_refused(monkeypatch, engine):
    _pretend(monkeypatch, "Windows")

    assert platform_support.engine_support(engine).supported


def test_only_vllm_is_dropped_on_windows(monkeypatch):
    _pretend(monkeypatch, "Windows")

    assert platform_support.supported_engines() == [
        InferenceEngine.TRANSFORMERS,
        InferenceEngine.LLAMA_SERVER,
    ]


# --------------------------------------------------------------------------
# The endpoints
# --------------------------------------------------------------------------


def test_training_start_refuses_before_spawning_a_worker(monkeypatch):
    """A 501 must come back without training_manager being touched at all."""
    _pretend(monkeypatch, "Windows")

    def fail(*args, **kwargs):
        raise AssertionError("training_manager.start_training must not be reached")

    monkeypatch.setattr(app_module.training_manager, "start_training", fail)

    response = client.post("/training/start", json=_training_payload())

    assert response.status_code == 501
    assert "Linux" in response.json()["detail"]


def test_training_start_proceeds_on_linux(monkeypatch):
    _pretend(monkeypatch, "Linux")
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False, raising=False)
    monkeypatch.setattr(
        app_module.training_manager,
        "start_training",
        lambda config: {"status": "starting", "session_id": "job-1"},
    )

    response = client.post("/training/start", json=_training_payload())

    assert response.status_code == 200
    assert response.json()["result"]["session_id"] == "job-1"


def test_load_model_refuses_vllm_off_linux(monkeypatch):
    """The engine is rejected before the worker process is started."""
    _pretend(monkeypatch, "Windows")

    def fail(*args, **kwargs):
        raise AssertionError("model_manager.start_loading must not be reached")

    # The real method is start_loading -- ModelManager has no load_model, so
    # patching that name (previously with raising=False, which let the typo
    # through silently) never actually intercepted anything. This assertion
    # happened to still pass because the platform gate refuses before the
    # manager either way, but it was not testing what it claimed to.
    monkeypatch.setattr(app_module.model_manager, "start_loading", fail)

    response = client.post("/inference/load_model", json=_inference_payload(engine="vllm"))

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "vLLM" in detail
    assert "Windows" in detail


def test_load_model_allows_llama_server_off_linux(monkeypatch):
    """Only vLLM is platform-bound; the other engines must stay reachable."""
    _pretend(monkeypatch, "Windows")
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    # Same fix as above: this used to patch a nonexistent "load_model" attribute
    # (raising=False let the typo through), so the request actually fell through
    # to the real start_loading() -- spawning a real worker process against
    # _inference_payload()'s made-up "test-model" with no model_path, which
    # failed inside the worker ("llama model file not found: test-model") and
    # left that error sitting in the process-wide ModelManager singleton for
    # every test that ran after this one in the same pytest session (see
    # tests/smoke/test_transformers_worker_smoke.py, which inherited it).
    monkeypatch.setattr(app_module.model_manager, "start_loading", lambda config: True)

    response = client.post("/inference/load_model", json=_inference_payload(engine="llama_server"))

    assert response.status_code != 501


# --------------------------------------------------------------------------
# /health, which is what clients gate their UI on
# --------------------------------------------------------------------------


def test_health_reports_what_this_host_supports(monkeypatch):
    _pretend(monkeypatch, "Windows")
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    monkeypatch.setattr(
        app_module.training_manager,
        "get_status",
        lambda: SimpleNamespace(is_training=False),
    )

    body = client.get("/health").json()

    assert body["platform"] == "Windows"
    assert body["training_supported"] is False
    assert body["supported_engines"] == ["transformers", "llama_server"]


def test_health_reports_everything_supported_on_linux(monkeypatch):
    _pretend(monkeypatch, "Linux")
    _vllm(monkeypatch, installed=True)  # whether this host really has it is not the point
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    monkeypatch.setattr(
        app_module.training_manager,
        "get_status",
        lambda: SimpleNamespace(is_training=False),
    )

    body = client.get("/health").json()

    assert body["platform"] == "Linux"
    assert body["training_supported"] is True
    assert "vllm" in body["supported_engines"]
