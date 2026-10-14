"""
Behaviour when the Hub cannot be reached.

Reading the config from the Hub is what makes the accurate path work for a model
that has not been downloaded yet — the endpoint's main use case. But the same change
made an air-gapped host wait: huggingface_hub retries five times with backoff, and
one un-downloaded model measured 159 seconds before the estimator gave up and
guessed from the name. It could have guessed immediately.

Reachability is therefore probed once per process and remembered, HF_HUB_OFFLINE is
honoured before any socket is opened, and the safetensors cross-check — an accuracy
refinement, not a source of truth — is skipped rather than waited on.

That process-level probe cannot cover a link that dies after it answered, so the
remote fetch is additionally gated on a bounded check of the file itself. Without
it the 159 seconds come straight back, because huggingface_hub cannot be given a
deadline from outside.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import service.inference.memory_estimator  # noqa: E402, F401  (registers the module)
from service.inference.memory_estimator import MemoryEstimator  # noqa: E402

# service.inference re-exports an instance under the same name, shadowing the module
# for attribute lookups; reach the module through sys.modules.
ME = sys.modules["service.inference.memory_estimator"]

pytestmark = pytest.mark.inference


@pytest.fixture(autouse=True)
def _forget_reachability(monkeypatch):
    """The probe result is process-wide; each test starts without one."""
    monkeypatch.setattr(ME, "_hub_reachable", None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


@pytest.fixture
def estimator():
    return MemoryEstimator()


def test_hf_hub_offline_is_honoured_without_touching_the_network(monkeypatch):
    """The standard opt-out must not cost even the probe."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fail(*args, **kwargs):
        raise AssertionError("no request may be made when HF_HUB_OFFLINE is set")

    import httpx2

    monkeypatch.setattr(httpx2, "head", fail)

    assert ME._hub_is_reachable() is False


def test_reachability_is_probed_once_per_process(monkeypatch):
    """
    Otherwise every model pays the timeout again.

    The cost that matters is per-estimate, and a service answering a page full of
    models would multiply it.
    """
    calls = []

    import httpx2

    monkeypatch.setattr(httpx2, "head", lambda *a, **k: calls.append(1))

    ME._hub_is_reachable()
    ME._hub_is_reachable()
    ME._hub_is_reachable()

    assert len(calls) == 1


def test_an_unreachable_hub_is_remembered_as_such(monkeypatch):
    import httpx2

    def refuse(*args, **kwargs):
        raise httpx2.ConnectError("no route to host")

    monkeypatch.setattr(httpx2, "head", refuse)

    assert ME._hub_is_reachable() is False
    assert ME._hub_reachable is False


def test_the_remote_config_attempt_is_skipped_when_offline(estimator, monkeypatch):
    """
    The 159-second case: only the local lookup should run.

    transformers is asked exactly once, with local_files_only=True, so nothing waits
    on a socket that will never answer.
    """
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: False)
    attempts = []

    def record(name, **kwargs):
        attempts.append(kwargs.get("local_files_only"))
        raise OSError("not in the local cache")

    monkeypatch.setattr(ME.AutoConfig, "from_pretrained", record)

    assert estimator._load_model_config("org/not-downloaded") is None
    assert attempts == [True]


def test_the_remote_config_attempt_runs_when_the_hub_answers(estimator, monkeypatch):
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: True)
    monkeypatch.setattr(ME, "_remote_config_available", lambda name, **k: True)
    attempts = []

    def record(name, **kwargs):
        attempts.append(kwargs.get("local_files_only"))
        if kwargs.get("local_files_only"):
            raise OSError("not in the local cache")
        return object()

    monkeypatch.setattr(ME.AutoConfig, "from_pretrained", record)

    assert estimator._load_model_config("org/not-downloaded") is not None
    assert attempts == [True, False]


def test_the_cross_check_is_skipped_when_offline(estimator, monkeypatch):
    """Accuracy refinement, not a source of truth — drop it rather than wait."""
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: False)

    import httpx2

    monkeypatch.setattr(
        httpx2, "get", lambda *a, **k: pytest.fail("no request when the Hub is unreachable")
    )

    assert estimator._count_via_safetensors_index("org/model") is None


def test_a_mirror_endpoint_is_respected(estimator, monkeypatch):
    """
    HF_ENDPOINT points an on-premise install at its own mirror.

    Hardcoding huggingface.co would send the cross-check somewhere the deployment
    cannot reach, which on an isolated network is the same failure as being offline.
    """
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.internal.example")
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: True)
    urls = []

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"safetensors": {"total": 7_000_000_000}}

    import httpx2

    monkeypatch.setattr(httpx2, "get", lambda url, **k: (urls.append(url), _Response())[1])

    assert estimator._count_via_safetensors_index("org/model") == pytest.approx(7.0)
    assert urls == ["https://hf.internal.example/api/models/org/model"]


# --------------------------------------------------------------------------
# A link that dies between the probe and the fetch
# --------------------------------------------------------------------------


def test_the_remote_fetch_is_gated_on_a_bounded_check(estimator, monkeypatch):
    """
    The process-level probe cannot cover a link that drops after it answered.

    Measured: with the probe cached as reachable and the endpoint then unroutable,
    transformers took 159 seconds — five retries, each waiting on the OS connect
    timeout, which HF_HUB_ETAG_TIMEOUT does not shorten. Asking about the file
    ourselves first, with a timeout we control, decides it in three.
    """
    monkeypatch.setattr(ME, "_hub_reachable", True)
    monkeypatch.setattr(ME, "_remote_config_available", lambda name, **k: False)
    attempts = []

    def record(name, **kwargs):
        attempts.append(kwargs.get("local_files_only"))
        raise OSError("not in the local cache")

    monkeypatch.setattr(ME.AutoConfig, "from_pretrained", record)

    assert estimator._load_model_config("org/model") is None
    assert attempts == [True], "transformers must not be asked once the check has failed"


def test_a_failed_fetch_stops_later_models_from_waiting(monkeypatch):
    """One unreliable link is enough; the next model should not pay for it again."""
    import httpx2

    monkeypatch.setattr(ME, "_hub_reachable", True)
    monkeypatch.setattr(
        httpx2, "head", lambda *a, **k: (_ for _ in ()).throw(httpx2.ConnectError("gone"))
    )

    assert ME._remote_config_available("org/model") is False
    assert ME._hub_reachable is False


def test_a_refused_repo_does_not_mark_the_hub_down(monkeypatch):
    """
    A gated repo answers 401 — the network is fine, only this model is out of reach.

    Treating that as "the Hub is down" would drop every later model to name guessing
    over one model the deployment simply has no token for.
    """
    import httpx2

    monkeypatch.setattr(ME, "_hub_reachable", True)

    class _Refused:
        status_code = 401

    monkeypatch.setattr(httpx2, "head", lambda *a, **k: _Refused())

    assert ME._remote_config_available("meta-llama/Meta-Llama-3-8B") is False
    assert ME._hub_reachable is True
