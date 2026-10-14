"""
Whose code the estimator is willing to run.

``trust_remote_code`` is not a compatibility switch. A checkpoint may declare an
``auto_map``, and then both ``AutoConfig.from_pretrained`` and
``AutoModel*.from_config`` import and execute Python from that repository.
``init_empty_weights()`` skips allocating tensor storage; it does not sandbox the
import.

That matters here more than in a loading path, because of who chooses the model id.
``/inference/estimate_memory`` takes ``model_name`` as a free-form string, has no
authentication, and the service runs with ``allow_origins=["*"]`` — so any page the
operator visits can name a repository and have the backend fetch it. Answering "will
this fit" must not become "run this".

The line drawn is locality, not trust in the Hub: a checkpoint already on this disk
was put there by the operator and loading it would execute the same code anyway, so
it keeps the old behaviour. Anything fetched over the network does not, and a remote
architecture that needs custom code degrades to the name estimate — which the
response labels via ``size_source``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import service.inference.memory_estimator  # noqa: E402, F401  (registers the module)
from service.inference.memory_estimator import MemoryEstimator  # noqa: E402

ME = sys.modules["service.inference.memory_estimator"]

pytestmark = pytest.mark.inference


@pytest.fixture
def estimator():
    return MemoryEstimator()


@pytest.fixture
def recorded_config_calls(monkeypatch):
    """Capture the kwargs every AutoConfig.from_pretrained call was made with."""
    calls: list[dict] = []

    class _StubAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("not in the local cache")
            return object()

    monkeypatch.setattr(ME, "AutoConfig", _StubAutoConfig)
    return calls


def test_a_hub_fetch_does_not_execute_repo_code(estimator, monkeypatch, recorded_config_calls):
    """The remote attempt must ask for the config without agreeing to run anything."""
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: True)
    monkeypatch.setattr(ME, "_remote_config_available", lambda *a, **k: True)

    estimator._load_model_config("attacker/evil-model")

    remote = [c for c in recorded_config_calls if not c["local_files_only"]]
    assert remote, "expected a remote attempt"
    assert all(c["trust_remote_code"] is False for c in remote)


def test_a_local_checkpoint_keeps_its_custom_architecture(estimator, monkeypatch):
    """
    A model already on disk is unchanged from before.

    Refusing here would break every locally converted checkpoint with a custom
    architecture, and would buy nothing: loading that model runs the same code.
    """
    calls: list[dict] = []

    class _StubAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(ME, "AutoConfig", _StubAutoConfig)
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: False)

    estimator._load_model_config("/models/my-finetune")

    assert calls[0]["local_files_only"] is True
    assert calls[0]["trust_remote_code"] is True


def test_the_meta_device_build_inherits_the_same_decision(estimator, monkeypatch):
    """
    Counting builds the model, which imports modelling code for a custom architecture.

    _load_model_config gating the config alone would leave that second door open, so
    the locality it recorded has to reach _count_via_meta_device too.
    """
    seen: list[bool] = []
    monkeypatch.setattr(
        estimator,
        "_count_via_meta_device",
        lambda config, *, trust_remote_code=False: seen.append(trust_remote_code) or 7.0,
    )
    monkeypatch.setattr(estimator, "_count_via_safetensors_index", lambda name: None)

    estimator._config_is_local["remote/model"] = False
    estimator._count_parameters("remote/model", object())

    estimator._config_is_local["/local/model"] = True
    estimator._count_parameters("/local/model", object())

    assert seen == [False, True]


def test_an_unknown_model_id_is_not_treated_as_a_dead_link(estimator, monkeypatch):
    """
    A refusal is not a network fault.

    transformers raises ValueError when it parsed a real response and would not build
    from it — an unrecognised architecture, or one whose custom code is now declined.
    Marking the Hub down for that would disable remote lookups for every later model
    in the process because of one bad request.
    """

    class _StubAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not in the local cache")
            raise ValueError("requires you to execute the configuration file")

    monkeypatch.setattr(ME, "AutoConfig", _StubAutoConfig)
    monkeypatch.setattr(ME, "_hub_is_reachable", lambda *a, **k: True)
    monkeypatch.setattr(ME, "_remote_config_available", lambda *a, **k: True)
    monkeypatch.setattr(ME, "_hub_reachable", True)

    assert estimator._load_model_config("org/custom-arch") is None
    assert ME._hub_reachable is True


def test_a_declined_remote_model_falls_back_to_the_name(estimator, monkeypatch):
    """The user still gets an answer, labelled as the guess it is."""
    monkeypatch.setattr(estimator, "_load_model_config", lambda name: None)

    result = estimator.estimate_memory_requirements("org/custom-arch-7b")

    assert result["size_source"] == "model_name"
    assert result["model_size_billions"] > 0
