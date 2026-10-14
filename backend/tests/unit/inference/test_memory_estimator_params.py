"""
How the estimator arrives at a parameter count.

The count feeds model_memory in estimate_memory_requirements(), so an error becomes a
wrong "will it fit" answer — and under-reporting is the one that hurts, because it
says a model fits when it does not.

Counting is done by building the model on the meta device (shapes and dtypes, no
storage) and cross-checking against the checkpoint's safetensors index. Both halves
matter, and each covers a failure the other has:

  meta device      exact for MoE experts, vision towers and tied embeddings, which
                   hand-written arithmetic got wrong by 27-36%; but num_parameters()
                   counts nn.Parameters only, so weights kept in buffers go missing
                   (Qwen3-Next, 2% low)
  safetensors      what is actually on disk, so it catches the above; but it needs
                   the Hub, and does not exist for local or private checkpoints

Measured against real checkpoints, the combination lands within 0.11% on SmolVLM
(multimodal), Qwen2.5-0.5B (tied), Mixtral-8x7B (MoE), Qwen1.5-MoE-A2.7B (MoE whose
name says 2.7B) and Qwen3-Next-80B (buffers).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import model_fixtures as mf  # noqa: E402
from service.inference.memory_estimator import (  # noqa: E402
    NAME_ESTIMATE_MARGIN,
    MemoryEstimator,
)

pytestmark = pytest.mark.inference

TOLERANCE = 0.03  # 3%


@pytest.fixture
def estimator():
    """A fresh estimator, so neither cache leaks between tests."""
    return MemoryEstimator()


# --------------------------------------------------------------------------
# Against real checkpoints
# --------------------------------------------------------------------------


def _count(estimator, path):
    config = estimator._load_model_config(str(path))
    if config is None:
        pytest.skip(f"config unreadable for {path}")
    return estimator._count_parameters(str(path), config)


@pytest.fixture(scope="module")
def text_model():
    try:
        _, path = mf.ensure_hf_model()
    except mf.ModelUnavailable as e:
        pytest.skip(f"text test model unavailable: {e}")
    return path


@pytest.fixture(scope="module")
def vlm_model():
    try:
        _, path = mf.ensure_vlm_model()
    except mf.ModelUnavailable as e:
        pytest.skip(f"multimodal test model unavailable: {e}")
    return path


def test_tied_embeddings_are_not_counted_twice(estimator, text_model):
    """
    Qwen2.5-0.5B is 0.494B on disk and ties its embeddings.

    from_config does not tie on the meta device, so without an explicit tie_weights()
    the lm_head is counted as a second copy of the token table — and that table is
    151936 x 896 = 0.136B, over a quarter of this model.
    """
    assert _count(estimator, text_model) == pytest.approx(0.494, rel=TOLERANCE)


def test_the_vision_tower_is_counted(estimator, vlm_model):
    """
    SmolVLM-256M is 0.2565B on disk, of which the vision tower is ~0.087B.

    Counting only the language model (config.text_config) reported 0.163B — 36% under
    — while the tower is loaded into the same device memory.
    """
    counted = _count(estimator, vlm_model)

    assert counted == pytest.approx(0.2565, rel=TOLERANCE)
    # Guard the specific regression: text-only would land near 0.163.
    assert counted > 0.20


# --------------------------------------------------------------------------
# The two halves, and what each is for
# --------------------------------------------------------------------------


def test_the_checkpoint_wins_when_the_meta_count_is_low(estimator, monkeypatch):
    """
    An architecture keeping weights in buffers counts low; the index catches it.

    This is Qwen3-Next: 79.67B from the meta device against 81.32B on disk. Taking the
    larger is deliberate — the failure to avoid is claiming a model fits.
    """
    monkeypatch.setattr(estimator, "_count_via_meta_device", lambda config, **kwargs: 79.67)
    monkeypatch.setattr(estimator, "_count_via_safetensors_index", lambda name: 81.32)

    assert estimator._count_parameters("org/model", object()) == 81.32


def test_the_meta_count_is_used_when_the_hub_is_unreachable(estimator, monkeypatch):
    """Local and private checkpoints have no index to consult."""
    monkeypatch.setattr(estimator, "_count_via_meta_device", lambda config, **kwargs: 7.5)
    monkeypatch.setattr(estimator, "_count_via_safetensors_index", lambda name: None)

    assert estimator._count_parameters("/local/path", object()) == 7.5


def test_no_count_at_all_is_reported_as_such(estimator, monkeypatch):
    """Neither half available: say so rather than inventing a number."""
    monkeypatch.setattr(estimator, "_count_via_meta_device", lambda config, **kwargs: None)
    monkeypatch.setattr(estimator, "_count_via_safetensors_index", lambda name: None)

    assert estimator._count_parameters("org/model", object()) is None


def test_local_paths_do_not_hit_the_hub(estimator):
    """A filesystem path is not a repo id; asking about it would be a wasted request."""
    assert estimator._count_via_safetensors_index("/models/my-finetune") is None
    assert estimator._count_via_safetensors_index("C:/models/my-finetune") is None


# --------------------------------------------------------------------------
# What the result says about its own provenance
# --------------------------------------------------------------------------


def test_a_name_based_estimate_says_so(estimator, monkeypatch):
    """
    size_source separates a measurement from a guess.

    On the name path the parameter count, hidden size and layer count are all
    inferred, so a client presenting it identically to a counted figure would be
    overstating what is known. The name fallback itself is covered in
    test_memory_estimator.py.
    """
    monkeypatch.setattr(estimator, "_load_model_config", lambda name: None)

    result = estimator.estimate_memory_requirements("some/mystery-7b")

    assert result["size_source"] == "model_name"
    assert result["model_size_billions"] == pytest.approx(7 * NAME_ESTIMATE_MARGIN, rel=0.01)


def test_an_unusable_model_returns_an_actionable_error(estimator, monkeypatch):
    """No config and an unreadable name: the message has to say what would fix it."""
    monkeypatch.setattr(estimator, "_load_model_config", lambda name: None)

    result = estimator.estimate_memory_requirements("mistralai/Mixtral-8x7B-v0.1")

    assert "error" in result
    assert "HF_TOKEN" in result["suggestion"]
