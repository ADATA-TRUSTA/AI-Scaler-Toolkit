"""Unit tests for MemoryEstimator.extract_model_size.

`extract_model_size` counts the parameters from the checkpoint when a config can be
read, and only otherwise falls back to reading a size out of the model id. These
tests mock the config loader so the fallback can be exercised in isolation; the
counting path is covered in test_memory_estimator_params.py.

The fallback deliberately does not answer for every name. It used to, and this file
asserted the result — including that "Mixtral-8x7b" yields 7.0, which is the active
size of a 46.7B checkpoint. That expectation is now inverted: an MoE name gets no
answer at all, because a 6.7x under-estimate reported confidently is worse for
"will it fit" than saying the size is unknown.
"""

import pytest

from service.inference.memory_estimator import NAME_ESTIMATE_MARGIN, MemoryEstimator

pytestmark = pytest.mark.models


@pytest.fixture
def estimator(monkeypatch):
    est = MemoryEstimator()
    # Force the fallback: this file is about what happens with no config.
    monkeypatch.setattr(est, "_load_model_config", lambda name: None)
    return est


# --------------------------------------------------------------------------
# Reading a size out of the name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, advertised",
    [
        ("meta-llama/Llama-2-7b-chat-hf", 7.0),
        ("model-13B", 13.0),  # case-insensitive
        ("microsoft/phi-1.5b", 1.5),  # decimal
        ("some-70-billion-model", 70.0),  # "-billion" form
        ("custom-405b-instruct", 405.0),
    ],
)
def test_size_is_read_from_the_name_with_a_margin(estimator, name, advertised):
    """
    The name carries the marketing figure, which sits below the real total.

    Measured against safetensors totals, gpt-oss-20b is 21.5B and gemma-3-4b is 4.3B
    — both 7.5% above their names — so returning the bare reading under-reports.
    """
    assert estimator.extract_model_size(name) == pytest.approx(
        advertised * NAME_ESTIMATE_MARGIN
    )


def test_a_name_without_a_size_gets_no_answer(estimator):
    assert estimator.extract_model_size("plain-model-name") is None


@pytest.mark.parametrize(
    "name",
    [
        "mistralai/Mixtral-8x7b",  # 7B in the name, 46.7B on disk
        "mistralai/Mixtral-8x22B-Instruct-v0.1",  # 22B in the name, 141B on disk
        "Qwen/Qwen1.5-MoE-A2.7B",  # 2.7B in the name, 14.3B on disk
    ],
)
def test_moe_names_get_no_answer_rather_than_the_active_size(estimator, name):
    """
    An MoE name advertises active or per-expert size; the total is not in it.

    This is the case the old regex answered confidently and wrongly. The margin
    cannot cover a 5-7x gap, so the honest result is None.
    """
    assert estimator.extract_model_size(name) is None


# --------------------------------------------------------------------------
# The counted path wins whenever it is available
# --------------------------------------------------------------------------


def test_the_counted_size_beats_the_name(monkeypatch):
    est = MemoryEstimator()
    monkeypatch.setattr(est, "_load_model_config", lambda name: object())
    monkeypatch.setattr(est, "_count_parameters", lambda name, config: 10.0)

    # The name says 7b; the checkpoint says 10.
    assert est.extract_model_size("whatever-7b") == 10.0


def test_a_failed_count_falls_back_to_the_name(monkeypatch):
    """A config that loads but cannot be built must not lose the estimate entirely."""
    est = MemoryEstimator()
    monkeypatch.setattr(est, "_load_model_config", lambda name: object())
    monkeypatch.setattr(est, "_count_parameters", lambda name, config: None)

    assert est.extract_model_size("fallback-7b") == pytest.approx(7 * NAME_ESTIMATE_MARGIN)
