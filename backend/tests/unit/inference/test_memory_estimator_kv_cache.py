"""
KV cache sizing.

Unlike activations and runtime overhead, this one is not a rule of thumb: an
attention layer caches exactly 2 (K and V) x num_key_value_heads x head_dim values
per token, so the answer follows from the config and can be asserted outright.

It used to be computed from hidden_size and then scaled by 0.6 ("about 60% is used
on average in practice"), which was wrong twice over:

  - hidden_size assumes MHA. Under GQA the cache is smaller by the grouping factor,
    so the estimate ran 2.4-4.2x high on every model measured.
  - the 0.6 removed 40% regardless, which on a genuine MHA model turns the result
    into an under-estimate — the direction that says a model fits when it does not.
"""

import pytest

from service.inference.memory_estimator import NAME_ESTIMATE_MARGIN, MemoryEstimator

pytestmark = pytest.mark.inference

GIB = 1024**3


def kv_bytes(layers: int, tokens: int, kv_heads: int, head_dim: int) -> float:
    """2 (K and V) x layers x tokens x cached width x 2 bytes, in GiB."""
    return 2 * layers * tokens * kv_heads * head_dim * 2 / GIB


@pytest.fixture
def estimator():
    return MemoryEstimator()


@pytest.mark.parametrize(
    "name, layers, heads, kv_heads, head_dim",
    [
        ("Qwen2.5-0.5B", 24, 14, 2, 64),
        ("Mixtral-8x7B", 32, 32, 8, 128),
        ("Qwen3-32B", 64, 64, 8, 128),
        ("gpt-oss-20b", 24, 64, 8, 64),
    ],
)
def test_gqa_models_cache_only_their_kv_heads(
    estimator, name, layers, heads, kv_heads, head_dim
):
    """
    The cached width is kv_heads x head_dim, not hidden_size.

    Head layouts are taken from the real configs of these checkpoints. Each has
    fewer KV heads than attention heads, which is exactly what the old formula
    ignored.
    """
    result = estimator._calculate_kv_cache_memory_with_config(
        num_layers=layers, batch_size=1, sequence_length=2048, kv_width=kv_heads * head_dim
    )

    assert result == pytest.approx(kv_bytes(layers, 2048, kv_heads, head_dim))
    # And it must be strictly smaller than the MHA reading it replaced.
    assert result < kv_bytes(layers, 2048, heads, head_dim)


def test_the_cache_scales_with_batch_and_sequence(estimator):
    """No averaging factor: the caller asks for a size and gets that size in full."""
    one = estimator._calculate_kv_cache_memory_with_config(32, 1, 1024, kv_width=1024)
    twice_the_tokens = estimator._calculate_kv_cache_memory_with_config(32, 1, 2048, 1024)
    two_sequences = estimator._calculate_kv_cache_memory_with_config(32, 2, 1024, 1024)

    assert twice_the_tokens == pytest.approx(one * 2)
    assert two_sequences == pytest.approx(one * 2)


def test_an_mha_model_is_not_scaled_down(estimator):
    """
    With kv_heads == heads the cached width is hidden_size, and that is the answer.

    The old 0.6 factor made this case 40% light, which is the dangerous direction.
    """
    result = estimator._calculate_kv_cache_memory_with_config(32, 1, 2048, kv_width=4096)

    assert result == pytest.approx(kv_bytes(32, 2048, 32, 128))


def test_without_a_config_it_assumes_the_largest_layout(estimator):
    """
    The fallback cannot know the head layout, so it assumes MHA.

    That over-estimates a GQA model, which is the safe direction for "will it fit".
    """
    hidden = estimator._estimate_hidden_size(7.0)
    layers = estimator._estimate_num_layers(7.0)

    result = estimator._calculate_kv_cache_memory(7.0, batch_size=1, sequence_length=2048)

    assert result == pytest.approx(
        estimator._calculate_kv_cache_memory_with_config(layers, 1, 2048, kv_width=hidden)
        * NAME_ESTIMATE_MARGIN
    )


def test_the_guessed_geometry_carries_a_margin(estimator):
    """
    The MHA ceiling protects GQA models; it leaves a genuinely-MHA model nothing.

    Measured against real configs, the guessed layers x hidden lands within 1.00-2.67x
    of the truth, and the 1.00x was phi-2 sitting exactly on a table row. Since each
    table row covers a wide band (one value for everything from 13B to 30B), a deep
    narrow model just under a threshold is the case that can turn negative, and for an
    MHA model there is no other headroom to absorb it.
    """
    exact = estimator._calculate_kv_cache_memory_with_config(
        estimator._estimate_num_layers(7.0), 1, 2048, kv_width=estimator._estimate_hidden_size(7.0)
    )

    assert estimator._calculate_kv_cache_memory(7.0, 1, 2048) > exact


def test_a_config_backed_estimate_carries_no_such_margin(estimator):
    """
    The margin belongs to the guess, not to the formula.

    Padding a figure counted from the checkpoint would put back the kind of unexplained
    fudge factor the 0.6 multiplier was.
    """
    assert estimator._calculate_kv_cache_memory_with_config(32, 1, 2048, 1024) == pytest.approx(
        kv_bytes(32, 2048, 8, 128)
    )


def test_the_kv_width_comes_from_the_config(estimator, monkeypatch):
    """
    End to end: a GQA config must not produce an MHA-sized cache.

    Guards the wiring as well as the formula — reading the right fields and then
    passing hidden_size anyway would still be wrong.
    """

    class _Config:
        architectures = ["Qwen3ForCausalLM"]
        hidden_size = 5120
        num_hidden_layers = 64
        num_attention_heads = 64
        num_key_value_heads = 8
        head_dim = 128
        intermediate_size = 25600
        vocab_size = 151936

    monkeypatch.setattr(estimator, "_load_model_config", lambda name: _Config())
    monkeypatch.setattr(estimator, "_count_parameters", lambda name, config: 32.0)

    result = estimator.estimate_memory_requirements("Qwen/Qwen3-32B", sequence_length=2048)

    assert result["memory_breakdown_gb"]["kv_cache"] == pytest.approx(
        kv_bytes(64, 2048, 8, 128), rel=0.01
    )
