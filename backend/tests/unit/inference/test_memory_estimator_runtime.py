"""
The two runtime terms, against measurements rather than rules of thumb.

Both were guesses with no recorded derivation, and both were wrong by more than an
order of magnitude — in opposite directions, which is presumably why the total
looked plausible enough that nobody checked.

  activation   was sequence x hidden x layers / 30, with no logits term at all.
               The logits tensor is the whole story: the output projection produces
               one score per vocabulary entry per position. Measured 606.6 MiB where
               the formula said 2.8 — 216x low, and low is the direction that OOMs.

  overhead     charged 2.15 GB of GPU context against 90-181 MiB measured, and added
               a further 1.3 GB of host-side framework cost to the same GPU figure.

Numbers below come from torch's allocator and nvidia-smi on an RTX 5060 Ti
(torch 2.11+cu130), weights subtracted.
"""

import pytest

from service.inference.memory_estimator import MemoryEstimator

pytestmark = pytest.mark.inference

MIB = 1 / 1024  # GB per MiB


@pytest.fixture
def estimator():
    return MemoryEstimator()


# --------------------------------------------------------------------------
# Activation: the logits tensor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vocab, hidden, layers, seq, measured_mib",
    [
        # Windows, torch 2.11+cu — Qwen2.5-0.5B-Instruct
        (151936, 896, 24, 128, 47.3),
        (151936, 896, 24, 512, 158.4),
        (151936, 896, 24, 2048, 606.6),
        # Windows — SmolLM2-135M-Instruct, a third of the vocabulary
        (49152, 576, 30, 128, 12.1),
        (49152, 576, 30, 512, 48.6),
        (49152, 576, 30, 2048, 195.0),
        # Linux (WSL2), torch 2.13+cu129, same card. Qwen carries a ~33 MiB constant
        # workspace here against Windows' 10-13, which is what sets the floor below.
        (151936, 896, 24, 128, 70.2),
        (151936, 896, 24, 512, 181.3),
        (151936, 896, 24, 2048, 629.5),
        (151936, 896, 24, 8192, 2420.1),
        # Linux — SmolLM2 tracks the logits tensor as closely as it does on Windows
        (49152, 576, 30, 128, 12.1),
        (49152, 576, 30, 2048, 194.3),
        (49152, 576, 30, 8192, 777.1),
    ],
)
def test_activation_covers_the_measured_peak(
    estimator, vocab, hidden, layers, seq, measured_mib
):
    """
    Never below the measurement, and not wildly above it.

    Both platforms are in the table because they disagree: a 10% margin plus the old
    16 MiB floor covered Windows and came out *under* the Linux measurement at 128 and
    512 tokens. The floor is 64 MiB, set by the larger of the two constants.

    Measurements are taken with use_cache=False. With the default the forward pass also
    allocates a KV cache, which the estimator counts separately — including it here
    would double-count it and make the residual look like it scaled with sequence
    length when it does not.

    The floor makes the short-sequence cases read high in relative terms while staying
    negligible in absolute ones. What matters is that the term can no longer come out
    200x low.
    """
    estimated = estimator._calculate_activation_memory_with_config(
        hidden, layers, 1, seq, "none", vocab_size=vocab
    )
    measured = measured_mib * MIB

    assert estimated >= measured, "activation must not under-report the measured peak"
    assert estimated <= measured * 1.15 + 70 * MIB, "over-reporting beyond the stated margin"


def test_activation_tracks_the_vocabulary_not_the_layers(estimator):
    """
    Doubling the vocabulary doubles the term; doubling the layer count does not.

    The old formula had it exactly backwards — layers and hidden size were the only
    inputs, and the vocabulary did not appear at all.
    """
    base = estimator._calculate_activation_memory_with_config(896, 24, 1, 2048, "none", 151936)
    twice_vocab = estimator._calculate_activation_memory_with_config(
        896, 24, 1, 2048, "none", 303872
    )
    twice_layers = estimator._calculate_activation_memory_with_config(
        896, 48, 1, 2048, "none", 151936
    )

    assert twice_vocab > base * 1.9
    assert twice_layers == pytest.approx(base)


def test_activation_without_a_config_assumes_a_large_vocabulary(estimator):
    """The fallback cannot read the vocabulary, so it must not assume a small one."""
    fallback = estimator._calculate_activation_memory(7.0, 1, 2048, "none")
    small_vocab = estimator._calculate_activation_memory_with_config(
        4096, 32, 1, 2048, "none", vocab_size=32000
    )

    assert fallback > small_vocab


# --------------------------------------------------------------------------
# Overhead: host and device are different budgets
# --------------------------------------------------------------------------


def test_host_overhead_is_not_charged_to_the_device(estimator):
    """
    The Python process and the framework live in system RAM.

    Summing them into the GPU requirement added ~1.3 GB to every answer, which for a
    small model is larger than the model.
    """
    overhead = estimator._estimate_runtime_overhead("none")

    assert overhead["host_total"] >= 1.0, "the framework cost is real, just not on the GPU"
    assert overhead["device_total"] < overhead["host_total"]
    assert overhead["total"] == pytest.approx(
        overhead["host_total"] + overhead["device_total"], abs=0.01
    )


def test_device_overhead_stays_near_what_was_measured(estimator, monkeypatch):
    """
    90-181 MiB measured on CUDA; the old constants claimed 2.15 GB.

    Kept above the measurement for headroom, but an order of magnitude closer.
    """
    monkeypatch.setattr(estimator, "_detect_accelerator", lambda: "cuda")

    overhead = estimator._estimate_runtime_overhead("none")

    assert 0.18 <= overhead["device_total"] <= 0.6


def test_a_non_cuda_gpu_is_not_treated_as_cpu_only(estimator, monkeypatch):
    """
    An Intel XPU has a context to pay for; it used to be charged nothing.

    The old branch keyed on torch.cuda.is_available(), so every other accelerator
    fell into the CPU path and got zero device overhead — an under-estimate on the
    exact machines this product ships on.
    """
    monkeypatch.setattr(estimator, "_detect_accelerator", lambda: "xpu")
    xpu = estimator._estimate_runtime_overhead("none")

    monkeypatch.setattr(estimator, "_detect_accelerator", lambda: "cpu")
    cpu = estimator._estimate_runtime_overhead("none")

    assert xpu["device_total"] > 0
    assert cpu["device_total"] == 0
    assert xpu["accelerator"] == "xpu"


def test_quantisation_kernels_are_charged_to_the_device(estimator, monkeypatch):
    """bitsandbytes loads onto the GPU, so it belongs in the device share."""
    monkeypatch.setattr(estimator, "_detect_accelerator", lambda: "cuda")

    plain = estimator._estimate_runtime_overhead("none")
    quantised = estimator._estimate_runtime_overhead("int4")

    assert quantised["device_total"] > plain["device_total"]
