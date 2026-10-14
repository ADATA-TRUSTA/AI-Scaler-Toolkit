"""
Unit tests for the GGUF / llama.cpp memory estimator.

The estimator reproduces llama.cpp's own tensor placement and KV-cache sizing from the GGUF
tensor table. These tests drive it with a synthetic `GgufModelInfo` instead of a real file so
they need neither a downloaded model nor the llama.cpp binaries, and so the byte arithmetic
can be asserted exactly.

The synthetic model is sized so every quantity lands on a whole MiB:
  - 4 blocks, each 1 MiB attention + 4 MiB MoE experts + 0.5 MiB dense/router = 5.5 MiB
  - 2 MiB token embeddings (always host) and 2 MiB output projection
  - 8 KV heads x 128 key/value length => exactly 1 MiB of f16 KV per layer at n_ctx 256

A real cross-check against `llama-fit-params` lives at the bottom and skips itself when the
model or the binary is unavailable.
"""

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from service.inference.gguf_estimator import (
    MAX_PLAN_CELLS,
    RECOMMEND_SOLVED_KEYS,
    SWEEP_SOLVED_KEYS,
    GgufMemoryEstimator,
    apply_cli_args,
    kv_bytes_per_element,
    parse_llama_server_args,
)
from service.inference.gguf_inspect import (
    GgufModelInfo,
    GgufTensor,
    _build_swa_layers,
    classify_tensor,
    register_extra_ggml_types,
)

pytestmark = pytest.mark.models

MIB = 1024 * 1024
ATTN_B = 1 * MIB
EXPS_B = 4 * MIB
FFN_B = MIB // 2
EMBD_B = 2 * MIB
OUTPUT_B = 2 * MIB

# n_head_kv * (key_length + value_length) * 2 bytes = 4096 B/token, x256 cells = 1 MiB.
CTX = 256
KV_PER_LAYER_MIB = 1.0


def make_info(  # noqa: PLR0913 - one switch per llama.cpp behaviour under test
    n_block=4,
    moe=True,
    n_swa=0,
    swa_layers=None,
    *,
    key_length_swa=0,
    n_layer_kv_from_start=-1,
    recurrent_layers=None,
    n_embd_r=0,
    n_embd_s=0,
    tied=False,
):
    """Build a synthetic GgufModelInfo with predictable, MiB-aligned tensor sizes."""
    tensors = []
    for il in range(n_block):
        tensors.append(GgufTensor(f"blk.{il}.attn_q.weight", il, "attn", ATTN_B))
        tensors.append(GgufTensor(f"blk.{il}.ffn_norm.weight", il, "ffn", FFN_B))
        if moe:
            tensors.append(GgufTensor(f"blk.{il}.ffn_gate_exps.weight", il, "moe_exps", EXPS_B))
        else:
            tensors.append(GgufTensor(f"blk.{il}.ffn_gate.weight", il, "ffn", EXPS_B))
    tensors.append(GgufTensor("token_embd.weight", -1, "input", EMBD_B))
    if tied:
        # What the reader appends for a model whose output projection is the embedding table.
        tensors.append(
            GgufTensor("output.weight (tied to token_embd)", n_block, "output", EMBD_B, True)
        )
    else:
        tensors.append(GgufTensor("output.weight", n_block, "output", OUTPUT_B))

    return GgufModelInfo(
        path="/synthetic/model.gguf",
        arch="qwen3moe" if moe else "qwen3",
        n_block=n_block,
        n_embd=1024,
        n_head=8,
        n_head_kv=[8] * n_block,
        key_length=128,
        value_length=128,
        n_vocab=1000,
        n_ctx_train=CTX,
        n_expert=2 if moe else 0,
        n_expert_used=1 if moe else 0,
        n_swa=n_swa,
        swa_layers=swa_layers or [False] * n_block,
        file_type=7,
        tensors=tensors,
        # Gemma 4 requires the SWA key and value dimensions to match, so mirror them.
        key_length_swa=key_length_swa,
        value_length_swa=key_length_swa,
        n_layer_kv_from_start=n_layer_kv_from_start,
        recurrent_layers=recurrent_layers or [False] * n_block,
        n_embd_r=n_embd_r,
        n_embd_s=n_embd_s,
        tied_embedding=tied,
    )


@pytest.fixture
def estimator(monkeypatch):
    """Estimator wired to the synthetic model, so no file or binary is touched."""
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info())
    return est


def gpu(result, key="model"):
    return result["memory_breakdown_mib"]["gpu_total"][key]


def host(result, key="model"):
    return result["memory_breakdown_mib"]["host"][key]


def run(est, **kwargs):
    kwargs.setdefault("n_ctx", CTX)
    kwargs.setdefault("include_per_layer", False)
    return est.estimate("/synthetic/model.gguf", **kwargs)


# --------------------------------------------------------------------------
# Tensor classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("token_embd.weight", (-1, "input")),
        ("output.weight", (4, "output")),
        ("output_norm.weight", (4, "output")),
        ("blk.0.attn_q.weight", (0, "attn")),
        ("blk.3.attn_k_norm.weight", (3, "attn")),
        ("blk.2.ffn_gate.weight", (2, "ffn")),
        ("blk.2.ffn_gate_inp.weight", (2, "ffn")),  # router stays with the dense weights
        ("blk.2.ffn_gate_exps.weight", (2, "moe_exps")),
        ("blk.2.ffn_down_exps.weight", (2, "moe_exps")),
        ("blk.2.ffn_gate_up_chexps.weight", (2, "moe_exps")),
        ("blk.2.ffn_up_shexp.weight", (2, "shexp")),  # shared expert is not a routed expert
        ("blk.1.ssm_a", (1, "other")),
    ],
)
def test_classify_tensor(name, expected):
    assert classify_tensor(name, n_block=4) == expected


@pytest.mark.parametrize(
    "cache_type, expected",
    [("f32", 4.0), ("f16", 2.0), ("bf16", 2.0), ("q8_0", 34 / 32), ("q4_0", 18 / 32)],
)
def test_kv_bytes_per_element(cache_type, expected):
    assert kv_bytes_per_element(cache_type) == pytest.approx(expected)


@pytest.mark.parametrize("cache_type", ["q3_k", "q6_k", "nonsense"])
def test_kv_bytes_per_element_rejects_types_llama_cpp_will_not_accept(cache_type):
    """Q3_K/Q6_K are real ggml types but are not valid for -ctk / -ctv."""
    with pytest.raises(ValueError, match="Unsupported KV cache type"):
        kv_bytes_per_element(cache_type)


# --------------------------------------------------------------------------
# Layer placement: i_gpu_start = max(n_layer + 1 - ngl, 0)
# --------------------------------------------------------------------------


def test_ngl_zero_puts_every_weight_on_host(estimator):
    result = run(estimator, n_gpu_layers=0)
    assert gpu(result) == 0.0
    assert host(result) == pytest.approx(4 * 5.5 + 2 + 2)


def test_ngl_one_offloads_the_output_layer_before_any_block(estimator):
    """llama.cpp fills the last of n_layer+1 slots first, and that slot is the output layer."""
    result = run(estimator, n_gpu_layers=1)
    assert gpu(result) == pytest.approx(OUTPUT_B / MIB)
    assert result["placement"]["output_on_gpu"] is True
    assert result["placement"]["blocks_on_gpu"] == 0
    # No block on the GPU means no KV cache on the GPU either.
    assert gpu(result, "context") == 0.0


def test_ngl_two_adds_the_last_block(estimator):
    result = run(estimator, n_gpu_layers=2)
    assert gpu(result) == pytest.approx(2 + 5.5)
    assert result["placement"]["blocks_on_gpu"] == 1
    assert gpu(result, "context") == pytest.approx(KV_PER_LAYER_MIB)


@pytest.mark.parametrize("n_gpu_layers", [-1, 5, 999])
def test_full_offload_variants_agree(estimator, n_gpu_layers):
    result = run(estimator, n_gpu_layers=n_gpu_layers)
    assert result["settings"]["n_gpu_layers_effective"] == 5
    assert gpu(result) == pytest.approx(4 * 5.5 + 2)
    assert gpu(result, "context") == pytest.approx(4 * KV_PER_LAYER_MIB)


def test_token_embeddings_never_leave_the_host(estimator):
    result = run(estimator, n_gpu_layers=999)
    assert host(result) == pytest.approx(EMBD_B / MIB)
    assert result["placement"]["input_on_gpu"] is False


def test_no_gpu_puts_everything_on_the_host(estimator):
    result = run(estimator, n_gpu_layers=-1, n_gpu=0)
    assert result["memory_breakdown_mib"]["gpu"] == []
    assert host(result) == pytest.approx(4 * 5.5 + 2 + 2)
    assert host(result, "context") == pytest.approx(4 * KV_PER_LAYER_MIB)


# --------------------------------------------------------------------------
# --n-cpu-moe / --cpu-moe
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_cpu_moe, moved", [(0, 0), (1, 1), (2, 2), (4, 4), (99, 4)])
def test_n_cpu_moe_moves_expert_weights_of_the_first_n_blocks(estimator, n_cpu_moe, moved):
    result = run(estimator, n_gpu_layers=-1, n_cpu_moe=n_cpu_moe)
    assert gpu(result) == pytest.approx(4 * 5.5 + 2 - moved * 4)
    assert host(result) == pytest.approx(2 + moved * 4)


def test_cpu_moe_matches_n_cpu_moe_over_all_layers(estimator):
    everything = run(estimator, n_gpu_layers=-1, cpu_moe=True)
    per_layer = run(estimator, n_gpu_layers=-1, n_cpu_moe=4)
    assert gpu(everything) == gpu(per_layer)


def test_n_cpu_moe_keeps_attention_and_kv_cache_on_the_gpu(estimator):
    """This is the whole point of -ncmoe over lowering -ngl."""
    result = run(estimator, n_gpu_layers=-1, n_cpu_moe=4)
    assert gpu(result, "context") == pytest.approx(4 * KV_PER_LAYER_MIB)
    assert host(result, "context") == 0.0
    # Attention weights (1 MiB per block) and the dense/router tensors stay on the GPU.
    assert gpu(result) == pytest.approx(4 * 1.5 + 2)


def test_n_cpu_moe_override_does_not_match_a_longer_block_index(monkeypatch):
    r"""blk\.1\. must not match blk.10, blk.11, ..."""
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(n_block=12))
    result = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX, n_cpu_moe=2, include_per_layer=False
    )
    # Only blocks 0 and 1 lose their experts, not 10 and 11.
    assert host(result) == pytest.approx(2 + 2 * 4)


def test_n_cpu_moe_is_a_no_op_on_a_dense_model(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(moe=False))
    plain = est.estimate("/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX)
    with_ncmoe = est.estimate("/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX, n_cpu_moe=4)
    assert gpu(with_ncmoe) == gpu(plain)
    assert any("no routed experts" in note for note in with_ncmoe["notes"])


def test_n_cpu_moe_only_affects_layers_that_are_on_the_gpu(estimator):
    """An override on a block that already lives on the host changes nothing."""
    without = run(estimator, n_gpu_layers=2)
    with_override = run(estimator, n_gpu_layers=2, n_cpu_moe=2)
    assert gpu(with_override) == gpu(without)


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_ctx, cells", [(256, 256), (512, 512), (1, 256), (300, 512)])
def test_n_ctx_is_padded_to_256(estimator, n_ctx, cells):
    result = run(estimator, n_gpu_layers=-1, n_ctx=n_ctx)
    assert result["settings"]["n_ctx_padded"] == cells
    assert gpu(result, "context") == pytest.approx(4 * cells / 256)


def test_kv_quantization_scales_by_bytes_per_element(estimator):
    f16 = run(estimator, n_gpu_layers=-1, cache_type_k="f16", cache_type_v="f16")
    q8 = run(estimator, n_gpu_layers=-1, cache_type_k="q8_0", cache_type_v="q8_0")
    # Reported values are rounded to two decimals, hence the absolute tolerance.
    assert gpu(q8, "context") == pytest.approx(gpu(f16, "context") * (34 / 32) / 2, abs=0.01)


def test_kv_cache_follows_layer_placement(estimator):
    result = run(estimator, n_gpu_layers=3)  # output layer + last 2 blocks
    assert gpu(result, "context") == pytest.approx(2 * KV_PER_LAYER_MIB)
    assert host(result, "context") == pytest.approx(2 * KV_PER_LAYER_MIB)


def test_no_kv_offload_moves_the_whole_cache_to_the_host(estimator):
    result = run(estimator, n_gpu_layers=-1, no_kv_offload=True)
    assert gpu(result, "context") == 0.0
    assert host(result, "context") == pytest.approx(4 * KV_PER_LAYER_MIB)
    assert gpu(result) == pytest.approx(4 * 5.5 + 2)  # weights unaffected


def test_sliding_window_layers_use_a_smaller_cache(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(
        est,
        "load_info",
        lambda path: make_info(n_swa=256, swa_layers=[False, True, True, True]),
    )
    result = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=4096, include_per_layer=False
    )
    # 1 full layer at 4096 cells + 3 SWA layers capped at pad256(256 + 512) = 768 cells.
    assert gpu(result, "context") == pytest.approx(4096 / 256 + 3 * 768 / 256)


def test_swa_full_disables_the_sliding_window_saving(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(
        est,
        "load_info",
        lambda path: make_info(n_swa=256, swa_layers=[False, True, True, True]),
    )
    result = est.estimate(
        "/synthetic/model.gguf",
        n_gpu_layers=-1,
        n_ctx=4096,
        swa_full=True,
        include_per_layer=False,
    )
    assert gpu(result, "context") == pytest.approx(4 * 4096 / 256)


# --------------------------------------------------------------------------
# Architecture quirks transcribed from llama.cpp
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "period", "expected"),
    [
        # Default: the last layer of each group is the dense one.
        ("gemma3", 2, [True, False, True, False, True, False]),
        ("gemma3", 3, [True, True, False, True, True, False]),
        # dense_first architectures put the dense layer at the start of each group.
        ("smallthinker", 2, [False, True, False, True, False, True]),
        ("modern-bert", 3, [False, True, True, False, True, True]),
    ],
)
def test_swa_pattern_phase_matches_set_swa_pattern(arch, period, expected):
    """The phase matters: it decides whose KV cache lands on the GPU under partial offload."""
    assert _build_swa_layers(arch, 1024, period, len(expected)) == expected


def test_swa_pattern_falls_back_to_the_architecture_default():
    # gpt-oss is implemented in models/openai-moe.cpp with a period of 2.
    assert _build_swa_layers("gpt-oss", 128, None, 4) == [True, False, True, False]
    assert _build_swa_layers("qwen3", 0, None, 4) == [False] * 4


def test_sliding_window_layers_can_use_a_narrower_head(monkeypatch):
    """Gemma 4 halves the key/value head on SWA layers; ignoring that doubles their cache."""
    est = GgufMemoryEstimator()
    monkeypatch.setattr(
        est,
        "load_info",
        lambda path: make_info(n_swa=4096, swa_layers=[False, True, True, True], key_length_swa=64),
    )
    result = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=4096, include_per_layer=False
    )
    # 1 full layer at 128-wide heads + 3 SWA layers at 64-wide heads, all 4096 cells.
    assert gpu(result, "context") == pytest.approx(16 + 3 * 8)


def test_layers_that_share_a_kv_cache_allocate_none_of_their_own(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(n_layer_kv_from_start=2))
    result = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX, include_per_layer=False
    )
    assert gpu(result, "context") == pytest.approx(2 * KV_PER_LAYER_MIB)
    assert any("share an earlier layer" in note for note in result["notes"])


def test_linear_attention_layers_hold_fixed_state_instead_of_a_kv_cache(monkeypatch):
    """A hybrid model's recurrent layers must not scale with the context."""
    est = GgufMemoryEstimator()
    state_rows = 256 * 1024  # 1 MiB of f32 state per recurrent layer
    monkeypatch.setattr(
        est,
        "load_info",
        lambda path: make_info(recurrent_layers=[True, True, True, False], n_embd_s=state_rows),
    )
    short = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX, include_per_layer=False
    )
    long = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX * 8, include_per_layer=False
    )
    # 3 recurrent layers at 1 MiB each, plus one attention layer that does grow.
    assert gpu(short, "context") == pytest.approx(3 + 1)
    assert gpu(long, "context") == pytest.approx(3 + 8)
    assert any("Hybrid attention" in note for note in short["notes"])


def test_recurrent_state_scales_with_parallel_sequences(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(
        est,
        "load_info",
        lambda path: make_info(recurrent_layers=[True] * 4, n_embd_s=256 * 1024),
    )
    result = est.estimate(
        "/synthetic/model.gguf",
        n_gpu_layers=-1,
        n_ctx=CTX,
        n_parallel=4,
        include_per_layer=False,
    )
    assert gpu(result, "context") == pytest.approx(4 * 4)


def test_tied_embeddings_are_counted_as_a_second_copy(monkeypatch):
    """llama.cpp duplicates token_embd as the output projection; that copy is real VRAM."""
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(tied=True))
    info = est.load_info("/synthetic/model.gguf")
    assert info.file_bytes == info.total_bytes - EMBD_B
    assert info.output_weight_bytes == EMBD_B

    full = est.estimate(
        "/synthetic/model.gguf", n_gpu_layers=-1, n_ctx=CTX, include_per_layer=False
    )
    # Every block plus the duplicate on the GPU; the original stays on the host.
    assert gpu(full) == pytest.approx(4 * 5.5 + 2)
    assert host(full) == pytest.approx(2)
    assert any("Tied embeddings" in note for note in full["notes"])

    none = est.estimate("/synthetic/model.gguf", n_gpu_layers=0, n_ctx=CTX, include_per_layer=False)
    assert host(none) == pytest.approx(4 * 5.5 + 2 + 2)


def test_extra_ggml_types_are_registered_with_the_right_geometry():
    """gpt-oss ships as MXFP4, which the pinned gguf package does not know."""
    gguf = pytest.importorskip("gguf")
    register_extra_ggml_types(gguf)
    mxfp4 = gguf.constants.GGMLQuantizationType(39)
    assert mxfp4.name == "MXFP4"
    assert gguf.constants.GGML_QUANT_SIZES[mxfp4] == (32, 17)
    # Idempotent: a second call adds nothing.
    assert register_extra_ggml_types(gguf) == []


# --------------------------------------------------------------------------
# Per-layer breakdown and multi-GPU split
# --------------------------------------------------------------------------


def test_per_layer_breakdown_splits_weights_by_role(estimator):
    result = estimator.estimate("/synthetic/model.gguf", n_gpu_layers=3, n_ctx=CTX)
    layers = result["per_layer_mib"]
    assert len(layers) == 4
    assert [layer["device"] for layer in layers] == ["host", "host", "gpu0", "gpu0"]
    assert layers[0] == {
        "index": 0,
        "device": "host",
        "attn": 1.0,
        "ffn": 0.5,
        "moe_exps": 4.0,
        "shexp": 0.0,
        "other": 0.0,
        "weights": 5.5,
        "kv": KV_PER_LAYER_MIB,
        "swa": False,
    }


def test_multi_gpu_splits_layers_across_devices(estimator):
    result = run(estimator, n_gpu_layers=-1, n_gpu=2)
    devices = result["memory_breakdown_mib"]["gpu"]
    assert len(devices) == 2
    assert sum(d["model"] for d in devices) == pytest.approx(4 * 5.5 + 2)
    assert all(d["model"] > 0 for d in devices)


# --------------------------------------------------------------------------
# recommend()
# --------------------------------------------------------------------------


def test_recommend_prefers_cpu_moe_over_dropping_layers(estimator):
    """Full offload needs ~30 MiB; -ncmoe frees 4 MiB per layer while keeping KV on GPU."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=25,
        host_budget_mib=10_000,
        margin_mib=0,
        verify=False,
    )
    assert result["recommended"]["n_gpu_layers"] == -1
    assert result["recommended"]["n_cpu_moe"] >= 1
    assert result["recommended"]["constraint"] == "n_cpu_moe"
    assert "-ncmoe" in result["llama_server_args"]


@pytest.mark.parametrize("flash_attn, expected", [(True, "-fa on"), (False, "-fa off")])
def test_recommend_states_flash_attention_explicitly(estimator, flash_attn, expected):
    """The args must pin -fa, since the compute estimate depends on which path runs."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=10_000,
        host_budget_mib=10_000,
        flash_attn=flash_attn,
        verify=False,
    )
    assert expected in result["llama_server_args"]


def test_recommend_uses_full_offload_when_it_fits(estimator):
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=10_000,
        host_budget_mib=10_000,
        verify=False,
    )
    assert result["recommended"]["n_cpu_moe"] == 0
    assert result["recommended"]["n_gpu_layers"] == -1


# --------------------------------------------------------------------------
# Filling a manually set GPU cap
# --------------------------------------------------------------------------


def test_recommend_grows_context_into_spare_budget(estimator):
    """A generous cap must be spent on context, not left idle at n_ctx_min."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        gpu_budget_mib=200,
        host_budget_mib=10_000,
        margin_mib=0,
        n_ctx_min=256,
        n_ctx_max=65536,
        verify=False,
    )
    assert result["recommended"]["n_ctx"] > 256
    assert result["utilization"]["utilization_pct"] >= 90
    assert result["utilization"]["within_budget"] is True


def test_recommend_never_exceeds_the_cap(estimator):
    """Whatever it picks, the analytic allocation must stay inside budget minus margin."""
    for cap in (20, 40, 80, 200, 1000):
        result = estimator.recommend(
            "/synthetic/model.gguf",
            gpu_budget_mib=cap,
            host_budget_mib=10_000,
            margin_mib=2,
            n_ctx_min=256,
            n_ctx_max=65536,
            verify=False,
        )
        if "error" in result:
            continue
        assert result["utilization"]["allocated_mib"] <= cap - 2, f"cap {cap} exceeded"


def test_recommend_context_search_uses_256_cell_steps(estimator):
    """llama.cpp pads the cache to 256, so a finer step could not change the allocation."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        gpu_budget_mib=200,
        host_budget_mib=10_000,
        margin_mib=0,
        n_ctx_min=256,
        n_ctx_max=65536,
        verify=False,
    )
    assert result["recommended"]["n_ctx"] % 256 == 0


def test_recommend_stops_at_the_context_ceiling(estimator):
    """With a huge cap the context cannot grow past n_ctx_max, and that is reported."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        gpu_budget_mib=100_000,
        host_budget_mib=10_000_000,
        n_ctx_min=256,
        n_ctx_max=2048,
        verify=False,
    )
    assert result["recommended"]["n_ctx"] == 2048
    assert result["recommended"]["constraint"] == "n_ctx ceiling"
    assert result["utilization"]["meets_target"] is False
    assert any("ceiling" in note for note in result["notes"])


def test_recommend_does_not_grow_a_pinned_context(estimator):
    """A pinned n_ctx is honoured even when that leaves the budget under-used."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=100_000,
        host_budget_mib=10_000_000,
        verify=False,
    )
    assert result["recommended"]["n_ctx"] == CTX
    assert any("pinned" in note for note in result["notes"])


def test_recommend_reports_utilization_fields(estimator):
    result = estimator.recommend(
        "/synthetic/model.gguf",
        gpu_budget_mib=200,
        host_budget_mib=10_000,
        margin_mib=10,
        n_ctx_min=256,
        n_ctx_max=65536,
        verify=False,
    )
    util = result["utilization"]
    assert util["gpu_budget_mib"] == 200
    assert util["usable_budget_mib"] == 190
    assert util["allocated_source"] == "analytic"
    assert util["correction_mib"] == 0.0
    assert util["headroom_mib"] == pytest.approx(190 - util["allocated_mib"], abs=0.01)


def test_recommend_lowers_n_gpu_layers_on_a_dense_model(monkeypatch):
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(moe=False))
    result = est.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=20,
        host_budget_mib=10_000,
        margin_mib=0,
        verify=False,
    )
    assert 0 < result["recommended"]["n_gpu_layers"] < 5
    assert result["recommended"]["constraint"] == "n_gpu_layers"


def test_recommend_reports_when_nothing_fits(estimator):
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=1,
        host_budget_mib=10_000,
        margin_mib=0,
        verify=False,
    )
    assert "error" in result


def test_recommend_flags_insufficient_host_memory(estimator):
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=25,
        host_budget_mib=1,
        margin_mib=0,
        verify=False,
    )
    assert result["host_fits"] is False
    assert any("Host RAM is short" in note for note in result["notes"])


def test_recommend_requires_a_gpu_budget(estimator, monkeypatch):
    monkeypatch.setattr(
        GgufMemoryEstimator,
        "resolve_budgets",
        staticmethod(
            lambda gpu_budget_mib, host_budget_mib, n_gpu=None, device=None, shared_memory_pool=False: {
                "gpu_budget_mib": None,
                "host_budget_mib": None,
                "source": "unavailable",
            }
        ),
    )
    result = estimator.recommend("/synthetic/model.gguf", verify=False)
    assert "No GPU memory budget" in result["error"]


# --------------------------------------------------------------------------
# plan(): a menu of candidates per budget
# --------------------------------------------------------------------------


def make_plan(estimator, **kwargs):
    kwargs.setdefault("host_budget_mib", 10_000)
    kwargs.setdefault("margin_mib", 0)
    kwargs.setdefault("n_ctx_min", 256)
    kwargs.setdefault("n_ctx_max", 65536)
    kwargs.setdefault("verify", False)
    return estimator.plan("/synthetic/model.gguf", **kwargs)


def test_plan_offers_one_candidate_per_kv_type_per_budget(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[60, 200])
    assert [p["gpu_budget_mib"] for p in result["plans"]] == [60, 200]
    assert result["kv_cache_types"] == ["f16", "q8_0", "q4_0"]
    for plan in result["plans"]:
        assert {c["kv_quant"] for c in plan["candidates"]} == {"f16", "q8_0", "q4_0"}
        assert plan["recommended"] == plan["candidates"][0]["llama_server_args"]


def test_plan_budgets_are_deduplicated_and_sorted(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[200, 60, 200])
    assert [p["gpu_budget_mib"] for p in result["plans"]] == [60, 200]


def test_plan_ranks_more_offload_above_more_context(estimator):
    """A cap that forces --n-cpu-moe on f16 but not on q4_0 must rank q4_0 first."""
    plan = make_plan(estimator, gpu_budgets_mib=[30])["plans"][0]
    best, worst = plan["candidates"][0], plan["candidates"][-1]
    assert best["kv_quant"] == "q4_0"
    assert best["n_cpu_moe"] == 0
    assert worst["kv_quant"] == "f16"
    assert worst["n_cpu_moe"] > 0
    assert best["gpu_weight_pct"] > worst["gpu_weight_pct"]


def test_plan_prefers_a_longer_context_when_offload_is_near_equal(monkeypatch):
    """One marginal layer must not outrank a much longer context."""
    est = GgufMemoryEstimator()
    monkeypatch.setattr(est, "load_info", lambda path: make_info(n_block=30, moe=False))
    plan = est.plan(
        "/synthetic/model.gguf",
        gpu_budgets_mib=[80],
        host_budget_mib=10_000,
        margin_mib=0,
        n_ctx_min=256,
        n_ctx_max=65536,
        verify=False,
    )["plans"][0]
    fitting = [c for c in plan["candidates"] if c["fits"]]
    top = fitting[0]
    for other in fitting[1:]:
        near_equal = abs(other["gpu_weight_pct"] - top["gpu_weight_pct"]) < 5.0
        assert not (near_equal and other["n_ctx"] > top["n_ctx"])


def test_plan_quantized_kv_buys_context_at_equal_offload(estimator):
    """With every layer offloaded either way, the cheaper cache must reach further."""
    plan = make_plan(estimator, gpu_budgets_mib=[200])["plans"][0]
    by_kv = {c["kv_quant"]: c for c in plan["candidates"]}
    assert by_kv["f16"]["gpu_weight_pct"] == by_kv["q4_0"]["gpu_weight_pct"]
    assert by_kv["q4_0"]["n_ctx"] > by_kv["q8_0"]["n_ctx"] > by_kv["f16"]["n_ctx"]
    assert [c["kv_quant"] for c in plan["candidates"]] == ["q4_0", "q8_0", "f16"]


def test_plan_flags_candidates_that_buy_nothing(estimator):
    """At the context ceiling a lossier cache gains neither layers nor context."""
    plan = make_plan(estimator, gpu_budgets_mib=[100_000], n_ctx_max=2048)["plans"][0]
    by_kv = {c["kv_quant"]: c for c in plan["candidates"]}
    assert by_kv["f16"]["dominated"] is False
    assert by_kv["q8_0"]["dominated"] is True
    assert by_kv["q4_0"]["dominated"] is True


def test_plan_keeps_every_candidate_inside_its_own_budget(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[30, 60, 200], margin_mib=2)
    for plan in result["plans"]:
        for candidate in plan["candidates"]:
            if not candidate["fits"]:
                continue
            assert candidate["gpu_mib"] <= plan["gpu_budget_mib"] - 2
            assert candidate["within_budget"] is True


def test_plan_reports_budgets_where_nothing_fits(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[1, 200])
    tight, roomy = result["plans"]
    assert tight["fits"] is False
    assert tight["recommended"] is None
    assert all(not c["fits"] and c["reason"] for c in tight["candidates"])
    assert roomy["fits"] is True
    assert any("No configuration fits" in note for note in result["notes"])


def test_plan_candidate_args_state_the_kv_type(estimator):
    plan = make_plan(estimator, gpu_budgets_mib=[200])["plans"][0]
    for candidate in plan["candidates"]:
        args = candidate["llama_server_args"]
        assert f"-c {candidate['n_ctx']}" in args
        if candidate["kv_quant"] == "f16":
            assert "-ctk" not in args
        else:
            assert f"-ctk {candidate['kv_quant']} -ctv {candidate['kv_quant']}" in args


def test_plan_pinning_a_context_makes_offload_the_only_difference(estimator):
    plan = make_plan(estimator, gpu_budgets_mib=[30], n_ctx=1024)["plans"][0]
    assert {c["n_ctx"] for c in plan["candidates"] if c["fits"]} == {1024}


def test_plan_rejects_a_kv_type_llama_cpp_will_not_accept(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[200], kv_cache_types=["f16", "q3_k"])
    assert "q3_k" in result["error"]


def test_plan_refuses_a_grid_larger_than_the_cell_limit(estimator):
    budgets = [1000 + 100 * i for i in range(11)]
    result = make_plan(estimator, gpu_budgets_mib=budgets, kv_cache_types=["f16", "q8_0", "q4_0"])
    assert str(MAX_PLAN_CELLS) in result["error"]


def test_plan_falls_back_to_free_vram_when_no_budget_is_given(estimator, monkeypatch):
    monkeypatch.setattr(
        GgufMemoryEstimator,
        "resolve_budgets",
        staticmethod(
            lambda gpu_budget_mib, host_budget_mib, n_gpu=None, device=None, shared_memory_pool=False: {
                "gpu_budget_mib": 200,
                "host_budget_mib": 10_000,
                "source": "system_monitor",
            }
        ),
    )
    result = estimator.plan("/synthetic/model.gguf", margin_mib=0, n_ctx_max=65536, verify=False)
    assert [p["gpu_budget_mib"] for p in result["plans"]] == [200]


def test_plan_marks_itself_unverified_when_the_binary_is_not_used(estimator):
    result = make_plan(estimator, gpu_budgets_mib=[200])
    assert result["verified"] is False
    assert all(c["gpu_source"] == "analytic" for p in result["plans"] for c in p["candidates"])


# --------------------------------------------------------------------------
# sweep()
# --------------------------------------------------------------------------


def test_sweep_covers_the_grid_and_flags_what_fits(estimator):
    result = estimator.sweep(
        "/synthetic/model.gguf",
        n_gpu_layers_grid=[0, 5],
        n_ctx_grid=[CTX],
        kv_quant_grid=["f16", "q8_0"],
        gpu_budget_mib=20,
        host_budget_mib=10_000,
        margin_mib=0,
    )
    assert len(result["rows"]) == 4
    assert result["truncated"] is False
    by_ngl = {(row["n_gpu_layers"], row["kv_quant"]): row for row in result["rows"]}
    assert by_ngl[(0, "f16")]["fits_gpu"] is True
    assert by_ngl[(5, "f16")]["fits_gpu"] is False
    # Quantizing the KV cache always shrinks the GPU footprint.
    assert by_ngl[(5, "q8_0")]["gpu_mib"] < by_ngl[(5, "f16")]["gpu_mib"]


def test_sweep_defaults_stay_within_the_trained_context(estimator):
    result = estimator.sweep("/synthetic/model.gguf", gpu_budget_mib=100, host_budget_mib=100)
    assert result["grid"]["n_ctx"] == [CTX]
    assert result["grid"]["kv_quant"] == ["f16", "q8_0"]


# --------------------------------------------------------------------------
# Exact probe fallback
# --------------------------------------------------------------------------


def test_probe_exact_returns_none_without_the_binary(estimator, monkeypatch):
    monkeypatch.setattr(estimator, "_resolve_fit_binary", lambda: None)
    assert estimator.probe_exact("/synthetic/model.gguf") is None


def test_estimate_still_works_without_the_binary(estimator, monkeypatch):
    monkeypatch.setattr(estimator, "_resolve_fit_binary", lambda: None)
    result = run(estimator, n_gpu_layers=-1)
    assert result["source"] == "analytic"
    assert gpu(result) > 0


def test_estimate_reports_a_missing_file():
    result = GgufMemoryEstimator().estimate("/definitely/not/here.gguf")
    assert "error" in result
    assert "not found" in result["error"]


# --------------------------------------------------------------------------
# llama-server command-line parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected",
    [
        ("-ngl 20", {"n_gpu_layers": 20}),
        ("--gpu-layers 20", {"n_gpu_layers": 20}),
        ("--n-gpu-layers 20", {"n_gpu_layers": 20}),
        ("--n-gpu-layers=20", {"n_gpu_layers": 20}),
        ("-ngl -1", {"n_gpu_layers": -1}),  # negative value must not read as a flag
        ("-ngl all", {"n_gpu_layers": -1}),
        ("-c 32768", {"n_ctx": 32768}),
        ("--ctx-size 32768", {"n_ctx": 32768}),
        ("-ctk q8_0 -ctv q8_0", {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}),
        ("-ncmoe 24", {"n_cpu_moe": 24}),
        ("--n-cpu-moe 24", {"n_cpu_moe": 24}),
        ("--cpu-moe", {"cpu_moe": True}),
        ("-cmoe", {"cpu_moe": True}),
        ("-ub 1024 -b 4096", {"n_ubatch": 1024, "n_batch": 4096}),
        ("-np 4", {"n_parallel": 4}),
        ("-fa on", {"flash_attn": True}),
        ("-fa off", {"flash_attn": False}),
        ("-fa", {"flash_attn": True}),  # older builds treat it as a bare flag
        ("--no-kv-offload", {"no_kv_offload": True}),
        ("-nkvo", {"no_kv_offload": True}),
        ("--swa-full", {"swa_full": True}),
        ("-ts 3,1", {"tensor_split": [3.0, 1.0]}),
        ("-m /models/a.gguf", {"model_path": "/models/a.gguf"}),
    ],
)
def test_parse_llama_server_args_recognises_every_alias(args, expected):
    assert parse_llama_server_args(args)["settings"] == expected


def test_parse_accepts_an_argv_list_as_well_as_a_string():
    as_list = parse_llama_server_args(["-ngl", "20", "-c", "8192"])
    as_string = parse_llama_server_args("-ngl 20 -c 8192")
    assert as_list["settings"] == as_string["settings"] == {"n_gpu_layers": 20, "n_ctx": 8192}


def test_parse_skips_a_leading_binary_name():
    result = parse_llama_server_args("./build/bin/llama-server -m /x.gguf -ngl 5")
    assert result["settings"] == {"model_path": "/x.gguf", "n_gpu_layers": 5}
    assert result["ignored"] == []


def test_parse_reports_unrelated_flags_instead_of_dropping_them():
    result = parse_llama_server_args("--host 0.0.0.0 --port 5001 --jinja -ngl 5")
    assert result["settings"] == {"n_gpu_layers": 5}
    assert set(result["ignored"]) == {"--host", "--port", "--jinja"}


def test_parse_later_occurrences_win():
    """Matches llama.cpp, and llama_server_extra_args is appended last at runtime."""
    assert parse_llama_server_args("-ngl 10 -ngl 30")["settings"]["n_gpu_layers"] == 30


def test_parse_override_tensor_into_cpu_patterns():
    result = parse_llama_server_args(r'-ot ".ffn_(up|down)_exps=CPU"')
    assert result["settings"]["extra_overrides"] == [".ffn_(up|down)_exps"]


def test_parse_override_tensor_to_a_non_cpu_target_is_reported():
    result = parse_llama_server_args("-ot .ffn_gate_exps=CUDA0")
    assert "extra_overrides" not in result["settings"]
    assert any("CUDA0" in w for w in result["warnings"])


def test_parse_warns_on_ngl_auto():
    result = parse_llama_server_args("-ngl auto")
    assert result["settings"]["n_gpu_layers"] == -1
    assert any("auto" in w for w in result["warnings"])


def test_parse_warns_on_unparsable_value():
    result = parse_llama_server_args("-c banana")
    assert "n_ctx" not in result["settings"]
    assert any("banana" in w for w in result["warnings"])


def test_override_tensor_moves_matching_weights_to_the_host(estimator):
    """-ot must reach the same placement machinery as --n-cpu-moe."""
    result = run(estimator, n_gpu_layers=-1, extra_overrides=[r"\.ffn_gate_exps"])
    assert host(result) == pytest.approx(2 + 4 * 4)
    assert gpu(result, "context") == pytest.approx(4 * KV_PER_LAYER_MIB)


# --------------------------------------------------------------------------
# Merging parsed args with structured fields
# --------------------------------------------------------------------------


def test_apply_cli_args_overrides_the_structured_fields():
    merged, report = apply_cli_args({"n_gpu_layers": 10, "n_ctx": 4096}, "-ngl 30")
    assert merged == {"n_gpu_layers": 30, "n_ctx": 4096}
    assert report["parsed"] == {"n_gpu_layers": 30}


def test_apply_cli_args_without_args_is_a_passthrough():
    merged, report = apply_cli_args({"n_ctx": 4096}, None)
    assert merged == {"n_ctx": 4096}
    assert report == {}


@pytest.mark.parametrize(
    "solved_keys, args, dropped",
    [
        (SWEEP_SOLVED_KEYS, "-ngl 20", "n_gpu_layers"),
        (SWEEP_SOLVED_KEYS, "-ctk q8_0", "cache_type_k"),
        (RECOMMEND_SOLVED_KEYS, "-ncmoe 8", "n_cpu_moe"),
        (RECOMMEND_SOLVED_KEYS, "-ngl 20", "n_gpu_layers"),
    ],
)
def test_apply_cli_args_reports_values_the_endpoint_solves_for(solved_keys, args, dropped):
    merged, report = apply_cli_args({}, args, solved_keys=solved_keys)
    assert dropped not in merged
    assert any(dropped in w for w in report["warnings"])


def test_apply_cli_args_keeps_values_the_endpoint_does_not_solve_for():
    merged, _ = apply_cli_args({}, "-ngl 20 -ub 256", solved_keys=SWEEP_SOLVED_KEYS)
    assert merged == {"n_ubatch": 256}


def test_parsed_args_reach_the_estimate(estimator):
    """A pasted command line must produce the same result as the structured fields."""
    merged, _ = apply_cli_args(
        {"n_gpu_layers": -1, "n_ctx": CTX}, "-ngl 2 -ctk q8_0 -ctv q8_0 -ncmoe 1"
    )
    from_args = estimator.estimate("/synthetic/model.gguf", include_per_layer=False, **merged)
    structured = run(
        estimator, n_gpu_layers=2, cache_type_k="q8_0", cache_type_v="q8_0", n_cpu_moe=1
    )
    assert from_args["memory_breakdown_mib"] == structured["memory_breakdown_mib"]


# --------------------------------------------------------------------------
# Probe binary resolution and live hardware budgets
# --------------------------------------------------------------------------


def _make_executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    "name, expected_args",
    [("llama-fit-params", []), ("llama", ["fit-params"]), ("llama.exe", ["fit-params"])],
)
def test_fit_cmd_prefix_adds_the_subcommand_only_for_the_unified_binary(name, expected_args):
    binary = f"/opt/bin/{name}"
    assert GgufMemoryEstimator._fit_cmd_prefix(binary) == [binary, *expected_args]


def test_resolve_fit_binary_falls_back_to_the_prebuilt_unified_llama(tmp_path, monkeypatch):
    """setup_env only sparse-checks out the Python convert tooling, so no build/bin exists."""
    from service.inference import gguf_estimator as module

    unified = _make_executable(tmp_path / "llama")
    monkeypatch.delenv("LLAMA_FIT_PARAMS_BIN", raising=False)
    monkeypatch.setattr(module.settings, "LLAMA_CPP_DIR", tmp_path / "llama.cpp")
    monkeypatch.setattr(module.settings, "LLAMA_SERVER_BINARY", unified)

    assert GgufMemoryEstimator()._resolve_fit_binary() == unified


def test_resolve_fit_binary_ignores_a_llama_server_that_has_no_fit_params(tmp_path, monkeypatch):
    """A self-built llama-server accepts no fit-params subcommand, so it must not be used."""
    from service.inference import gguf_estimator as module

    server = _make_executable(tmp_path / "llama-server")
    monkeypatch.delenv("LLAMA_FIT_PARAMS_BIN", raising=False)
    # The resolver also looks beside LLAMA_SERVER_BINARY, and tmp_path holds only the server
    # binary - no llama-fit-params sibling - which is the case being asserted.
    monkeypatch.setattr(module.settings, "LLAMA_CPP_DIR", tmp_path / "llama.cpp")
    monkeypatch.setattr(module.settings, "LLAMA_SERVER_BINARY", server)

    assert GgufMemoryEstimator()._resolve_fit_binary() is None


def test_resolve_fit_binary_prefers_a_standalone_source_build(tmp_path, monkeypatch):
    from service.inference import gguf_estimator as module

    build_bin = tmp_path / "llama.cpp" / "build" / "bin"
    build_bin.mkdir(parents=True)
    # Matches the name the resolver looks for; it is suffixed on Windows.
    standalone = _make_executable(
        build_bin / ("llama-fit-params.exe" if os.name == "nt" else "llama-fit-params")
    )
    monkeypatch.delenv("LLAMA_FIT_PARAMS_BIN", raising=False)
    monkeypatch.setattr(module.settings, "LLAMA_CPP_DIR", tmp_path / "llama.cpp")
    monkeypatch.setattr(
        module.settings, "LLAMA_SERVER_BINARY", _make_executable(tmp_path / "llama")
    )

    assert GgufMemoryEstimator()._resolve_fit_binary() == standalone


class _FakeGpu:
    def __init__(self, free_gb, name="FakeGPU", is_integrated=False, vendor=None):
        self.free_gb = free_gb
        self.name = name
        self.is_integrated = is_integrated
        self.vendor = vendor


class _FakeGpuResource:
    def __init__(self, gpus):
        self.available = True
        self.gpus = gpus


def _patch_system_monitor(monkeypatch, gpus, dram_free_gb=8.0):
    from service.utils import system_monitor as module

    class _FakeMonitor:
        @staticmethod
        def get_gpu_resource(_mode):
            return _FakeGpuResource(gpus)

        @staticmethod
        def get_cpu_resource(_mode):
            return type("_Cpu", (), {"dram": _FakeGpu(dram_free_gb)})()

    monkeypatch.setattr(module, "system_monitor", _FakeMonitor())


def test_resolve_budgets_skips_gpus_whose_free_memory_is_unreadable(monkeypatch):
    """Intel/DXGI and the generic GPU path report free_gb=None; summing it would raise."""
    _patch_system_monitor(monkeypatch, [_FakeGpu(4.0), _FakeGpu(None)])

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] == pytest.approx(4.0 * 1024)
    assert result["gpu_count"] == 1
    assert result["source"] == "system_monitor"


def test_resolve_budgets_leaves_the_gpu_budget_unset_when_no_gpu_is_readable(monkeypatch):
    _patch_system_monitor(monkeypatch, [_FakeGpu(None), _FakeGpu(None)])

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] is None
    assert "gpu_count" not in result
    assert result["host_budget_mib"] == pytest.approx(8.0 * 1024)


def test_resolve_budgets_excludes_integrated_gpus(monkeypatch):
    """An iGPU's free "VRAM" is system RAM; counting it invents a budget nothing fits in.

    Measured on a 5060 Ti box with Intel graphics: the card had ~14.6 GB free and
    the iGPU claimed ~100 GB, which summed to a 117 GB budget and made the solver
    recommend a full offload that could not load.
    """
    _patch_system_monitor(
        monkeypatch,
        [
            _FakeGpu(14.64, name="NVIDIA GeForce RTX 5060 Ti"),
            _FakeGpu(100.34, name="Intel(R) Graphics", is_integrated=True),
        ],
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] == pytest.approx(14.64 * 1024)
    assert result["gpu_count"] == 1
    assert result["gpu_devices"] == ["NVIDIA GeForce RTX 5060 Ti"]
    assert result["gpu_excluded_integrated"] == ["Intel(R) Graphics"]


def test_resolve_budgets_pools_only_as_many_gpus_as_take_part(monkeypatch):
    """Pooling VRAM is real only when the model is split across cards.

    The request already says how many GPUs participate (n_gpu, default 1), so a
    single-GPU load must be sized against one card -- the largest -- not against
    the sum of every card present.
    """
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(8.0, name="GPU-small"), _FakeGpu(20.0, name="GPU-big")],
    )

    one = GgufMemoryEstimator.resolve_budgets(None, None, 1)
    assert one["gpu_budget_mib"] == pytest.approx(20.0 * 1024)
    assert one["gpu_count"] == 1
    assert one["gpu_devices"] == ["GPU-big"]
    assert one["gpu_devices_not_counted"] == ["GPU-small"]

    both = GgufMemoryEstimator.resolve_budgets(None, None, 2)
    assert both["gpu_budget_mib"] == pytest.approx(28.0 * 1024)
    assert both["gpu_count"] == 2
    assert "gpu_devices_not_counted" not in both


def test_resolve_budgets_without_n_gpu_keeps_summing(monkeypatch):
    """No declared count -- keep the old behaviour rather than guessing at 1."""
    _patch_system_monitor(
        monkeypatch, [_FakeGpu(8.0, name="GPU-small"), _FakeGpu(20.0, name="GPU-big")]
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] == pytest.approx(28.0 * 1024)
    assert result["gpu_count"] == 2


def test_resolve_budgets_falls_back_to_an_igpu_only_machine(monkeypatch):
    """With no discrete card there is still a target -- an iGPU box must get a budget.

    The exclusion exists to stop shared memory being ADDED to a real card, not to
    leave a machine that only has integrated graphics with no answer at all.
    """
    _patch_system_monitor(
        monkeypatch, [_FakeGpu(30.0, name="Intel(R) Graphics", is_integrated=True)]
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] == pytest.approx(30.0 * 1024)
    assert result["gpu_count"] == 1
    assert result["gpu_budget_is_shared_memory"] is True
    assert "gpu_excluded_integrated" not in result


def test_resolve_budgets_keeps_discrete_intel_gpus(monkeypatch):
    """A discrete Arc is a real load target -- only shared-memory adapters are dropped."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(20.0, name="Intel(R) Arc(TM) Pro B70", is_integrated=False)],
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None)

    assert result["gpu_budget_mib"] == pytest.approx(20.0 * 1024)
    assert result["gpu_count"] == 1
    assert "gpu_excluded_integrated" not in result


# --------------------------------------------------------------------------
# Cross-check against llama.cpp itself (skipped when unavailable)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _fit_binary_compute(binary: str) -> str | None:
    """
    Which compute backend the resolved llama-fit-params has, via scripts/llama_backend.py.

    The baselines below were measured on CUDA/Vulkan. A CPU-only install is a legitimate
    configuration - `-LlamaBackend cpu` / TRUSTA_LLAMA_BACKEND=cpu - and would compare GPU
    expectations against CPU numbers, so those tests skip rather than fail. Returns None when
    the probe itself cannot run, which also means "do not judge".
    """
    probe = Path(__file__).resolve().parents[3] / "scripts" / "llama_backend.py"
    if not probe.is_file():
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(probe), binary, "--json"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return json.loads(out.stdout).get("compute")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _require_gpu_fit_binary(est: GgufMemoryEstimator) -> str:
    """Skip unless a llama-fit-params with a GPU backend is installed; returns its path."""
    binary = est._resolve_fit_binary()
    if binary is None:
        pytest.skip("llama-fit-params is not built")
    compute = _fit_binary_compute(binary)
    if compute in {"cpu", "other"}:
        pytest.skip(f"llama-fit-params reports the {compute} backend; these baselines are GPU ones")
    return binary


@pytest.mark.parametrize(
    "n_gpu_layers, n_ctx, kv",
    [(0, 4096, "f16"), (1, 4096, "f16"), (12, 8192, "f16"), (-1, 4096, "q8_0"), (-1, 2048, "f16")],
)
def test_analytic_matches_llama_fit_params(n_gpu_layers, n_ctx, kv):
    """Weights and KV cache must match llama.cpp exactly; compute may only over-estimate."""
    import model_fixtures

    est = GgufMemoryEstimator()
    _require_gpu_fit_binary(est)
    try:
        model_path = model_fixtures.ensure_gguf()
    except model_fixtures.ModelUnavailable as e:
        pytest.skip(str(e))

    kwargs = {"n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx, "cache_type_k": kv, "cache_type_v": kv}
    analytic = est.estimate(model_path, include_per_layer=False, **kwargs)
    exact = est.probe_exact(model_path, **kwargs)
    if exact is None:
        pytest.skip("llama-fit-params probe failed on this machine")

    a = analytic["memory_breakdown_mib"]["gpu_total"]
    x = exact["memory_breakdown_mib"]["gpu_total"]
    # llama.cpp truncates to whole MiB, so allow one unit of rounding slack.
    assert a["model"] == pytest.approx(x["model"], abs=1.0)
    assert a["context"] == pytest.approx(x["context"], abs=1.0)
    assert a["compute"] >= x["compute"] - 8.0, "compute estimate must not under-report"


@pytest.mark.parametrize("n_ctx", [4096, 8192, 16384, 32768])
def test_disabling_flash_attention_does_not_add_to_the_buffer(n_ctx):
    """
    The score matrix shares the compute buffer rather than adding to it.

    -fa off materialises QK^T, but that tensor does not coexist with the logits and
    mask ones, so ggml's allocator reuses the space: the buffer tracks the largest of
    the three. Adding the score term instead put the estimate 101% high at n_ctx 8192,
    where measurement shows turning flash attention off costs nothing at all — the
    buffer stays at its baseline until the scores tensor grows past it.

    Measured on Qwen2.5-0.5B-Q8_0 at n_ubatch 512, identical on CUDA and Vulkan:

        n_ctx    -fa on    -fa off
         4096       298        298
         8192       298        298
        16384       298        487
        32768       298        967
    """
    import model_fixtures

    est = GgufMemoryEstimator()
    _require_gpu_fit_binary(est)
    try:
        model_path = model_fixtures.ensure_gguf()
    except model_fixtures.ModelUnavailable as e:
        pytest.skip(str(e))

    kwargs = {"n_gpu_layers": -1, "n_ctx": n_ctx, "include_per_layer": False}
    off = est.estimate(model_path, flash_attn=False, **kwargs)
    exact = est.probe_exact(model_path, n_gpu_layers=-1, n_ctx=n_ctx, flash_attn=False)
    if exact is None:
        pytest.skip("llama-fit-params probe failed on this machine")

    estimated = off["memory_breakdown_mib"]["gpu_total"]["compute"]
    measured = exact["memory_breakdown_mib"]["gpu_total"]["compute"]

    assert estimated >= measured - 8.0, "compute estimate must not under-report"
    # Summing the terms rather than taking the max lands near double at small n_ctx;
    # 40% leaves room for the safety factor without letting that back in.
    assert estimated <= measured * 1.4, (
        f"compute over-reported: {estimated:.0f} vs {measured:.0f} MiB measured"
    )


# --------------------------------------------------------------------------
# Pinned --device: the budget is that one adapter's, or nothing is claimed
# --------------------------------------------------------------------------

_TWO_ADAPTERS = [
    ("NVIDIA GeForce RTX 5060 Ti", 14.6, False, "nvidia"),
    ("Intel(R) Graphics", 100.3, True, "intel"),
]


@pytest.mark.parametrize(
    "device, expected",
    [
        ("CUDA0", ("NVIDIA GeForce RTX 5060 Ti", 14.6, False, "nvidia")),
        # Pinning the iGPU is a real choice; the record carries that it is one.
        ("SYCL0", ("Intel(R) Graphics", 100.3, True, "intel")),
        ("HIP0", None),  # no AMD device present
        ("Vulkan0", None),  # Vulkan does not narrow the vendor
        ("Vulkan1", None),
        ("", None),
        ("not a device", None),
    ],
)
def test_pick_pinned_device(device, expected):
    assert GgufMemoryEstimator._pick_pinned_device(device, _TWO_ADAPTERS) == expected


def test_resolve_budgets_uses_only_the_pinned_device(monkeypatch):
    """--device CUDA0 means the load runs on that card, so that card is the budget."""
    _patch_system_monitor(
        monkeypatch,
        [
            _FakeGpu(14.6, name="NVIDIA GeForce RTX 5060 Ti", vendor="nvidia"),
            _FakeGpu(100.3, name="Intel(R) Graphics", is_integrated=True, vendor="intel"),
        ],
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1, "CUDA0")

    assert result["gpu_budget_mib"] == pytest.approx(14.6 * 1024)
    assert result["gpu_devices"] == ["NVIDIA GeForce RTX 5060 Ti"]
    assert result["gpu_budget_pinned_device"] == "CUDA0"


def test_resolve_budgets_honours_a_deliberately_pinned_igpu(monkeypatch):
    """Excluding integrated adapters is about not ADDING them, not about banning them."""
    _patch_system_monitor(
        monkeypatch,
        [
            _FakeGpu(14.6, name="NVIDIA GeForce RTX 5060 Ti", vendor="nvidia"),
            _FakeGpu(100.3, name="Intel(R) Graphics", is_integrated=True, vendor="intel"),
        ],
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1, "SYCL0")

    assert result["gpu_budget_mib"] == pytest.approx(100.3 * 1024)
    assert result["gpu_devices"] == ["Intel(R) Graphics"]


def test_resolve_budgets_reports_an_unresolvable_device_instead_of_guessing(monkeypatch):
    """Vulkan enumerates its own order, so an index cannot be trusted -- say so."""
    _patch_system_monitor(
        monkeypatch,
        [
            _FakeGpu(14.6, name="NVIDIA GeForce RTX 5060 Ti", vendor="nvidia"),
            _FakeGpu(100.3, name="Intel(R) Graphics", is_integrated=True, vendor="intel"),
        ],
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1, "Vulkan1")

    assert result["gpu_budget_device_unresolved"] == "Vulkan1"
    # Falls back to the discrete card rather than refusing to answer at all.
    assert result["gpu_budget_mib"] == pytest.approx(14.6 * 1024)
    assert "gpu_budget_pinned_device" not in result


# --------------------------------------------------------------------------
# Shared memory: an iGPU's budget is a slice of DRAM, not memory on top of it
#
# Measured shape of the bug: a 60 GB "VRAM" budget and a 100 GB host budget on a
# box with 100 GB of DRAM. Each ceiling passed on its own, so a load needing 60 +
# 60 was reported as fitting twice over.
# --------------------------------------------------------------------------


def test_resolve_budgets_reserves_a_shared_gpu_budget_out_of_dram(monkeypatch):
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(60.0, name="Intel(R) Arc(TM) 140V", is_integrated=True)],
        dram_free_gb=100.0,
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1)

    assert result["gpu_budget_is_shared_memory"] is True
    assert result["gpu_budget_mib"] == pytest.approx(60.0 * 1024)
    assert result["host_budget_reserved_for_gpu_mib"] == pytest.approx(60.0 * 1024)
    assert result["host_budget_mib"] == pytest.approx(40.0 * 1024)
    # The two ceilings together may never promise more than the box holds.
    assert result["gpu_budget_mib"] + result["host_budget_mib"] == pytest.approx(100.0 * 1024)


def test_resolve_budgets_reserves_for_a_deliberately_pinned_igpu(monkeypatch):
    """Pinning an iGPU is allowed; it does not stop that memory from being DRAM."""
    _patch_system_monitor(
        monkeypatch,
        [
            _FakeGpu(14.6, name="NVIDIA GeForce RTX 5060 Ti", vendor="nvidia"),
            _FakeGpu(60.0, name="Intel(R) Graphics", is_integrated=True, vendor="intel"),
        ],
        dram_free_gb=100.0,
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1, "SYCL0")

    assert result["gpu_budget_is_shared_memory"] is True
    assert result["host_budget_mib"] == pytest.approx(40.0 * 1024)


def test_resolve_budgets_keeps_discrete_pools_side_by_side(monkeypatch):
    """VRAM and DRAM really are two pools; subtracting one from the other would lie."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(14.6, name="NVIDIA GeForce RTX 5060 Ti")],
        dram_free_gb=100.0,
    )

    result = GgufMemoryEstimator.resolve_budgets(None, None, 1)

    assert result["host_budget_mib"] == pytest.approx(100.0 * 1024)
    assert "host_budget_reserved_for_gpu_mib" not in result
    assert "gpu_budget_is_shared_memory" not in result


def test_resolve_budgets_never_reserves_out_of_a_stated_host_budget(monkeypatch):
    """A budget the caller stated is theirs to reconcile, shared pool or not."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(60.0, name="Intel(R) Graphics", is_integrated=True)],
        dram_free_gb=100.0,
    )

    result = GgufMemoryEstimator.resolve_budgets(None, 8192.0, 1)

    assert result["host_budget_mib"] == 8192.0
    assert "host_budget_reserved_for_gpu_mib" not in result


def test_budget_charges_move_host_state_to_the_gpu_side_on_a_shared_pool():
    """Every byte is charged once, so the two ceilings can still be checked apart."""
    est = {
        "memory_breakdown_mib": {
            "gpu_total": {"total": 100.0},
            "host": {"model": 500.0, "context": 30.0, "compute": 7.0, "total": 537.0},
        }
    }

    assert GgufMemoryEstimator.budget_charges(est, shared_pool=False) == (100.0, 537.0)
    # The KV cache and compute buffers cross over; the mmapped weights do not,
    # and nothing is counted on both sides.
    assert GgufMemoryEstimator.budget_charges(est, shared_pool=True) == (137.0, 500.0)


def test_recommend_keeps_a_shared_pool_inside_one_budget(estimator, monkeypatch):
    """On an iGPU the host side eats the same DRAM, so it has to fit in the budget too."""
    budget_mib = 25.0
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(budget_mib / 1024, name="Intel(R) Graphics", is_integrated=True)],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.recommend("/synthetic/model.gguf", n_ctx=CTX, margin_mib=0, verify=False)

    breakdown = result["memory_breakdown_mib"]
    pool_used = (
        breakdown["gpu_total"]["total"]
        + breakdown["host"]["context"]
        + breakdown["host"]["compute"]
    )
    assert pool_used <= budget_mib
    assert result["utilization"]["includes_host_nonreclaimable"] is True
    assert any("shares system RAM" in note for note in result["notes"])
    # The reported GPU figure stays the GPU's, not the pool's.
    assert result["recommended"]["gpu_mib"] == pytest.approx(breakdown["gpu_total"]["total"])


def test_recommend_leaves_a_discrete_budget_measured_against_the_gpu_alone(estimator, monkeypatch):
    """The shared-pool rule must not leak into the normal two-pool case."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(25.0 / 1024, name="NVIDIA GeForce RTX 5060 Ti")],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.recommend("/synthetic/model.gguf", n_ctx=CTX, margin_mib=0, verify=False)

    assert result["utilization"]["includes_host_nonreclaimable"] is False
    assert result["utilization"]["analytic_mib"] == pytest.approx(
        result["memory_breakdown_mib"]["gpu_total"]["total"]
    )


# --------------------------------------------------------------------------
# The property has to survive being passed as numbers
#
# Every endpoint resolves budgets once and then hands floats to the solver. While
# shared-pool lived only in the dict resolve_budgets returned, that hand-off took
# the early "source: caller" return, the flag vanished, and the whole shared-pool
# rule switched itself off on exactly the paths the API uses.
# --------------------------------------------------------------------------


def test_recommend_keeps_the_pool_rule_when_the_budgets_arrive_as_numbers(estimator):
    budget_mib = 25.0

    stated = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=budget_mib,
        host_budget_mib=10_000.0,
        margin_mib=0,
        verify=False,
        shared_memory_pool=True,
    )

    breakdown = stated["memory_breakdown_mib"]
    pool_used = (
        breakdown["gpu_total"]["total"]
        + breakdown["host"]["context"]
        + breakdown["host"]["compute"]
    )
    assert pool_used <= budget_mib
    assert stated["utilization"]["includes_host_nonreclaimable"] is True
    assert stated["budgets_mib"]["gpu_budget_is_shared_memory"] is True


def test_recommend_without_the_flag_sizes_against_the_gpu_alone(estimator):
    """The counterpart: nobody said shared, so the normal two-pool rule applies."""
    result = estimator.recommend(
        "/synthetic/model.gguf",
        n_ctx=CTX,
        gpu_budget_mib=25.0,
        host_budget_mib=10_000.0,
        margin_mib=0,
        verify=False,
    )

    assert result["utilization"]["includes_host_nonreclaimable"] is False
    assert result["memory_breakdown_mib"]["gpu_total"]["total"] <= 25.0


def test_plan_hands_the_pool_rule_to_every_candidate(estimator, monkeypatch):
    """plan() solves through recommend(), so the property has to reach each call."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(30.0 / 1024, name="Intel(R) Graphics", is_integrated=True)],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.plan("/synthetic/model.gguf", n_ctx=CTX, margin_mib=0, verify=False)

    plan = result["plans"][0]
    assert plan["gpu_budget_mib"] == pytest.approx(30.0)
    for candidate in plan["candidates"]:
        if candidate["fits"]:
            assert candidate["gpu_mib"] <= 30.0


def test_plan_gives_every_candidate_the_same_host_budget(estimator, monkeypatch):
    """Passing the raw argument down made each candidate re-read live DRAM."""
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(30.0 / 1024, name="Intel(R) Graphics", is_integrated=True)],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.plan("/synthetic/model.gguf", n_ctx=CTX, margin_mib=0, verify=False)

    # The reservation is visible at the top and is what the candidates were solved
    # against: 10000 MiB of DRAM less the 30 MiB the adapter borrows from it.
    assert result["host_budget_mib"] == pytest.approx(10_000.0 - 30.0)


def test_sweep_judges_a_shared_pool_row_against_one_pool(estimator, monkeypatch):
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(25.0 / 1024, name="Intel(R) Graphics", is_integrated=True)],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.sweep(
        "/synthetic/model.gguf",
        n_gpu_layers_grid=[-1],
        n_ctx_grid=[CTX],
        kv_quant_grid=["f16"],
        margin_mib=0,
    )

    row = result["rows"][0]
    assert row["gpu_charge_mib"] > row["gpu_mib"]  # host KV and compute crossed over
    assert row["host_charge_mib"] < row["host_mib"]  # ...and left the host side
    assert row["fits_gpu"] is (row["gpu_charge_mib"] <= 25.0)


def test_sweep_leaves_a_discrete_row_charged_to_its_own_pools(estimator, monkeypatch):
    _patch_system_monitor(
        monkeypatch,
        [_FakeGpu(25.0 / 1024, name="NVIDIA GeForce RTX 5060 Ti")],
        dram_free_gb=10_000 / 1024,
    )

    result = estimator.sweep(
        "/synthetic/model.gguf",
        n_gpu_layers_grid=[-1],
        n_ctx_grid=[CTX],
        kv_quant_grid=["f16"],
        margin_mib=0,
    )

    row = result["rows"][0]
    assert row["gpu_charge_mib"] == row["gpu_mib"]
    assert row["host_charge_mib"] == row["host_mib"]
