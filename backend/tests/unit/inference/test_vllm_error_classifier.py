"""Unit tests for service.inference.engines.vllm_error_classifier.

The classifier is a pure, dependency-free function that turns raw vLLM stderr
lines into a structured VllmErrorReport. These tests focus on edge cases:
empty / None / whitespace input, category priority (first matching line +
first matching rule wins), case-insensitivity, the important-vs-tail fallback,
line caps, hint-line selection for the summary, and to_text() precedence.
"""

import pytest

from service.inference.engines.vllm_error_classifier import (
    classify_stderr,
    VllmErrorReport,
    ERROR_OOM,
    ERROR_CONFIG,
    ERROR_PORT_BUSY,
    ERROR_MODEL_NOT_FOUND,
    ERROR_CUDA_MISMATCH,
    ERROR_GPU_FAULT,
    ERROR_SHARED_LIBRARY_MISSING,
    ERROR_TEMPLATE,
    ERROR_TOOLCHAIN_MISSING,
    ERROR_QUANTIZATION,
    ERROR_UNKNOWN,
)


# --------------------------------------------------------------------------
# Empty / degenerate input
# --------------------------------------------------------------------------

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]


def test_empty_iterable_returns_unknown_with_no_logs_summary():
    report = classify_stderr([])
    assert report.category == ERROR_UNKNOWN
    assert report.summary == "No recent vLLM stderr logs found."
    assert report.important_lines == []
    assert report.tail_lines == []
    # to_text() falls back to the summary when nothing else is present.
    assert report.to_text() == "No recent vLLM stderr logs found."


def test_none_and_blank_lines_are_dropped_and_treated_as_empty():
    report = classify_stderr([None, "", "   ", "\n", "\t "])
    assert report.category == ERROR_UNKNOWN
    assert report.summary == "No recent vLLM stderr logs found."


def test_none_entries_are_skipped_but_real_lines_kept():
    report = classify_stderr([None, "  RuntimeError: CUDA out of memory  ", None])
    assert report.category == ERROR_OOM
    # Whitespace/newlines are stripped from retained lines.
    assert report.important_lines == ["RuntimeError: CUDA out of memory"]


# --------------------------------------------------------------------------
# Per-category detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line, expected",
    [
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2GB", ERROR_OOM),
        # Config, not OOM: vLLM's own remedy for this one is "increase
        # gpu_memory_utilization" -- a setting, not a bigger card.
        ("ValueError: No available memory for the cache blocks", ERROR_CONFIG),
        ("Please try lowering max_num_seqs when warming up sampler failed", ERROR_OOM),
        ("OSError: [Errno 98] Address already in use", ERROR_PORT_BUSY),
        ("ImportError: libcudnn.so.8: cannot open shared object file", ERROR_SHARED_LIBRARY_MISSING),
        ("OSError: model-x is not a local folder and repository not found", ERROR_MODEL_NOT_FOUND),
        ("RuntimeError: CUDA error: no kernel image is available for execution", ERROR_CUDA_MISMATCH),
        ("jinja2.exceptions.TemplateError: chat template render failed", ERROR_TEMPLATE),
        ("ValueError: Unknown quantization method awq requested", ERROR_QUANTIZATION),
        # Seen on an Intel laptop without intel-ocloc; it used to read as a missing model.
        (
            "torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code"
            " block: [Errno 2] No such file or directory: 'ocloc'",
            ERROR_TOOLCHAIN_MISSING,
        ),
        ("RuntimeError: Could not find nvcc and default cuda_home doesn't exist", ERROR_TOOLCHAIN_MISSING),
        (
            "ValueError: Free memory on device xpu:0 (7.03/28.56 GiB) on startup is less than"
            " desired GPU memory utilization (0.5, 14.28 GiB).",
            ERROR_CONFIG,
        ),
    ],
)
def test_single_line_category_detection(line, expected):
    assert classify_stderr([line]).category == expected


def test_detection_is_case_insensitive():
    assert classify_stderr(["FATAL: OUT OF MEMORY"]).category == ERROR_OOM


def test_unmatched_lines_yield_unknown_category():
    report = classify_stderr(["INFO: engine failed to start for reasons"])
    assert report.category == ERROR_UNKNOWN
    # But "failed" is an important keyword, so the line is still surfaced.
    assert report.important_lines == ["INFO: engine failed to start for reasons"]


# --------------------------------------------------------------------------
# Category priority: first categorisable line wins, and within a line the
# rule order (OOM before CUDA_MISMATCH, etc.) decides the label.
# --------------------------------------------------------------------------

def test_oom_rule_takes_priority_over_cuda_on_same_line():
    # "cuda out of memory" matches OOM (earlier rule) not CUDA_MISMATCH.
    assert classify_stderr(["CUDA out of memory"]).category == ERROR_OOM


def test_first_categorisable_line_determines_category():
    lines = [
        "ValueError: quantization config awq is invalid",  # -> quantization (first)
        "RuntimeError: CUDA out of memory",                # OOM, but comes later
    ]
    assert classify_stderr(lines).category == ERROR_QUANTIZATION


# --------------------------------------------------------------------------
# important vs tail fallback
# --------------------------------------------------------------------------

def test_category_line_without_important_keyword_uses_tail_fallback():
    # "address already in use" matches PORT_BUSY but contains no important
    # keyword (error/failed/exception/...), so important_lines stays empty and
    # tail_lines carries the context instead.
    report = classify_stderr(["address already in use"])
    assert report.category == ERROR_PORT_BUSY
    assert report.important_lines == []
    assert report.tail_lines == ["address already in use"]
    assert report.to_text() == "address already in use"
    assert report.summary == "vLLM port already bound: address already in use"


def test_tail_is_empty_when_important_lines_present():
    report = classify_stderr(["Traceback (most recent call last):", "RuntimeError: boom"])
    assert report.important_lines  # non-empty
    assert report.tail_lines == []


# --------------------------------------------------------------------------
# Line caps
# --------------------------------------------------------------------------

def test_important_lines_are_capped_to_last_max_important():
    lines = [f"Error line {i}" for i in range(20)]
    report = classify_stderr(lines)  # default max_important=16
    assert len(report.important_lines) == 16
    # Keeps the *last* 16 (indices 4..19).
    assert report.important_lines[0] == "Error line 4"
    assert report.important_lines[-1] == "Error line 19"


def test_custom_max_important_is_respected():
    lines = [f"failed step {i}" for i in range(5)]
    report = classify_stderr(lines, max_important=2)
    assert report.important_lines == ["failed step 3", "failed step 4"]


def test_tail_lines_are_capped_to_last_max_tail():
    lines = [f"info step {i}" for i in range(30)]  # none important, none match a rule
    report = classify_stderr(lines)  # default max_tail=25
    assert report.category == ERROR_UNKNOWN
    assert report.important_lines == []
    assert len(report.tail_lines) == 25
    assert report.tail_lines[0] == "info step 5"
    assert report.tail_lines[-1] == "info step 29"


def test_custom_max_tail_is_respected():
    lines = [f"info step {i}" for i in range(30)]
    report = classify_stderr(lines, max_tail=3)
    assert report.tail_lines == ["info step 27", "info step 28", "info step 29"]


# --------------------------------------------------------------------------
# Summary hint-line selection
# --------------------------------------------------------------------------

def test_summary_prefers_actionable_hint_line_over_last_line():
    lines = [
        "torch.OutOfMemoryError: CUDA out of memory",
        # "ValueError" makes this line important; it also carries an actionable hint.
        "ValueError: to serve, please try lowering gpu_memory_utilization",
    ]
    report = classify_stderr(lines)
    assert report.category == ERROR_OOM
    assert report.summary.startswith("GPU/CPU memory exhausted: ")
    assert "please try lowering gpu_memory_utilization" in report.summary


def test_summary_falls_back_to_last_line_without_hint():
    report = classify_stderr(["RuntimeError: something broke", "Exception: final boom"])
    assert report.summary.endswith("Exception: final boom")


# --------------------------------------------------------------------------
# VllmErrorReport.to_text precedence: important > tail > summary
# --------------------------------------------------------------------------

def test_to_text_prefers_important_lines():
    report = VllmErrorReport(
        category=ERROR_OOM,
        summary="ignored summary",
        important_lines=["a", "b"],
        tail_lines=["x", "y"],
    )
    assert report.to_text() == "a\nb"


def test_to_text_uses_tail_when_no_important_lines():
    report = VllmErrorReport(summary="ignored", tail_lines=["x", "y"])
    assert report.to_text() == "x\ny"


def test_to_text_uses_summary_as_last_resort():
    report = VllmErrorReport(summary="only summary")
    assert report.to_text() == "only summary"


# --------------------------------------------------------------------------
# Config errors are not OOM
#
# Four observed misclassifications. Three startup ValueErrors were reported as
# `oom`, which sends the operator after a bigger card when the fix is one
# setting, and a GPU fault was reported as `cuda_mismatch`, whose remedy
# (reinstall a matching build) has nothing to do with the actual problem.
# Message text is taken verbatim from vLLM 0.28.
# --------------------------------------------------------------------------


def test_max_num_seqs_over_the_mamba_cache_limit_is_config_not_oom():
    report = classify_stderr(
        [
            "ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (128). "
            "Each decode sequence requires one Mamba cache block, so CUDA graph capture "
            "cannot proceed. Please lower max_num_seqs to at most 128 or increase "
            "gpu_memory_utilization."
        ]
    )
    assert report.category == ERROR_CONFIG


def test_kv_cache_too_small_for_max_model_len_is_config_not_oom():
    report = classify_stderr(
        [
            "ValueError: To serve at least one request with the model's max seq len "
            "(131072), (18.50 GiB KV cache is needed, which is larger than the available "
            "KV cache memory (6.20 GiB). Try increasing `gpu_memory_utilization` or "
            "decreasing `max_model_len`."
        ]
    )
    assert report.category == ERROR_CONFIG


def test_unmergeable_kv_specs_is_config_not_oom():
    report = classify_stderr(
        ["ValueError: Failed to promote local KV cache specs to one unified type."]
    )
    assert report.category == ERROR_CONFIG


def test_no_memory_for_cache_blocks_is_config_not_oom():
    report = classify_stderr(
        [
            "ValueError: No available memory for the cache blocks. Try increasing "
            "`gpu_memory_utilization` when initializing the engine."
        ]
    )
    assert report.category == ERROR_CONFIG


@pytest.mark.parametrize(
    "line",
    [
        "RuntimeError: CUDA error: uncorrectable ECC error encountered",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "RuntimeError: CUDA error: device-side assert triggered",
        "RuntimeError: CUDA error: unspecified launch failure",
    ],
)
def test_gpu_faults_are_not_reported_as_a_build_mismatch(line):
    assert classify_stderr([line]).category == ERROR_GPU_FAULT


@pytest.mark.parametrize(
    "line",
    [
        "RuntimeError: CUDA error: no kernel image is available for execution on the device",
        "NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible",
        "RuntimeError: The detected CUDA version mismatch with the version that was used to compile",
    ],
)
def test_real_mismatches_still_classify_as_cuda_mismatch(line):
    assert classify_stderr([line]).category == ERROR_CUDA_MISMATCH


@pytest.mark.parametrize(
    "line",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA out of memory",
        "Error while warming up sampler: please try lowering max_num_seqs",
    ],
)
def test_genuine_oom_still_classifies_as_oom(line):
    """Narrowing the OOM rule must not stop it catching a real allocation failure."""
    assert classify_stderr([line]).category == ERROR_OOM


def test_config_wins_over_oom_when_a_line_could_match_both():
    """
    A config error naming a memory setting must not be relabelled OOM.

    This is the ordering the whole fix rests on: the config rules sit ahead of
    the OOM rule precisely so a line mentioning memory *and* a setting to change
    reports the setting.
    """
    report = classify_stderr(
        [
            "ValueError: 20.0 GiB KV cache is needed, which is larger than the available "
            "KV cache memory (4.0 GiB). Try increasing `gpu_memory_utilization`."
        ]
    )
    assert report.category == ERROR_CONFIG
    assert "rejected the configuration" in report.summary


# --------------------------------------------------------------------------
# Verbatim from a real Qwen3.5-35B-A3B bring-up on a 16GB card
#
# Every line below was actually produced during that session. Synthetic samples
# had already passed while two of these still classified as `unknown`, which is
# the argument for keeping the real ones.
# --------------------------------------------------------------------------


def test_a_quantization_layout_this_build_cannot_serve():
    """
    An AWQ-8bit MoE with group_size 32; vLLM's INT8 MoE path requires -1.

    The assert names neither the format nor the file, so without a keyword it
    read as `unknown` -- and the remedy is "use a different checkpoint", which
    is exactly what the quantization label conveys.
    """
    report = classify_stderr(["AssertionError: group_size must be -1 for INT8 MoE, got 32"])
    assert report.category == ERROR_QUANTIZATION


def test_a_card_that_needs_a_reset_is_a_gpu_fault():
    """
    Xid 62 as EngineCore exited, then NV_ERR_RESET_REQUIRED until reboot.

    Classified as `unknown` before, which reads like an ordinary startup failure
    worth retrying -- against a card that will fail every retry.
    """
    assert (
        classify_stderr(
            ["nvidia driver reported NV_ERR_RESET_REQUIRED after EngineCore exit"]
        ).category
        == ERROR_GPU_FAULT
    )
    assert (
        classify_stderr(["NVRM: Xid (PCI:0000:01:00): 62, pid=1234, name=VLLM"]).category
        == ERROR_GPU_FAULT
    )


def test_non_hma_connector_on_a_hybrid_model_is_a_config_error():
    """
    What vLLM says when a KV connector without SupportsHMA meets a hybrid model.

    A connector passed through vllm_server_extra_args that cannot do hybrid
    allocation leaves vLLM unable to unify the KV specs of a model like Qwen3.5.
    This was reported as `oom` before, which points at the memory budget
    instead of at the connector choice.
    """
    report = classify_stderr(
        ["ValueError: Failed to promote local KV cache specs to one unified type."]
    )
    assert report.category == ERROR_CONFIG


def test_max_num_seqs_over_the_real_mamba_block_count():
    """The observed numbers: 256 requested, 184 blocks available."""
    report = classify_stderr(
        [
            "ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (184). "
            "Please lower max_num_seqs to at most 184 or increase gpu_memory_utilization."
        ]
    )
    assert report.category == ERROR_CONFIG


def test_a_bf16_weight_load_that_really_did_run_out_of_memory():
    """Still OOM: 67GB of weights, 32GB offloaded, 35GB left for a 16GB card."""
    report = classify_stderr(
        ["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 35.00 GiB"]
    )
    assert report.category == ERROR_OOM


def test_traceback_frames_survive_into_the_summary():
    """
    A summary without the `File "..."` frames cannot be acted on.

    Frame lines carry none of the other important keywords, so they used to be
    dropped -- leaving the Traceback header and the final exception with the
    whole "where" removed between them.
    """
    report = classify_stderr(
        [
            "Traceback (most recent call last):",
            '  File "/opt/vllm/entrypoints/openai/api_server.py", line 1841, in run_server',
            "    await init_app_state(vllm_config)",
            '  File "/opt/vllm/config/vllm.py", line 1731, in __post_init__',
            "    raise ValueError(msg)",
            "ValueError: Free memory on device is less than desired GPU memory utilization",
        ]
    )

    text = report.to_text()
    assert "config/vllm.py" in text, "the frame naming the failing file was dropped"
    assert "line 1731" in text
    assert "ValueError" in text


def test_a_deep_traceback_keeps_the_innermost_frames_and_the_exception():
    """Truncation is tail-biased, so the frames nearest the failure survive."""
    lines = ["Traceback (most recent call last):"]
    for i in range(40):
        lines.append(f'  File "/opt/vllm/frame_{i}.py", line {i}, in fn_{i}')
    lines.append("RuntimeError: engine core initialization failed")

    report = classify_stderr(lines)
    text = report.to_text()

    assert "RuntimeError: engine core initialization failed" in text
    assert "frame_39.py" in text, "the innermost frame must survive truncation"


def test_report_defaults():
    report = VllmErrorReport()
    assert report.category == ERROR_UNKNOWN
    assert report.important_lines == []
    assert report.tail_lines == []
    # No summary and nothing else -> the hard-coded fallback string.
    assert report.to_text() == "No recent vLLM stderr logs found."
