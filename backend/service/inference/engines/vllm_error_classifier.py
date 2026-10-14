"""
Central classifier for vLLM stderr output, producing structured error reports.

`vllm_engine._get_error_reason` used to grep stderr for keywords inline, which
was hard to extend and gave callers no way to reason about the error category.
This module keeps the matching rules in one place, shared by `VllmEngine` and
any other caller that needs to interpret vLLM output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# Error category constants (plain strings rather than Enum, so they serialize
# easily into status_queue / data_queue)
ERROR_OOM = "oom"
ERROR_CONFIG = "config"
ERROR_PORT_BUSY = "port_busy"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_CUDA_MISMATCH = "cuda_mismatch"
ERROR_GPU_FAULT = "gpu_fault"
ERROR_SHARED_LIBRARY_MISSING = "shared_library_missing"
# A compiler vLLM or torch.compile needs at startup is not on the machine.
ERROR_TOOLCHAIN_MISSING = "toolchain_missing"
ERROR_TEMPLATE = "chat_template"
ERROR_QUANTIZATION = "quantization"
ERROR_UNKNOWN = "unknown"


# Matching rules: (category, keyword tuple); any keyword hit assigns that category.
# Matching is lower-case substring containment; keep rules tight to avoid false positives.
#
# Order matters: for a line matching several rules, the first one here wins. The
# config rules come before OOM on purpose -- they name a specific setting to
# change, which is a strictly more useful answer than "you ran out of memory".
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        # vLLM refused the configuration and said which knob to turn. These are
        # ValueErrors raised during startup validation, not allocation failures,
        # and they used to be reported as OOM -- which sends the operator after
        # a bigger card when the fix is one setting.
        #
        # Keywords are taken verbatim from vLLM 0.28's message text rather than
        # guessed, so they stay tight: a bare "kv cache" or "max_num_seqs" is
        # what over-matched before.
        ERROR_CONFIG,
        (
            # config/compilation.py: max_num_seqs exceeds Mamba cache blocks
            "exceeds available mamba cache",
            "please lower max_num_seqs",
            # v1/core/kv_cache_utils.py: model len needs more KV cache than exists
            "is larger than the available kv cache memory",
            "no available memory for the cache blocks",
            # v1/core/kv_cache_utils.py: heterogeneous KV specs cannot be merged
            "failed to promote local kv cache specs",
            "cannot be unified",
            # config/scheduler.py: batched-token vs sequence budget disagreement
            "exceeds max_num_seqs",
            # v1/worker: less free device memory than gpu_memory_utilization asks for.
            # Seen on an Intel iGPU, whose memory is the system's and was already
            # in use; the remedy is that setting, not a bigger card.
            "is less than desired gpu memory utilization",
        ),
    ),
    (
        ERROR_OOM,
        (
            "out of memory",
            "cuda out of memory",
            # vLLM v1 sampler warm-up OOM: message contains
            # `Please try lowering max_num_seqs or gpu_memory_utilization ...`.
            # Matched on the phase, not on the setting names it suggests.
            "warming up sampler",
        ),
    ),
    (
        ERROR_PORT_BUSY,
        (
            "address already in use",
            "errno 98",
            "port is already in use",
        ),
    ),
    (
        ERROR_SHARED_LIBRARY_MISSING,
        (
            "importerror: libcudnn.so",
            "importerror: libcublas.so",
            "importerror: libcudart.so",
            "cannot open shared object file",
            "error while loading shared libraries",
            "libcudnn.so",
            "libcublas.so",
            "libcudart.so",
        ),
    ),
    (
        # Before model_not_found, which matches any "no such file or directory":
        # torch.compile on Intel XPU runs Intel's offline compiler and fails with
        # "[Errno 2] No such file or directory: 'ocloc'" when intel-ocloc is not
        # installed, and that was reported as a missing model. flashinfer's JIT
        # build fails the same way without nvcc.
        ERROR_TOOLCHAIN_MISSING,
        (
            "no such file or directory: 'ocloc'",
            "could not find nvcc",
        ),
    ),
    (
        ERROR_MODEL_NOT_FOUND,
        (
            "no such file or directory",
            "is not a local folder",
            "huggingfaceh4 is not a valid model identifier",
            "could not locate config.json",
            "repository not found",
        ),
    ),
    (
        # Hardware or driver faulted at runtime. Distinct from cuda_mismatch,
        # which is a build/hardware incompatibility: the remedies share nothing.
        # A fault means reset the GPU or check the card; a mismatch means
        # reinstall a build that matches it. A bare "cuda error" used to land in
        # cuda_mismatch and sent every fault down the wrong path.
        ERROR_GPU_FAULT,
        (
            "uncorrectable ecc error",
            "illegal memory access",
            "device-side assert triggered",
            "gpu has fallen off the bus",
            "unspecified launch failure",
            "cuda error: unknown error",
            "misaligned address",
            # The driver's own words for a card that needs a reset. Observed for
            # real: an Xid 62 as the EngineCore exited, then NV_ERR_RESET_REQUIRED
            # on every subsequent call until the host was rebooted. Without these
            # the driver message alone classifies as `unknown`, and a dead card
            # reads like an ordinary startup failure worth retrying.
            "nv_err_reset_required",
            "nvrm: xid",
        ),
    ),
    (
        ERROR_CUDA_MISMATCH,
        (
            "cuda capability",
            "no kernel image is available",
            "compute capability",
            "version mismatch",
        ),
    ),
    (
        ERROR_TEMPLATE,
        (
            "chat template",
            "jinja2",
            "templateerror",
        ),
    ),
    (
        ERROR_QUANTIZATION,
        (
            "quantization",
            "awq",
            "gptq",
            "fp8",
            # A checkpoint whose quantization layout this build cannot serve, which
            # surfaces as a bare assert naming neither the format nor the file.
            # Observed: an AWQ-8bit MoE with group_size 32, where vLLM's INT8 MoE
            # path requires -1. The remedy is a different checkpoint, so saying
            # "quantization" is the whole value of the message.
            "group_size must be",
            "group size must be",
            "w8a8",
        ),
    ),
)


# Generic keywords marking a line as "important"; any hit counts as a signal.
#
# `file "` is what keeps a traceback readable. A Python frame line --
# `  File "vllm/config/vllm.py", line 1731, in __post_init__` -- contains none of
# the other keywords, so the summary used to keep the `Traceback (most recent
# call last):` header and the final `SomeError: ...` and silently drop every
# frame between them. That is precisely the part that says *where* it broke, and
# without it the summary cannot be acted on.
_IMPORTANT_KEYWORDS: tuple[str, ...] = (
    "error",
    "failed",
    "exception",
    "traceback",
    'file "',
    "out of memory",
    "abort",
    "fatal",
)


@dataclass
class VllmErrorReport:
    """
    Error report for the vLLM subprocess.

    Attributes:
        category: Error category constant; ``ERROR_UNKNOWN`` when nothing matched.
        summary: Human-readable summary (for logs / API responses).
        important_lines: Key lines extracted from stderr, at most ``max_important``.
        tail_lines: Last N stderr lines kept verbatim as fallback context.
    """

    category: str = ERROR_UNKNOWN
    summary: str = ""
    important_lines: list[str] = field(default_factory=list)
    tail_lines: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Build the text summary used for RuntimeError messages."""
        if self.important_lines:
            return "\n".join(self.important_lines)
        if self.tail_lines:
            return "\n".join(self.tail_lines)
        return self.summary or "No recent vLLM stderr logs found."


def classify_stderr(
    lines: Iterable[str],
    *,
    max_important: int = 16,
    max_tail: int = 25,
) -> VllmErrorReport:
    """
    Scan stderr lines and produce a structured error report.

    Args:
        lines: stderr line sequence (raw strings, newlines allowed; no need to lower-case).
        max_important: Max lines kept in ``important_lines``, preferring the tail.
            16, not the 8 it was: traceback frames now count as important, and a
            vLLM startup failure runs deep enough that 8 truncated the frames back
            off again, which is the thing keeping them was meant to fix. Tail-biased
            selection means the innermost frames and the final exception survive.
        max_tail: When no important keyword matched, keep the last N lines in ``tail_lines``.

    Returns:
        :class:`VllmErrorReport` — ``category`` and ``summary`` are always set.
    """
    materialised: list[str] = []
    for raw in lines:
        if raw is None:
            continue
        stripped = str(raw).strip()
        if stripped:
            materialised.append(stripped)

    if not materialised:
        return VllmErrorReport(
            category=ERROR_UNKNOWN,
            summary="No recent vLLM stderr logs found.",
        )

    important: list[str] = []
    matched_category: str | None = None

    for line in materialised:
        low = line.lower()
        if any(k in low for k in _IMPORTANT_KEYWORDS):
            important.append(line)

        if matched_category is None:
            for category, keywords in _RULES:
                if any(k in low for k in keywords):
                    matched_category = category
                    break

    important_tail = important[-max_important:] if important else []
    tail = materialised[-max_tail:] if not important_tail else []

    category = matched_category or ERROR_UNKNOWN
    summary = _build_summary(category, important_tail, tail)

    return VllmErrorReport(
        category=category,
        summary=summary,
        important_lines=important_tail,
        tail_lines=tail,
    )


# Summaries prefer actionable hint lines (common vLLM / PyTorch phrasings)
_HINT_KEYWORDS: tuple[str, ...] = (
    "please try lowering",
    "please try reducing",
    "please reduce",
    "please try",
    "please set",
    "please use",
    "please increase",
    "consider lowering",
    "consider reducing",
    "try setting",
    "try increasing",
    "try lowering",
    "try reducing",
)


def _pick_summary_line(important_tail: list[str], tail: list[str]) -> str | None:
    """Pick the most informative candidate line: a hint line first, else the last one."""
    pool = important_tail or tail
    if not pool:
        return None
    # Scan backwards for a line containing a hint phrase (usually vLLM's own advice)
    for line in reversed(pool):
        low = line.lower()
        if any(k in low for k in _HINT_KEYWORDS):
            return line
    return pool[-1]


def _build_summary(category: str, important_tail: list[str], tail: list[str]) -> str:
    """Build a human-readable one-line summary for logger / status_queue."""
    label = {
        ERROR_OOM: "GPU/CPU memory exhausted",
        ERROR_CONFIG: "vLLM rejected the configuration",
        ERROR_PORT_BUSY: "vLLM port already bound",
        ERROR_MODEL_NOT_FOUND: "model artifact not found",
        ERROR_CUDA_MISMATCH: "CUDA / GPU capability mismatch",
        ERROR_GPU_FAULT: "GPU fault (hardware or driver), not a build mismatch",
        ERROR_SHARED_LIBRARY_MISSING: "missing CUDA shared library runtime",
        ERROR_TOOLCHAIN_MISSING: "a GPU compiler vLLM needs is missing (Intel: install intel-ocloc)",
        ERROR_TEMPLATE: "chat template parse error",
        ERROR_QUANTIZATION: "quantization configuration error",
        ERROR_UNKNOWN: "vLLM startup/runtime failure",
    }.get(category, "vLLM startup/runtime failure")

    picked = _pick_summary_line(important_tail, tail)
    if picked:
        return f"{label}: {picked}"
    return label
