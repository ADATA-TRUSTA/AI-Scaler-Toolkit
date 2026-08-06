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
ERROR_PORT_BUSY = "port_busy"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_CUDA_MISMATCH = "cuda_mismatch"
ERROR_SHARED_LIBRARY_MISSING = "shared_library_missing"
ERROR_TEMPLATE = "chat_template"
ERROR_QUANTIZATION = "quantization"
ERROR_UNKNOWN = "unknown"


# Matching rules: (category, keyword tuple); any keyword hit assigns that category.
# Matching is lower-case substring containment; keep rules tight to avoid false positives.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        ERROR_OOM,
        (
            "out of memory",
            "cuda out of memory",
            "no available memory",
            "kv cache",  # common when vLLM fails to allocate the KV cache
            # vLLM v1 sampler warm-up OOM: message contains
            # `Please try lowering max_num_seqs or gpu_memory_utilization ...`
            "warming up sampler",
            "max_num_seqs",
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
        ERROR_CUDA_MISMATCH,
        (
            "cuda error",
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
        ),
    ),
)


# Generic keywords marking a line as "important"; any hit counts as a signal.
_IMPORTANT_KEYWORDS: tuple[str, ...] = (
    "error",
    "failed",
    "exception",
    "traceback",
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
    max_important: int = 8,
    max_tail: int = 25,
) -> VllmErrorReport:
    """
    Scan stderr lines and produce a structured error report.

    Args:
        lines: stderr line sequence (raw strings, newlines allowed; no need to lower-case).
        max_important: Max lines kept in ``important_lines``, preferring the tail.
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
        ERROR_PORT_BUSY: "vLLM port already bound",
        ERROR_MODEL_NOT_FOUND: "model artifact not found",
        ERROR_CUDA_MISMATCH: "CUDA / GPU capability mismatch",
        ERROR_SHARED_LIBRARY_MISSING: "missing CUDA shared library runtime",
        ERROR_TEMPLATE: "chat template parse error",
        ERROR_QUANTIZATION: "quantization configuration error",
        ERROR_UNKNOWN: "vLLM startup/runtime failure",
    }.get(category, "vLLM startup/runtime failure")

    picked = _pick_summary_line(important_tail, tail)
    if picked:
        return f"{label}: {picked}"
    return label
