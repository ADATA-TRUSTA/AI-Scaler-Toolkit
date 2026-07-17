"""集中分類 vLLM stderr 輸出，產生結構化的錯誤摘要。

過去 `vllm_engine._get_error_reason` 直接在 stderr 裡 grep 關鍵字，難以擴充
也難以在外層判斷錯誤類別。本模組把比對規則集中於此，供 `VllmEngine` 與其他
需要解讀 vLLM 輸出的呼叫端共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# 錯誤類別常數（字串而非 Enum，方便序列化進 status_queue / data_queue）
ERROR_OOM = "oom"
ERROR_PORT_BUSY = "port_busy"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_CUDA_MISMATCH = "cuda_mismatch"
ERROR_SHARED_LIBRARY_MISSING = "shared_library_missing"
ERROR_TEMPLATE = "chat_template"
ERROR_QUANTIZATION = "quantization"
ERROR_UNKNOWN = "unknown"


# 比對規則：(類別, 關鍵字 tuple)；任一關鍵字命中即歸入該類別。
# 比對採 lower-case 包含；保持規則精簡，避免誤判。
_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        ERROR_OOM,
        (
            "out of memory",
            "cuda out of memory",
            "no available memory",
            "kv cache",  # vLLM 在 KV cache 配置失敗時常出現
            # vLLM v1 sampler warm-up OOM：訊息中含
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


# 用來判斷一行是否「重要」的通用關鍵字；命中任一就視為重要訊號。
_IMPORTANT_KEYWORDS: Tuple[str, ...] = (
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
    """vLLM 子程序錯誤報告。

    Attributes:
        category: 錯誤類別常數，未匹配時為 ``ERROR_UNKNOWN``。
        summary: 人類可讀摘要（用於 log / API response）。
        important_lines: 從 stderr 萃取的關鍵行，最多保留 ``max_important`` 行。
        tail_lines: stderr 最後 N 行原樣保留，作為 fallback 上下文。
    """

    category: str = ERROR_UNKNOWN
    summary: str = ""
    important_lines: List[str] = field(default_factory=list)
    tail_lines: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """產生供 RuntimeError 使用的文字摘要。"""
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
    """掃描 stderr 行列，產生結構化錯誤報告。

    Args:
        lines: stderr 行序列（原樣字串，可含換行；不需事先 lower-case）。
        max_important: ``important_lines`` 最多保留行數，取尾端優先。
        max_tail: 找不到任何重要關鍵字時，``tail_lines`` 保留尾端 N 行。

    Returns:
        :class:`VllmErrorReport` — 至少 ``category`` 與 ``summary`` 有值。
    """
    materialised: List[str] = []
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

    important: List[str] = []
    matched_category: Optional[str] = None

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


# 摘要優先抓「使用者可採取行動」的 hint 行（vLLM / PyTorch 常見字眼）
_HINT_KEYWORDS: Tuple[str, ...] = (
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




def _pick_summary_line(important_tail: List[str], tail: List[str]) -> Optional[str]:
    """從候選行中挑最有資訊量的一行：優先含 hint 的，其次是最後一行。"""
    pool = important_tail or tail
    if not pool:
        return None
    # 從尾端往前找含 hint 字眼的行（通常是 vLLM 給的可採取建議）
    for line in reversed(pool):
        low = line.lower()
        if any(k in low for k in _HINT_KEYWORDS):
            return line
    return pool[-1]


def _build_summary(
    category: str, important_tail: List[str], tail: List[str]
) -> str:
    """組出人類可讀的單行摘要，供 logger / status_queue 使用。"""
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
