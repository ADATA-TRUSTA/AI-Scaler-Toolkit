"""llama-server prefill benchmark via existing backend APIs.

用途：
1. 透過 `/inference/load_model` 載入指定 GGUF 模型（engine=llama_server）。
2. 使用 `/v1/chat/completions` 串流量測：模型載入時間、TTFT、TPS。
3. 同一輪內比較 cold / warm / changed-tail。
4. 每輪結束呼叫 `/inference/unload_model` 後重新測試，觀察 reload 後差異。

執行：
    /home/test/project/Trusta_AST_Backend/service/.venv/bin/python tests/benchmark_llama_server_prefill.py

可覆蓋環境變數：
    BACKEND_URL=http://127.0.0.1:8000
    LLAMA_SERVER_URL=http://127.0.0.1:5001
    BENCH_LLAMA_SERVER_AUTO_START=true
    LLAMA_SERVER_PRECHECK_TIMEOUT=15
    REQUEST_TIMEOUT=1800
    LOAD_TIMEOUT=7200
    POLL_INTERVAL=1.0
    BENCH_N_CTX=4096
    BENCH_N_BATCH=512
    BENCH_NP=4
    BENCH_N_GPU_LAYERS=-1
    GPT_OSS_NGLS=-1
    QWEN_NGLS=-1
    PREFIX_REPEAT=220
    MAX_TOKENS=50
    TEMPERATURE=0.2
    TOP_P=0.9
    TOP_K=40
    REPETITION_PENALTY=1.05
    RESULTS_JSON=tests/benchmark_llama_server_prefill_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:5001").rstrip("/")
BENCH_LLAMA_SERVER_AUTO_START = str(os.getenv("BENCH_LLAMA_SERVER_AUTO_START", "true")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLAMA_SERVER_PRECHECK_TIMEOUT = max(2, int(os.getenv("LLAMA_SERVER_PRECHECK_TIMEOUT", "15")))
REQUEST_TIMEOUT = max(60, int(os.getenv("REQUEST_TIMEOUT", "1800")))
LOAD_TIMEOUT = max(60, int(os.getenv("LOAD_TIMEOUT", "7200")))
POLL_INTERVAL = max(0.2, float(os.getenv("POLL_INTERVAL", "1.0")))
BENCH_N_CTX = max(1024, int(os.getenv("BENCH_N_CTX", "4096")))
BENCH_N_BATCH = max(32, int(os.getenv("BENCH_N_BATCH", "512")))
BENCH_NP = max(1, int(os.getenv("BENCH_NP", "4")))
BENCH_N_GPU_LAYERS = int(os.getenv("BENCH_N_GPU_LAYERS", "-1"))
PREFIX_REPEAT = max(40, int(os.getenv("PREFIX_REPEAT", "220")))
MAX_TOKENS = max(16, int(os.getenv("MAX_TOKENS", "50")))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TOP_K = int(os.getenv("TOP_K", "40"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))
RESULTS_JSON = os.getenv(
    "RESULTS_JSON",
    str(Path("/home/test/project/Trusta_AST_Backend/tests/benchmark_llama_server_prefill_results.json")),
)


BASE_MODELS: List[Dict[str, Any]] = [
    {
        "model_name": "unsloth/gpt-oss-120b-GGUF",
        "model_path": "/media/test/eee156b9-166c-48ab-a252-ee619815c845/home/trusta/.cache/huggingface/hub/models--unsloth--gpt-oss-120b-GGUF/snapshots/ff1a82da6ad466e32284fa3d2b86694db3204789/gpt-oss-120b-F16.gguf",
        "label": "GPT-OSS 120B F16 (Local)",
        "size": "75GB",
        "max_context_length": 131072,
        "is_multimodal": False,
        "ngl_env": "GPT_OSS_NGLS",
    },
    {
        "model_name": "unsloth/Qwen3.5-35B-GGUF",
        "model_path": "/media/test/eee156b9-166c-48ab-a252-ee619815c845/home/trusta/.cache/huggingface/hub/Qwen3.5-35B-A3B-Q4_K/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "label": "Qwen3.5-35B-A3B-Q4_K_M-GGUF",
        "size": "16.7GB",
        "max_context_length": 262144,
        "is_multimodal": True,
        "ngl_env": "QWEN_NGLS",
    },
]


@dataclass
class StreamMetrics:
    name: str
    request_id: str
    total_latency_sec: float
    ttft_sec: Optional[float]
    generation_latency_sec: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    output_tps: Optional[float]
    end_to_end_tps: Optional[float]
    text_preview: str
    content_chars: int
    chunk_count: int
    session_user: str
    ok: bool
    error: Optional[str] = None


@dataclass
class RoundResult:
    round_name: str
    load_time_sec: float
    unload_time_sec: float
    ngl: int
    loaded_n_ctx: Optional[int]
    loaded_slot_n_ctx: Optional[int]
    prefill_strategy: Optional[str]
    llama_capabilities: List[str]
    slot_restore_summary: Optional[Dict[str, Any]]
    cases: List[StreamMetrics]


def _parse_ngl_values(raw_value: Optional[str], default_value: int) -> List[int]:
    values: List[int] = []
    for item in str(raw_value or str(default_value)).split(","):
        text = item.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"無法解析 NGL 值: {text}") from exc

    if not values:
        values = [default_value]

    deduped: List[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _expand_model_variants() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for model in BASE_MODELS:
        env_name = str(model.get("ngl_env") or "").strip()
        ngl_values = _parse_ngl_values(os.getenv(env_name), BENCH_N_GPU_LAYERS)
        for ngl in ngl_values:
            variant = dict(model)
            variant["ngl"] = ngl
            variant["variant_label"] = f"{model['label']} | ngl={ngl}"
            variants.append(variant)
    return variants


MODEL_VARIANTS = _expand_model_variants()


def _estimate_text_tokens(text: str) -> int:
    """粗估 token 數，寧可高估，避免超出 n_ctx。"""
    estimated = 0.0
    for ch in text:
        code = ord(ch)
        if ch.isspace():
            estimated += 0.1
        elif 0x4E00 <= code <= 0x9FFF:
            estimated += 1.8
        elif ch.isascii() and ch.isalnum():
            estimated += 0.35
        elif ch.isascii():
            estimated += 0.25
        else:
            estimated += 1.2
    return max(1, int(estimated) + 1)


def _make_prefix_text(model_label: str, n_ctx: int) -> str:
    header = (
        f"你是一個用於 TTFT / prefix cache / slot restore 基準測試的助手。\n"
        f"目前測試模型：{model_label}。\n"
        "請嚴格遵守以下固定規則：\n"
        "1. 回答使用繁體中文。\n"
        "2. 請勿輸出思考過程。\n"
        "3. 若要求列表，僅輸出最終結果。\n"
        "4. 若要求 JSON，僅輸出合法 JSON。\n"
        "5. 回答應簡潔。\n"
        "以下是固定長前綴，用於模擬 Agent 大量前置指令與工具規格。\n"
    )
    available_budget = max(96, n_ctx - MAX_TOKENS - 320)
    if n_ctx <= 2048:
        target_prompt_tokens = min(available_budget, max(96, int(n_ctx * 0.22)))
    else:
        target_prompt_tokens = min(available_budget, max(160, int(n_ctx * 0.35)))
    fragments: List[str] = []
    current_text = header

    for idx in range(PREFIX_REPEAT):
        fragment = f"規格片段 {idx:03d}: 工具定義、角色限制、輸出格式與安全規則必須保持穩定。"
        candidate = current_text + fragment + "\n"
        if _estimate_text_tokens(candidate) >= target_prompt_tokens:
            break
        fragments.append(fragment)
        current_text = candidate

    return header + "\n".join(fragments)


def _build_messages(model_label: str, variant: str, n_ctx: int) -> List[Dict[str, str]]:
    system_prompt = _make_prefix_text(model_label, n_ctx=n_ctx)
    user_suffix_map = {
        "cold_exact": "請根據上述規格，用一句話總結本輪測試目的。",
        "warm_exact": "請根據上述規格，用一句話總結本輪測試目的。",
        "warm_variant": "請根據上述規格，改用三點條列說明本輪測試目的。",
        "reload_exact": "請根據上述規格，用一句話總結本輪測試目的。",
        "reload_variant": "請根據上述規格，輸出一個 JSON 物件，包含 keys: purpose, focus。",
    }
    user_text = user_suffix_map[variant]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def _ensure_healthy(client: httpx.Client) -> None:
    resp = client.get(f"{BACKEND_URL}/health")
    resp.raise_for_status()


def _ensure_llama_server_ready(client: httpx.Client) -> None:
    try:
        resp = client.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=LLAMA_SERVER_PRECHECK_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "llama-server 尚未手動啟動或 URL 不可達。"
            f" 請先確認 {LLAMA_SERVER_URL} 可回應 /v1/models，再執行 benchmark。"
        ) from exc


def _wait_until_unloaded(client: httpx.Client, timeout_sec: int = 120) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = client.get(f"{BACKEND_URL}/inference/status")
        resp.raise_for_status()
        body = resp.json()
        if not body.get("loaded") and not body.get("is_loading"):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("等待模型卸載完成超時")


def _load_model(client: httpx.Client, model_spec: Dict[str, Any]) -> Dict[str, Any]:
    if not BENCH_LLAMA_SERVER_AUTO_START:
        _ensure_llama_server_ready(client)

    payload = {
        "model_name": model_spec["model_name"],
        "model_path": model_spec["model_path"],
        "engine": "llama_server",
        "quantization": "none",
        "llama_server_url": LLAMA_SERVER_URL,
        "llama_server_auto_start": BENCH_LLAMA_SERVER_AUTO_START,
        "llama_server_timeout": REQUEST_TIMEOUT,
        "llama_server_health_timeout": LOAD_TIMEOUT,
        "llama_server_np": BENCH_NP,
        "n_ctx": min(BENCH_N_CTX, int(model_spec["max_context_length"])),
        "n_batch": BENCH_N_BATCH,
        "n_gpu_layers": int(model_spec["ngl"]),
        "use_cache": True,
        "trust_remote_code": True,
    }

    t0 = time.perf_counter()
    resp = client.post(f"{BACKEND_URL}/inference/load_model", json=payload)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") not in {"loading", "already_loaded"}:
        raise RuntimeError(f"load_model 回傳非預期: {body}")

    deadline = time.time() + LOAD_TIMEOUT
    while time.time() < deadline:
        status_resp = client.get(f"{BACKEND_URL}/inference/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("loaded") is True:
            status["load_time_sec"] = time.perf_counter() - t0
            return status
        if status.get("loading_error"):
            raise RuntimeError(f"模型載入失敗: {status.get('loading_error')}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError("等待模型載入完成超時")


def _unload_model(client: httpx.Client) -> float:
    t0 = time.perf_counter()
    resp = client.post(f"{BACKEND_URL}/inference/unload_model")
    resp.raise_for_status()
    _wait_until_unloaded(client)
    return time.perf_counter() - t0


def _stream_chat_completion(
    client: httpx.Client,
    *,
    model_name: str,
    case_name: str,
    messages: List[Dict[str, str]],
    session_user: str,
) -> StreamMetrics:
    request_id = f"bench-{case_name}-{uuid.uuid4().hex[:10]}"
    body = {
        "model": model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": messages,
        "user": session_user,
        "request_id": request_id,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "enable_thinking": False,
        "total_timeout": REQUEST_TIMEOUT,
    }

    started_at = time.perf_counter()
    ttft_sec: Optional[float] = None
    chunk_count = 0
    content_parts: List[str] = []
    usage: Dict[str, Any] = {}
    error_message: Optional[str] = None

    try:
        with client.stream("POST", f"{BACKEND_URL}/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue
                payload = line.split("data:", 1)[1].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue

                item = json.loads(payload)
                if item.get("error"):
                    error_message = json.dumps(item.get("error"), ensure_ascii=False)
                    break

                if item.get("usage"):
                    usage = item.get("usage") or {}
                    continue

                choices = item.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    if ttft_sec is None:
                        ttft_sec = time.perf_counter() - started_at
                    chunk_count += 1
                    content_parts.append(str(text))
    except Exception as exc:
        return StreamMetrics(
            name=case_name,
            request_id=request_id,
            total_latency_sec=time.perf_counter() - started_at,
            ttft_sec=ttft_sec,
            generation_latency_sec=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            output_tps=None,
            end_to_end_tps=None,
            text_preview="",
            content_chars=0,
            chunk_count=chunk_count,
            session_user=session_user,
            ok=False,
            error=str(exc),
        )

    total_latency_sec = time.perf_counter() - started_at
    content_text = "".join(content_parts)
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
    total_tokens = usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None

    generation_latency_sec: Optional[float] = None
    if ttft_sec is not None:
        generation_latency_sec = max(total_latency_sec - ttft_sec, 1e-9)

    output_tps: Optional[float] = None
    if completion_tokens is not None and generation_latency_sec is not None:
        output_tps = completion_tokens / generation_latency_sec

    end_to_end_tps: Optional[float] = None
    if completion_tokens is not None and total_latency_sec > 0:
        end_to_end_tps = completion_tokens / total_latency_sec

    ok = error_message is None and bool(content_text.strip())
    return StreamMetrics(
        name=case_name,
        request_id=request_id,
        total_latency_sec=total_latency_sec,
        ttft_sec=ttft_sec,
        generation_latency_sec=generation_latency_sec,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_tps=output_tps,
        end_to_end_tps=end_to_end_tps,
        text_preview=content_text.replace("\n", " ")[:160],
        content_chars=len(content_text),
        chunk_count=chunk_count,
        session_user=session_user,
        ok=ok,
        error=error_message,
    )


def _run_round(client: httpx.Client, model_spec: Dict[str, Any], round_index: int) -> RoundResult:
    round_name = f"round_{round_index}"
    status = _load_model(client, model_spec)
    load_time_sec = float(status.get("load_time_sec", 0.0))
    loaded_n_ctx = status.get("n_ctx") if isinstance(status.get("n_ctx"), int) else None
    prefill_strategy = status.get("prefill_strategy")
    llama_capabilities = status.get("llama_capabilities") or []
    slot_restore_summary = status.get("slot_restore_summary")

    session_user = f"bench-{Path(model_spec['model_path']).stem}-{round_index}"
    effective_n_ctx = loaded_n_ctx or min(BENCH_N_CTX, int(model_spec["max_context_length"]))
    slot_n_ctx = max(128, effective_n_ctx // max(1, BENCH_NP))
    cases = [
        _stream_chat_completion(
            client,
            model_name=model_spec["model_name"],
            case_name="cold_exact",
            messages=_build_messages(model_spec["label"], "cold_exact", slot_n_ctx),
            session_user=session_user,
        ),
        _stream_chat_completion(
            client,
            model_name=model_spec["model_name"],
            case_name="warm_exact",
            messages=_build_messages(model_spec["label"], "warm_exact", slot_n_ctx),
            session_user=session_user,
        ),
        _stream_chat_completion(
            client,
            model_name=model_spec["model_name"],
            case_name="warm_variant",
            messages=_build_messages(model_spec["label"], "warm_variant", slot_n_ctx),
            session_user=session_user,
        ),
    ]
    unload_time_sec = _unload_model(client)

    return RoundResult(
        round_name=round_name,
        load_time_sec=load_time_sec,
        unload_time_sec=unload_time_sec,
        ngl=int(model_spec["ngl"]),
        loaded_n_ctx=loaded_n_ctx,
        loaded_slot_n_ctx=slot_n_ctx,
        prefill_strategy=prefill_strategy,
        llama_capabilities=list(llama_capabilities),
        slot_restore_summary=slot_restore_summary,
        cases=cases,
    )


def _print_case(case: StreamMetrics) -> None:
    status = "OK" if case.ok else "FAIL"
    ttft = f"{case.ttft_sec:.3f}s" if case.ttft_sec is not None else "n/a"
    out_tps = f"{case.output_tps:.2f}" if case.output_tps is not None else "n/a"
    e2e_tps = f"{case.end_to_end_tps:.2f}" if case.end_to_end_tps is not None else "n/a"
    print(
        f"    - [{status}] {case.name}: total={case.total_latency_sec:.3f}s, "
        f"TTFT={ttft}, output_tps={out_tps}, e2e_tps={e2e_tps}, "
        f"prompt_tokens={case.prompt_tokens}, completion_tokens={case.completion_tokens}, preview={case.text_preview!r}"
    )
    if case.error:
        print(f"      error={case.error}")


def _print_round(round_result: RoundResult) -> None:
    print(f"  * {round_result.round_name}")
    print(
        f"    load_time={round_result.load_time_sec:.2f}s, unload_time={round_result.unload_time_sec:.2f}s, "
        f"ngl={round_result.ngl}, loaded_n_ctx={round_result.loaded_n_ctx}, loaded_slot_n_ctx={round_result.loaded_slot_n_ctx}, "
        f"prefill_strategy={round_result.prefill_strategy}, capabilities={round_result.llama_capabilities}, "
        f"slot_restore_summary={round_result.slot_restore_summary}"
    )
    for case in round_result.cases:
        _print_case(case)


def _build_summary_payload(model_spec: Dict[str, Any], rounds: List[RoundResult]) -> Dict[str, Any]:
    return {
        "model": {
            "model_name": model_spec["model_name"],
            "model_path": model_spec["model_path"],
            "label": model_spec["label"],
            "variant_label": model_spec["variant_label"],
            "size": model_spec["size"],
            "max_context_length": model_spec["max_context_length"],
            "is_multimodal": model_spec["is_multimodal"],
            "ngl": model_spec["ngl"],
        },
        "rounds": [
            {
                **asdict(round_result),
                "cases": [asdict(case) for case in round_result.cases],
            }
            for round_result in rounds
        ],
    }


def main() -> int:
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=30.0)
    results: List[Dict[str, Any]] = []

    with httpx.Client(timeout=timeout) as client:
        _ensure_healthy(client)
        if not BENCH_LLAMA_SERVER_AUTO_START:
            _ensure_llama_server_ready(client)

        try:
            _unload_model(client)
        except Exception:
            pass

        for model_spec in MODEL_VARIANTS:
            print("=" * 100)
            print(
                f"[MODEL] {model_spec['variant_label']} | multimodal={model_spec['is_multimodal']} | "
                f"path={model_spec['model_path']}"
            )

            rounds: List[RoundResult] = []
            try:
                rounds.append(_run_round(client, model_spec, round_index=1))
                rounds.append(_run_round(client, model_spec, round_index=2))
            finally:
                try:
                    _unload_model(client)
                except Exception:
                    pass

            for round_result in rounds:
                _print_round(round_result)

            model_summary = _build_summary_payload(model_spec, rounds)
            results.append(model_summary)

    output = {
        "backend_url": BACKEND_URL,
        "llama_server_url": LLAMA_SERVER_URL,
        "bench_n_ctx": BENCH_N_CTX,
        "bench_n_batch": BENCH_N_BATCH,
        "bench_np": BENCH_NP,
        "prefix_repeat": PREFIX_REPEAT,
        "max_tokens": MAX_TOKENS,
        "results": results,
    }

    output_path = Path(RESULTS_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"[DONE] benchmark results saved to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] benchmark stopped by user", file=sys.stderr)
        raise SystemExit(130)
