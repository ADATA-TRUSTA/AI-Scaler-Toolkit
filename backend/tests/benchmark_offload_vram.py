"""Offload VRAM / throughput benchmark via the backend APIs.

用途 (zh):
    量測「offload 到 SSD/DRAM」帶來的實際效益，用來取代 wiki 中的估算數字。
    對每個模型：
      1. 以 `/inference/estimate_memory` 取得「全程放 GPU 所需的 VRAM」（基準需求）。
      2. 記錄載入前的 GPU 已用量。
      3. 以指定的 offload 設定 `/inference/load_model` 載入模型。
      4. 跑一次短生成，期間輪詢 `/system/resources?mode=usage`，取得峰值 GPU VRAM。
      5. 以串流 `/v1/chat/completions` 量測 output tok/s 與 TTFT。
      6. 計算 VRAM 降幅 = 1 - (offload 後峰值 VRAM / 全程放 GPU 需求)。
    結果寫入 RESULTS_JSON，wiki 直接引用實測值，不再用估算。

Purpose (en):
    Measure the real benefit of offloading to SSD/DRAM, to replace the estimated
    numbers in the wiki. For each model this records the full-GPU VRAM requirement
    (from `/inference/estimate_memory`), the peak GPU VRAM actually used with
    offload enabled, output tok/s and TTFT, and derives the VRAM-reduction ratio.
    Results are written to RESULTS_JSON for the wiki to cite as measured data.

執行 / Run:
    python tests/benchmark_offload_vram.py

可覆蓋環境變數 / Overridable env vars:
    BACKEND_URL=http://127.0.0.1:8000
    REQUEST_TIMEOUT=1800
    LOAD_TIMEOUT=7200
    POLL_INTERVAL=1.0
    MAX_TOKENS=128
    TEMPERATURE=0.2
    TOP_P=0.9
    RESULTS_JSON=tests/benchmark_offload_vram_results.json

備註 / Notes:
    - 本腳本只量測「本服務自己」的數字；不驅動 Ollama / 獨立 vLLM 等外部工具，
      因此不產生競品對比欄。
      This script measures only THIS service; it does not drive external tools
      (Ollama, standalone vLLM), so it produces no competitor-comparison columns.
    - 需在有 GPU 且模型可取得的機器上執行。
      Must run on a machine with a GPU and access to the models.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "1800"))
LOAD_TIMEOUT = float(os.getenv("LOAD_TIMEOUT", "7200"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
RESULTS_JSON = os.getenv("RESULTS_JSON", "tests/benchmark_offload_vram_results.json")

# 要量測的模型清單。依實際環境調整 model_name / model_path / engine / offload 設定。
# Models to benchmark. Adjust model_name/model_path/engine/offload settings for
# your environment. `n_gpu_layers` < total layers (or a device_map with an
# offload_folder / cpu entry) is what actually triggers offload.
MODELS: List[Dict[str, Any]] = [
    {
        "label": "Qwen3-14B (transformers, bf16, device_map=auto + CPU/disk offload)",
        # 本地權重目錄；全量基準（全程放 GPU 所需）= 這裡 *.safetensors 總大小。
        # If model_dir is set, the full-GPU baseline is the sum of its *.safetensors
        # sizes (works for local paths, where the estimate_memory endpoint can't).
        "model_dir": "./models/Qwen3-14B",
        "estimate_model_name": None,
        "estimate_quantization": None,
        "load_payload": {
            "model_name": "Qwen3-14B-local",
            "model_path": "./models/Qwen3-14B",
            "engine": "transformers",
            "quantization": "none",
            # device_map='auto' + offload_folder 讓放不下 GPU 的層落到 CPU/磁碟
            "device_map": "auto",
            "offload_folder": "./offload",
        },
    },
]

_GB_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def _parse_gb(value: Optional[str]) -> Optional[float]:
    """把 '20GB' / '20 GB' 之類字串轉成 float GB。"""
    if not value:
        return None
    m = _GB_RE.search(str(value))
    return float(m.group(1)) if m else None


def _gpu_used_gb(client: httpx.Client) -> Optional[float]:
    """回傳目前所有 GPU 已用 VRAM 的總和 (GB)。"""
    resp = client.get(f"{BACKEND_URL}/system/resources", params={"mode": "usage"})
    resp.raise_for_status()
    gpus = (resp.json().get("gpu") or {}).get("gpus") or []
    used = [g.get("used_gb") for g in gpus if g.get("used_gb") is not None]
    return round(sum(used), 3) if used else None


def _estimate_full_gpu_gb(client: httpx.Client, spec: Dict[str, Any]) -> Optional[float]:
    """全程放 GPU 所需 VRAM（基準需求）。

    優先用本地權重檔總大小（適用本地路徑）；否則退回 estimate_memory 端點。
    """
    model_dir = spec.get("model_dir")
    if model_dir and os.path.isdir(model_dir):
        total = 0
        for fn in os.listdir(model_dir):
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(model_dir, fn))
        if total:
            return round(total / (1024 ** 3), 3)

    name = spec.get("estimate_model_name")
    if not name:
        return None
    params = {}
    if spec.get("estimate_quantization"):
        params["quantization"] = spec["estimate_quantization"]
    try:
        resp = client.get(
            f"{BACKEND_URL}/inference/estimate_memory/{name}", params=params
        )
        resp.raise_for_status()
        return _parse_gb(resp.json().get("model_total_memory"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] estimate_memory failed for {name}: {exc}")
        return None


def _status(client: httpx.Client) -> Dict[str, Any]:
    resp = client.get(f"{BACKEND_URL}/inference/status")
    resp.raise_for_status()
    return resp.json() or {}


def _wait_loaded(client: httpx.Client, timeout: float) -> None:
    """Poll /inference/status until the model is loaded (fields: loaded / is_loading / loading_error)."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = _status(client)
        if s.get("loading_error"):
            raise RuntimeError(f"model load failed: {s.get('loading_error')}")
        if s.get("loaded") is True:
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("timed out waiting for model to load")


def _wait_unloaded(client: httpx.Client, timeout: float) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = _status(client)
        if not s.get("loaded") and not s.get("is_loading"):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("timed out waiting for model to unload")


def _load(client: httpx.Client, payload: Dict[str, Any]) -> None:
    resp = client.post(f"{BACKEND_URL}/inference/load_model", json=payload)
    resp.raise_for_status()
    body = resp.json() or {}
    if body.get("status") not in {"loading", "already_loaded"}:
        raise RuntimeError(f"unexpected load_model response: {body}")
    _wait_loaded(client, LOAD_TIMEOUT)


def _unload(client: httpx.Client) -> None:
    try:
        client.post(f"{BACKEND_URL}/inference/unload_model")
        _wait_unloaded(client, 120)
    except Exception:  # noqa: BLE001
        pass


def _measure_generation(client: httpx.Client) -> Dict[str, Any]:
    """串流一次生成，量 TTFT / output tok/s，並在過程中取樣峰值 VRAM。"""
    messages = [
        {"role": "user", "content": "請用三段文字說明什麼是模型 offload，以及它如何節省 GPU 記憶體。"}
    ]
    body = {
        "model": "trusta-ast-default",
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    peak_vram = _gpu_used_gb(client) or 0.0
    ttft_sec: Optional[float] = None
    completion_tokens = 0
    started = time.perf_counter()
    last_poll = started

    with client.stream(
        "POST", f"{BACKEND_URL}/v1/chat/completions", json=body
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft_sec is None:
                    ttft_sec = time.perf_counter() - started
                completion_tokens += 1
            now = time.perf_counter()
            if now - last_poll >= POLL_INTERVAL:
                cur = _gpu_used_gb(client)
                if cur is not None:
                    peak_vram = max(peak_vram, cur)
                last_poll = now

    total_latency = time.perf_counter() - started
    gen_latency = (total_latency - ttft_sec) if ttft_sec is not None else None
    output_tps = (
        completion_tokens / gen_latency
        if gen_latency and gen_latency > 0 and completion_tokens
        else None
    )
    return {
        "ttft_sec": ttft_sec,
        "total_latency_sec": total_latency,
        "completion_tokens": completion_tokens,
        "output_tps": output_tps,
        "peak_gpu_vram_gb": round(peak_vram, 3),
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        client.get(f"{BACKEND_URL}/health").raise_for_status()

        for spec in MODELS:
            label = spec["label"]
            print(f"\n=== {label} ===")
            entry: Dict[str, Any] = {"label": label, "load_payload": spec["load_payload"]}

            entry["estimated_full_gpu_gb"] = _estimate_full_gpu_gb(client, spec)

            _unload(client)
            # Baseline GPU usage AFTER unload (excludes the model, reflects other processes).
            entry["gpu_used_before_gb"] = _gpu_used_gb(client)
            try:
                _load(client, spec["load_payload"])
                gen = _measure_generation(client)
                entry.update(gen)

                full = entry["estimated_full_gpu_gb"]
                peak = gen["peak_gpu_vram_gb"]
                if full and peak is not None and full > 0:
                    entry["vram_reduction_pct"] = round((1 - peak / full) * 100, 1)
                else:
                    entry["vram_reduction_pct"] = None
                entry["ok"] = True
                entry["error"] = None
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = str(exc)
                print(f"[error] {label}: {exc}")
            finally:
                _unload(client)

            results.append(entry)
            print(json.dumps(entry, ensure_ascii=False, indent=2))

    out = {
        "backend_url": BACKEND_URL,
        "max_tokens": MAX_TOKENS,
        "results": results,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
