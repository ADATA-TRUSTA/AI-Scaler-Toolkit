"""Offload VRAM / throughput benchmark via the backend APIs.

Purpose:
    Measure the real benefit of offloading to SSD/DRAM, so the VRAM-reduction and
    throughput figures quoted elsewhere rest on measurement rather than estimation.
    For each model this:
      1. Derives the full-GPU VRAM requirement (the baseline) from the local weight
         files, falling back to `/inference/estimate_memory`.
      2. Records GPU usage before the load, after unloading anything already resident.
      3. Loads the model through `/inference/load_model` with the offload settings.
      4. Runs one short generation, polling `/system/resources?mode=usage` throughout
         to capture peak GPU VRAM.
      5. Measures TTFT and output tok/s over a streaming `/v1/chat/completions`,
         taking the token count from the server's own usage rather than counting
         SSE chunks, which only agree when the server emits one token per chunk.
      6. Derives VRAM reduction = 1 - (the model's own peak / full-GPU requirement),
         where the model's own peak is the whole-device peak minus what other
         processes already held before the load.
    Results are written to RESULTS_JSON.

Run it from the repo root, with the service already up:
    .venv/bin/python tests/e2e/benchmark_offload_vram.py

The weights are far too large to ship, so their location comes from the environment.
Leave it unset to skip that model:
    QWEN3_14B_MODEL_DIR=/path/to/Qwen3-14B

Other overridable environment variables:
    BACKEND_URL=http://127.0.0.1:8000
    REQUEST_TIMEOUT=1800
    LOAD_TIMEOUT=7200
    POLL_INTERVAL=1.0
    MAX_TOKENS=128
    TEMPERATURE=0.2
    TOP_P=0.9
    OFFLOAD_FOLDER=./offload
    RESULTS_JSON=tests/results/benchmarks/offload_vram.json

Notes:
    - This script measures only THIS service; it does not drive external tools
      (Ollama, standalone vLLM), so it produces no competitor-comparison columns.
    - Must run on a machine with a GPU and access to the models.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx2

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "1800"))
LOAD_TIMEOUT = float(os.getenv("LOAD_TIMEOUT", "7200"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "./offload")
QWEN3_14B_MODEL_DIR = os.getenv("QWEN3_14B_MODEL_DIR", "")
# Results are test output: they live under tests/results/ (gitignored), never in the source
# tree, and the default is anchored at the repo layout instead of one developer's path.
RESULTS_JSON = os.getenv(
    "RESULTS_JSON",
    str(Path(__file__).resolve().parent.parent / "results" / "benchmarks" / "offload_vram.json"),
)

# Models to benchmark. Adjust model_name/model_path/engine/offload settings for your
# environment. `n_gpu_layers` < total layers (or a device_map with an offload_folder /
# cpu entry) is what actually triggers offload.
MODELS: List[Dict[str, Any]] = [
    {
        "label": "Qwen3-14B (transformers, bf16, device_map=auto + CPU/disk offload)",
        # Local weight directory. When set, the full-GPU baseline is the sum of its
        # *.safetensors sizes, which works for local paths where estimate_memory cannot.
        "model_dir": QWEN3_14B_MODEL_DIR,
        "path_env": "QWEN3_14B_MODEL_DIR",
        "estimate_model_name": None,
        "estimate_quantization": None,
        "load_payload": {
            "model_name": "Qwen3-14B-local",
            "model_path": QWEN3_14B_MODEL_DIR,
            "engine": "transformers",
            "quantization": "none",
            # device_map='auto' + offload_folder pushes layers that do not fit on the
            # GPU out to CPU/disk.
            "device_map": "auto",
            "offload_folder": OFFLOAD_FOLDER,
        },
    },
]

_GB_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def _parse_gb(value: Optional[str]) -> Optional[float]:
    """Convert a string like '20GB' / '20 GB' into a float number of GB."""
    if not value:
        return None
    m = _GB_RE.search(str(value))
    return float(m.group(1)) if m else None


def _gpu_used_gb(client: httpx2.Client) -> Optional[float]:
    """Return the summed used VRAM across all GPUs, in GB."""
    resp = client.get(f"{BACKEND_URL}/system/resources", params={"mode": "usage"})
    resp.raise_for_status()
    gpus = (resp.json().get("gpu") or {}).get("gpus") or []
    used = [g.get("used_gb") for g in gpus if g.get("used_gb") is not None]
    return round(sum(used), 3) if used else None


def _estimate_full_gpu_gb(client: httpx2.Client, spec: Dict[str, Any]) -> Optional[float]:
    """VRAM required to hold the model entirely on GPU (the baseline).

    Prefers the total size of the local weight files; falls back to the
    estimate_memory endpoint.
    """
    model_dir = spec.get("model_dir")
    if model_dir and os.path.isdir(model_dir):
        total = 0
        for fn in os.listdir(model_dir):
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(model_dir, fn))
        if total:
            return round(total / (1024**3), 3)

    name = spec.get("estimate_model_name")
    if not name:
        return None
    params = {}
    if spec.get("estimate_quantization"):
        params["quantization"] = spec["estimate_quantization"]
    try:
        resp = client.get(f"{BACKEND_URL}/inference/estimate_memory/{name}", params=params)
        resp.raise_for_status()
        return _parse_gb(resp.json().get("model_total_memory"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] estimate_memory failed for {name}: {exc}")
        return None


def _status(client: httpx2.Client) -> Dict[str, Any]:
    resp = client.get(f"{BACKEND_URL}/inference/status")
    resp.raise_for_status()
    return resp.json() or {}


def _wait_loaded(client: httpx2.Client, timeout: float) -> None:
    """Poll /inference/status until loaded (fields: loaded / is_loading / loading_error)."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = _status(client)
        if s.get("loading_error"):
            raise RuntimeError(f"model load failed: {s.get('loading_error')}")
        if s.get("loaded") is True:
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("timed out waiting for model to load")


def _wait_unloaded(client: httpx2.Client, timeout: float) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = _status(client)
        if not s.get("loaded") and not s.get("is_loading"):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("timed out waiting for model to unload")


def _load(client: httpx2.Client, payload: Dict[str, Any]) -> None:
    resp = client.post(f"{BACKEND_URL}/inference/load_model", json=payload)
    resp.raise_for_status()
    body = resp.json() or {}
    if body.get("status") not in {"loading", "already_loaded"}:
        raise RuntimeError(f"unexpected load_model response: {body}")
    _wait_loaded(client, LOAD_TIMEOUT)


def _unload(client: httpx2.Client) -> None:
    try:
        client.post(f"{BACKEND_URL}/inference/unload_model")
        _wait_unloaded(client, 120)
    except Exception:  # noqa: BLE001
        pass


def _measure_generation(client: httpx2.Client) -> Dict[str, Any]:
    """Stream one generation, measuring TTFT / output tok/s and sampling peak VRAM."""
    # The prompt is deliberately Chinese: it is the workload these numbers were
    # measured on, and changing it would change the tokenisation and the tok/s.
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
        # Ask the server for the real token accounting. Counting SSE chunks instead
        # only matches the token count when the server emits one token per chunk.
        "stream_options": {"include_usage": True},
    }

    peak_vram = _gpu_used_gb(client) or 0.0
    ttft_sec: Optional[float] = None
    content_chunks = 0
    usage: Dict[str, Any] = {}
    started = time.perf_counter()
    last_poll = started

    with client.stream("POST", f"{BACKEND_URL}/v1/chat/completions", json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk.get("usage") or {}
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft_sec is None:
                    ttft_sec = time.perf_counter() - started
                content_chunks += 1
            now = time.perf_counter()
            if now - last_poll >= POLL_INTERVAL:
                cur = _gpu_used_gb(client)
                if cur is not None:
                    peak_vram = max(peak_vram, cur)
                last_poll = now

    total_latency = time.perf_counter() - started
    gen_latency = (total_latency - ttft_sec) if ttft_sec is not None else None
    raw_completion = usage.get("completion_tokens")
    completion_tokens = raw_completion if isinstance(raw_completion, int) else None
    # No usage from the server means no defensible tok/s. Reporting the chunk count
    # here would look like a token count and quietly misstate the result.
    output_tps = (
        completion_tokens / gen_latency
        if completion_tokens and gen_latency and gen_latency > 0
        else None
    )
    return {
        "ttft_sec": ttft_sec,
        "total_latency_sec": total_latency,
        "completion_tokens": completion_tokens,
        "content_chunks": content_chunks,
        "output_tps": output_tps,
        "peak_gpu_vram_gb": round(peak_vram, 3),
    }


def main() -> None:
    # Resolve which models are actually runnable before touching the network, so an
    # unconfigured checkout reports what to set instead of a connection error.
    runnable: List[Dict[str, Any]] = []
    for spec in MODELS:
        model_dir = str(spec.get("model_dir") or "").strip()
        if not model_dir or not Path(model_dir).is_dir():
            print(f"[SKIP] {spec['label']}: set {spec['path_env']} to a local weight directory")
            continue
        runnable.append(spec)

    if not runnable:
        print("No models to benchmark; nothing to do.")
        return

    results: List[Dict[str, Any]] = []
    with httpx2.Client(timeout=REQUEST_TIMEOUT) as client:
        client.get(f"{BACKEND_URL}/health").raise_for_status()

        for spec in runnable:
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
                before = entry["gpu_used_before_gb"]
                # peak is whole-device usage, so it includes whatever other processes
                # were already holding. The baseline has to come off before comparing
                # it with full, which is model weights only.
                if peak is not None and before is not None:
                    model_peak = round(max(peak - before, 0.0), 3)
                else:
                    model_peak = peak
                entry["model_peak_gpu_vram_gb"] = model_peak
                if full and model_peak is not None and full > 0:
                    entry["vram_reduction_pct"] = round((1 - model_peak / full) * 100, 1)
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
