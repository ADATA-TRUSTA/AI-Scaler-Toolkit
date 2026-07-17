---
title: Trusta AST vs Market Solutions
summary: A qualitative comparison of the differences between Trusta AST and Ollama, LM Studio, vLLM and other solutions, focusing on the offload mechanism and fine-tuning integration. Competitor numbers have not been benchmarked by this project.
kind: comparison
sources:
  - wiki/sources/inference_manual.md
  - wiki/concepts/offload-mechanism.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
provenanceState: merged
---

# Trusta AST vs Market Solutions

This document compares **Trusta AST** with existing LLM service solutions (Ollama, LM Studio, vLLM, llama.cpp, etc.), focusing on the **DRAM/SSD offload mechanism** and **integrated fine-tuning**.

> ⚠️ **Important note**: The competitor columns below reflect general public understanding and have **not been benchmarked on this project's identical hardware/models**.
> This project only benchmarks "its own" numbers (see `tests/benchmark_llama_server_prefill.py`, `tests/benchmark_offload_vram.py`).
> Therefore this page does **not list competitors' tok/s, training time, or cost figures**.

## Qualitative Feature Comparison

| Feature | Trusta AST | Ollama | LM Studio | vLLM |
|---------|-----------|--------|-----------|------|
| Offload | DRAM/SSD hybrid (quantization + device_map / n_gpu_layers / DeepSpeed) | GPU-oriented | GPU-oriented | GPU-oriented |
| Fine-tuning | ✅ Integrated (LoRA/QLoRA, full) | ❌ Inference only | ❌ Inference only | Requires additional tools (e.g., Unsloth/Axolotl) |
| OpenAI API | ✅ `/v1/chat/completions`, `/v1/models` | Partially compatible | Partially compatible | ✅ Compatible |
| Process isolation | ✅ Inference runs in a dedicated worker process | Generally single-process | Desktop application | Depends on deployment |
| Number of models loaded simultaneously | Single model (loading a second returns HTTP 409) | Depends on implementation | Depends on implementation | Depends on deployment |

(The competitor columns are for quick reference only; for actual capabilities please refer to each project's official documentation.)

## Core Differences

### 1. GPU Offload Mechanism

Trusta AST performs offload using the capabilities of existing frameworks, depending on the engine/scenario:

- Inference (Transformers): `device_map` + `offload_folder` + bitsandbytes quantization.
- Inference (llama-server / GGUF): `n_gpu_layers` controls GPU/CPU layer split.
- Training: DeepSpeed ZeRO-3, offloading the optimizer/parameters to CPU RAM or NVMe (four profiles).

The benefit is **reducing the required GPU VRAM**, allowing larger models to run on smaller GPUs; the cost is reduced throughput (especially with disk offload).

### 2. Fine-tuning Integration (main difference)

```
Workflow: original model → offload loading → fine-tuning (LoRA/QLoRA/full) → save → automatic conversion to GGUF → inference deployment
```

- Inference and fine-tuning share the same platform, the same offload mechanism, and the same model registry.
- Pure inference tools (Ollama, LM Studio) usually do not include fine-tuning; vLLM requires pairing with external training tools.

### 3. Multi-Process Isolation

```
Main Process (FastAPI) ←→ Worker Process (Inference)
    ├─ Session management     ├─ Model loading
    ├─ Request routing        ├─ Inference computation
    └─ Status queries         └─ Memory management
```

- When a worker exits due to OOM or other reasons, the main HTTP service stays alive and the status transitions to `error`.
- **Recovery is manual**: you need to call `/inference/load_model` or `/inference/force_cleanup_gpu` again.

## Use Cases

**Suitable for Trusta AST:**
- You need to do inference + fine-tuning on the same platform.
- GPU resources are limited, and you want to use offload to reduce VRAM requirements in order to deploy larger models.
- You need an OpenAI-compatible API.

**Less suitable (other solutions recommended):**
- Ultra-high concurrency (a large number of requests per second) → vLLM + multi-GPU.
- Extremely low latency requirements → pure GPU deployment, model entirely in VRAM.

## Performance and Cost (how to obtain credible numbers)

- **This service's tok/s / TTFT / load time**: `tests/benchmark_llama_server_prefill.py` (results in `tests/benchmark_llama_server_prefill_results.json`).
- **VRAM reduction (before vs. after offload)**: `tests/benchmark_offload_vram.py`.
- **Fine-tuning time**: `tests/stress_tests/stress_test_finetune.py` can measure this service's own wall-clock time.
- **Competitor comparison**: this requires separately installing Ollama / standalone vLLM etc. and manually benchmarking under identical conditions; this project does not provide such data.

## Conclusion

Trusta AST's differentiation lies in **"offload reduces VRAM requirements" + "integrated fine-tuning on the same platform" + "multi-process isolation"**. These are capabilities this service definitely provides; any quantitative comparison against competitors should avoid giving specific numbers unless benchmarked under identical conditions.

---

**Related documents**: [[GPU Offload Mechanism]], [[Trusta AST Inference Service]], [[Inference Engine Comparison]]  
**Last updated**: 2026-07-16
