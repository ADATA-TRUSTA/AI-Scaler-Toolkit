---
title: TRUSTA-AST Core Differentiation
summary: Trusta AST's key differences: DRAM/SSD offload reduces GPU VRAM requirements, integrated fine-tuning on the same platform, and multi-process isolation.
kind: concept
sources:
  - wiki/sources/inference_manual.md
  - concepts/offload-mechanism.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
---

# TRUSTA-AST Core Differentiation

## 🎯 Core Value Proposition

1. **DRAM/SSD Offload Mechanism** — via device_map / offload_folder / n_gpu_layers / DeepSpeed, offload model layers or training state to DRAM/SSD, **reducing the required GPU VRAM**, allowing larger models to run on smaller GPUs.
2. **Integrated Fine-tuning** — not just an inference service; the same platform can perform LoRA/QLoRA and full fine-tuning, and can automatically export GGUF for inference.
3. **Multi-Process Isolation** — inference runs in a dedicated worker process, so OOM does not bring down the main HTTP service (recovery is manual, see below).

## 🆚 Differences from Market Solutions (Qualitative)

> The table below only compares capabilities that "this service definitely provides"; the competitor column reflects general understanding, has **not been benchmarked by this project**, and should not be cited as a benchmark.

| Aspect | Trusta AST | Typical pure inference tools (e.g., Ollama / LM Studio) |
|------|-----------|------------------------------------------|
| Main function | Inference + Fine-tuning (same platform) | Mostly inference only |
| Offload | DRAM/SSD hybrid (quantization + device_map / n_gpu_layers / DeepSpeed) | Mostly GPU-oriented |
| OpenAI API | Compatible with `/v1/chat/completions`, `/v1/models` | Varying degrees of compatibility |
| Process isolation | Inference in a dedicated worker process | Not necessarily |

## 💰 Cost-Benefit (How to Understand It)

The source of cost savings is "**a lower required GPU tier**": with offload enabled, models can run on machines with smaller VRAM, so cheaper hardware can be used.

- **VRAM reduction (measured)**: On an RTX 5060 Ti (16 GB), Qwen3-14B (bf16, ~27.5 GB of weights) — which does not fit fully on the card — was run via `device_map=auto` + `offload_folder`, using a peak of **~13.4 GB GPU VRAM** vs. the **~27.5 GB** needed to keep everything on GPU: a **~51% reduction**. The trade-off is throughput — with heavy CPU/disk offload, generation dropped to **~0.5 tok/s**, so offload is about *fitting a model that otherwise wouldn't run*, not speed. Reproduce with `tests/benchmark_offload_vram.py` (raw data: `tests/benchmark_offload_vram_results.json`).
- **Monetary savings**: an **estimate**, dependent on GPU pricing assumptions. Based on the current preliminary assessment, **training costs can be reduced by approximately ~80% (an estimate, not a benchmarked amount)**; this figure is for reference only, and actual results depend on the model, hardware, and hours of usage.

## 🔑 Technical Highlights

### 1. DRAM/SSD Offload
- Traditional: the model must be loaded entirely into GPU VRAM, requiring a higher-end GPU.
- Trusta AST: uses quantization + device_map/offload_folder (inference) or DeepSpeed ZeRO-3 (training) to offload part of the data to DRAM/SSD, reducing VRAM requirements.

### 2. Integrated Fine-tuning
- A single platform completes inference and fine-tuning, sharing the offload mechanism and model registry; supports LoRA/QLoRA and full fine-tuning.

### 3. Multi-Process Isolation
- Inference runs in a dedicated worker process; when the worker exits due to OOM or other reasons, the main service stays alive and the status transitions to `error`.
- **Recovery is manual**: you need to call `/inference/load_model` or `/inference/force_cleanup_gpu` again.

## 📊 Use Cases

- **Small-to-medium enterprises / constrained budgets**: deploy 7B–13B models on smaller GPUs with quantization + offload.
- **Development environments**: frequently fine-tune and test on the same platform for fast iteration.
- **Mixed workloads**: handle inference and fine-tuning within the same service.

## 💡 Summary

Trusta AST's positioning: **an LLM service that integrates "offload to reduce VRAM requirements" and "fine-tuning" on the same platform**, suitable for teams that need to do both inference and fine-tuning under limited GPU resources.

---

**Last updated**: 2026-07-16  
**Related documents**: [[GPU Offload Mechanism]], [[Trusta AST vs Market Solutions]]
