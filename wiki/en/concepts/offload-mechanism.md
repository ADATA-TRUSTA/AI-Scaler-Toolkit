---
title: GPU Offload Mechanism
summary: Trusta AST offloads models or training state to DRAM/SSD via device_map / offload_folder / n_gpu_layers / DeepSpeed ZeRO-3, reducing GPU VRAM requirements, and integrates fine-tuning capabilities.
kind: concept
sources:
  - wiki/sources/inference_manual.md
  - wiki/sources/finetune_manual.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
provenanceState: extracted
---

# GPU Offload Mechanism

The **GPU Offload Mechanism** is one of Trusta AST's core features: it offloads model layers or training state that do not fit in GPU VRAM to CPU DRAM or SSD/NVMe, allowing smaller GPUs to load larger models, and integrates fine-tuning on the same platform.

## Core Mechanism (Actual Implementation)

Offload is not a single custom module; instead it uses the capabilities of existing frameworks, depending on the engine/scenario:

### 1. Inference: Transformers Engine

- Uses HuggingFace `device_map` (e.g., `"auto"`) to automatically split the model across GPU / CPU; layers that do not fit fall to disk (SSD/NVMe) via `offload_folder`.
- Combined with bitsandbytes quantization (`int4` / `int8` / `nf4` / `fp4`) to further reduce memory usage.
- Related implementation: `service/inference/engines/transformers_engine.py`.

### 2. Inference: llama-server Engine (GGUF)

- Uses `n_gpu_layers` (`n_gpu_layers=-1` means all layers on GPU; other values mean some layers on GPU and the rest on CPU) to control the GPU/CPU allocation.
- Related implementation: `service/inference/engines/llama_server_engine.py`.

### 3. Training: DeepSpeed ZeRO-3 Offload

- For full-parameter training or LoRA on large models, the optimizer state and parameters can be offloaded to CPU RAM or NVMe Disk.
- Four built-in profiles (`service/configs/deepspeed/`):
  - `zero3_offload_cpu_cpu`, `zero3_offload_cpu_disk`, `zero3_offload_disk_cpu`, `zero3_offload_disk_disk`
- Note: Disk offload significantly lengthens training time and is a measure for "extremely low-memory environments."

> Quantization benchmarking: VRAM reduction and tok/s can be measured with `tests/benchmark_offload_vram.py`,
> with results written to `tests/benchmark_offload_vram_results.json`; any numbers cited on this page should be based on that benchmark.
>
> Measured example (RTX 5060 Ti, 16 GB): Qwen3-14B bf16 (~27.5 GB of weights) ran via `device_map=auto` + `offload_folder` at a peak of **~13.4 GB GPU VRAM** (**~51% less** than the ~27.5 GB it would need fully on GPU), at **~0.5 tok/s** under heavy CPU/disk offload — i.e. offload trades throughput to fit a model that otherwise would not load.

## Benefits

- **Reduced VRAM requirements**: models that originally required a high-end, large-VRAM GPU to fit entirely into VRAM can, with offload enabled, run on machines with smaller VRAM.
- **Ability to run larger models**: constrained environments can still load models exceeding a single card's VRAM (at the cost of reduced throughput).
- **Inference and training share the same mechanism**: everything is done on the same platform, with no need to switch tools.

> Cost-benefit: hardware cost savings come from "a lower required GPU tier." **The VRAM reduction is measurable** (see above),
> while the monetary savings converted from the VRAM reduction are an **estimate** (dependent on GPU pricing assumptions), so documents should label them as estimated values when citing them.

## Fine-tuning Integration Capability

Trusta AST is not just an inference service; it also provides fine-tuning on the same platform:

- Inference and fine-tuning share the same offload mechanism and model registry.
- Supported methods: LoRA / QLoRA, full fine-tuning.
- Upon completion of training, it can automatically export GGUF (Q4_K_M) for llama-server inference.

## Practical Use Cases

- **Deploying small-to-medium models (7B–13B) on constrained hardware**: reduce VRAM requirements with quantization + device_map/offload.
- **Large models in low-memory environments**: use `n_gpu_layers` for partial offload or DeepSpeed disk offload to trade for "being able to run it at all."
- **An integrated inference + fine-tuning workflow**: complete fine-tuning and deployment within the same service.

## Related Concepts

- [[Multi-Process Isolation]] - Multi-process isolation architecture
- [[Inference Engine Comparison]] - Inference engine comparison
- [[Trusta AST Inference Service]] - Overall service introduction

---

**Last updated**: 2026-07-16  
**Related documents**: [[Trusta AST Inference Service]], [[Inference Engine Comparison]]
