---
title: Measured results
summary: Measurements tied directly to AST's core — DRAM/SSD offload, engine choice, memory estimation, GPU handoff, low-VRAM training — each with its hardware, model and settings.
updatedAt: 2026-10-29
---

# Measured results

**Every number holds only under its conditions.** A different GPU, model, quantization or version
can give different results. Each page states its conditions first and its limits last.

| Page | Conclusion in one sentence |
|---|---|
| [vLLM native KV cache offload to DRAM/NVMe](vllm-kv-offload.md) | Revisiting a full-context conversation with a 35B model on a 16 GB card went from 445 s to 6 s; the DRAM tier must hold the prefix to reuse |
| [vLLM vs llama.cpp on an 8 GB VRAM budget](vllm-vs-llamacpp-8gb.md) | Shared prefixes, long prompts and reuse after restart favour vLLM; single-user decode and startup favour llama.cpp |
| [Accuracy of the GGUF memory estimate](gguf-estimation-accuracy.md) | Weights and KV within 1 MiB and never under; `recommend`'s settings use about 97% of the budget |
| [What Gen5 NVMe does for local AI workloads](gen5-ssd.md) | The gain depends on how much time is spent waiting for the disk; pre-warming the page cache doubles MoE decode |
| [Handing the GPU to another program: three ways](gpu-handoff.md) | At a 256K conversation, Strata sleep/wake takes about 1.8 s from getting the GPU back to the first answer; reloading takes 26–33 s |
| [LoRA training a 35B MoE on a 16 GB GPU](deepspeed-lora-16gb.md) | 3.08 GiB peak with DeepSpeed 0.19.2; OOM on the first step from 0.19.3 |

## The usual test machine

Unless a page says otherwise:

| Item | Spec |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16 GB (PCIe Gen5 x8, also driving the desktop) |
| CPU | Intel Core Ultra 5 225 (6 P-cores + 4 E-cores) |
| RAM | 125 GB DDR5 |
| SSD | ADATA T7P5 Gen5 NVMe |
| OS | Ubuntu 24.04 |

Laptop tests (RTX 5080 Laptop 16 GB, Core Ultra 9 275HX) are marked on their pages.
