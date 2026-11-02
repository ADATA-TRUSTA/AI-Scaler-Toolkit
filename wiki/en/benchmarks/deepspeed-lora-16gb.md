---
title: LoRA training a 35B MoE on a 16 GB GPU
summary: A Qwen3.5-35B-A3B LoRA trained on a 16 GB card with ZeRO-3 CPU offload peaks at 3.08 GiB with DeepSpeed 0.19.2; from 0.19.3 frozen weights are not released from the GPU and the first step runs out of memory. AST pins DeepSpeed at 0.19.2 for this reason.
updatedAt: 2026-10-29
measuredAt: 2026-09-02
---

# LoRA training a 35B MoE on a 16 GB GPU

## Conditions

| Item | Value |
|---|---|
| Hardware | RTX 5060 Ti 16 GB, 125 GB DRAM, Linux |
| Software | torch 2.13.0+cu130, transformers 5.15.0 |
| Training | Qwen3.5-35B-A3B, LoRA, profile `zero3_offload_cpu_cpu_tight`, seq 512 |

## Results: only the DeepSpeed version changes

| | Gradient checkpointing on | Off |
|---|---|---|
| DeepSpeed **0.19.2** | **Trains · peak 3.08 GiB** (sawtooth 3.1 ↔ 2.5, no build-up) | OOM · 12.07 GiB |
| DeepSpeed 0.19.6 | OOM · 11.99 GiB (staircase 1.5 → 8.3 → 12.0) | OOM · 12.07 GiB |

Turning checkpointing off is no way around it: it trades this problem for activation memory blowing up.

## Mechanism

- To fix a crash, 0.19.3 also records "the current backward owner" on frozen parameters during
  checkpoint recomputation; a recomputed layer's own hook removes only its own record, so the
  parameters are released only when the owner's backward finishes.
- Upstream assumed frozen parameters are few. **LoRA is the opposite**: the whole base is frozen,
  so every weight of every layer stays on the GPU.
- The numbers match: fused experts are 1.50 GiB per layer; 1.5 + 1.50 × 7 = 12.0 GiB.
- Disabling that mechanism on 0.19.6 restores the sawtooth: 4.03 GiB, 13 steps completed.

AST's training already uses reentrant checkpointing with DeepSpeed, where that crash cannot
happen — so in this configuration the 0.19.3 change is cost with no benefit.

## Who is affected

| Case | 0.19.5 → 0.19.2 |
|---|---|
| Qwen3.5-35B MoE · LoRA | OOM → 3,237 MB |
| Qwen3-4B dense · LoRA | 7,834 → 4,777 MB (−39%) |
| gemma-4-E4B · LoRA (images, NVMe offload) | Unchanged |
| SmolVLM 256M · full fine-tune | Unchanged (no frozen parameters) |

It builds up only when all four hold: LoRA/PEFT, gradient checkpointing, ZeRO-3 offload, and frozen
weights large per layer.

## Related

- [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md)
- [Fine-tuning overview](../training/index.md)
