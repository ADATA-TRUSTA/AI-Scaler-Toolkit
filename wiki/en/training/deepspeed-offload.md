---
title: DeepSpeed ZeRO-3 offload
summary: Five offload profiles, the NVMe path and the pinned-DRAM budget, two saving traps, and why DeepSpeed is pinned at 0.19.2.
updatedAt: 2026-10-29
---

# DeepSpeed ZeRO-3 offload

For the four kinds of offload see [Offload](../memory/offload.md); this page is about the training one.

## Five profiles

In `backend/service/configs/deepspeed/`, named `<where parameters go>_<where the optimizer goes>`:

| Profile | Parameters | Optimizer |
|---|---|---|
| `zero3_offload_cpu_cpu` | CPU | CPU |
| `zero3_offload_cpu_cpu_tight` | CPU | CPU (smaller buffers, leaner) |
| `zero3_offload_cpu_disk` | CPU | NVMe |
| `zero3_offload_disk_cpu` | NVMe | CPU |
| `zero3_offload_disk_disk` | NVMe | NVMe |

Choose with `deepspeed_profile`, or pass your own file with `deepspeed_config`.

## Where NVMe goes: `AUTO`

The NVMe profiles say `AUTO` for the path, resolved at run time from `DEEPSPEED_NVME_DIR` (by
default `.deepspeed_offload` under the project). A job's `offload_folder` takes precedence. Point
it at a fast NVMe drive.

## The pinned-DRAM budget

`DEEPSPEED_NVME_PINNED_BUDGET_GB`, default 24.

Each swap buffer must hold **the largest single parameter**, and DeepSpeed pins them whatever
`pin_memory` says; uncapped, the size derived from the model can be tens of GB. The backend keeps
within the budget by **using fewer buffers** (a buffer cannot be smaller than the largest parameter).

> **The DRAM NVMe offload uses is not a leak.** Measured: it is real RSS (not page cache), from
> swap buffers plus fp32 staging. This budget is the knob.

## Two saving traps (handled by the backend)

| Trap | Handling |
|---|---|
| DeepSpeed + LoRA: `gather=True` OOMs, `gather=False` hits `FileExists` | Save only the adapter for PEFT models |
| `Cannot partition a param in flight`: saving mid-loop while ZeRO-3 is prefetching the next parameters | Drain in-flight parameters before saving |

## Version pins: do not upgrade casually

| Package | Version | Why |
|---|---|---|
| `deepspeed` | **0.19.2** | From 0.19.3, with LoRA + gradient checkpointing + ZeRO-3 offload, **frozen weights are not released from the GPU**. Qwen3.5-35B-A3B on a 16 GB card went from a 3.08 GiB peak to OOM on the first step. See [the measurement](../benchmarks/deepspeed-lora-16gb.md) |
| `transformers`, `trl` | Deliberately not the latest | Newer versions rename layers, and name-based lookups then **silently** find nothing |

## Long runs

| Risk | What to do |
|---|---|
| On a GPU that also drives the display, long kernels get killed by the driver (measured at about 61 minutes, `cudaErrorLaunchTimeout`) | Keep `save_steps` small; or drive the display from another card or the iGPU |
| The API has no resume | You restart from the last checkpoint |

## Related

- [Offload](../memory/offload.md)
- [Fine-tuning overview](index.md)
- [LoRA training a 35B MoE on a 16 GB GPU](../benchmarks/deepspeed-lora-16gb.md)
