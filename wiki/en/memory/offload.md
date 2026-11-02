---
title: "Offload: four different things"
summary: In AST "offload" means four things — transformers weight placement, vLLM expert-weight offload, vLLM's tiered KV cache, and DeepSpeed's training state. Settings, timing and failure modes all differ.
updatedAt: 2026-10-29
---

# Offload: four different things

> **When you see offload, first ask "offload what?"**

| # | Name | What is offloaded | Settings | When |
|---|---|---|---|---|
| 1 | `device_map` | **Weights** (whole layers to GPU/CPU/disk) | `device_map`, `max_memory`, `offload_folder` | transformers inference |
| 2 | vLLM weight offload | **Weights**, optionally only MoE experts | `vllm_cpu_offload_gb`, `vllm_cpu_offload_params` | vLLM inference |
| 3 | vLLM KV offload | **KV cache** | `vllm_kv_offloading_size`, `vllm_kv_offloading_disk_path` | vLLM inference |
| 4 | DeepSpeed ZeRO-3 | **Parameters, optimizer state, gradients** | DeepSpeed profile, `offload_folder` | **Training** |

llama.cpp's `n_cpu_moe` (`-ncmoe`) is closest to #2; Strata keeps experts in DRAM by design.

## 1. `device_map` (transformers)

| Value | Result |
|---|---|
| `cuda:0` | Everything on the GPU |
| `auto` | accelerate splits; **when quantizing with `max_memory`**, the structure is measured on the meta device first and placed against the quantized budget |
| `cpu` | Everything on the CPU |
| Omitted | **Lands on the CPU**, not the GPU |

See [transformers](../engines/transformers.md).

## 2. vLLM weight offload

| Field | Effect |
|---|---|
| `vllm_cpu_offload_gb` | **Budget**: GB of weights kept in pinned CPU memory and read over PCIe on every forward |
| `vllm_cpu_offload_params` | **Scope**: offload only parameters whose names contain these segments |

**`["experts"]` decides whether a MoE larger than the GPU loads at all**: experts on the CPU,
attention on the GPU. Without a scope, weights are moved in order, attention included, and
everything gets slower. Whole segments match: `experts` matches `mlp.experts.w13_qweight`;
`expert` does not. A scope with a zero budget is refused.

> Measured: expert offload can use UVA or Prefetch. On a 16 GB card at PCIe Gen5 x8, UVA gave
> prefill 831 tok/s and decode 17.8 tok/s; Prefetch gave 2,000 and **0.76** tok/s. **Use UVA.**
> See [vLLM native KV cache offload](../benchmarks/vllm-kv-offload.md).

## 3. vLLM KV offload: two tiers, DRAM is the gate

```
GPU ──evict──► DRAM tier (vllm_kv_offloading_size, GB) ──► NVMe tier (vllm_kv_offloading_disk_path)
```

- **No skipping a tier**: a block on NVMe must be copied into the DRAM tier before the GPU can use it.
  The DRAM tier must hold the longest prefix you want to reuse, or the hit rate is 0% even though
  the data was read.
- With several GPUs, `vllm_kv_offloading_size` is **the total across all of them**.
- KV offload turns on prefix caching; **prefix caching must stay on**, or offload silently does nothing.
- **The NVMe tier never deletes on its own.** Set a cap with `vllm_kv_offloading_disk_max_gb`, or
  call `POST /inference/kv_cache/disk/cleanup`.
- **Offload saves recomputation time, not concurrent capacity**: a request being computed still
  needs all of its KV on the GPU.

Measured: revisiting a full 146K-context conversation that had been evicted went from 444.9 s to
6.04 s. See [vLLM native KV cache offload](../benchmarks/vllm-kv-offload.md).

## 4. DeepSpeed training offload

During training, parameters, optimizer state and gradients are sharded and offloaded to CPU or
NVMe. Five profiles are built in, named `<where parameters go>_<where the optimizer goes>`.

| Watch out for | Notes |
|---|---|
| NVMe offload uses a lot of DRAM | **Real RSS, not page cache, and not a leak**: swap buffers plus fp32 staging. The knob is `DEEPSPEED_NVME_PINNED_BUDGET_GB` |
| Multimodal full fine-tuning does not fit in 16 GB | OOM in backward; **enough DRAM does not help when VRAM runs out**. Use LoRA |

See [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md).

## Related

- [Memory estimation](estimation.md)
- [Choosing an inference engine](../engines/index.md)
- [Glossary](../glossary.md)
