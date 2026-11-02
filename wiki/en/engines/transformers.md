---
title: The transformers engine
summary: The only engine that generates inside the worker and the only one that needs no conversion, which makes it the place to check a fresh fine-tune. Includes device_map and quantization caveats.
updatedAt: 2026-10-29
---

# The transformers engine

## Its value: no conversion

| Engine | What a fine-tuning result needs |
|---|---|
| **transformers** | **A LoRA/QLoRA adapter loads from its path** (base and adapter given separately) |
| llama.cpp | Merge, then convert to GGUF |
| vLLM | Merge |

So it is the place to **check a fine-tune right away**. To judge what fine-tuning did, loading the
f16 adapter here is the cleanest test: no quantization loss and no conversion step in between.

The cost is concurrency: one forward pass at a time, no batching.

## `device_map`: the only placement control

Carried out by `accelerate`.

| Value | Result |
|---|---|
| `cuda:0` | Everything on the GPU |
| `auto` | accelerate splits across GPU/CPU |
| `cpu` | Everything on the CPU |
| **Omitted** | **Lands on `cpu`**, not the GPU |
| `none` | ⚠️ Passed to `from_pretrained` as-is and fails (`ValueError`) |

> ⚠️ `GET /config/offload_types` lists `none` ("Keep all in GPU"), but in 0.3.0 sending it fails
> the load. **To keep everything on the GPU, use `cuda:0`**; to split automatically, use `auto`.

### Placement is derived only when quantizing

Three conditions must all hold: **quantization is requested** (`int4`/`int8`),
`device_map == "auto"`, and **`max_memory` is given**. The model is then built once on the
`meta` device to measure its structure, and placed against a budget scaled for the quantized size.
Without quantization, accelerate's `"auto"` is used as-is.

### Intel GPUs load to CPU first

`accelerate` does not recognise `"xpu:N"`, so when the target is an XPU the model is loaded to the
CPU and moved afterwards.

## Placement reported after loading

| Field | Content |
|---|---|
| `device` | Main device |
| `device_map_summary` | e.g. `cuda:0:30, cpu:10` |
| `layer_lines` | Sampled layer placement |
| `memory_usage` | GPU usage |

## No `reasoning_content`

The upstream servers of the server engines split thinking into `reasoning_content`; this path has
no server-side reasoning parser, so **the field does not appear**. That is a difference between
paths, not a bug.

## Freeing memory

| Method | What it does |
|---|---|
| `POST /inference/cleanup_generation_memory` | Clears what generation left behind |
| Unload | **Ends the whole worker process**; VRAM always comes back |
| `POST /inference/force_cleanup_gpu` | Forced recovery after a worker failure |

## When to choose it

| Situation | Use it? |
|---|---|
| Just fine-tuned, want to try the adapter | ✅ Its best use |
| One user, low concurrency, an HF model | ✅ |
| Windows | ✅ |
| Serving several users | ❌ No batching |
| Model larger than the GPU | ❌ Use llama.cpp, vLLM expert offload, or Strata |

## Related

- [Choosing an inference engine](index.md)
- [Offload](../memory/offload.md)
