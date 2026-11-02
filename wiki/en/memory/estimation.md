---
title: Memory estimation
summary: Two estimators (HF and GGUF models), the analytic and oracle tracks for GGUF, what each of the five GGUF endpoints answers, and the placement rules taken from llama.cpp.
updatedAt: 2026-10-29
---

# Memory estimation

## Two estimators; do not mix them

| | HF models | GGUF models |
|---|---|---|
| For | Models loaded by transformers | GGUF served by llama.cpp |
| Endpoints | `POST /inference/estimate_memory` | `POST /inference/estimate_memory/gguf` and four variants |
| Answers | Roughly how much memory | **Which layers go on which device, and whether this configuration fits** |

Their precision is entirely different.

## HF models: three fallbacks

Tried in order: compute from the config → build the model once on the `meta` device → read the
safetensors index. The first goes wrong on unusual architectures; the second needs the model to be
instantiable; the third needs it downloaded.

If Hugging Face is unreachable, it probes once (2 s) and remembers, so requests do not each wait
for a timeout. For gated repos or offline use, the error tells you to provide `HF_TOKEN` or
download the model first.

## GGUF models: two tracks

| Track | How | Cost | Precision |
|---|---|---|---|
| **analytic** (default) | Reproduces llama.cpp's placement and KV arithmetic in pure Python from the GGUF tensor table | Practically zero | Weights and KV **exact**; compute buffer from a calibrated formula, **deliberately high** |
| **oracle** (`verify: true`) | Runs `llama-fit-params -fitp on`, llama.cpp's own dry-run load | One process per data point (about 0.4 s) | **Exact** |

A sweep tries many combinations, and a process for each is not practical, so analytic is the
default; add `verify: true` for the final check of one configuration.

Accuracy: against `llama-fit-params` on 6 architectures, weights and KV are within 1 MiB and never
underestimated; the compute buffer is over by 0–26%. See
[Accuracy of the GGUF memory estimate](../benchmarks/gguf-estimation-accuracy.md).

## The five GGUF endpoints

All under `/inference/estimate_memory/`.

| Endpoint | Question | Key response fields |
|---|---|---|
| `gguf` | How much does this configuration need | `memory_breakdown_mib`, `placement`, `per_layer_mib` |
| `gguf/check` | **Does this configuration fit** | `fits`, `verdict`, `summary`, `suggestion` |
| `gguf/recommend` | **Give me a configuration** | `recommended`, `utilization`, **`llama_server_args`** |
| `gguf/sweep` | A table of combinations | `rows`, `truncated` |
| `gguf/plan` | One plan per KV cache type | `plans`, `kv_cache_types` |

**The `llama_server_args` from `recommend` can be pasted straight into `llama_server_extra_args`**,
the shortest answer to "it does not load". Measured, its suggestions use 96.6–97.8% of the budget.

When the target is not a GGUF, `gguf/check` returns `400` with a dict `detail` containing a `hint`
that this endpoint only checks GGUF. The per-device margin follows `llama-fit-params`' `-fitt`
default.

## llama.cpp's placement rules

These rules follow llama.cpp's source one by one, and they explain some of its less intuitive behaviour:

| Rule | Consequence |
|---|---|
| There are `n_layer + 1` slots and **the GPU takes the last `n_gpu_layers`**; the last slot is the output layer | **The output layer reaches the GPU before any block**; `-ngl 1` places only the output layer, not "the first layer" |
| Token embeddings **always stay on the CPU** | Not counted in the GPU budget |
| `--n-cpu-moe N` moves the expert tensors of the first N layers to the CPU | Experts go to RAM; **attention and KV stay on the GPU** |
| The KV cache follows the device owning its layer | |
| `n_ctx` is rounded up to a multiple of 256 | |

> These rules are copied, so they go stale if llama.cpp changes them; the compute-buffer
> calibration is also one-off. Re-check after upgrading llama.cpp.

### Measured behaviour that is easy to misread

| Behaviour | Measured |
|---|---|
| Without flash attention the compute buffer explodes | 32B, 32K ctx: 344 → 5,256 MiB, so `recommend` always says `-fa on` |
| `-ncmoe` does not grow the compute buffer; lowering `-ngl` does | |
| Models with tied embeddings carry an extra `token_embd` | gemma-4-26B-A4B Q4_K_M: at `-ngl 1` the GPU holds exactly 748 MiB more |
| Hybrid linear attention has KV on some layers only | Qwen3.5-9B, c=4096: KV 178 MiB, not the 512 MiB of counting every layer as attention |

## Related

- [Offload](offload.md)
- [The llama.cpp engine](../engines/llama-server.md)
- [Accuracy of the GGUF memory estimate](../benchmarks/gguf-estimation-accuracy.md)
