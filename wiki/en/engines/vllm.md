---
title: The vLLM engine
summary: The most constrained and the fastest of the four: Linux only, runs in its own venv, detects its tool parser automatically, two-tier KV offload, sleep/wake, classified startup failures.
updatedAt: 2026-10-29
---

# The vLLM engine

The backend manages a `vllm serve` child process. The version is **0.28.0**.

## Can it be used: platform and install

1. **Not Linux** → not offered
2. **Linux but not installed** → not offered either

| `TRUSTA_SETUP_VLLM` | NVIDIA host | Intel host |
|---|---|---|
| Unset (`auto`) | **Installed** | Not installed |
| `1` | Installed | **Installed** |
| `0` | Not installed | Not installed |

There are three reasons it is not installed on Intel by default: an integrated GPU has no memory of
its own, so vLLM's "device memory" is system RAM; the kernel's TTM page pool keeps up to half of
RAM after use, which vLLM's startup check counts as used, so **the first load works and the second
fails**; and vLLM's official XPU support does not cover integrated GPUs.

When refused, `POST /inference/load_model` returns `501` with the reason. The web interface builds
its menu from `GET /health`'s `supported_engines`, so normally you never see the option.

## It runs in its own venv

vLLM has its own project directory and venv (`VLLM_SERVER_PROJECT_DIR`), separate from the
backend's. **That way vLLM's version is not tied to the backend's**; their dependencies conflict.
The backend finds vLLM's interpreter and the NVIDIA shared-library paths itself.

## Tool-call parser: asked, not looked up

At load time the backend renders one tool call with the model's own chat template, then lets each
of vLLM's parsers try to read it back. **The one that can is the answer.** The answer comes from
the two authorities involved (the model's template and vLLM's parsers), so it stays right when vLLM
renames a parser or changes its syntax.

`vllm_tool_call_parser` and `vllm_reasoning_parser` override the detection.

## With a reasoning parser, thinking defaults to off

Condition: the engine is vLLM, a reasoning parser is active, and the request does not set
`enable_thinking`.

The symptom it fixes: **every reply after a tool result is empty**. The reasoning parser treats
everything before `</think>` as reasoning, which with no `</think>` is the whole reply; and some
hybrid templates (MiniCPM5, for example) prefill nothing when the switch is unset, so the model often
answers directly — and the answer, tool call included, is filed as reasoning.

It is pinned to "off" rather than "on" because fine-tuning data usually has no think block in the
answers. A developer measurement: one MiniCPM5-1B LoRA answered 27 of 40 with thinking off, 18
with it on.

It only affects loads with a reasoning parser; models whose template thinks anyway (Qwen3, for
example) are unaffected.

## MoE expert offload: how a MoE larger than the GPU loads

| Field | Effect |
|---|---|
| `vllm_cpu_offload_gb` | How many GB of weights go to pinned CPU memory |
| `vllm_cpu_offload_params` | Offload **only** parameters whose names contain these segments |

`["experts"]` puts only the MoE experts on the CPU and keeps attention on the GPU. Without it,
weights are moved in order, attention included, and everything gets slower. **Whole segments
match**: `experts` matches `mlp.experts.w13_qweight`; `expert` does not.

See [Offload](../memory/offload.md).

## KV offload: two tiers, DRAM is the gate

```
GPU ──evict──► DRAM tier (vllm_kv_offloading_size) ──► NVMe tier (vllm_kv_offloading_disk_path)
```

**A block on disk must be copied into the DRAM tier before the GPU can use it.** So the DRAM tier
must hold the longest prefix you want to reuse, and without `vllm_kv_offloading_size` the disk
tier does nothing at all.

| Field | Notes |
|---|---|
| `vllm_kv_offloading_size` | DRAM tier size (GB). **With several GPUs, the total across all of them** |
| `vllm_kv_offloading_disk_path` | Directory of the NVMe tier |
| `vllm_kv_offloading_disk_max_gb` | Cap for the NVMe tier (GiB) |

**vLLM never deletes from the disk tier**; the directory only grows (one test directory reached
65 GB). The backend therefore offers a cap and a cleanup:

```
GET  /inference/kv_cache/disk           # how much is used
POST /inference/kv_cache/disk/cleanup   # delete least-recently-used blocks to below 90% of the cap
```

Deleting while it runs is safe: vLLM looks a block up by checking whether its file exists, so a
deleted block is a miss and is recomputed; writes go to a temporary file and are renamed, so a
`.bin` is always complete.

The backend fixes `PYTHONHASHSEED` when it starts vLLM, **so the KV can be reused across restarts**;
otherwise hashes start from a different seed each time and the cache silently misses.

Measurements: [vLLM native KV cache offload](../benchmarks/vllm-kv-offload.md).

## Sleep / wake: release the GPU without unloading

```
POST /inference/sleep   {"mode": "wait" | "abort"}
POST /inference/wake
```

Load with `vllm_enable_sleep_mode`. vLLM's sleep moves the weights to pinned CPU memory and
**drops the KV cache**.

| Behaviour | Notes |
|---|---|
| `mode: "wait"` (default) | Waits for running generations to finish |
| `mode: "abort"` | Aborts them immediately |
| A generation request while asleep | Wakes first; with `vllm_auto_wake` off, `503` with `Retry-After: 1` |
| Sleep while already asleep | Does nothing |
| Engine does not support sleep | `409` |
| Sleep or wake itself fails | `502` |

> ⚠️ **Wake can fail, and commonly does**: another program took the GPU memory while it slept,
> so it cannot come back → `502`, and `/inference/status` still says asleep.
> **A coordinator must re-read the status rather than assume wake succeeded.**

The backend has **no idle timer**: the caller (an agent skill, for instance) decides when to sleep
and wake; `auto_wake` is only a safety net so a request does not fail when the coordinator forgot.

### The admin endpoints are guarded

Sleep needs vLLM's admin endpoints, and the same switch also mounts dangerous ones such as
`/collective_rpc` and `/update_weights`. The backend adds a vLLM middleware that decides who gets in:

| Endpoint | Local programs | Web pages |
|---|---|---|
| `/sleep`, `/wake_up`, `/is_sleeping`, cache reset | ✅ | ❌ |
| Other admin endpoints | 404 | 404 |

When `VLLM_SERVER_HOST` is not loopback, the admin endpoints are always off, and asking for
`vllm_enable_sleep_mode` is refused before anything loads.

## Intel GPUs: `gpu_memory_utilization` is lowered

On XPU builds only: the interface sends `0.9`, but an integrated GPU's "device memory" is system
RAM; on a 32 GB laptop 0.9 is 25.7 GB, which is never free, and vLLM refuses to start. The backend
uses **the fraction that is actually free now**, keeps 2 GiB for the system, and logs a warning.
NVIDIA builds pass the value through unchanged.

## Startup failures in eleven classes

```
oom · config · port_busy · model_not_found · cuda_mismatch · gpu_fault
shared_library_missing · toolchain_missing · chat_template · quantization · unknown
```

They exist because **every failure used to be guessed as OOM**; mistake a missing shared library
for OOM and users keep shrinking the model and keep failing.

`toolchain_missing` is checked before `model_not_found` because its message looks like a missing
file: without `intel-ocloc` on Intel it is `No such file or directory: 'ocloc'`; without `nvcc` on
CUDA it is `could not find nvcc`.

## Leftover cleanup

| What | When |
|---|---|
| Leftover `vllm serve` processes | At service startup (`VLLM_STARTUP_SWEEP`), so they do not hold the port |
| Offload regions in `/dev/shm` | Left by a killed server; removed only after checking no live process maps them |

## Where it fits

| Fact | Notes |
|---|---|
| On a 16 GB card, 8B-class bf16 models do not fit in vLLM | Only Qwen3-4B-class has headroom; larger models need quantization or expert offload |
| Qwen3.5-35B-A3B with 131K context in 8 GB of VRAM | GPTQ-Int4 + experts on CPU + fp8 KV. See [the 8 GB comparison](../benchmarks/vllm-vs-llamacpp-8gb.md) |
| vLLM 0.27.1 cannot load gemma-4 | Fixed in 0.28.0 |

## Related

- [Choosing an inference engine](index.md)
- [Offload](../memory/offload.md)
- [Chat](../api/chat.md)
- [Error model](../api/errors.md)
