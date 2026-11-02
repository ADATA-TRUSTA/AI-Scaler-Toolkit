---
title: Pitfalls found by running it
summary: Things you cannot tell from reading the code, found by running it; each marked as measured or as an incident.
updatedAt: 2026-10-29
---

# Pitfalls found by running it

> **What goes here: things the code does not tell you.** Each entry says how it is known:
> **measured** = backed by numbers; **incident** = it happened and the cause is recorded, but no
> measurements were kept.

## Wrong order, no error

The most dangerous kind: nothing reports it. The backend fixes these orders; know them if you change the code.

| Pitfall | Consequence |
|---|---|
| Importing a module that touches CUDA before `spawn` is set | `fork` copies an initialised CUDA context (undefined behaviour) |
| Importing transformers before settings are read | `HF_HOME` is not yet in effect, and the model cache lands elsewhere |
| Loading the model before creating the DeepSpeed configuration | ZeRO-3 silently gives up sharding at load time |
| Loading the model before enabling image training | The multimodal model has no vision tower |

## Hardware and long runs

| Pitfall | Detail | Known by |
|---|---|---|
| **A GPU that drives the display kills long training runs** | RTX 5060 Ti also driving the desktop: a ZeRO-3 run was killed after about 61 minutes with `cudaErrorLaunchTimeout`. The API has no resume | Measured |
| **Omitting `device_map` lands on the CPU** | Not the GPU | Measured |
| **`device_map="none"` raises `ValueError`** | Even though `GET /config/offload_types` lists it | Measured |
| **Multimodal full fine-tuning does not fit in 16 GB** | OOM in backward; enough DRAM does not help | Measured |
| **Loading a 50 GiB Strata model logged out the desktop** | The page cache held memory pressure above oomd's 50% threshold (69.8%). Unbuffered loading is now the default, peaking at 0.2% | Measured |
| **NVMe offload's DRAM use is real** | Swap buffers + fp32 staging, not page cache | Measured |

## Measurement traps

| Trap | Notes |
|---|---|
| **llama-server measured outside AST decodes about 16% slower** | AST's environment happens to use PyTorch's libgomp. Set `OMP_WAIT_POLICY=active` when measuring standalone; keep both sides the same in an A/B |
| **"Cold reads" on ZFS were page cache** | A 22 GB/s reading was ARC cache. Verify cold reads with `mincore()` |
| **`posix_fadvise(DONTNEED)` does not evict mmapped pages** | Only real memory pressure reclaims them |
| **The `/dev/shm` KV tier counted twice on Linux** | One process maps it twice; one vLLM DRAM figure was overstated as 47.6 GB, actually about 30 GB |

## Settings that exist but are not read

| Setting | Problem | Now |
|---|---|---|
| `HF_HUB_DISABLE_XET` set in the settings module | `huggingface_hub` reads it only once, at import | The download module changes the hub's constants directly |
| `DEFAULT_MAX_NEW_TOKENS` | Nothing read it | Replaced by `MAX_OUTPUT_TOKENS_CAP`, unset by default |

## Dependencies you do not upgrade casually

| Package | Version | Why |
|---|---|---|
| `deepspeed` | 0.19.2 | From 0.19.3 frozen weights are not released under LoRA. See [the measurement](../benchmarks/deepspeed-lora-16gb.md) |
| `transformers`, `trl` | Deliberately not the latest | Newer versions rename layers and name-based lookups silently fail |
| CUDA compiler toolchain | Pinned together to one CUDA minor | Pinning `nvcc` alone is not enough: `cicc` and the headers come from two other packages with no lower bound, and a wrong mix gives `ptxas fatal: Unsupported .version`. It only hits **machines with the driver but no system CUDA toolkit** — most of them — and only after many GB of downloads |

## Deployment

| Pitfall | Detail |
|---|---|
| A proxy's timeout must keep up with the backend | The backend's generation timeout is 1800 s, and when streaming the wait for the first byte is the prefill |
| SSE needs proxy buffering off | Otherwise training logs are not live |
| Without `.env`, `.env.example` is the configuration | The service looks for `.env` and falls back to `.env.example` |

## Related

- [Troubleshooting by symptom](troubleshooting.md)
- [Behind a reverse proxy](reverse-proxy.md)
