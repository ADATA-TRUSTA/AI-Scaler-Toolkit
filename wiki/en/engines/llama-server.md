---
title: The llama.cpp engine (llama_server)
summary: Serves GGUF through a managed llama-server; how its command line is built, which flags you may not set, slot cache, mmproj, and measured tuning.
updatedAt: 2026-10-29
---

# The llama.cpp engine (`llama_server`)

**Serves GGUF.** The backend manages a `llama-server` child process; generation goes from the
main process straight to it over HTTP.

## Two kinds of binary

| Kind | Source | How it is called |
|---|---|---|
| Unified `llama` | Official prebuilt release | With the `serve` subcommand |
| Standalone `llama-server` | Built from source | Arguments directly |

The arguments are the same; the backend tells them apart by file name. `LLAMA_SERVER_BINARY`
decides which one is used, and the service logs it and where the value came from at startup.

## How the command line is built

```
[binary] [serve] -m <model> --host --port -np -c -b --alias
         [-ngl] [-ncmoe] [--device] [--slot-save-path] [--mmproj] [--jinja]
         --api-key <internal key> --cors-origins <this service>
         <llama_server_extra_args>
```

**`llama_server_extra_args` always goes last.** llama-server takes the last of a repeated flag, so
your values override the automatic ones.

| Added automatically | Not added when |
|---|---|
| `-ncmoe` | `n_cpu_moe` is 0 |
| `--device` | `llama_server_device` is not set |
| `--mmproj` | None was found, or you already wrote one in the extra args |
| `--jinja` | You already wrote `--jinja` or `--no-jinja` |

### Flags you may not set

These flags belong to the service; putting them in `llama_server_extra_args` is refused before
anything starts (422): `--host`, `--port`, `--api-key`, `--cors-*`, `--ssl-*`, `--path`,
`--api-prefix`, `--slot-save-path`, `--log-file`, and a few more.

Why: llama-server accepts any origin by default, and these flags could move it off loopback, open
its CORS, or make it read and write files. For the same reason `llama_server_url` accepts only a
loopback address.

### The internal API key

The backend always starts llama-server with a key (`LLAMA_SERVER_API_KEY`; **a new random one on
every start** when unset), so that not even a web page on this machine can use that llama-server
directly.

> Side effect: a program that calls llama-server's port directly loses access whenever the backend
> restarts. Either set a fixed `LLAMA_SERVER_API_KEY`, or go through the backend's `/v1/`.

### Why `--jinja` is passed explicitly

**The thinking switch (`enable_thinking`) and the extraction of `reasoning_content` both need the
GGUF's jinja chat template.** Newer llama-server enables it by default, older ones do not, so it is
passed explicitly. Without `--jinja` there is no `reasoning_content`.

### What `--device` is for

Pinning the offload device. For example, on a machine with both an NVIDIA and an Intel GPU, choose
`Vulkan1` to use the Intel one; otherwise Vulkan picks the NVIDIA card first.

## Slot cache: text-only conversations can be saved

| Model | `prefill_strategy` | Behaviour |
|---|---|---|
| Text-only | **`slot`** | Restore each slot's KV on load, save it on unload |
| Multimodal | `cache_prompt` | Slots untouched; prompt cache only |

The restore result is in `/inference/status` as `slot_restore_summary` (`{restored, total, slots}`).

> Measured: after a restore you must **continue the same conversation** for the cache to apply.
> Asking the same question again requires rewinding recurrent state, which a slot file has no
> checkpoint for, so the whole prefix is recomputed. See
> [three ways to hand off the GPU](../benchmarks/gpu-handoff.md).

## Reported metadata

| Field | What it is |
|---|---|
| `llama_capabilities` | What the server says it supports |
| `llama_model_meta` | Includes `n_ctx_train` (the model's own limit) |
| **`served_context_length`** | **The window this load actually got** |
| `prefill_strategy`, `slot_restore_summary` | See above |

Put `served_context_length` next to `n_ctx_train` to see whether a limit comes from the engine or the model.

## mmproj: without it, vision does not work

The backend finds a multimodal GGUF's vision projector from the model file and the extra args.
**If one is detected but the file is missing, loading fails** instead of silently continuing.

> For gemma-4 in production, add `--image-min-tokens 280` to the extra args (measured).

## Measured tuning

Conditions: RTX 5060 Ti 16 GB, Intel Core Ultra 5 225 (6 P-cores + 4 E-cores), Linux.

| Fact | Measured |
|---|---|
| **On hybrid CPUs, set `-t` yourself, to the P-core count** | Qwen3.5-35B-A3B Q4_K_M, `-ncmoe 20`: default 14.1 tok/s → `-t 6` 23.5 → **`-t 10` drops to 9.8**. llama.cpp splits work evenly, so E-cores become each layer's critical path. Add `-t <P-cores>` via `llama_server_extra_args` |
| **Do not feed it BF16 GGUF** | Qwen3-4B fully on GPU: BF16 49 tok/s, Q8_0 88 tok/s. Both reach about 83% of the memory-bandwidth limit; BF16 is simply twice the size |
| **Use `n_cpu_moe` when a MoE does not fit** | Expert weights go to RAM; attention and KV stay on the GPU. See [Memory estimation](../memory/estimation.md) |
| **Benchmarking llama-server outside AST: set `OMP_WAIT_POLICY=active`** | With experts on the CPU, decode inside AST is about 16% faster than standalone (47.5 vs 40.7 tok/s). The difference is OpenMP's wait policy: AST's environment happens to load PyTorch's libgomp. Keep both sides the same when comparing |

## Startup failures

The backend keeps the child's recent stderr (up to 12 lines / 2000 characters), builds a one-line
summary from the exit code, and checks whether it was an OOM rather than calling every failure one.

## Related

- [Choosing an inference engine](index.md)
- [Memory estimation](../memory/estimation.md) — how `-ngl` and `-ncmoe` are computed
- [GGUF conversion](../models/gguf-conversion.md)
- [Chat](../api/chat.md) — thinking and `reasoning_content`
