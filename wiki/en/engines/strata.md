---
title: The Strata engine
summary: Experts of a large MoE live in DRAM, hot experts are cached on the GPU; deep sleep releases the GPU in about a second and keeps the conversation. Optional, NVIDIA only.
updatedAt: 2026-10-29
---

# The Strata engine

[Strata](https://github.com/Niko1221/Strata) is an inference server built for large MoE models.
AST uses a build with **deep sleep** added.

## The problem it solves

A MoE's experts live in **DRAM**, and **hot experts are cached on the GPU**, with the cache sized to
fill the card.

| | vLLM expert offload | Strata |
|---|---|---|
| Where experts live | Chosen with `vllm_cpu_offload_params=["experts"]` | That is the design |
| What is on the GPU | KV cache + non-expert weights | **A cache of hot experts** |
| Releasing the GPU | sleep: **KV dropped** | deep sleep: **KV and conversation kept** |

## Can it be used

Optional, **NVIDIA only**. Backend 0.3.0 does not contain Strata: `strata` appears in
`GET /health`'s `supported_engines` only when `STRATA_HOME` points to an installed Strata whose
engine is a deep-sleep build supporting `--sleep-mode`. The settings are in the Strata section of
[`backend/.env.example`](../../../backend/.env.example).

> The backend **always** starts Strata with `--sleep-mode`, so the prebuilt engine Strata's own
> setup downloads (which lacks that flag) cannot be used; `STRATA_ENGINE_BINARY` must point to a
> deep-sleep build.

## How a model is described: the run config

A Strata model is not described by a list of arguments but by the **run config** Strata's setup
writes (`strata-<model>.json`: engine binary, expert pack, GGUF shards, vision encoder).

```
your strata-<model>.json ──copied, this load's overrides applied──► STRATA_RUNTIME_DIR/ ──started from that copy
```

**Your file is never modified.** Without `strata_config_path`, the backend looks in `STRATA_HOME`
for the run config whose first GGUF shard is `model_path`.

## Only four load fields

```
strata_config_path · strata_max_context · strata_vram_limit_gb · strata_auto_wake
```

The other startup flags (KV format, KV placement, speculative window, expert cache) stay in the run
config; edit that file to tune them. This is the opposite of vLLM: **a small API, a rich file**.

### `strata_vram_limit_gb`: less is slower; leave it unset

Strata fills the GPU with its expert cache; this value is room kept for other programs.

| Available VRAM | Decode |
|---|---|
| About 15 GB | 72–84 tok/s |
| About 8 GB | 45–54 tok/s |

**Leave it unset, and `POST /inference/sleep` when another program needs the GPU.**

## Deep sleep: release the GPU in about a second, keep the conversation

Always on; it costs nothing until the first sleep.

| | vLLM sleep | Strata deep sleep |
|---|---|---|
| Weights | Moved to pinned CPU memory | The engine's VRAM is copied to pinned host RAM |
| KV cache | **Dropped** | **Kept** |

Measured (Windows, RTX 5060 Ti 16 GB, GSQ-RCO IQ3_S, two shards of about 78 GB): sleep 1.3 s,
VRAM from 15,866 to 1,498 MiB; wake 1.0 s, and the conversation continued. The endpoints are shared
with vLLM: `POST /inference/sleep`, `POST /inference/wake`; see the sleep/wake section of
[vLLM](vllm.md) for their behaviour.

For a comparison of three ways to give up the GPU at 256K context, see
[three ways to hand off the GPU](../benchmarks/gpu-handoff.md).

### The conversation is saved and restored on the next load

On unload the slot is saved to `STRATA_RUNTIME_DIR/slots/<model>.slot-0.bin` and restored on the
next load. No saved file is not an error: `{"restored": false, "reason": "no saved conversation"}`.

> This is not the server-side chat history of `session_id`: what is saved here is the engine's own
> KV, not a message list.

## Two environment defaults at load

```
STRATA_UNBUFFERED_LOAD=1    STRATA_NO_ARENA_THP=1
```

Applied only when you have not set them yourself. The reason is a measured incident: loading a
50 GiB expert arena through the page cache kept memory pressure above systemd-oomd's 50% threshold
for long enough (69.8% measured) that **oomd logged out the whole desktop session**. Unbuffered,
with transparent huge pages off, the same load peaked at 0.2% and **read faster**
(9.1–10.0 GiB/s against 6.4).

## How it runs

Like llama-server and vLLM: the worker starts the server (listening only on
`127.0.0.1:STRATA_PORT`) and the main process talks to it over HTTP. The wait for the model to load
is capped by `STRATA_HEALTH_TIMEOUT`, 900 s by default (it reads tens of GB). On failure the last 40
lines of the server log are attached.

## Settings

| Variable | Default | Notes |
|---|---|---|
| `STRATA_HOME` | empty | Strata's install directory (holding `serve/server.py`); empty means the engine is not offered |
| `STRATA_PYTHON` | — | Interpreter for `serve/server.py`, normally Strata's own `.venv` |
| `STRATA_ENGINE_BINARY` | — | **Required**: the deep-sleep engine build |
| `STRATA_PORT` | `5002` | Loopback port |
| `STRATA_HEALTH_TIMEOUT` | `900` | Seconds to wait for the model to load |
| `STRATA_RUNTIME_DIR` | `./.cache/strata` | Run-config copies, server log, saved conversations |

## Related

- [Choosing an inference engine](index.md)
- [vLLM](vllm.md) — the other engine that can sleep
- [Three ways to hand off the GPU](../benchmarks/gpu-handoff.md)
