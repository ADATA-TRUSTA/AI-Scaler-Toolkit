---
title: Service architecture
summary: The backend's three processes, two inference paths, where state is kept, and which limits are enforced versus only warned about.
updatedAt: 2026-10-29
---

# Service architecture

This page is about **processes, state, and mutual exclusion**: the "why can't I do that here" layer.
For the API list see [API overview](api/index.md); for the training flow see
[Fine-tuning overview](training/index.md).

## The one-minute version

```
                 ┌──────────────────────────────────────┐
                 │  FastAPI main process (one uvicorn)  │
                 └────┬──────────────────────┬──────────┘
                      │ spawn                │ spawn
                      ▼                      ▼
          ┌─────────────────────────┐  ┌─────────────────────────┐
          │ Inference worker         │  │ Training worker          │
          │ transformers runs here   │  │ one job at a time        │
          └─────────────────────────┘  └─────────────────────────┘

   llama-server / vLLM / Strata do not use the worker:
   the main process talks to "the server it manages" over async HTTP
```

**Three processes, two inference paths.** Everything else is detail.

## Why separate processes

**With the model in its own process, VRAM can be reclaimed after an OOM.** When the process
ends, its CUDA context goes with it; `del model` plus `torch.cuda.empty_cache()` in the same
process cannot do that, and the state after an OOM is not clean anyway. Training gets its own
process so that a stall or OOM there cannot freeze the API.

So **unloading a transformers model ends the worker process**, and VRAM always comes back.

### `spawn` is mandatory

`app.py` calls `multiprocessing.set_start_method("spawn")` before almost every import. `fork`
copies an initialised CUDA context into the child, which is undefined behaviour; once any module
has touched CUDA, it is too late to change the start method.

The cost is that children inherit nothing: each worker re-imports and re-reads its settings.

## Two inference paths

| Engine | Path | How the main process gets results |
|---|---|---|
| `transformers` | **Worker queue** | Through a multiprocessing queue; one dispatcher thread routes each response to its request |
| `llama_server`, `vllm`, `strata` | **Async HTTP** | The main process `await`s the server directly |

The three server engines already speak OpenAI-compatible HTTP, so the main process calls them
directly; a request is one `await` on the event loop. Both paths return **the same shape** to the
caller.

### Where the difference shows

| What you see | Why |
|---|---|
| `reasoning_content` only from server engines | The upstream servers split out reasoning; the transformers path has no server-side reasoning parser. **Not a bug** |
| `/inference/error_details` empty for server engines | It reads errors recorded by the worker, and server-engine generation does not go through the worker |
| `/inference/status` fields differ by engine | transformers reports per-module placement; llama.cpp reports layer split and `served_context_length`; vLLM reports its endpoint; Strata reports only the device. **Callers must tolerate missing fields** |

### The port must be settled in the main process

If llama-server's default port is taken, another is chosen. That has to happen in the main
process, **before the worker gets its copy of the settings**. Done inside the engine, the worker
would change only its own copy while the main process kept the old port: **the model loads, and
every generation fails to connect**.

That is the price of spawn: two processes, two copies of the settings, so **any decision both
must agree on is made in the main process and passed down**.

## Where state lives

Ask this first when troubleshooting:

| State | Stored in | After a service restart |
|---|---|---|
| Loaded model and its settings | Main process memory + worker | **Gone** |
| Download tasks | Main process memory | **Gone** |
| Live training status | Memory | **Gone** |
| Training event stream | **Disk**, `LOG_DIR/training/{job_id}/` | Kept |
| Chat history (`session_id`) | Redis; **memory when not configured** | Depends on Redis |
| Registered models | JSON on disk | Kept |

Redis is used only when configured. **Without it, memory is the store, not a fallback for a
failed connection**, so no connection warning appears when Redis is absent.

The service **loads no model at startup**. A worker starts the first time it is needed.

## Mutual exclusion: enforced versus warned

### Enforced

| Rule | Behaviour |
|---|---|
| One inference model at a time | Same model, same key settings → `200 already_loaded`; a different model or settings → **`409`**; while loading → `409` |
| One training job at a time | Refused |
| Concurrent generations capped | `MAX_CONCURRENT_GENERATIONS`, default 8 |
| Per-generation timeout | `DEFAULT_GENERATION_TIMEOUT`, default 1800 s |

> **"Same key settings" compares only six fields**: `quantization`, `device_map`, `max_memory`,
> `offload_folder`, `torch_dtype`, `model_total_memory` (plus the model path or name).
> **Reloading with only other fields changed returns "already loaded" and does not reload.**
> To apply other settings, unload and load again.

### Only warned

**Training and inference can share a GPU.** If a model is loaded when training starts, the server
writes one warning line to its log; the API caller never sees it. On a 16 GB card that is usually
an OOM. **Unload the model before training.**

## Who can connect

Two middleware layers decide before any route runs: CORS accepts only pages served by this
machine; the other applies three sets of rules by source — this machine, other machines (key
required, `/v1/` only by default), and web pages in a browser. See
[Remote access and the trust boundary](api/remote-access.md).

That layer is deliberately plain ASGI middleware, **so streamed responses (chat, training logs)
pass through untouched**.

## Related

- [Choosing an inference engine](engines/index.md)
- [API overview](api/index.md)
- [Fine-tuning overview](training/index.md)
- [Remote access and the trust boundary](api/remote-access.md)
