---
title: API overview
summary: The backend's 54 routes by group and purpose; the route list itself is in docs/usage.en.md, and this page adds each group's behaviour and limits.
updatedAt: 2026-10-29
---

# API overview

The full route list with command-line examples is in
[`docs/usage.en.md`](../../../docs/usage.en.md#api-routes); interactive docs are at the service's
`/docs` (reachable from this machine only). This page adds **each group's behaviour and limits**.

## Basics

| Fact | Notes |
|---|---|
| Address | Desktop app: `http://127.0.0.1:8571`; from source: `SERVICE_PORT`, default `8000` |
| Path prefix | **No `/api`**; everything is at the root. A prefix behind a reverse proxy is added by the proxy |
| Who can call | Everything from this machine; other machines need a key and get only `/v1/` by default. See [Remote access](remote-access.md) |
| Error format | See [Error model](errors.md) |

## Groups

| Group | Paths | Notes |
|---|---|---|
| Service | `/health`, `/` | `/health` reports available engines and whether fine-tuning runs; it is **the only path open to the network** |
| Chat | `/v1/chat/completions`, `/v1/models` | OpenAI-compatible. See [Chat](chat.md) |
| Inference lifecycle | `/inference/load_model`, `unload_model`, `status`, `stop_generation`… | One model at a time |
| Sleep / wake | `/inference/sleep`, `/inference/wake` | Release the GPU without unloading (vLLM, Strata) |
| Disk KV tier | `/inference/kv_cache/disk`, `/cleanup` | vLLM's NVMe KV usage and cleanup |
| Memory estimation | `/inference/estimate_memory`, `/gguf`, `/gguf/{check,recommend,sweep,plan}` | See [Memory estimation](../memory/estimation.md) |
| Models | `/config/models`, `/download`, `/convert`… | Download, delete, convert to GGUF |
| Enums and examples | `/config/{quantization,offload}_types`, `/config/training_methods`, `/examples/*` | |
| Training | `/training/*` | Linux. See [Fine-tuning overview](../training/index.md) |
| Documents (RAG) | `/rag/docs`, `/rag/search` | |
| System | `/system/resources` | CPU, RAM, GPU, disk; `?mode=spec|usage` |

`POST /inference/chat` is deprecated and returns `410`; use `/v1/chat/completions`.

## Inference lifecycle

### One model at a time

| Situation | `POST /inference/load_model` returns |
|---|---|
| Same model, same key settings | `200 {"status": "already_loaded"}` |
| A different model or settings | **`409`**; unload first |
| Still loading | `409` |
| This machine cannot run that engine | `501` with the reason |

> **"Key settings" are only six fields** (`quantization`, `device_map`, `max_memory`,
> `offload_folder`, `torch_dtype`, `model_total_memory`). Reloading with only other fields changed
> returns `already_loaded` and **does not take effect**; unload, then load.

When the worker fails, status becomes `error` and **recovery is manual**: call `load_model` again,
or `POST /inference/force_cleanup_gpu`. `GET /inference/error_details` returns the error the worker
recorded (with `is_oom` and the traceback).

### Sleep / wake

Supported by `vllm` (load with `vllm_enable_sleep_mode`) and `strata` only; other engines return
`409`. The difference is the KV: vLLM drops it, Strata keeps the conversation.

| Request | Response |
|---|---|
| `POST /inference/sleep {"mode": "wait"}` | Sleeps after running generations finish |
| `POST /inference/sleep {"mode": "abort"}` | Aborts them, then sleeps |
| The engine fails | `502` |
| A generation request while asleep | Wakes automatically; with `auto_wake` off, `503` and `Retry-After: 1` |

**The backend never decides on its own when to sleep.** The caller coordinates (an agent handing
the GPU to image generation, for example). Wake can fail because another program took the GPU
memory (`502`); re-read `/inference/status`.

## Model downloads

| Behaviour | Notes |
|---|---|
| `POST /config/models/download` | Returns a `task_id` at once and downloads in the background. **A mistyped model id does not fail immediately**; it fails when the task reaches the Hub |
| A GGUF repo without `filename` | **Downloads every quantization** (can be many GB) |
| `POST /config/models/download/{task_id}/cancel` | The task goes to `cancelling`, then `cancelled` once it stops |
| Download tasks | Kept in memory only; **gone after a restart** |
| `DELETE /config/models/{label}` | Returns `202` and deletes in the background; the model is `deleting`, or `delete_failed` on failure; deleting the same one again is `409` |

> With `HF_XET_ENABLED`, a Xet transfer cannot be interrupted midway, so a cancel may wait for the current file to finish.

## RAG

`/rag/*` manages documents and search; `/v1/chat/completions` also accepts `use_rag`,
`rag_top_k`, `rag_query` and `rag_include_sources`. The web interface has no RAG screen at present.

## Related

- [Chat](chat.md)
- [Error model](errors.md)
- [Remote access and the trust boundary](remote-access.md)
- [Service architecture](../architecture.md)
