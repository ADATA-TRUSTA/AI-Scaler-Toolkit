---
title: Error model
summary: What status codes mean in this service, the three shapes of detail, how errors are sent once a stream has started, and how OOM and vLLM startup failures are classified.
updatedAt: 2026-10-29
---

# Error model

## `detail` comes in three shapes

**Callers must handle all three**, or error messages get lost.

```jsonc
// string (most common)
{"detail": "Model with label 'foo' not found"}

// list[dict]: 422 validation errors; the sentence is in msg, the field in loc
{"detail": [{"loc": ["body", "vllm_max_model_len"],
             "msg": "Input should be a valid integer, got a number with a fractional part",
             "type": "int_parsing"}]}

// dict: 507 OOM, 410 deprecated, 400 GGUF check
{"detail": {"error": "Out of Memory (OOM)", "message": "...", "is_oom": true,
            "suggestions": ["Use POST /inference/force_cleanup_gpu ...", "..."]}}
```

> A `422` `msg` often starts with pydantic's `"Value error, "`; cross-field rules (for example
> "this offload setting would do nothing") have a `loc` that points at no field.

## What status codes mean

| Code | Meaning |
|---|---|
| `400` | Invalid parameters, or a domain rule fails (GGUF estimation on a non-GGUF) |
| `404` | The named resource does not exist (task, job, model, file) |
| **`409`** | **The current state does not allow it**: a model is loaded or loading, training is running, a repeated delete, sleep/wake in the wrong state, an engine without sleep |
| `410` | `POST /inference/chat` is deprecated |
| `422` | Request validation failed |
| `500` | Unexpected exception |
| **`501`** | **This machine cannot do it**: fine-tuning off Linux, vLLM not installed |
| `502` | The engine failed during sleep/wake |
| `503` | The model is asleep and auto-wake is off; with `Retry-After: 1` |
| **`507`** | **Out of memory**; `detail` carries `is_oom` and `suggestions` |

Three deliberate choices:

- **`409`, not `400`**: the request is fine; sent again later, it succeeds.
- **`501`, not `403`**: an unsupported platform is not a permission problem. `detail` is a sentence
  you can show users as is.
- **`507`, not `500`**: OOM is a lack of resources, and there is advice to give.

> Most `500` `detail`s are the text of an underlying exception and **are not necessarily fit to show users**.

### Codes from the access layer

Remote-access checks run before any route, and their `detail` is always a string:

| Code | When |
|---|---|
| `401` | Another machine sent no key or a wrong key |
| `403` | A page from another site, a `Host` that is neither localhost nor an IP, a path outside the allow-list, or remote access not enabled |
| `429` | Too many wrong keys from one address; locked out |

Each message is explained in [Troubleshooting by symptom](../operations/troubleshooting.md#another-machine-cannot-connect).

## Errors after a stream has started

Once a stream starts, the HTTP status is already 200. Later errors are sent as an OpenAI-style error
object in the event stream, and the stream ends:

```
data: {"error": {"message": "CUDA out of memory...", "type": "server_error"}}
```

| Path | Where the error is |
|---|---|
| Ordinary requests (including non-streaming chat) | HTTP status + `detail` |
| Mid-stream chat | The SSE event's `error.message` |

A client disconnect and a generation failure are kept apart on purpose: only connection-type
exceptions count as "the client went away"; everything else becomes a structured SSE error —
otherwise a failed generation would look like a connection dropping for no reason.

## Where OOM shows up

| Source | Shape |
|---|---|
| Non-streaming chat | `507` + `detail.is_oom: true` + `suggestions` |
| `GET /inference/error_details` | `{"has_error", "is_oom", "error_traceback", "process_alive"}` |
| vLLM startup failure | Classified as `oom` |

## vLLM startup failures in eleven classes

```
oom · config · port_busy · model_not_found · cuda_mismatch · gpu_fault
shared_library_missing · toolchain_missing · chat_template · quantization · unknown
```

**Look at the class first; do not assume OOM.** See [vLLM](../engines/vllm.md).

## Related

- [Chat](chat.md)
- [Remote access and the trust boundary](remote-access.md)
- [Troubleshooting by symptom](../operations/troubleshooting.md)
