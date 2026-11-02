---
title: Chat (OpenAI-compatible)
summary: Fields /v1/chat/completions adds to the OpenAI protocol, how thinking and reasoning_effort are handled, the shape of tool calls and reasoning_content, and the Qwen3.5 sampling overrides.
updatedAt: 2026-10-29
---

# Chat (OpenAI-compatible)

`POST /v1/chat/completions`, with streaming (SSE) and tool calls. Any OpenAI SDK works:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1` (desktop app) |
| Model | `trusta-ast-default`, or omit it — the server serves whatever is loaded |
| API key | Any string locally; from another computer, the key shown in Settings |

## Fields beyond OpenAI's

| Field | Purpose |
|---|---|
| `top_k`, `repetition_penalty` | Sampling |
| `enable_thinking` | Thinking switch (below) |
| `chat_template_kwargs` | Passed straight to the chat template |
| `session_id`, `reset_history` | Server-side chat history (Redis or memory) |
| `total_timeout` | Timeout for this request (seconds) |
| `use_rag`, `rag_top_k`, `rag_query`, `rag_include_sources` | RAG |
| `request_id` | The caller's own tracking id |

With the OpenAI SDK, put these in `extra_body`.

## Two things to know about responses

| What | Notes |
|---|---|
| **Thinking is in `reasoning_content`** | Not `<think>` tags inside `content`. Only server engines (llama.cpp, vLLM, Strata) produce it; the transformers path does not |
| **Streamed tool calls always carry `index`** | Required by the OpenAI protocol; without it some SDKs drop the whole tool call |

With `stream_options: {"include_usage": true}`, one extra chunk with token counts is sent before the stream ends.

## Thinking: three sources, in order

```
chat_template_kwargs["enable_thinking"]  →  enable_thinking  →  reasoning_effort
```

### `reasoning_effort`

A standard OpenAI field; the backend does two things with it.

**1. Without an explicit switch, it is folded into `enable_thinking`.** Many templates understand only a boolean.

| `reasoning_effort` | Becomes |
|---|---|
| `none`, `minimal` | `enable_thinking = false` |
| Anything else (`low`, `medium`, `high`, `max`…) | `enable_thinking = true` |
| Absent | The template is left alone |

"Off" is the allow-list and everything else is "on", because the vocabulary of levels keeps
growing; listing "on" instead would silently turn any new level off.

**2. It is forwarded to the engine, limited to `low` / `medium` / `high`.**

| Received | Forwarded |
|---|---|
| `none`, `minimal` | **Not sent** (thinking is turned off with `enable_thinking=false` instead) |
| `low`, `medium`, `high` | As is |
| `max` or higher | `high` |

Why: vLLM renders gpt-oss accepting only those three and errors on anything else, and a model that
cannot stop thinking rejects `none`. The cost: a model with a level above `high` is held at `high`.

### vLLM with a reasoning parser defaults to off

See [vLLM](../engines/vllm.md).

## Qwen3.5-family sampling overrides

The backend recognises the Qwen3.5 family by model name and architecture and **overrides what the
caller sent**:

| Override | When | Why |
|---|---|---|
| `repetition_penalty > 1.0` → `1.0` (with a warning) | **Always**, tool requests included | Qwen3.5 (especially MoE) produces mixed-language garbage above 1.0 |
| `temperature=1.0`, `top_p=0.95`, `top_k=20` | Ordinary requests | The official generation_config |
| No sampling override | **Requests with `tools`** | A tool call is structured output; drifting from the caller's sampling only costs reliability |

The last row was measured: the same request and tools on Qwen3.5-35B-A3B produced 153 tool-call
deltas from llama-server directly and 0 with the sampling overridden (it answered with a markdown
code block instead).

For a tool request without `enable_thinking`: **thinking is on only when `max_tokens` is large
enough (or absent)**. Thinking is why tool calls happen at all, but it also burns the budget — a
request with `max_tokens=512` can spend all of it inside `<think>` and never answer.

Qwen3.5's thinking is triggered by the template prefilling `<think>`; it responds only to
`chat_template_kwargs` and has no `/think` soft switch.

## Sending images: JSON or multipart

The standard way is image data URLs in `messages[].content`. The backend also accepts
**`multipart/form-data`** for callers that do not want to base64-encode:

| Form field | Content |
|---|---|
| `payload` (or `request`) | A JSON string: the usual request body |
| `image`, `images`, `file`, `files`, `attachment`, `attachments` | Files; attached to the last user message |

Form values are all strings, so the backend converts by field (`max_tokens`, `top_k` are integers;
`temperature`, `top_p` are floats).

## When the model is asleep

| Situation | Response |
|---|---|
| Put to sleep through the backend, `auto_wake` on (default) | Wakes, then generates |
| Put to sleep through the backend, `auto_wake` off | `503` + `Retry-After: 1` |
| Another program called vLLM's `/sleep` directly | Fails at once saying the model is asleep, instead of waiting for a timeout |

## Privacy

At INFO level the service logs only the **shape** of a request (model, message count, streaming,
`max_tokens`), not the conversation; full content is logged only at DEBUG.

## Related

- [Error model](errors.md) — errors mid-stream
- [The llama.cpp engine](../engines/llama-server.md) — `--jinja` and `reasoning_content`
- [vLLM](../engines/vllm.md) — tool-parser detection
