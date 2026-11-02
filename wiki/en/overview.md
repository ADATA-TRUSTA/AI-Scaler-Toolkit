---
title: Overview
summary: AI Scaler Toolkit is a local platform for running and fine-tuning LLMs on one machine; this page covers its components, the three layers of hardware fitting, and its most important limits.
updatedAt: 2026-10-29
---

# Overview

## In one sentence

**A platform for running LLMs on your own computer**: download models, chat, convert and
quantize, and fine-tune, all on local hardware. Your data does not leave the machine.

**The core of the product is "it runs on a GPU that is not big enough."** That goal shapes most
of the design: four inference engines, layered offload, GGUF quantization, DeepSpeed ZeRO-3.
The NVMe SSD is treated as **part of the memory hierarchy**, not only as a place to keep files.

## Components

| Component | What it is | Where in the repo |
|---|---|---|
| **Backend** | FastAPI service: 54 routes, four inference engines, fine-tuning, RAG | `backend/` |
| **Web interface** | A prebuilt single-page app, served by the backend | `frontend/` |
| **Console** | A headless Python client for the same API | `console/` |
| **Desktop app** | Launcher, Windows installer, Linux AppImage | `deploy/` |

```
┌─ Web interface ─┐   ┌─ console ─┐   ┌─ Other apps (Hermes, coding assistants…) ─┐
└────────┬────────┘   └─────┬─────┘   └──────────────────────┬────────────────────┘
         │ HTTP / SSE        │ HTTP                          │ OpenAI-compatible /v1/
         └───────────────────┴───────────────┬───────────────┘
                                             ▼
                               ┌─ Backend (FastAPI) ─┐
                               │ main process+workers │
                               └──────────┬──────────┘
                                          │ manages child processes
                  ┌───────────────────────┼────────────────────────┐
                  ▼                       ▼                        ▼
            llama-server              vllm serve                 Strata
              (GGUF)              (HF safetensors)           (GGUF, MoE)
```

The `transformers` engine is not in the picture: it runs **inside the backend's own worker
process**, not as a separate server. That difference is the subject of
[Service architecture](architecture.md).

## Three layers of hardware fitting

| Layer | Means | See |
|---|---|---|
| **Weights** | Quantization (int4/int8/GGUF), CPU offload, MoE expert offload | [Offload](memory/offload.md) |
| **KV cache** | vLLM's DRAM tier → NVMe tier; llama.cpp's slot cache | [Offload](memory/offload.md), [vLLM](engines/vllm.md) |
| **Training** | DeepSpeed ZeRO-3 with CPU/NVMe offload | [DeepSpeed ZeRO-3 offload](training/deepspeed-offload.md) |

## The most misunderstood limit: one at a time

| Resource | How many at once | Beyond that |
|---|---|---|
| Inference model | **1** | `409`; unload first |
| Training job | **1** | Refused |
| Generation requests | 8 (`MAX_CONCURRENT_GENERATIONS`) | Queued |
| **Training and inference on the same GPU** | **Not prevented** | Only a warning in the server log; on a 16 GB card this is usually an OOM |

Mind the last row: **unload the inference model before you start training.**

## What each platform can do

| Feature | Windows | Linux |
|---|---|---|
| transformers, llama.cpp | ✅ | ✅ |
| vLLM | ❌ | ✅ (installed by default with NVIDIA; Intel needs `TRUSTA_SETUP_VLLM=1`) |
| Strata | Depends on the package | Depends on the package; **NVIDIA only** |
| Fine-tuning | ❌ | ✅ (WSL2 counts as Linux) |

The backend reports the engines and fine-tuning this machine can actually run through
`GET /health`, and the web interface shows options accordingly. Calling an unsupported feature
returns `501` with a sentence explaining why.

## Where to go next

| You want to know | Go to |
|---|---|
| The backend's processes and state | [Service architecture](architecture.md) |
| Which inference engine to use | [Choosing an inference engine](engines/index.md) |
| How to build on the API | [API overview](api/index.md) |
| How to let another computer connect | [Remote access and the trust boundary](api/remote-access.md) |
| Terms | [Glossary](glossary.md) |
