# AI Scaler Toolkit Wiki (English)

> 繁體中文: [`../zh/index.md`](../zh/index.md)
>
> Written against **backend 0.3.0**. For installation and day-to-day use, see
> [`docs/`](../../docs/installation.en.md). This wiki covers why it behaves the way it does,
> how to tune it, and how to build on its API.

## Where to start

| You want to know | Read |
|---|---|
| What AST is and what it is made of | [Overview](overview.md) |
| Which inference engine to use for your model | [Choosing an inference engine](engines/index.md) |
| How to run a model larger than your GPU | [Offload: four different things](memory/offload.md) |
| Whether a configuration will fit | [Memory estimation](memory/estimation.md) |
| Something went wrong | [Troubleshooting by symptom](operations/troubleshooting.md) |
| How fast it actually runs | [Measured results](benchmarks/index.md) |

## Overview

- [Overview](overview.md) — a local LLM platform on one machine; four components, three layers of hardware fitting, one model at a time
- [Service architecture](architecture.md) — three processes, two inference paths, where state lives
- [Glossary](glossary.md) — terms that mean something specific here, or that mean different things in different places

## Inference engines

- [Choosing an inference engine](engines/index.md) — the four engines' trade-offs at a glance
- [transformers](engines/transformers.md) — load a fresh adapter directly; `device_map` caveats
- [llama.cpp (llama_server)](engines/llama-server.md) — GGUF, slot cache, mmproj, `--jinja`
- [vLLM](engines/vllm.md) — concurrency, KV offload, sleep/wake, automatic tool-parser detection
- [Strata](engines/strata.md) — experts of a large MoE in DRAM; deep sleep keeps the conversation (NVIDIA; Intel Arc without deep sleep)

## Memory and model formats

- [Offload: four different things](memory/offload.md) — weights, MoE experts, KV cache, training state
- [Memory estimation](memory/estimation.md) — two estimators for HF and GGUF models, five GGUF endpoints
- [GGUF conversion and LoRA merge](models/gguf-conversion.md) — shard-by-shard merge, quantization, conversion pitfalls

## Fine-tuning (Linux)

- [Fine-tuning overview](training/index.md) — flow, phases, the event stream and SSE, two ordering invariants
- [Multimodal fine-tuning](training/multimodal.md) — `multimodal_scope` and four model-specific pitfalls
- [DeepSpeed ZeRO-3 offload](training/deepspeed-offload.md) — five profiles, NVMe and pinned DRAM, version pins

## API

- [API overview](api/index.md) — route groups, one model at a time, sleep/wake
- [Chat (OpenAI-compatible)](api/chat.md) — extension fields, thinking and `reasoning_effort`, tool calls, model-specific overrides
- [Error model](api/errors.md) — what status codes mean here, the three shapes of `detail`, errors mid-stream
- [Remote access and the trust boundary](api/remote-access.md) — separate rules for this machine, other machines, and web pages

## Operations

- [Troubleshooting by symptom](operations/troubleshooting.md) — see X, check Y
- [Pitfalls found by running it](operations/known-pitfalls.md) — things the code does not tell you
- [Behind a reverse proxy](operations/reverse-proxy.md) — SSE buffering, timeouts, `X-Forwarded-For`

## Measured results

- [Index of measured results](benchmarks/index.md) — each with its hardware and conditions
  - [vLLM native KV cache offload to DRAM/NVMe](benchmarks/vllm-kv-offload.md)
  - [vLLM vs llama.cpp on an 8 GB VRAM budget](benchmarks/vllm-vs-llamacpp-8gb.md)
  - [Accuracy of the GGUF memory estimate](benchmarks/gguf-estimation-accuracy.md)
  - [What Gen5 NVMe does for local AI workloads](benchmarks/gen5-ssd.md)
  - [Handing the GPU to another program: three ways](benchmarks/gpu-handoff.md)
  - [LoRA training a 35B MoE on a 16 GB GPU](benchmarks/deepspeed-lora-16gb.md)

## Products

- [TRUSTA enterprise SSDs](products/trusta-ssd.md) — product lines and use cases

## About this wiki

- [About this wiki](about.md) — scope, conventions, version
- [Changelog](log.md)
