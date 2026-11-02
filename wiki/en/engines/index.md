---
title: Choosing an inference engine
summary: The trade-offs between transformers, llama.cpp, vLLM and Strata, and the behaviour they share.
updatedAt: 2026-10-29
---

# Choosing an inference engine

## Start with this table

| | `transformers` | `llama_server` (llama.cpp) | `vllm` | `strata` |
|---|---|---|---|---|
| Model format | HF safetensors | **GGUF** | HF safetensors | GGUF (large MoE) |
| Platform | All | All | **Linux** | **NVIDIA**, optional component |
| Runs as | Inside the backend's worker | A managed child process | A managed child process | A managed child process |
| Concurrency | Weak (one forward at a time) | `-np` slots | **Strong** (continuous batching) | Mostly single-user |
| Quantization | bitsandbytes int8/int4 | In the GGUF | AWQ / GPTQ / FP8 etc. | In the GGUF |
| Multimodal | ✅ | ✅ (needs mmproj) | ✅ | Depends on the model |
| Fine-tuning output | **Load the adapter directly** | Merge, then convert to GGUF | Merge first (or vLLM's LoRA) | Convert to GGUF |
| Release the GPU without unloading | ❌ | ❌ | ✅ sleep (**drops the KV**) | ✅ deep sleep (**keeps the conversation**) |
| Best for | Trying a model right after fine-tuning | One machine, tight memory, big models | Many users, long context, KV reuse | A large MoE for one user, sharing the GPU with other programs |

**In one sentence**: check a fresh fine-tune with `transformers`, serve with one of the other three.
Many concurrent users → `vllm`; one machine with tight memory → `llama_server`; a large MoE that
must give the GPU to other programs → `strata`.

For numbers, see [vLLM vs llama.cpp on an 8 GB budget](../benchmarks/vllm-vs-llamacpp-8gb.md)
and [three ways to hand off the GPU](../benchmarks/gpu-handoff.md).

## The most important structural fact

All four engine classes have `load_model`, `unload`, `apply_chat_template` and
`cleanup_generation_memory`, but **only transformers implements generation**. The other three
only start and stop their server and collect its metadata; generation goes straight from the main
process over async HTTP. See [Service architecture](../architecture.md).

## What all four share

**One model at a time.** Loading a different model or different settings returns `409`; unload first.

**"Unload" means different things.** For transformers it ends the worker process (that is how VRAM
comes back); for server engines it stops the child process. llama.cpp and Strata save the
conversation's KV at that point.

**The `quantization` field only applies to transformers.** GGUF quantization is in the file; vLLM
uses `vllm_quantization`.

**`/inference/status` fields differ by engine.** Callers must tolerate missing fields.

## Engine pages

- [transformers](transformers.md)
- [llama.cpp (llama_server)](llama-server.md)
- [vLLM](vllm.md)
- [Strata](strata.md)

## Related

- [Offload: four different things](../memory/offload.md)
- [Memory estimation](../memory/estimation.md)
- [Service architecture](../architecture.md)
