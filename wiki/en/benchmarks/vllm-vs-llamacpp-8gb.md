---
title: vLLM vs llama.cpp on an 8 GB VRAM budget
summary: Qwen3.5-35B-A3B using about 8 GB of VRAM. Shared prefixes, long cold prompts and cache reuse after restart favour vLLM (four 100K prompts at once, 125 s vs 1,070 s); single-user decode, startup and reclaimable DRAM favour llama.cpp.
updatedAt: 2026-10-29
measuredAt: 2026-09-24
---

# vLLM vs llama.cpp on an 8 GB VRAM budget

Goal: keep the LLM to about 8 GB of VRAM and leave the rest for image generation (ComfyUI).
Both engines were measured through AST's `/v1/chat/completions`.

## Conditions

Hardware: RTX 5060 Ti 16 GB, 125 GB DRAM, Linux.

| Item | vLLM | llama.cpp |
|---|---|---|
| Version | 0.28.0 | b10107 |
| Weights | GPTQ-Int4 (attention kept bf16) | Q4_K_M GGUF |
| Experts | On the CPU | On the CPU |
| KV | fp8 | q8_0 (no fp8) |
| Context / concurrent requests | 131,072 · 4 | Same (`-np 4`, 32K per slot) |

## Results

| Item | vLLM | llama.cpp |
|---|---|---|
| Model load | 82 s | 7–12 s |
| Single-user decode (20K context) | 29 tok/s | **48 tok/s** |
| 100K cold TTFT | **104 s** | 145–259 s |
| The same 20K prefix after restart | **1.34 s** (hit from NVMe) | 26.0 s (fully recomputed) |
| Four 100K prompts sharing a prefix, started together | **125 s** (all four TTFTs 105 s) | 1,070 s (prefilled one after another) |
| GPU | 7.8 idle, 8.7 peak GB | 6.8–7.7 GB |
| DRAM | About 30 GB, **pinned** | 17–21 GB of mmapped weights, **reclaimable** |

- **Why decode differs by 1.7×**: llama.cpp computes the experts on the CPU (six P-cores at full
  load); vLLM ships the experts to the GPU over PCIe (Gen5 x8, about 22 GB/s measured) for every token.
- vLLM's three KV tiers: NVMe→DRAM about 7–8 GB/s, DRAM→GPU about 26 GB/s; moving the data takes
  under 0.2 s, and the rest of a hit is recomputing the unaligned tail (this hybrid model's prefix
  unit is 2,096 tokens).

## How to choose

| Situation | Choose |
|---|---|
| 1–2 agents, service always running | llama.cpp |
| Frequent restarts, many users starting at once, context often above 32K | vLLM |

With `-np 4`, llama.cpp gives each slot only 32K; one agent with a video-generation skill needs about 37K.

## Limits

Prompts were synthetic; each scenario ran once; the two engines used different quantization formats.

> The vLLM DRAM figure was first recorded as 47.6 GB; the `/dev/shm` KV tier had been counted twice
> (one process maps it twice). It is about 30 GB.

## Related

- [Choosing an inference engine](../engines/index.md)
- [vLLM native KV cache offload](vllm-kv-offload.md)
