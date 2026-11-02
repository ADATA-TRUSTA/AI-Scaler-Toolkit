---
title: vLLM native KV cache offload to DRAM/NVMe
summary: On a 16 GB card, Qwen3.5-35B at full context, 8 concurrent requests, and many users sharing one document all worked; revisits went from 445 s to 6 s and from 1,676 s to 37.6 s. When the DRAM tier cannot hold the prefix to bring back, the hit rate is 0% even though the data was read.
updatedAt: 2026-10-29
measuredAt: 2026-09-23
---

# vLLM native KV cache offload to DRAM/NVMe

## Conditions

| Item | Value |
|---|---|
| Hardware | RTX 5060 Ti 16 GB (PCIe Gen5 x8), 125 GB DRAM, ADATA T7P5 3.84 TB Gen5 NVMe (ext4), Linux |
| vLLM | 0.28.0, offline `LLM()` API (`vllm serve` not tested) |
| Main model | Qwen3.5-35B-A3B GPTQ-8bit (hybrid architecture), `--cpu-offload-gb 32 --cpu-offload-params experts --max-num-seqs 64`; only 6.4 GB of weights on the GPU |
| Second model | Gemma-4-E4B fp8 (SWA hybrid) |
| Measurement | Disk reads from device counters, not the page cache |

## Results

| Scenario | No offload | With offload | Notes |
|---|---|---|---|
| Two full 146K contexts evicting each other, then revisited | 444.9 s each | **6.04 s each** (8 GB DRAM + NVMe) | 3,049 MB read from NVMe |
| 8 concurrent × 18K, 16 rotating, second revisit | 1,676 s | **37.6 s (44.5×)** | 98.1% hit |
| 8 users asking about one 135K document, GPU cleared, asked again | TTFT 190–194 s | **6.9–13.8 s** (6 GB DRAM) | 99.3% hit |
| Same, with only 2 GB of DRAM tier | — | 191–196 s | **8,954 MB read, 0% hit** |
| NVMe I/O (20.6 MB blocks) | — | write 7.1, read 8.1 GB/s | |
| Expert offload: UVA vs Prefetch | — | prefill 831 vs 2,000 tok/s; decode **17.8 vs 0.76** tok/s | **Use UVA** |

## Conclusions

1. **The DRAM tier is the gate.** An NVMe block must be promoted into DRAM before it reaches the
   GPU, so the DRAM tier must hold the longest prefix you want to reuse. The fourth row shows what
   happens when it cannot: the data is read and none of it is used.
2. **Offload saves recomputation time, not concurrent capacity.** A request being computed still
   needs all of its KV on the GPU.
3. **Prefix caching must be on**, or offload silently does nothing.
4. **The NVMe tier never deletes on its own** and does not `fsync`. The backend offers a cap and a
   cleanup endpoint; see [vLLM](../engines/vllm.md).
5. **Reuse across restarts needs a fixed `PYTHONHASHSEED`**, which the backend sets when starting
   vLLM; after a full machine restart the measured hit rate was 99.3%.

## Limits

- Rows two and three used `max_num_batched_tokens=2048` (a wrong script default), so **absolute times
  are about 4× slower than default settings**; the ratios still hold.
- Long sustained writes and a full disk were not tested; prompts were random tokens.

## Related

- [Offload](../memory/offload.md)
- [vLLM](../engines/vllm.md)
- [vLLM vs llama.cpp on an 8 GB VRAM budget](vllm-vs-llamacpp-8gb.md)
