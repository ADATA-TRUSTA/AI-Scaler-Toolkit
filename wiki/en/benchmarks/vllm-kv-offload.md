---
title: vLLM native KV cache offload to DRAM/NVMe
summary: On a 16 GB card, Qwen3.5-35B at full context, 8 concurrent requests, and many users sharing one document all worked; revisits went from 445 s to 6 s and from 1,676 s to 37.6 s. When the DRAM tier cannot hold the prefix to bring back, the hit rate is 0% even though the data was read.
updatedAt: 2026-11-20
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

## Update: vLLM 0.32, 100 users, DRAM and SSD (2026-11)

The same question on a bigger setup: the GPU's KV cache is made deliberately small, and 100 people
take turns.

| Item | Value |
|---|---|
| Hardware | two GPUs with 48 GB each (tensor parallel 2), 20-core workstation CPU, 503 GiB DRAM; the KV tier on a Gen5 NVMe SSD, separate from the model disk |
| vLLM | 0.32.0, `vllm serve`; requests to `/v1/completions`, streaming; reuse read from `cached_tokens` |
| Model | Qwen3.5-35B-A3B GPTQ-8bit (41 GB), all weights on the GPUs |
| Settings | model length 262,144, `max_num_seqs` 128, **GPU KV limited to 816,244 tokens** (8 GiB; unrestricted it would be 2,139,300 and the working set would fit, so offload would not be used) |
| Workload | 100 users, each a different 16,000-token document, all sending at once with the whole history plus a new question, 3 rounds; a working set of about 1.6 M tokens, twice the GPU KV. `max_tokens` 16 |
| SSD runs | the page cache dropped every second, so reads really come from the disk |

Time for a round of 100 requests, and how much was reused:

| Setting | Round 1 | Round 2 | Round 3 | Round 2 hit | Users hit |
|---|---|---|---|---|---|
| No offload | 111.1 s | 111.2 s | 111.5 s | 0% | 0 |
| DRAM 8 GiB | 114.0 s | 113.8 s | 114.6 s | 0% | 0 |
| DRAM 32 GiB | 113.5 s | 113.5 s | 112.9 s | 0% | 0 (1 in round 3) |
| DRAM 8 GiB + SSD | 113.8 s | 70.1 s | 71.6 s | 40.5% | 41 |
| **DRAM 32 GiB + SSD** | 114.2 s | **8.1 s** | **5.7 s** | **98.7%** | **100** |

With DRAM 32 GiB + SSD the mean time to first token in round 2 was 4.5 s (p95 7.8 s), against 56.2 s
(p95 105.5 s) with no offload. One user's 200,000-token document, asked about again after the
100-user rounds had pushed it out of the GPU: 25.5 s without the SSD tier, **1.8 s** with it
(199,584 tokens reused).

- **All the saving comes from the SSD tier.** With DRAM only, even 32 GiB, the hit rate stays near 0:
  the working set is cycled through and the DRAM LRU, smaller than it, evicts it all in turn.
- **A DRAM tier that is too small cripples the SSD tier.** With 8 GiB, the SSD tier read 529 GB but
  only 30.6 GB reached the GPU; 100 simultaneous requests load blocks that are evicted before they
  are used. With 32 GiB it read 47.3 GB (wrote 52.7 GB) and all 100 hit.
- Cost: round 1 is 2–3% slower (it writes the tier); SSD reads peaked at about 3.1–3.6 GB/s.
- Without offload, the time is recomputing: about 8,400 tok/s for this 35B (a 200K document cold in 23.7 s).

Only 8 and 32 GiB were tried; that the DRAM tier must hold the blocks of the requests loading at once
is inferred from the counters, not from a sweep. A GGUF model could not be tried: vLLM 0.28 and 0.32
have no GGUF loader.

## Limits

- Rows two and three used `max_num_batched_tokens=2048` (a wrong script default), so **absolute times
  are about 4× slower than default settings**; the ratios still hold.
- Long sustained writes and a full disk were not tested; prompts were random tokens.

## Related

- [Offload](../memory/offload.md)
- [vLLM](../engines/vllm.md)
- [vLLM vs llama.cpp on an 8 GB VRAM budget](vllm-vs-llamacpp-8gb.md)
