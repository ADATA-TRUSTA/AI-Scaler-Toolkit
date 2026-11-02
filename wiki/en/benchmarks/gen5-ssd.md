---
title: What Gen5 NVMe does for local AI workloads
summary: Two measurements. Swapping only the system drive (Gen3 → Gen5) and rerunning image and video generation changed times by −43% to +2%, depending on how much time was spent waiting for the disk; and on Linux, pre-warming the page cache doubled a MoE model's decode speed.
updatedAt: 2026-10-29
measuredAt: 2026-09-21
---

# What Gen5 NVMe does for local AI workloads

## 1. Swapping only the system drive: Gen3 → Gen5

### Conditions

| Item | Value |
|---|---|
| Hardware | RTX 5080 Laptop 16 GB, 32 GB RAM, Windows |
| System drive | PCIe Gen3 x4 → **ADATA T7P5 7.68 TB Gen5 x4** |
| Method | Whole-disk clone: OS, drivers, Python environment and models bit-identical; same seeds, prompts and start images |
| Check | All 24 images had the same sha256 as the Gen3 run — the drive changed only the speed |

### Results

| Workload | Gen3 | Gen5 | Change |
|---|---|---|---|
| Z-Image (8 steps), warm | 10.6 s | 6.1 s | **−42.5%** |
| FLUX.1-schnell (4 steps), warm | 48.5 s | 41.1 s | −15.3% |
| RealVisXL, warm (no disk; control) | 8.6 s | 8.5 s | −1.2% |
| Qwen-Image (20 steps), warm | 154.1 s | 157.6 s | +2.3% |
| Wan 14B Lightning, agent polling every 60 s | 460.0 s | 358.1 s | −22.2% |
| Wan 14B, no polling (`--wait`) | 325.8 s | 314.9 s | −3.3% |
| Wan 14B, full steps | 1,342.0 s | 1,302.7 s | −2.9% |

- **The gain depends on compute density**: few steps, time spent streaming weights and paging →
  Gen5 helps; compute-bound work barely notices.
- **32 GB of RAM was the real bottleneck**: each image moves 12–31 GB and each video 116 GB, almost
  all of it paging. Polling every 60 s swaps the LLM out and back; **the old drive without polling
  (325.8 s) beat the new drive with polling (358.1 s).**
- The drive itself: 10.98 GB/s peak multi-threaded, but 2.96 GB/s for single-threaded 1 MiB
  sequential reads — the same as Gen3. The real workload peaked at about 4.6 GB/s; the limit was the
  access pattern (single thread, shallow queue), not the interface.

### Recommendations

1. Have the agent wait for results instead of polling — free, and worth more than the drive.
2. Spend the next hardware budget on **RAM**.
3. Load the LLM with mmap, so swapping it out drops clean pages instead of writing the pagefile.

## 2. Pre-warming the page cache: cold reads on Linux

### Conditions

Intel Core Ultra 9 275HX, 30.7 GB RAM, RTX 5080 Laptop, Linux on bare metal, ADATA T7P5 Gen5 x4 (ext4).
Reading a 20.5 GB Qwen3.5-35B-A3B GGUF; every "cold" figure verified with `mincore()`, three repeats.

### Results

| Item | Value |
|---|---|
| Device limit (O_DIRECT, QD32, 128 KB scattered reads) | 8.41 GB/s |
| Best pre-warm: 4-thread `read()` | **5.95 GB/s** (93% of the 6.37 GB/s software-path ceiling) |
| Same method on Windows | 5.36 GB/s |
| `mmap(MAP_POPULATE)` | 3.88 GB/s |
| `readahead(2)`, `fadvise(WILLNEED)` | The kernel honoured 0.8% / 0% |

Against a running llama-server: real memory pressure pushed the GGUF out of the page cache
(simulating image generation taking the RAM and giving it back).

| Stage | Not pre-warmed | Pre-warmed | Ratio |
|---|---|---|---|
| Prefill (32 tok) | 1,980 ms | 651 ms | **3.0×** |
| Decode (64 tok) | 21.7 tok/s | 44.4 tok/s | **2.0×** |

Without pre-warming the device ran at only 814 MB/s: MoE inference pulls in experts scattered by
routing, which is bound by IOPS, not bandwidth.

### Two things about method

- An early run on ZFS showed 22 GB/s; it was the ARC cache. **Always verify cold reads.**
- `posix_fadvise(DONTNEED)` cannot evict pages a running process has mmapped; only real memory pressure can.

## Related

- [Offload](../memory/offload.md)
- [Handing the GPU to another program](gpu-handoff.md)
- [TRUSTA enterprise SSDs](../products/trusta-ssd.md)
