---
title: "Handing the GPU to another program: three ways"
summary: With a 255.5K-token conversation, Strata sleep/wake takes about 1.8 s from getting the GPU back to the first answer, Strata unload plus slot restore about 25.9 s, llama.cpp unload plus slot restore about 32.7 s. All three resume the conversation intact; the difference is that sleep still holds about 63 GB of RAM while the GPU is away.
updatedAt: 2026-11-20
measuredAt: 2026-10-20
---

# Handing the GPU to another program: three ways

Scenario: one GPU taking turns between an LLM and image generation (ComfyUI, for example), with a
long LLM conversation you do not want to recompute.

## Conditions

| Item | Value |
|---|---|
| Hardware | RTX 5060 Ti 16 GB, 125 GB DRAM, NVMe SSD, Linux |
| Model | A community IQ3_XXS quantization of a Qwen3.8-Flash-Next derivative (512 experts, 49.8 GiB of expert weights) + mmproj F16 |
| Context | 262,144, 8-bit KV; a 1024×1024 image at the start of the conversation, 255.5K tokens in total |
| Strata | Deep-sleep build, full GPU, MTP on, unbuffered load |
| llama.cpp | Build 11348, `-ncmoe 48`, no MTP |

This round did **not** actually run image generation between giving up and taking back the GPU.

## Handoff results

| Stage | Strata sleep/wake | Strata unload + slot restore | llama.cpp unload + slot restore |
|---|---|---|---|
| Give up the GPU | 1.96 s | 4.3 s | 4.7 s |
| Take it back | wake 0.71 s | load 22.1 s + restore 3.48 s | load 30.4 s + restore 0.93 s |
| **Back to first answer** | **about 1.8 s** | about 25.9 s | about 32.7 s |
| RAM held by the LLM meanwhile | 63.1 GB (incl. a 5.0 GB pinned backup) | 0.25 GB | 0 |
| SSD | Not needed | 3.9 GB slot file | 4.1 GB slot file |

**How to choose**: with plenty of DRAM (80 GB or more suggested), sleep/wake; with 64 GB or less,
unload plus slot restore.

## Inference while awake

| | Strata | llama.cpp |
|---|---|---|
| Reading all 255.5K (summed in segments) | 177.8 s | about 589 s |
| Decode at 255K | 58.1 tok/s | 11.7–11.9 tok/s |

At 256K, llama.cpp's KV takes the GPU and only about one layer of experts still fits.

> llama.cpp's slot restore has a limitation: after restoring you must continue the same
> conversation. Re-asking the same question needs recurrent state rewound, which a slot file has no
> checkpoint for, so the whole prefix is recomputed (about 10 minutes).

## Other checks

- After all three handoffs, questions about the image were answered correctly — the image at the
  very start of the 255K conversation was still remembered. A functional check, not a vision-quality
  benchmark.
- OCR and region location: clean, small, photographed and tabular text almost all correct, mean IoU
  about 0.9; with `min_tokens 1024`, 8 px text was all correct, at 0.4–0.7 s extra per image.
- On the image-generation side: for a video of the same length, dropping available VRAM from about
  15.5 GB to about 8 GB was only 3–6% slower; VRAM decides how long a clip can be (720p single
  segment 81 frames → 65), not how fast it renders.

## Strata sleep with the backup on an SSD (2026-11)

The results above are for a 16 GB card. What happens when the GPU memory in use is 16 to 46 GB, and
the sleep backup is written to an SSD (`--sleep-backup file`) and read back on wake?

| Item | Value |
|---|---|
| Hardware | one GPU with 48 GB, 20-core workstation CPU, 503 GiB DRAM; the backup on a PCIe 3.0 x4 NVMe SSD, the model on a PCIe 5.0 x4 one; Ubuntu |
| Engine | Strata 0.1.52 (v0.2.7 plus the sleep patches); started directly, not through AST |
| Model | Qwen3.8-Flash-Next Q8_0, MTP on, 256K context, int8 KV, vision off |
| Method | Used GPU memory set with `--vram-reserve-mib`; each start: sleep and wake straight away ("hot"), then twice sleep, force the backup to disk and out of RAM, wake ("ssd"). After every wake the same conversation went on: 30 of 30 answers right |

**What the default backup contains.** The expert cache is **not** in it: it is dropped at sleep and
refilled at wake from the experts' RAM copy or the model file. Everything else of 1 MiB or more is
(KV, prompt cache, MTP, buffers, the non-expert weights). Of the 45.8 GB card, 36.75 GiB was expert
cache and the backup was 7.07 GiB. `STRATA_SLEEP_KEEP_CACHE=1` backs up the experts too.

**Default (expert cache refilled from the model file):**

| GPU memory in use | Backup | Experts refilled | Sleep + flush | Wake | of which refill |
|---|---|---|---|---|---|
| 16 GB | 7.07 GiB | 7.6 GiB | 7.3 s | 6.1 s | 2.8 s |
| 32 GB | 7.07 GiB | 23.2 GiB | 7.3–7.4 s | 11.8–12.0 s | 8.5 s |
| 46 GB | 7.07 GiB | 36.7 GiB | 7.1–8.0 s | 16.6–17.1 s | 13.3–13.6 s |

**With the experts backed up too (`STRATA_SLEEP_KEEP_CACHE=1`), backup on the Gen3 disk:**

| GPU memory in use | Backup | Sleep (HTTP) | + flush | Wake (from SSD) | Wake (pages still in RAM) |
|---|---|---|---|---|---|
| 16.0 GB | 14.7 GiB | 10.1 s | 3.6–4.8 s | 5.9 s | 2.9 s |
| 24.0 GB | 22.5 GiB | 15.2–16.4 s | 5.7–7.4 s | 8.4–8.5 s | 4.3 s |
| 32.0 GB | 30.3 GiB | 20.6–21.0 s | 8.0–8.8 s | 10.8–11.2 s | 6.1 s |
| 40.0 GB | 38.1 GiB | 27.3–27.6 s | 8.4–8.6 s | 13.4 s | 12.3 s (one run, cause not found) |
| 45.8 GB | 43.8 GiB | 34.0–34.7 s | 4.9–8.6 s | 15.3–15.9 s | 8.3 s |

- **Wake grows about linearly with the memory in use, 0.34–0.36 s per GiB, in both modes.** A 16 GB
  card wakes in about 6 s, a full 48 GB card in about 15.5–17 s.
- **The modes differ at sleep.** By default sleep writes only 7 GiB, in about 7 s. Backing up the
  experts writes the whole card: about 40 s at 46 GB. Copying VRAM into the mapped file runs at only
  about 1.3 GB/s.
- **A faster disk helps little.** On a PCIe 5.0 disk (16 GB case) the flush took 1.6 s against 3.6–4.8 s,
  but the wake only 11% faster (5.24 s against 5.9 s): reading back runs at about 3 GB/s whichever
  disk, so the read-back path is the limit, not the disk.

Not covered: through AST, with vision on, with the model file still in the page cache
(`STRATA_UNBUFFERED_LOAD=0`), or with the v0.2.11 engine (the sleep code is the same).

## Stability

Loading 50 GB of expert weights, systemd-oomd (memory pressure 69.8% > 50% for 20 s) logged out the
desktop; with unbuffered loading and arena THP off, pressure dropped to 0.2% and loading sped up
from 6.40 to 9.1–10.0 GiB/s. This is now AST's default; see [Strata](../engines/strata.md).

## Related

- [Strata](../engines/strata.md)
- [The llama.cpp engine](../engines/llama-server.md)
- [vLLM sleep/wake](../engines/vllm.md)
