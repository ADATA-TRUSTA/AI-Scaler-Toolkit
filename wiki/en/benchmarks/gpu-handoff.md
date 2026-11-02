---
title: "Handing the GPU to another program: three ways"
summary: With a 255.5K-token conversation, Strata sleep/wake takes about 1.8 s from getting the GPU back to the first answer, Strata unload plus slot restore about 25.9 s, llama.cpp unload plus slot restore about 32.7 s. All three resume the conversation intact; the difference is that sleep still holds about 63 GB of RAM while the GPU is away.
updatedAt: 2026-10-29
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

## Stability

Loading 50 GB of expert weights, systemd-oomd (memory pressure 69.8% > 50% for 20 s) logged out the
desktop; with unbuffered loading and arena THP off, pressure dropped to 0.2% and loading sped up
from 6.40 to 9.1–10.0 GiB/s. This is now AST's default; see [Strata](../engines/strata.md).

## Related

- [Strata](../engines/strata.md)
- [The llama.cpp engine](../engines/llama-server.md)
- [vLLM sleep/wake](../engines/vllm.md)
