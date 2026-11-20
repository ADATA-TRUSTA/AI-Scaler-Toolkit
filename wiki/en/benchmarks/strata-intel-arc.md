---
title: Strata on an Intel Arc integrated GPU
summary: A 58 GB Gyro-S model (Q1, TQ1_0) runs on a laptop with 30–32 GB of shared memory: about 11 tok/s decode and 175 tok/s prefill on Windows with MTP, 6–8 tok/s decode on Linux without it; conversations of 64K–256K tokens can be saved and restored after a restart.
updatedAt: 2026-11-20
measuredAt: 2026-11-03 to 2026-11-20
---

# Strata on an Intel Arc integrated GPU

Scenario: a thin laptop with an integrated Arc GPU and no VRAM of its own, running a model far larger
than its memory. The Intel package has **no deep sleep** (see [Strata](../engines/strata.md)), so what
is measured is speed, long conversations, and saving a conversation across a restart.

## Conditions

| Item | Value |
|---|---|
| Hardware | Laptop CPU with 16 cores (4 performance + 12 efficiency); Intel Arc integrated GPU (Xe3), which uses system memory and has no VRAM of its own |
| Memory | Linux: 30 GB shared. Windows: 32 GB |
| OS | Linux: Ubuntu 24.04. Windows: 11 |
| Model | Qwen3.8 Flash-Next **Gyro-S TQ1_0** (the lowest precision; 58.5 GB file, about 27.6 GB of weights) |
| Engine | Strata Intel (SYCL) package; the Linux measurements used the engine with the three fixes that now ship in the strata-sleep packages |
| Linux settings | 8K context unless stated, `--mmap-experts --resident-budget-gib 12`, MTP off unless stated |
| Windows settings | 32K context, MTP on (`--mtp`), experts streamed by the engine's own choice |

Each answer was also checked automatically (expected keywords, garbled text, repetition loops).

## Speed

**Windows** (streamed requests, each prompt different so nothing comes from the prefix cache; prefill
is prompt tokens divided by the time to the first output, decode is the rate after it):

| Prompt | Time to first output | Prefill | Decode (200 tokens) |
|---|---|---|---|
| About 560 tokens | 8.4–8.7 s | 64–66 tok/s | 11.0, 12.6 tok/s |
| About 2,800 tokens | 20.5–21.0 s | 133–136 tok/s | 11.9, 11.3 tok/s |
| About 8,700 tokens | 54.1–54.2 s | 160 tok/s | 11.6, 9.3 tok/s |

- **Steady prefill is about 175 tok/s**, from the difference between the 2,800- and 8,700-token runs.
  The short prompt looks slower because every request has a fixed 5–8 s before the first output.
- **Decode is about 11.3 tok/s** on average (9.3–12.6); the experts cache hit rate was 77–93%.

**Linux** (8K context, MTP off, after the fixes; short prompt about 25 tokens, long answer about
470 tokens, cold = first read, nothing cached):

| Setting | Short-prompt decode | Long-answer decode | Cold 1,903-token prompt |
|---|---|---|---|
| DRAM tier 12 GiB | 5.5–5.7 tok/s | 7.0 tok/s | 121 tok/s |
| DRAM tier 6 GiB, more experts computed on the GPU | 5.9–6.0 tok/s | 7.6 tok/s | 120 tok/s |
| With MTP | 4.8–5.2 tok/s | 6.2–7.4 tok/s | 122–123 tok/s |

- On an earlier run of the first row the same prompt read at 184 tok/s. The cause was not found.
- **MTP does not help here.** Each token picks 10 of 512 experts at every layer, and the tokens
  checked together mostly pick different ones, so the work the CPU has to do roughly triples. (Windows
  above was run with MTP on and is faster; the two are different machines, memory sizes and settings,
  so they are not a like-for-like comparison.)
- The bottleneck is the **SSD tier**: 30 GB cannot hold 23 GiB of experts, and the ones outside the
  DRAM tier are read from the GGUF each time.

## Long conversations, saved and restored (Linux, MTP on)

Two facts hidden at 30% and 70% of the context; after filling it: ask the first, save the
conversation, restart the whole server, restore, ask the second.

| Context | Prompt | Read rate | Saved | Restore after restart | The second fact |
|---|---|---|---|---|---|
| 64K | 58,338 tokens | 162 tok/s | 1.88 GB, 0.6 s | 1.7 s | Right, follow-up about 13 s |
| 128K | 120,778 tokens | 155 tok/s | 3.64 GB, 1.0 s | 3.1 s | Right, about 15 s |
| 256K | 245,658 tokens (33 min) | 124 tok/s | 7.16 GB, 1.6 s | 6.0 s | Right, about 13 s |

- After a restore the whole conversation is reused from the cache; only the new tokens are read.
- Restore reads the file into RAM and keeps some free: the DRAM tier had to be made smaller for it
  to fit at every length, and at 256K the free-memory floor lowered as well.
- Without restoring, re-reading the same conversation takes 360 s (64K), 783 s (128K) and 1,982 s (256K).
- Asked which day a given number first appeared in a list of thousands, the model answered wrongly at
  all three lengths. That is the model's limit, not the cache.

## Agent runs (Linux, 64K, MTP off)

| Run | Result |
|---|---|
| Three small coding tasks (write, fix a bug, add a feature), three times each, judged by the task's own tests | 9 of 9 passed; median 166 s; no engine restarts |
| One agent session of 15 turns building a small project, four sessions, with the server restarted after turn 8 | 58 of 60 turns passed; restore after restart 0.6–0.8 s each; 94% of tokens reused from the cache; no unplanned engine restarts in 17 hours |

The two failures were a check that expected an exact error string and a turn that ran past its
60-minute limit with the work already correct: neither was the engine.

## Limits

1. One model and one quantization, the lowest precision. Answer quality at Q1 is the model's.
2. Each number is one run unless a range is given; the Windows decode differs by about 5–8% from an
   older package in a single comparison, which may be noise.
3. Windows and Linux were different machines and settings; do not read one as the other's speedup.
4. Memory is shared with the desktop: on Linux, a graphics-memory page pool that fills up (to half
   of RAM) can leave the next load with a much smaller expert cache, and a reboot clears it.
5. The agent runs went directly to Strata, not through AST.

## Related

- [Strata](../engines/strata.md)
- [Three ways to hand off the GPU](gpu-handoff.md) — the NVIDIA deep-sleep numbers
