---
title: Strata and vLLM on one 48 GB GPU, experts in DRAM
summary: Same prompts, same machine, a 173 GB MoE whose experts live in DRAM: four users reading 4K each finish in 23 s on Strata and 150 s on vLLM; four 32K prompts in 54 s and 504 s; a 4K follow-up with a KV cache in 2.3 s and 16–19 s.
updatedAt: 2026-11-20
measuredAt: 2026-11-19 to 2026-11-20
---

# Strata and vLLM on one 48 GB GPU, experts in DRAM

Scenario: a MoE far larger than the GPU, served to several people from one card, with the experts and
the PLE table (the large per-token embedding table) kept in DRAM.

## Conditions

| Item | Value |
|---|---|
| GPU | one with 48 GB of VRAM, PCIe Gen4 x16 |
| CPU, RAM | 20-core workstation CPU with AMX, 503 GiB (shared with another model loaded on a second GPU) |
| Model | Qwen3.8-Flash-Next. Strata: **Q8_0** GGUF. vLLM: the official **FP8** (173 GB). Both 8-bit, but not the same weights; vLLM 0.32 has no GGUF loader |
| Strata | `--resident-budget-gib 125`: 95 GiB RAM copy of the experts, PLE table locked in RAM, a cache of hot experts on the GPU (36.5 GiB); MTP on. "AMX" = the CPU computes a share of the experts, "upstream" = without |
| vLLM | 0.32.0; `--cpu-offload-gb 120 --cpu-offload-params experts` (110 GiB of experts in DRAM, read by the GPU through UVA); the PLE table in pinned host memory (`cpu_offload`); 11.9 GiB of weights and 29.5 GiB of KV on the GPU; vision off |
| Requests | `max_tokens` 256 for the multi-user runs, the same prompts for both. "Batch" = the four requests sent together; "queued" = one after another |

## One user

| | Strata (AMX off / on) | vLLM |
|---|---|---|
| 4K prompt, time to first token | 4.1 s / 2.0 s (1,012 / 2,148 tok/s) | 13.6 s (302 tok/s) |
| 8K prompt, time to first token | not measured | 24.0 s (315 tok/s) |
| Decode | not measured separately | about 9.5 tok/s (16 tokens) |

## Four users, 4K each (time to first token / time to finish, mean of 3 runs)

| | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| Strata, batch, AMX | 2.3 / 23.1 s | 6.3 / 23.1 s | 10.1 / 23.2 s | 13.9 / 23.1 s |
| Strata, batch, upstream | 4.1 / 26.0 s | 8.3 / 25.9 s | 12.5 / 26.1 s | 16.7 / 26.2 s |
| Strata, queued, AMX | 2.1 / 6.1 s | 8.6 / 12.3 s | 14.3 / 18.2 s | 20.2 / 24.0 s |
| vLLM | 16.7 / 149.1 s | 30.4 / 150.0 s | 44.0 / 150.6 s | 54.6 / 150.9 s |

## Four users, 32K each

| | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| Strata, batch (2 runs) | 9.3 / 21.9 s | 22.6 / 39.9 s | 35.9 / 49.1 s | 49.3 / 53.6 s |
| vLLM (1 run) | 107.7 / 491.9 s | 212.3 / 498.0 s | 316.7 / 502.2 s | 418.2 / 504.4 s |

AMX is not used at 32K (see [prompt reading with AMX](strata-amx-prefill.md)), so the two Strata
builds are the same here. vLLM reads a 32K prompt in about 105 s (about 310 tok/s).

## A conversation with a KV cache

First turn: a document. Then each turn adds new text. Time to first token; Strata: median of 3
conversations × 3 follow-ups; vLLM: 1 conversation × 3 follow-ups.

| History + new text | Strata, AMX | Strata, upstream | vLLM |
|---|---|---|---|
| First turn, 32K document | 9.2 s | 9.3 s | 104.1 s |
| First turn, 64K document | 17.0 s | 16.9 s | 213.0 s |
| 32K + 4K follow-up | 2.34 s | 4.12 s | 19.3 s |
| 64K + 4K follow-up | 2.33 s | 4.40 s | 16.5 s |
| 32K + 8K follow-up | 3.58 s | 4.43 s | 34.9 s |
| 64K + 8K follow-up | 3.51 s | 4.47 s | 31.3 s |

Both reuse the history: Strata's reused tokens equal the history every time; vLLM's cached tokens
are counted in blocks, so it re-reads a few hundred to a few thousand tokens more. With a cache the
gap is 5–9×; for the first turn, with no cache yet, 11–13×.

## Why (not profiled)

vLLM's offload makes the GPU read the experts from DRAM over PCIe on every forward pass and keeps no
expert cache on the GPU. Strata keeps 36.5 GiB of hot experts on the GPU and the CPU computes much of
the rest. This is inferred from the settings; neither engine was profiled for this comparison, and
vLLM's other offload settings (prefetching, other batch sizes) were not tried.

## Running it

vLLM was ended twice while loading, together with the desktop session it was started in, by
systemd-oomd (memory pressure above 50% for 20 s): it had been started inside the session's own
scope. Started with `systemd-run --user --scope`, with the other model unloaded first, it loaded in
165 s and ran normally.

## Limits

1. Different weights (FP8 against Q8_0); the outputs were not compared.
2. One vLLM configuration (4 sequences, vision off; 32K model length for the one-user runs, 98,304 for
   the long and multi-turn runs). Its 32K and multi-turn runs are a single run each.
3. RAM was shared with another loaded model; neither engine was measured alone.
4. Strata's decode speed was not measured separately, so its finishing times mix prefill and decode.

## Related

- [Prompt reading with AMX](strata-amx-prefill.md)
- [Strata](../engines/strata.md)
- [vLLM](../engines/vllm.md)
