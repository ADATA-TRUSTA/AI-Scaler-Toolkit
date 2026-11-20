---
title: Strata prompt reading with the CPU's AMX units
summary: With the CPU computing a share of the experts during prefill, a 4K prompt reads about 90–110% faster (792 → 1,507 tok/s on a 16 GB card, about 1,000 → 2,000 on a 48 GB card); at 32K there is no gain. With the engine's default settings AMX does nothing above about 3K tokens.
updatedAt: 2026-11-20
measuredAt: 2026-11-18 to 2026-11-19
---

# Strata prompt reading with the CPU's AMX units

Strata keeps a large MoE's experts in DRAM. Reading a long prompt (prefill) needs every expert, and the
experts that are not on the GPU either cross PCIe or are computed by the CPU. A CPU with AMX (Intel's
matrix units) can compute a share of them. This page measures what that is worth.

## Conditions

| Item | Machine A | Machine B |
|---|---|---|
| GPU | 16 GB VRAM, PCIe Gen5 x8 (27.8 GB/s host→device measured) | 48 GB VRAM, PCIe Gen4 x16 (25.0 GB/s measured) |
| CPU | server CPU with AMX | 20-core workstation CPU with AMX (`amx_int8`) |
| RAM | 251 GB | 503 GiB |
| OS | Ubuntu 24.04 | Ubuntu 24.04 |
| Engine | Strata 0.1.57 (v0.2.11 plus the strata-sleep patches), built locally for the card | the same |
| Model | Qwen3.8-Flash-Next **Q8_0** (176 GB, 6 shards) | the same |
| Settings | `--expert-cache auto --prefill auto --spec 4 --mtp`, 128K context, int8 KV | `--spec 4 --spec-min-p 0.5 --mtp`, 256K context, vision off unless stated |

One user at a time, `max_tokens` 1 (so the time is the prompt), a different first line in every
request so nothing comes from the prefix cache, one warm-up, then AMX off and on alternately.
Speed is the engine's own `read in … tok/s`.

## Machine A: the default limit hides AMX

| CPU share settings | Prompt | AMX off | AMX on |
|---|---|---|---|
| Defaults (upper limit 3,072 tokens) | about 4K | 786–807 tok/s | 795–807 tok/s: **AMX did not run** |
| Upper limit raised to 16,000 | about 4K | 788–797 (mean 792) | 1,404–1,587 (mean 1,507): **+91%** |
| same | about 8K | 1,474–1,494 (mean 1,485) | 1,761–1,879 (mean 1,820): **+23%** |

The CPU share only applies to chunks below an upper limit, 3,072 tokens by default; a 4K prompt
exceeds it and takes the all-streaming path where the CPU takes no part. The limit is the engine's
`STRATA_PREFILL_CPU_SHARE_MAX`, used with `STRATA_PREFILL_CPU_SHARE=auto`.

## Machine B: it depends on how many experts have a copy in RAM

Prompt reading speed, tok/s. "Budget" is `--resident-budget-gib`: how much of the experts' RAM copy
the engine keeps.

| Setting | 4K AMX off | 4K AMX on | 32K AMX off | 32K AMX on |
|---|---|---|---|---|
| Budget 80 GiB (part of the experts read from the model file) | 810–871 | 1,182–1,322 | 2,536 | 2,638 |
| **Budget 125 GiB** (96 GiB RAM copy, nothing read from file) | **985–1,013** | **1,815–2,156** | **3,642** | 3,616 |
| Budget 80, prefill chunk 8,192 instead of 32,768 | 858–873 | 1,331–1,350 | 1,717 | 2,070 |
| Budget 125, through AST with vision on | 1,006–1,031 | 1,862–2,205 | 3,595 | 3,519 |

- **At 4K the limit is PCIe.** With AMX off, waiting for expert copies took 58–66% of the GPU's
  timeline. With AMX on that fell to 0.2–0.5 s, and the CPU took about 64–83% of the experts.
- **At 32K it is not.** Copy waits were 12–15%; the GPU's own compute dominates, so AMX adds about
  4% with the smaller budget and nothing with the larger.
- **Keeping all the experts in RAM mattered more than AMX at 32K:** budget 125 made 32K reading 44%
  faster with AMX off, and 4K 17% faster. It cost about 16 GiB more RAM.
- **A smaller prefill chunk is worse:** it makes every chunk stream the experts again, so 32K
  dropped from 2,536 to 1,717 tok/s.
- **CPU worker threads matter little:** with 11 workers AMX reached about 97% of the speed with 18
  (the default), with 5 about 90%. Measured at 4K, budget 80.
- With AMX off and budget 80, the 48 GB card is barely faster than the 16 GB card at 4K (810–871
  against 786–807): its extra VRAM helps decode and cache hits more than prefill.

## Limits

1. One user, one model, one precision (Q8_0). Several users reading at the same time are measured in
   [Strata and vLLM on one 48 GB GPU](strata-vs-vllm.md).
2. Engines built on the test machines for their own GPU architecture, not the released package.
3. Output correctness was not checked: only the first token was requested. That AMX ran is inferred
   from the speed difference and the limit setting, not from the CPU's performance counters.
4. 16K prompts and above were measured only on machine B. Machine A's 8K is one comparison.
5. How a request or a setting turns AMX on through AST is not covered here.

## Related

- [Strata](../engines/strata.md)
- [Strata and vLLM on one 48 GB GPU](strata-vs-vllm.md)
