---
title: Accuracy of the GGUF memory estimate
summary: Against llama-fit-params on 6 architectures, weights and KV are within 1 MiB and never underestimated, and the compute buffer is always over by 0–26%; recommend's settings actually use 96.6–97.8% of the budget.
updatedAt: 2026-10-29
measuredAt: 2026-09-01
---

# Accuracy of the GGUF memory estimate

For how the estimator works see [Memory estimation](../memory/estimation.md). This page has only the check results.

## Against `llama-fit-params`

`llama-fit-params` is llama.cpp's own dry-run load, so it is the exact reference. Six architectures:
qwen2, qwen3, qwen35, qwen35moe, gemma4, gpt-oss (dense/MoE, tied/untied embeddings, SWA, hybrid
linear attention).

| Item | Error |
|---|---|
| GPU/host weights | ≤ 1 MiB (rounding only) |
| GPU/host KV cache and recurrent state | ≤ 1 MiB |
| GPU compute buffer | **+0% to +26%, always over** |

Weights and KV were **never underestimated**. The compute buffer is the only approximation; add
`verify: true` for an exact figure.

## How fully `recommend` fills the budget

Qwen2.5-0.5B, `n_ctx_max=262144`:

| Limit | Suggested arguments | llama.cpp actual | Use |
|---|---|---|---|
| 1000 MiB | `-ngl 25 -c 8960 -fa on` | 924 MiB | 97.3% |
| 1500 MiB | `-ngl 25 -c 47360 -fa on` | 1353 MiB | 96.6% |
| 3000 MiB | `-ngl 25 -c 158976 -fa on` | 2835 MiB | 97.8% |

## Measured behaviour that is easy to misread

| Behaviour | Measured |
|---|---|
| Without flash attention the compute buffer explodes | 32B, 32K ctx: 344 → 5,256 MiB |
| `-ncmoe` does not grow the compute buffer; lowering `-ngl` does | `-ngl 999` and `-ngl 999 -ncmoe 2` both 14 MiB; `-ngl 2` 18 MiB |
| Tied embeddings carry an extra `token_embd` | gemma-4-26B-A4B Q4_K_M: file 16,147 MiB, host 16,895 MiB at `-ngl 0`; at `-ngl 1` the GPU holds exactly 748 MiB more |
| Hybrid linear attention has KV on some layers only | Qwen3.5-9B, c=4096: KV 178 MiB (8 attention layers 128 + 24 linear 50), not 512 MiB |
| `-ngl 1` is not "the first layer" | It places only the output layer |

## Limits

The estimator follows llama.cpp's source; if llama.cpp changes placement or the compute-buffer
arithmetic, it needs checking again.

## Related

- [Memory estimation](../memory/estimation.md)
- [The llama.cpp engine](../engines/llama-server.md)
