# Wiki Log (English)

> Chronological record of knowledge-base operations. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`

## [2026-04-29] ingest | Trusta AST inference service
- Ingested `inference_manual.html` into the wiki.
- Created source page `inference_manual.md`.
- Created entity pages for `trusta-ast-inference-service`, `model-manager`, and `worker-process`.
- Created concept pages: `multi-process-isolation`, `qwen3-5-inference-optimization`, `inference-engine-comparison`, `openai-api-compatibility`.
- Created comparison page `legacy-vs-openai-api`.

## [2026-05-28] ingest | LLM Wiki methodology + TRUSTA SSD product line
- Ingested LLM Wiki format-conversion research and the TRUSTA enterprise SSD Q&A dataset.
- Created `concepts/llm-wiki-pattern.md`, `concepts/file-format-conversion.md`, `research-evidence.md`.
- Created the `synthesis/*` TRUSTA SSD product-line pages.

## [2026-07-16] update | Correctness pass + bilingual restructure
- Removed fabricated cost/benchmark figures; competitor-comparison numbers dropped (not benchmarked in this repo).
- Replaced the non-existent `OffloadConfig` code sample with the real mechanism (device_map / offload_folder / n_gpu_layers / DeepSpeed ZeRO-3).
- Corrected "auto-restart / auto-recovery" (recovery is manual) and "multi-model / load-balancing" (single model, HTTP 409 on a second load).
- Fixed `F8` → `FP8`, `ModelManager` method signatures, and stale `.html` source references.
- Flattened the nested `wiki/` layer and split the whole wiki into `en/` and `zh/`.
