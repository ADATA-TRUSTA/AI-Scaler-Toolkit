# Changelog

Newest first.

## 2026-11-20 — Strata on Intel Arc, and new Strata and vLLM results

- [Strata](engines/strata.md): the installer carries the package and installs it on the first load; Intel Arc (Windows and Linux) has a package too, without deep sleep; the supported GGUF list, the first-load flow and the load-time environment defaults are written down.
- Added the measured result [Strata on Intel Arc](benchmarks/strata-intel-arc.md).
- Added [Strata prompt reading with the CPU's AMX units](benchmarks/strata-amx-prefill.md) and [Strata and vLLM on one 48 GB GPU](benchmarks/strata-vs-vllm.md).
- [vLLM native KV cache offload](benchmarks/vllm-kv-offload.md): a section for vLLM 0.32 with 100 users and an SSD tier.
- [Handing the GPU to another program](benchmarks/gpu-handoff.md): a section on Strata sleep with the backup on an SSD at 16–46 GB of GPU memory in use.
- The rest of the wiki is still written against backend 0.3.0.

## 2026-10-29 — Rebuilt

The whole wiki was rewritten against the backend 0.3.0 code.

- Added: service architecture, the four inference engines (including Strata), the four kinds of
  offload, memory estimation, GGUF conversion, fine-tuning, the API (chat, errors, remote access),
  operations, and six measured results with their conditions.
- The TRUSTA SSD product Q&A became one page, [TRUSTA enterprise SSDs](products/trusta-ssd.md).
- Removed: the old pages dated 2026-04 to 07 (earlier architecture, engine comparison, offload
  notes, raw manual imports, the LLM-Wiki method, the competitor comparison). Installation and usage
  now live in [`docs/`](../../docs/installation.en.md).
