---
title: About this wiki
summary: What this wiki covers and does not, how it is written, and which version it describes.
updatedAt: 2026-10-29
---

# About this wiki

## What it covers

**How AI Scaler Toolkit works and what is worth knowing in practice**: architecture, the trade-offs
between the four inference engines, memory offload, memory estimation, fine-tuning, the API's
behaviour in detail, the causes of common problems, and measured numbers with their conditions.

## What it does not cover

| Content | Where it lives |
|---|---|
| Install, update, uninstall | [`docs/installation.en.md`](../../docs/installation.en.md) |
| Using the app, connecting other apps, the list of API routes | [`docs/usage.en.md`](../../docs/usage.en.md) |
| Common user questions | [`docs/faq.en.md`](../../docs/faq.en.md) |
| Every setting with its default | [`backend/.env.example`](../../backend/.env.example) |

**Each thing is written in one place.** When this wiki touches the topics above, it links to
them instead of repeating them.

## How it is written

- **The code is the reference.** Every page was checked against the backend source; module paths
  mentioned on a page are under `backend/service/`.
- **Numbers come with their conditions.** Performance and capacity figures state the hardware,
  model and settings. A number without conditions is left out.
- **Measured and inferred are kept apart.** What was run says so; what was read from code is
  marked as such.
- **No prices.** Cost comparisons use ratios only.

## Version

| Component | Version |
|---|---|
| Backend | **0.3.0** (vLLM 0.28.0, transformers 5.15.0, DeepSpeed 0.19.2) |

Each page's `updatedAt` is the date it was last checked. If a newer version behaves differently
from what is written here, **the code wins**.

## Languages

`en/` and `zh/` have the same page tree, and the two versions of a page say the same thing.
