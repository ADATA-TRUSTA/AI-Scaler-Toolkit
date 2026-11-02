---
title: Multimodal fine-tuning
summary: multimodal_scope for image fine-tuning, the load ordering that must hold, and four model-specific pitfalls for Qwen3.5, gemma-4 and others.
updatedAt: 2026-10-29
---

# Multimodal fine-tuning

For the overall flow see [Fine-tuning overview](index.md); this page covers only what is specific
to image training. Model families exercised include SmolVLM, Gemma 4 and Qwen3.5.

## `multimodal_scope`: how much training may change

| Value | Vision encoder | Bridge (projector) | Language model |
|---|---|---|---|
| `text_only` | Frozen | Frozen | Trained |
| `text_and_bridge` | Frozen | **Trained** | Trained |

`text_only` keeps image parsing exactly as pretrained and tunes only "how those features are read".
`text_and_bridge` is for images far from the pretraining distribution (industrial imagery, document scans).

> **Try `text_only` first.** In a face-identity experiment, the encoder already carried an identity
> signal; with negative examples added, held-out photos were answered 3/3 (including unseen photos),
> and `text_and_bridge` did not help in that case — the bridge was already full rank.
> "Train the bridge when the domain is far" is a hypothesis, not a rule.

## Use LoRA

Full fine-tuning a ~5B multimodal model on a 16 GB GPU **runs out of memory in backward**.
Enough DRAM does not help when VRAM is the limit; **CPU offload cannot save it**. Use LoRA for multimodal.

## Load ordering

Image training mode must be enabled **before** the model loads, or it is loaded with
`AutoModelForCausalLM`. Architectures such as Qwen3.5 and Gemma 3/4 **also register a text-only
mapping**, so the load does not fail — you just get a model **without its vision tower**. The
shared setup sequence guarantees the order.

## Four model-specific pitfalls (all handled by the backend)

### 1. Qwen3.5's weight rename under ZeRO-3

Some multimodal architectures (Qwen3.5) register a checkpoint rename meant for their text-only
sub-model, and under ZeRO-3 transformers applies it to the whole multimodal model unconditionally,
so **almost none of the language weights bind**. The backend adds a key mapping when it detects
this; on upstream versions with the fix the workaround does not trigger.

### 2. TRL's `mm_token_type_ids` has the wrong length

With prompt/completion data, TRL computes `mm_token_type_ids` from the prompt only; completion
tokens are appended to `input_ids` but not to it, which breaks Qwen3.5-VL's rope-index computation.
The backend recomputes it from the full `input_ids`.

### 3. ZeRO-3 skips some buffers (gemma-4)

transformers' ZeRO-3 loader skips buffers on modules that have no parameters of their own. gemma-4's
42 `layer_scalar` buffers therefore stayed at 1.0 for the whole run (true values 0.06–0.89), and
**the trained adapter could not reproduce the state it was trained in**. Fixed; when no key matches,
the backend warns instead of reporting all clear.

### 4. Produce mmproj from the merged model

When converting to GGUF, the vision projector must come from the merged model, not the base.

## `max_image_pixels`

Images are scaled down to at most this size before the processor, which directly sets each
image's token count and the VRAM peak.

> The inference side has a different setting: gemma-4 served by llama-server after GGUF conversion
> needs `--image-min-tokens 280`.

## Related

- [Fine-tuning overview](index.md)
- [DeepSpeed ZeRO-3 offload](deepspeed-offload.md)
- [GGUF conversion](../models/gguf-conversion.md)
