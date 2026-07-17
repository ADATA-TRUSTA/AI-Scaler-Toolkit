---
type: concept
title: Qwen3.5 Inference Optimization
created: 2026-04-29
updated: 2026-04-29
tags: [qwen, optimization, model-specific, parameters]
related: [trusta-ast-inference-service, model-aware-parameter-adjustment]
sources: ["inference_manual.md"]
---
# Qwen3.5 Inference Optimization

**Qwen3.5 Inference Optimization** refers to a set of hard-coded logic and parameter overrides implemented in the Trusta AST Inference Service specifically for the **Qwen3.5** model family. This optimization is necessary due to the model's sensitivity to generation parameters and its MoE (Mixture of Experts) architecture.

## The Problem

Running Qwen3.5 with default or non-standard generation parameters often results in generation artifacts, such as garbled text or multi-language output errors.

## Solution: Model-Aware Parameter Adjustment

The service detects when the loaded model name contains "qwen3.5" and automatically forces specific parameter values, overriding any user-provided settings:

| Parameter | Forced Value | Reason |
| :--- | :--- | :--- |
| `temperature` | `1.0` | Stabilizes output distribution. |
| `top_p` | `0.95` | Ensures optimal token selection. |
| `top_k` | `20` | Limits vocabulary sampling. |
| `repetition_penalty` | `1.0` | Prevents repetition issues. |
| `enable_thinking` | `false` | Disables "thinking" mode to prevent artifacts. |

## Impact on Users

*   **Transparency**: Users do not need to manually tune these parameters for Qwen3.5; the system handles it automatically.
*   **Limitations**: Power users cannot override these values via the API for Qwen3.5 models. This ensures reliability at the cost of fine-grained control.
*   **Applicability**: This logic is specific to the Qwen3.5 family and does not automatically apply to other MoE models unless similar logic is added.
