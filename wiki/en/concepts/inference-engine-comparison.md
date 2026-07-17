---
type: concept
title: Inference Engine Comparison (vLLM, Transformers, GGUF)
created: 2026-04-29
updated: 2026-04-29
tags: [comparison, vllm, transformers, gguf, llama-cpp]
related: [trusta-ast-inference-service, inference-config]
sources: ["inference_manual.md"]
---
# Inference Engine Comparison (vLLM, Transformers, GGUF)

The Trusta AST Inference Service supports three distinct inference engines, each optimized for different use cases and hardware constraints. This comparison outlines the trade-offs for each.

| Feature | Transformers | vLLM | llama-server (GGUF) |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | General purpose, CPU offload, low resource. | High-throughput production, multi-GPU. | Efficient CPU/GPU inference, quantized models. |
| **Quantization** | int4, int8, nf4, fp4 (via bitsandbytes). | AWQ, GPTQ, FP8. | GGUF (Q8_0, Q4_K_M, etc.). |
| **Throughput** | Moderate. | Very High (PagedAttention). | High (for GGUF models). |
| **Startup Time** | Slow (model loading). | Slow (subprocess init). | Fast (for small models). |
| **Hardware** | CPU/GPU flexible (via `device_map`). | GPU intensive (requires VRAM). | CPU/GPU flexible (via `n_gpu_layers`). |
| **Configuration** | `device_map`, `max_memory`. | `vllm_gpu_memory_utilization`, `tensor_parallel`. | `n_gpu_layers`, `n_ctx`. |

## Selection Guide

*   **Choose Transformers if**: You need maximum compatibility, want to run models on CPU, or require specific offloading strategies (Disk Offload). It is the default engine.
*   **Choose vLLM if**: You are deploying to production with sufficient GPU resources and need to handle many concurrent requests. It offers the highest throughput.
*   **Choose llama-server if**: You have models in GGUF format, want to run on limited hardware, or need fast startup times for small models.

## Configuration Differences

Each engine requires a specific set of parameters in the `InferenceConfig`:
*   **vLLM** requires settings like `vllm_gpu_memory_utilization` and `vllm_max_model_len`.
*   **llama-server** requires settings like `n_gpu_layers` and `llama_server_port`.
*   **Transformers** relies on standard Hugging Face parameters like `device_map` and `quantization`.
