---
type: concept
title: 推論引擎比較（vLLM、Transformers、GGUF）
created: 2026-04-29
updated: 2026-04-29
tags: [comparison, vllm, transformers, gguf, llama-cpp]
related: [trusta-ast-inference-service, inference-config]
sources: ["inference_manual.md"]
---
# 推論引擎比較（vLLM、Transformers、GGUF）

Trusta AST 推論服務支援三種不同的推論引擎，各自針對不同的使用情境與硬體限制進行最佳化。本比較概述各引擎的取捨。

| 功能特性 | Transformers | vLLM | llama-server（GGUF） |
| :--- | :--- | :--- | :--- |
| **主要使用情境** | 通用用途、CPU 卸載、低資源環境。 | 高吞吐量生產環境、多 GPU。 | 高效率的 CPU/GPU 推論、量化模型。 |
| **量化** | int4、int8、nf4、fp4（透過 bitsandbytes）。 | AWQ、GPTQ、FP8。 | GGUF（Q8_0、Q4_K_M 等）。 |
| **吞吐量** | 中等。 | 非常高（PagedAttention）。 | 高（對於 GGUF 模型）。 |
| **啟動時間** | 慢（模型載入）。 | 慢（子行程初始化）。 | 快（對於小型模型）。 |
| **硬體** | CPU/GPU 彈性配置（透過 `device_map`）。 | GPU 密集（需要 VRAM）。 | CPU/GPU 彈性配置（透過 `n_gpu_layers`）。 |
| **設定** | `device_map`、`max_memory`。 | `vllm_gpu_memory_utilization`、`tensor_parallel`。 | `n_gpu_layers`、`n_ctx`。 |

## 選擇指南

*   **選擇 Transformers 的時機**：您需要最大化的相容性、想在 CPU 上執行模型，或需要特定的卸載策略（磁碟卸載，Disk Offload）。它是預設引擎。
*   **選擇 vLLM 的時機**：您正部署至具備充足 GPU 資源的生產環境，並需要處理大量並行請求。它提供最高的吞吐量。
*   **選擇 llama-server 的時機**：您擁有 GGUF 格式的模型、想在有限硬體上執行，或需要小型模型的快速啟動時間。

## 設定差異

每種引擎都需要在 `InferenceConfig` 中設定一組特定的參數：
*   **vLLM** 需要 `vllm_gpu_memory_utilization` 與 `vllm_max_model_len` 等設定。
*   **llama-server** 需要 `n_gpu_layers` 與 `llama_server_port` 等設定。
*   **Transformers** 依賴標準的 Hugging Face 參數，例如 `device_map` 與 `quantization`。
