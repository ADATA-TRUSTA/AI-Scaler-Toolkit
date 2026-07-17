---
type: source
title: Trusta AST 推論服務手冊
created: 2026-04-29
updated: 2026-04-29
tags: [inference, architecture, fastapi, openai-api, vllm, llama-cpp, transformers]
related: [trusta-ast-inference-service-architecture, qwen3-5-inference-optimization, inference-engine-comparison, multi-process-isolation, openai-api-compatibility]
sources: ["inference_manual.md"]
---
# Trusta AST 推論服務手冊

這是 Trusta AST Inference Service 的完整技術文件，這是一個以 FastAPI 建構的高效能 LLM 推論伺服器。本文件詳述了系統架構、支援的推論引擎、API 規格、程式碼結構以及疑難排解程序。

## 概觀

本服務的設計旨在為各種大型語言模型（LLM）提供一個統一、穩定且高吞吐量的推論介面。它支援三種主要的推論引擎：**Hugging Face Transformers**、**vLLM** 以及 **llama-server**（用於 GGUF 模型）。

主要功能包含：
*   **Multi-process Isolation**：將 HTTP 伺服器（FastAPI）與模型推論分離，以確保穩定性與記憶體管理。
*   **OpenAI API Compatibility**：採用業界標準的 `/v1/chat/completions` 端點來進行聊天互動。
*   **Model Agnosticism**：透過統一的組態結構描述（`InferenceConfig`）抽象化後端差異。
*   **Session Management**：支援多輪對話，並可透過 Redis 或記憶體內儲存進行狀態持久化。

## 架構

核心架構模式為 **Singleton Manager & Process Isolation**（單例管理器與行程隔離）。
*   **Main Process (FastAPI)**：處理 HTTP 請求、路由、工作階段管理與使用者身分驗證。
*   **Worker Process**：一個專責載入模型、管理 VRAM 與執行推論運算的隔離行程。
*   **益處**：如果 Worker Process 中發生記憶體不足（OOM）錯誤，它會被侷限住而不影響主 HTTP 伺服器。恢復是手動的（重新發出 `load_model`，或呼叫 `force_cleanup_gpu`）。

## 支援的推論引擎

本服務支援多種後端，可透過組態中的 `engine` 欄位選擇：

1.  **Transformers**（`engine: transformers`）：
    *   預設引擎。
    *   支援 int4/int8 量化、CPU Offload 與 Disk Offload。
    *   適用於低資源環境與更廣泛的模型相容性。
2.  **vLLM**（`engine: vllm`）：
    *   針對高吞吐量的生產環境進行最佳化。
    *   支援原生的平行請求、AWQ/GPTQ/FP8 量化。
    *   需要特定的設定，但為並行請求提供了卓越的效能。
3.  **llama-server**（`engine: llama_server`）：
    *   針對 GGUF 格式模型（量化）進行最佳化。
    *   使用 `llama.cpp` 作為子行程。
    *   適合在 CPU 或有限 GPU 資源上進行高效率部署。

## 主要 API 端點

*   `POST /inference/load_model`：非同步地將模型載入 worker 行程。
*   `POST /inference/unload_model`：同步地卸載模型並釋放資源。
*   `GET /inference/status`：回傳目前的模型狀態、OOM 錯誤與裝置對應（device map）統計資訊。
*   `POST /v1/chat/completions`：**主要端點**。OpenAI 相容的聊天介面，支援串流、工作階段歷史與工具呼叫。
*   `POST /inference/estimate_memory`：在不載入模型的情況下，估算給定模型組態的 VRAM 需求。
*   `POST /inference/force_cleanup_gpu`：透過重新啟動 worker 行程來強制清理 GPU 記憶體。
*   `POST /inference/cleanup_generation_memory`：在不卸載模型的情況下，清除長對話期間累積的 KV cache。

**注意**：舊有端點 `POST /inference/chat` 已被淘汰（HTTP 410），不應再使用。

## 特定模型邏輯

本服務實作了 **Model-Aware Parameter Adjustment**（模型感知參數調整）來處理特定模型的怪癖。舉例來說，當載入 **Qwen3.5** 模型系列時，系統會自動覆寫使用者提供的生成參數，以防止生成瑕疵：
*   `temperature` 被強制設為 `1.0`。
*   `top_p` 被強制設為 `0.95`。
*   `enable_thinking` 被強制設為 `false`。

## 程式碼結構

*   `service/app.py`：主要進入點、API 路由與生命週期管理。
*   `service/model_manager.py`：API 與 Worker 行程之間的單例協調器。
*   `service/inference/model_inference_process.py`：隔離 Worker 行程的控制器。
*   `service/inference/engines/`：Transformers、vLLM 與 llama-server 的模組化實作。
*   `service/session_manager.py`：管理對話歷史與 Redis 持久化。
*   `service/model_registry.py`：管理可用模型清單（Base、Finetuned、GGUF）。
