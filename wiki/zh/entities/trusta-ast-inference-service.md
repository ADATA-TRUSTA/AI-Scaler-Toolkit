---
type: entity
title: Trusta AST Inference Service
created: 2026-04-29
updated: 2026-04-29
tags: [service, inference, llm, trusta]
related: [fastapi, model-manager, multi-process-isolation, vllm, llama-server, transformers-engine]
sources: ["inference_manual.md"]
---
# Trusta AST Inference Service

**Trusta AST Inference Service** 是專為 Trusta AST 後端設計的高效能 LLM 推論伺服器。它作為在生產環境中部署與提供大型語言模型服務的核心執行環境。

## 核心特性

*   **框架**：建構於 **FastAPI** 之上，以達成高效能的 HTTP 處理。
*   **架構**：採用 **Multi-process Isolation**，將 API 伺服器與模型推論邏輯分離，以強化穩定性與記憶體安全。
*   **相容性**：完全相容於 **OpenAI API** 標準，特別是 `/v1/chat/completions` 端點。
*   **引擎支援**：與底層推論引擎無關，支援 **Hugging Face Transformers**、**vLLM** 以及 **llama-server**（GGUF）。

## 主要元件

1.  **FastAPI Server**：處理傳入的 HTTP 請求、身分驗證與路由。它不會直接執行繁重的推論任務。
2.  **ModelManager**：一個單例元件，協調 API 層與隔離的 Worker 行程之間的通訊。
3.  **Worker Process**：一個專責載入模型、管理 VRAM 與執行推論的隔離子行程。此隔離確保 OOM 錯誤不會使整個服務當機。
4.  **SessionManager**：管理對話狀態，支援多輪對話，並可將狀態持久化至 Redis 或記憶體內儲存。

## API 標準化

本服務已淘汰其舊有的自訂端點（例如 `/inference/chat`），改採標準的 OpenAI 相容介面。所有聊天互動現在都必須使用 `POST /v1/chat/completions`，其支援：
*   使用 Server-Sent Events（SSE）進行串流回應。
*   使用 Session ID 保留上下文。
*   工具呼叫（Function Calling）。
*   多模態輸入（需特定引擎支援）。

## 部署情境

本服務設計上可靈活適應不同的硬體配置：
*   **高吞吐量**：使用具備張量平行（tensor parallelism）的 vLLM 來提供大型模型服務。
*   **資源受限**：使用具備 CPU 卸載的 Transformers，或搭配 llama-server 的 GGUF 模型來進行以 CPU 為基礎的推論。
