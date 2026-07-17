---
type: overview
title: 專案總覽
created: 2026-04-29
updated: 2026-04-29
tags: [overview, llm, inference, trusta]
related: []
sources: ["inference_manual.md"]
---
# 專案總覽

本 wiki 記載了 **Trusta AST Inference Service** 的架構、元件與使用模式，這是一個為穩定性與彈性而設計的高效能 LLM 推論伺服器。本專案旨在提供一個統一、可用於生產環境的介面，用以部署各種大型語言模型，同時抽象化不同推論後端的複雜性。

## 核心架構

本系統建構於 **Multi-process Isolation** 架構之上，HTTP 伺服器（FastAPI）與模型推論邏輯（Worker Process）被嚴格分離。此設計確保推論行程中記憶體密集的任務與潛在的記憶體不足（Out-Of-Memory，OOM）錯誤不會使主服務當機。當某個 worker 死亡時，其狀態會變為 `error`，且恢復是手動的（重新發出 `load_model`，或呼叫 `force_cleanup_gpu`）。**ModelManager** 扮演中央協調者的角色，管理模型的生命週期，並處理 API 層與隔離 worker 之間的通訊。

## 引擎無關性

本服務的一項關鍵優勢在於其 **Engine Agnostic**（引擎無關）的設計。它支援三種不同的推論引擎：
1.  **Hugging Face Transformers**：用於廣泛的相容性與 CPU 卸載。
2.  **vLLM**：用於具備 GPU 加速的高吞吐量生產環境。
3.  **llama-server**：用於在 CPU 或有限硬體上有效率地部署 GGUF 模型。

使用者可透過組態在這些引擎之間切換，而無需變更 API 介面。

## 標準化與相容性

本服務已完全採用 **OpenAI API** 標準來進行聊天互動（`POST /v1/chat/completions`），並淘汰了所有舊有的自訂端點。這確保了與更廣泛的 LLM 工具與用戶端生態系相容。此實作支援串流（SSE）、用於多輪對話的工作階段管理，以及如工具呼叫（Tool Calling）等進階功能。

## 特定最佳化

本 wiki 也詳述了某些模型系列所需的特定最佳化，例如 **Qwen3.5** 模型。由於 Qwen3.5（MoE）的獨特架構，本服務實作了 **Model-Aware Parameter Adjustment** 邏輯，以自動強制設定特定的生成參數（temperature、top_p 等），藉此防止輸出瑕疵並確保可靠的輸出。

## 未來範疇

未來的開發可能聚焦於擴展引擎支援、精進記憶體估算工具，以及為舊有用戶端建立遷移腳本。本 wiki 作為理解目前推論服務狀態並引導其演進的中央知識庫。
