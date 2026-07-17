---
type: concept
title: 多行程隔離
created: 2026-04-29
updated: 2026-04-29
tags: [architecture, concurrency, stability, memory-management]
related: [trusta-ast-inference-service, model-manager, worker-process, oom-recovery]
sources: ["inference_manual.md"]
---
# 多行程隔離

**多行程隔離（Multi-process Isolation）** 是 Trusta AST 推論服務所採用的核心架構模式，用以確保系統穩定性與高效的資源管理。它將應用程式拆分為兩個不同的行程：**主行程（Main Process）**（FastAPI）與 **工作行程（Worker Process）**（推論）。

## 理由

LLM 推論屬於運算密集型作業，且容易出現記憶體尖峰（Out-Of-Memory 錯誤）。在單體式行程中，OOM 錯誤會導致整個應用程式崩潰，需要完整重啟並造成停機。

## 實作

1.  **主行程（FastAPI）**：
    *   處理所有 HTTP 請求處理、路由與工作階段（Session）管理。
    *   透過 IPC（Inter-Process Communication，行程間通訊）佇列作為工作行程的用戶端。
    *   保持韌性；若工作行程當掉，API 伺服器仍持續接受請求並管理工作階段。

2.  **工作行程（模型推論）**：
    *   專用的子行程，負責載入模型權重並持有 VRAM/CPU 記憶體。
    *   執行所有繁重的運算（前向傳遞、KV cache 管理）。
    *   可安全地獨立終止並重啟，而不影響主行程。

## 優點

*   **穩定性**：OOM 錯誤被侷限在工作行程內。系統會偵測到該故障（狀態變為 `error`），但復原是 **手動的** — 工作行程僅在下一次 `load_model` 呼叫時，或透過 `force_cleanup_gpu` 才會重新建立。
*   **記憶體管理**：能夠精確控制 VRAM 使用量，並可啟用「KV Cache 清理」等機制，而無需卸載整個模型。
*   **避免 GIL**：隔離 Python 的全域直譯器鎖（Global Interpreter Lock，GIL）問題，確保繁重的推論任務不會阻塞 HTTP 伺服器的事件迴圈。
