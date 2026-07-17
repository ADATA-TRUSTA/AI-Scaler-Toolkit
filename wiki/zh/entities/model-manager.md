---
type: entity
title: ModelManager
created: 2026-04-29
updated: 2026-04-29
tags: [component, singleton, manager]
related: [multi-process-isolation, trusta-ast-inference-service, worker-process]
sources: ["inference_manual.md"]
---
# ModelManager

**ModelManager** 是 Trusta AST Inference Service 內的單例（singleton）類別，作為模型生命週期管理的中央協調者。它銜接了 FastAPI HTTP 伺服器與負責實際推論的隔離 Worker Process 之間的橋樑。

## 職責

*   **生命週期協調**：管理模型的載入（`start_loading`）、卸載（`unload_model`）以及狀態檢查。
*   **IPC 通訊**：透過行程間通訊（Inter-Process Communication，IPC）佇列系統與 Worker Process 通訊。它會傳送指令（載入、生成、停止）並接收狀態更新。
*   **並行控制**：確保同一時間只有一個模型被載入，並在使用者嘗試載入一個已在執行、或與現有模型衝突的模型時處理衝突。
*   **請求派發**：將生成請求路由到 Worker Process，並處理將 token 串流回傳給用戶端的流程。

## 架構角色

ModelManager 對於 **Multi-process Isolation** 模式至關重要。透過扮演中介者的角色，它讓 FastAPI 伺服器能夠持續回應 HTTP 請求，而繁重的模型推論工作則在另一個、可能容易當機的 Worker 行程中進行。

## 主要方法

*   `start_loading(config)`：以指定的組態啟動模型的非同步載入。
*   `generate_stream(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, ...)`：回傳一個用於串流模型輸出的迭代器。
*   `generate(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, ...)`：回傳單一的完成（completion）回應。
*   `stop_generation(request_id)`：終止特定的生成任務。
