---
type: comparison
title: 舊版 API 與 OpenAI API 之比較（遷移）
created: 2026-04-29
updated: 2026-04-29
tags: [migration, api, deprecated]
related: [trusta-ast-inference-service, openai-api-compatibility]
sources: ["inference_manual.md"]
---
# 舊版 API 與 OpenAI API 之比較（遷移）

本比較著重說明 Trusta AST 推論服務從舊版自訂 API 過渡至現代 OpenAI 相容標準的過程。

| 功能特性 | 舊版 API（`/inference/chat`） | OpenAI API（`/v1/chat/completions`） |
| :--- | :--- | :--- |
| **狀態** | **已淘汰**（HTTP 410） | **主要／標準** |
| **回應格式** | 自訂格式 | 標準 OpenAI JSON 格式 |
| **串流** | 自訂實作 | 標準 SSE（Server-Sent Events） |
| **工作階段處理** | 隱含或自訂 | 明確的 `session_id` 參數 |
| **工具呼叫** | 不支援 | 透過 `tools` 參數完整支援 |
| **錯誤碼** | 自訂錯誤碼 | 標準 HTTP 錯誤碼（舊版為 410） |

## 遷移需求

用戶端必須立即從舊版端點遷移。
*   **端點變更**：將基礎 URL 從 `POST /inference/chat` 更新為 `POST /v1/chat/completions`。
*   **酬載重構**：將自訂請求主體轉換為標準的 `messages` 陣列格式。
*   **串流處理**：若使用串流，請針對新格式實作 SSE 解析。
*   **工作階段管理**：在請求酬載中明確傳遞 `session_id`，以維持對話歷史記錄。

## 理由

轉向 OpenAI 相容性可確保服務與業界標準一致、改善與第三方工具的互通性，並簡化新用戶端的 API 介面。
