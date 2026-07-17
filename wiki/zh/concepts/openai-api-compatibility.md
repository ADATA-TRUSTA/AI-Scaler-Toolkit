---
type: concept
title: OpenAI API 相容性
created: 2026-04-29
updated: 2026-04-29
tags: [api, openai, standard, streaming, tool-calling]
related: [trusta-ast-inference-service, v1-chat-completions]
sources: ["inference_manual.md"]
---
# OpenAI API 相容性

Trusta AST 推論服務對所有對話互動嚴格遵循 **OpenAI API** 結構描述。此標準化簡化了用戶端整合，並確保與為 OpenAI 設計的廣泛工具與函式庫生態系相容。

## 主要端點

所有對話互動都必須使用 `POST /v1/chat/completions` 端點。舊版端點 `POST /inference/chat` 已被淘汰（HTTP 410）。

## 支援的功能

*   **串流**：支援 Server-Sent Events（SSE），可即時串流權杖（token）。服務會將工作行程的輸出封裝為標準的 OpenAI 分塊（chunk）格式。
*   **工作階段管理**：接受 `session_id` 參數，以在多次請求間維持對話歷史記錄。
*   **工具呼叫**：支援 OpenAI 的函式呼叫標準（Tool Call），允許模型以特定參數請求執行函式。
*   **多模態**：在引擎（vLLM 或 llama-server）提供該能力的情況下，支援影像輸入。

## 請求格式

請求遵循標準的 OpenAI 結構：
```json
{
  "model": "Qwen/Qwen3-4B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum mechanics."}
  ],
  "stream": true,
  "session_id": "user-123"
}
```

## 優點

*   **生態系整合**：與現有的 OpenAI 用戶端與 SDK 相容。
*   **標準化**：降低熟悉 OpenAI 的開發者的學習曲線。
*   **面向未來**：確保服務持續與不斷演進的 OpenAI 標準及工具相容。
