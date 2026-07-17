---
type: comparison
title: Legacy vs. OpenAI API (Migration)
created: 2026-04-29
updated: 2026-04-29
tags: [migration, api, deprecated]
related: [trusta-ast-inference-service, openai-api-compatibility]
sources: ["inference_manual.md"]
---
# Legacy vs. OpenAI API (Migration)

This comparison highlights the transition from the Trusta AST Inference Service's legacy custom API to the modern OpenAI-compatible standard.

| Feature | Legacy API (`/inference/chat`) | OpenAI API (`/v1/chat/completions`) |
| :--- | :--- | :--- |
| **Status** | **Deprecated** (HTTP 410) | **Primary/Standard** |
| **Response Format** | Custom format | Standard OpenAI JSON format |
| **Streaming** | Custom implementation | Standard SSE (Server-Sent Events) |
| **Session Handling** | Implicit or custom | Explicit `session_id` parameter |
| **Tool Calling** | Not supported | Full support via `tools` parameter |
| **Error Codes** | Custom codes | Standard HTTP codes (410 for legacy) |

## Migration Requirements

Clients must migrate from the legacy endpoint immediately.
*   **Endpoint Change**: Update base URL from `POST /inference/chat` to `POST /v1/chat/completions`.
*   **Payload Restructuring**: Convert custom request bodies to the standard `messages` array format.
*   **Streaming Handling**: If using streaming, implement SSE parsing for the new format.
*   **Session Management**: Explicitly pass `session_id` in the request payload to maintain conversation history.

## Rationale

The shift to OpenAI compatibility ensures the service aligns with industry standards, improves interoperability with third-party tools, and simplifies the API surface for new clients.
