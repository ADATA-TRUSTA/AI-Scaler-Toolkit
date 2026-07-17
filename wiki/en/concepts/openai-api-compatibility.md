---
type: concept
title: OpenAI API Compatibility
created: 2026-04-29
updated: 2026-04-29
tags: [api, openai, standard, streaming, tool-calling]
related: [trusta-ast-inference-service, v1-chat-completions]
sources: ["inference_manual.md"]
---
# OpenAI API Compatibility

The Trusta AST Inference Service adheres strictly to the **OpenAI API** schema for all chat interactions. This standardization simplifies client integration and ensures compatibility with the broader ecosystem of tools and libraries designed for OpenAI.

## Primary Endpoint

All chat interactions must use the `POST /v1/chat/completions` endpoint. The legacy endpoint `POST /inference/chat` has been deprecated (HTTP 410).

## Supported Features

*   **Streaming**: Supports Server-Sent Events (SSE) for real-time token streaming. The service wraps worker output into standard OpenAI chunk formats.
*   **Session Management**: Accepts a `session_id` parameter to maintain conversation history across multiple requests.
*   **Tool Calling**: Supports the OpenAI standard for function calling (Tool Call), allowing the model to request function execution with specific parameters.
*   **Multi-modal**: Supports image inputs where the engine (vLLM or llama-server) provides the capability.

## Request Format

Requests follow the standard OpenAI structure:
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

## Benefits

*   **Ecosystem Integration**: Compatible with existing OpenAI clients and SDKs.
*   **Standardization**: Reduces the learning curve for developers familiar with OpenAI.
*   **Future Proofing**: Ensures the service remains compatible with evolving OpenAI standards and tooling.
