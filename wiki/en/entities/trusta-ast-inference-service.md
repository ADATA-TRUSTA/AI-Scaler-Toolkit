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

The **Trusta AST Inference Service** is a high-performance LLM inference server designed for the Trusta AST backend. It serves as the core runtime environment for deploying and serving Large Language Models in production.

## Core Characteristics

*   **Framework**: Built on **FastAPI** for high-performance HTTP handling.
*   **Architecture**: Utilizes **Multi-process Isolation** to separate the API server from the model inference logic, enhancing stability and memory safety.
*   **Compatibility**: Fully compatible with the **OpenAI API** standard, specifically the `/v1/chat/completions` endpoint.
*   **Engine Support**: Agnostic to the underlying inference engine, supporting **Hugging Face Transformers**, **vLLM**, and **llama-server** (GGUF).

## Key Components

1.  **FastAPI Server**: Handles incoming HTTP requests, authentication, and routing. It does not perform heavy inference tasks directly.
2.  **ModelManager**: A singleton component that coordinates communication between the API layer and the isolated Worker processes.
3.  **Worker Process**: An isolated subprocess dedicated to loading models, managing VRAM, and executing inference. This isolation ensures that OOM errors do not crash the entire service.
4.  **SessionManager**: Manages conversation state, supporting multi-turn dialogues with persistence to Redis or in-memory storage.

## API Standardization

The service has deprecated its legacy custom endpoints (e.g., `/inference/chat`) in favor of the standard OpenAI-compatible interface. All chat interactions must now use `POST /v1/chat/completions`, which supports:
*   Server-Sent Events (SSE) for streaming responses.
*   Session IDs for context retention.
*   Tool calling (Function Calling).
*   Multi-modal inputs (with specific engine support).

## Deployment Scenarios

The service is designed to be flexible across different hardware configurations:
*   **High-Throughput**: Using vLLM with tensor parallelism for large model serving.
*   **Resource-Constrained**: Using Transformers with CPU offloading or GGUF models with llama-server for CPU-based inference.
