---
type: overview
title: Project Overview
created: 2026-04-29
updated: 2026-04-29
tags: [overview, llm, inference, trusta]
related: []
sources: ["inference_manual.md"]
---
# Project Overview

This wiki documents the architecture, components, and usage patterns of the **Trusta AST Inference Service**, a high-performance LLM inference server designed for stability and flexibility. The project aims to provide a unified, production-ready interface for deploying various Large Language Models while abstracting the complexities of different inference backends.

## Core Architecture

The system is built on a **Multi-process Isolation** architecture, where the HTTP server (FastAPI) is strictly separated from the model inference logic (Worker Process). This design ensures that memory-intensive tasks and potential Out-Of-Memory (OOM) errors in the inference process do not crash the main service. When a worker dies, its status becomes `error` and recovery is manual (re-issue `load_model`, or call `force_cleanup_gpu`). The **ModelManager** acts as the central coordinator, managing the lifecycle of models and handling communication between the API layer and the isolated workers.

## Engine Agnosticism

A key strength of the service is its **Engine Agnostic** design. It supports three distinct inference engines:
1.  **Hugging Face Transformers**: For broad compatibility and CPU offloading.
2.  **vLLM**: For high-throughput production environments with GPU acceleration.
3.  **llama-server**: For efficient GGUF model deployment on CPU or limited hardware.

Users can switch between these engines via configuration without changing the API interface.

## Standardization and Compatibility

The service has fully adopted the **OpenAI API** standard for chat interactions (`POST /v1/chat/completions`), deprecating all legacy custom endpoints. This ensures compatibility with the broader ecosystem of LLM tools and clients. The implementation supports streaming (SSE), session management for multi-turn conversations, and advanced features like Tool Calling.

## Specialized Optimizations

The wiki also details specific optimizations required for certain model families, such as the **Qwen3.5** model. Due to the unique architecture of Qwen3.5 (MoE), the service implements **Model-Aware Parameter Adjustment** logic to automatically force specific generation parameters (temperature, top_p, etc.) to prevent artifacts and ensure reliable output.

## Future Scope

Future development may focus on expanding engine support, refining memory estimation tools, and creating migration scripts for legacy clients. The wiki serves as the central knowledge base for understanding the current state of the inference service and guiding its evolution.
