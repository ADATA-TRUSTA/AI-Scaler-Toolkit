---
type: source
title: Trusta AST Inference Service Manual
created: 2026-04-29
updated: 2026-04-29
tags: [inference, architecture, fastapi, openai-api, vllm, llama-cpp, transformers]
related: [trusta-ast-inference-service-architecture, qwen3-5-inference-optimization, inference-engine-comparison, multi-process-isolation, openai-api-compatibility]
sources: ["inference_manual.md"]
---
# Trusta AST Inference Service Manual

A comprehensive technical documentation for the Trusta AST Inference Service, a high-performance LLM inference server built with FastAPI. This document details the system architecture, supported inference engines, API specifications, code structure, and troubleshooting procedures.

## Overview

The service is designed to provide a unified, stable, and high-throughput inference interface for various Large Language Models (LLMs). It supports three primary inference engines: **Hugging Face Transformers**, **vLLM**, and **llama-server** (for GGUF models).

Key features include:
*   **Multi-process Isolation**: Separates the HTTP server (FastAPI) from model inference to ensure stability and memory management.
*   **OpenAI API Compatibility**: Adopts the industry-standard `/v1/chat/completions` endpoint for chat interactions.
*   **Model Agnosticism**: Abstracts backend differences through a unified configuration schema (`InferenceConfig`).
*   **Session Management**: Supports multi-turn conversations with state persistence via Redis or in-memory storage.

## Architecture

The core architectural pattern is **Singleton Manager & Process Isolation**.
*   **Main Process (FastAPI)**: Handles HTTP requests, routing, session management, and user authentication.
*   **Worker Process**: A dedicated isolated process responsible for loading models, managing VRAM, and performing inference calculations.
*   **Benefit**: If an Out-Of-Memory (OOM) error occurs in the Worker Process, it is contained without affecting the main HTTP server. Recovery is manual (re-issue `load_model`, or call `force_cleanup_gpu`).

## Supported Inference Engines

The service supports multiple backends, selectable via the `engine` field in the configuration:

1.  **Transformers** (`engine: transformers`):
    *   Default engine.
    *   Supports int4/int8 quantization, CPU Offload, and Disk Offload.
    *   Suitable for low-resource environments and broader model compatibility.
2.  **vLLM** (`engine: vllm`):
    *   Optimized for high-throughput production environments.
    *   Supports native parallel requests, AWQ/GPTQ/FP8 quantization.
    *   Requires specific setup but offers superior performance for concurrent requests.
3.  **llama-server** (`engine: llama_server`):
    *   Optimized for GGUF format models (quantized).
    *   Uses `llama.cpp` as a subprocess.
    *   Ideal for efficient deployment on CPU or limited GPU resources.

## Key API Endpoints

*   `POST /inference/load_model`: Asynchronously loads a model into the worker process.
*   `POST /inference/unload_model`: Synchronously unloads the model and frees resources.
*   `GET /inference/status`: Returns current model status, OOM errors, and device map statistics.
*   `POST /v1/chat/completions`: **Primary Endpoint**. OpenAI-compatible chat interface supporting streaming, session history, and tool calls.
*   `POST /inference/estimate_memory`: Estimates VRAM requirements for a given model configuration without loading it.
*   `POST /inference/force_cleanup_gpu`: Forces a cleanup of GPU memory by restarting the worker process.
*   `POST /inference/cleanup_generation_memory`: Clears KV cache accumulated during long conversations without unloading the model.

**Note**: The legacy endpoint `POST /inference/chat` is deprecated (HTTP 410) and should not be used.

## Model-Specific Logic

The service implements **Model-Aware Parameter Adjustment** to handle specific model quirks. For example, when the **Qwen3.5** model family is loaded, the system automatically overrides user-provided generation parameters to prevent generation artifacts:
*   `temperature` is forced to `1.0`.
*   `top_p` is forced to `0.95`.
*   `enable_thinking` is forced to `false`.

## Code Structure

*   `service/app.py`: Main entry point, API routes, and lifecycle management.
*   `service/model_manager.py`: Singleton coordinator between the API and Worker processes.
*   `service/inference/model_inference_process.py`: Controller for the isolated Worker process.
*   `service/inference/engines/`: Modular implementations for Transformers, vLLM, and llama-server.
*   `service/session_manager.py`: Manages conversation history and Redis persistence.
*   `service/model_registry.py`: Manages the list of available models (Base, Finetuned, GGUF).
