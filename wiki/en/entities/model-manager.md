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

The **ModelManager** is a singleton class within the Trusta AST Inference Service that acts as the central coordinator for model lifecycle management. It bridges the gap between the FastAPI HTTP server and the isolated Worker Process responsible for actual inference.

## Responsibilities

*   **Lifecycle Coordination**: Manages the loading (`start_loading`), unloading (`unload_model`), and status checking of models.
*   **IPC Communication**: Communicates with the Worker Process via an Inter-Process Communication (IPC) Queue system. It sends instructions (load, generate, stop) and receives status updates.
*   **Concurrency Control**: Ensures that only one model is loaded at a time and handles conflicts if a user attempts to load a model that is already running or conflicts with an existing one.
*   **Request Dispatching**: Routes generation requests to the Worker Process and handles the streaming of tokens back to the client.

## Architecture Role

The ModelManager is critical to the **Multi-process Isolation** pattern. By acting as a mediator, it allows the FastAPI server to remain responsive to HTTP requests while the heavy lifting of model inference occurs in a separate, potentially crash-prone Worker process.

## Key Methods

*   `start_loading(config)`: Initiates the asynchronous loading of a model with a given configuration.
*   `generate_stream(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, ...)`: Returns an iterator for streaming model outputs.
*   `generate(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, ...)`: Returns a single completion response.
*   `stop_generation(request_id)`: Terminates a specific generation task.
