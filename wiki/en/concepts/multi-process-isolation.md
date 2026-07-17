---
type: concept
title: Multi-Process Isolation
created: 2026-04-29
updated: 2026-04-29
tags: [architecture, concurrency, stability, memory-management]
related: [trusta-ast-inference-service, model-manager, worker-process, oom-recovery]
sources: ["inference_manual.md"]
---
# Multi-Process Isolation

**Multi-process Isolation** is a core architectural pattern used in the Trusta AST Inference Service to ensure system stability and efficient resource management. It separates the application into two distinct processes: the **Main Process** (FastAPI) and the **Worker Process** (Inference).

## Rationale

LLM inference is computationally intensive and prone to memory spikes (Out-Of-Memory errors). In a monolithic process, an OOM error would crash the entire application, requiring a full restart and causing downtime.

## Implementation

1.  **Main Process (FastAPI)**:
    *   Handles all HTTP request handling, routing, and Session management.
    *   Acts as the client to the Worker Process via IPC (Inter-Process Communication) queues.
    *   Remains resilient; if the Worker dies, the API server continues to accept requests and manage sessions.

2.  **Worker Process (Model Inference)**:
    *   A dedicated subprocess that loads the model weights and holds the VRAM/CPU memory.
    *   Performs all heavy computation (forward passes, KV cache management).
    *   Can be safely terminated and restarted independently without affecting the Main Process.

## Benefits

*   **Stability**: OOM errors are contained within the Worker Process. The failure is detected (status becomes `error`), but recovery is **manual** — the worker is only re-created on the next `load_model` call or via `force_cleanup_gpu`.
*   **Memory Management**: Allows for precise control over VRAM usage and enables mechanisms like "KV Cache Cleanup" without unloading the entire model.
*   **GIL Avoidance**: Isolates Python's Global Interpreter Lock (GIL) issues, ensuring that heavy inference tasks do not block the event loop of the HTTP server.
