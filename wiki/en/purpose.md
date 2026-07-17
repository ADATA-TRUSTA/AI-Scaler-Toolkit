# Project Purpose

## Goal

To document and explain, in a structured wiki, **Trusta AST (AI Scaler Toolkit)** — an LLM inference/training service built around FastAPI: its architecture, inference engines, OpenAI-compatible API, offload mechanism, and fine-tuning capabilities; and to connect it with the **TRUSTA enterprise-grade SSD product line** (the hardware counterpart of offloading to SSD/DRAM).

## Key Questions

1. How does this service use offload (device_map / n_gpu_layers / DeepSpeed) to reduce the required GPU VRAM, allowing larger models to run on smaller GPUs?
2. What are the respective positioning and trade-offs of the three inference engines (Transformers / vLLM / llama-server)?
3. How does the service integrate inference and fine-tuning on a single platform while maintaining the stability of multi-process isolation?

## Scope

**In scope:**
- The architecture, API, engines, offload, and fine-tuning of the Trusta AST service (sources: the `docs/` manual and the `service/` code)
- An overview of the TRUSTA SSD product line (the offload hardware counterpart)
- The LLM Wiki methodology / format conversion used to build this wiki

**Out of scope:**
- Measured performance/cost numbers of competing products (Ollama / standalone vLLM, etc.) (this project has not measured them under identical conditions)
- Unimplemented features (such as automatic worker restart, simultaneous multiple models, dynamic load balancing)

## Thesis

> The core value of Trusta AST lies in "reducing GPU VRAM requirements through offload" + "integrating fine-tuning on the same platform" + "multi-process isolation." Performance and VRAM reduction should be based on this project's measurement scripts; cost savings are estimates and must be clearly labeled.
