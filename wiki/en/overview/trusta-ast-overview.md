---
title: TRUSTA-AST Overview
summary: TRUSTA-AST is a high-performance LLM inference service backend that provides a unified OpenAI API interface and support for multiple inference engines.
kind: overview
sources:
  - wiki/entities/trusta-ast-inference-service.md
  - wiki/overview.md
  - wiki/sources/inference_manual.md
createdAt: "2026-05-28"
updatedAt: "2026-05-28"
confidence: high
provenanceState: merged
---

# TRUSTA-AST Overview

**TRUSTA-AST** is a **high-performance LLM inference service backend**, designed for production environments, providing a unified OpenAI API interface and support for multiple inference engines.

## 🎯 Core Positioning

The main goals of TRUSTA-AST are:
1. **Unified interface**: provide a standard OpenAI API to simplify client integration
2. **Stability**: ensure high service availability through a multi-process isolation architecture
3. **Flexibility**: support multiple inference engines to adapt to different hardware and needs
4. **Production-grade**: designed for high-concurrency, high-reliability production environments

## 🏗️ Core Architecture Features

### 1. Multi-Process Isolation

```
┌─────────────────────────────────────────────────┐
│          Main Process (FastAPI)                 │
│  ├─ HTTP request handling                        │
│  ├─ Routing and authentication                   │
│  └─ Session management                           │
└─────────────┬───────────────────────────────────┘
              │ IPC Queue
┌─────────────▼───────────────────────────────────┐
│         Worker Process (Inference)              │
│  ├─ Model loading                                │
│  ├─ VRAM/CPU memory management                   │
│  └─ Inference computation                        │
└─────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ **Stability**: the worker's OOM errors do not affect the main service
- ✅ **Error isolation**: when the worker fails, the main service stays alive and the status transitions to `error`; recovery is manual (re-run `load_model` or `force_cleanup_gpu`)
- ✅ **Memory management**: precise control of VRAM usage
- ✅ **Avoids GIL issues**: inference tasks do not block the HTTP event loop

### 2. Engine Agnostic

TRUSTA-AST supports three main inference engines:

| Engine | Use case | Characteristics |
|------|---------|------|
| **Hugging Face Transformers** | General purpose, CPU offload, low resources | Default engine, supports int4/int8 quantization |
| **vLLM** | High-concurrency production environments | PagedAttention, ultra-high throughput |
| **llama-server (GGUF)** | Efficient CPU/GPU inference | Quantized models, fast startup |

### 3. OpenAI API Compatibility

Fully adheres to the OpenAI API standard:
- **Main endpoint**: `POST /v1/chat/completions`
- **Supported features**:
  - SSE streaming responses
  - Session ID context maintenance
  - Tool Calling (Function Calling)
  - Multimodal input (depending on engine support)

## 🔧 Core Components

### 1. FastAPI Server
- Handles HTTP requests, authentication, routing
- Does not directly execute heavy inference tasks
- Maintains service stability

### 2. ModelManager
- Singleton-pattern coordinator
- Manages communication between the API layer and the worker process
- Controls the model lifecycle

### 3. Worker Process
- An independent, isolated child process
- Responsible for model loading, VRAM management, inference execution
- Can be safely terminated and restarted

### 4. SessionManager
- Manages conversation state
- Supports multi-turn conversations
- Persists to Redis or in-memory storage

## 📊 Key Features

### Model Management
- `POST /inference/load_model` - Asynchronously load a model
- `POST /inference/unload_model` - Synchronously unload a model
- `POST /inference/estimate_memory` - Estimate VRAM requirements (no loading needed)
- `POST /inference/force_cleanup_gpu` - Forcibly clean up GPU memory

### Memory Management
- `POST /inference/cleanup_generation_memory` - Clean up the KV Cache
- Automatic OOM detection (recovery is manual)
- Precise VRAM usage monitoring

### Model-Specific Optimizations
- **Qwen3.5 model-specific optimizations**:
  - Automatically forces `temperature=1.0`
  - Automatically forces `top_p=0.95`
  - Automatically disables `enable_thinking`
  - Prevents artifact generation

## 🚀 Deployment Scenarios

### High-Throughput Scenarios
- Use vLLM + tensor parallelism
- Suitable for production deployment of large models
- Supports multi-GPU configurations

### Resource-Constrained Scenarios
- Use Transformers + CPU Offload
- Or use GGUF models + llama-server
- Suitable for CPU inference or limited hardware

### Deployment Flexibility
- A single service loads **one** model at a time (loading a second returns HTTP 409)
- You can choose different engines as needed (Transformers / vLLM / llama-server)
- One model is loaded at a time; switching models requires unloading first and then loading

## 📈 Technical Highlights

### 1. Model-Aware Parameter Adjustment
Automatically detects the model type and adjusts generation parameters:
```python
if "qwen3.5" in model_name.lower():
    params = {
        "temperature": 1.0,      # Stabilize the output distribution
        "top_p": 0.95,          # Optimal token selection
        "top_k": 20,            # Limit vocabulary sampling
        "repetition_penalty": 1.0,
        "enable_thinking": False # Prevent artifacts
    }
```

### 2. Memory Estimation Tool
- Estimate VRAM requirements without actually loading the model
- Avoid OOM errors
- Help with capacity planning

### 3. Session Management
- Supports multi-turn conversation context
- Redis persistent storage
- Automatically cleans up old sessions

## 🔍 Comparison with Other Systems

### vs Traditional RAG
| Aspect | TRUSTA-AST | Traditional RAG |
|------|-----------|---------|
| Knowledge management | Compiled knowledge base | Retrieval-based knowledge base |
| Cumulativeness | Knowledge accumulates | Retrieved anew each time |
| Structure | Structured wiki | Unstructured |
| Query speed | Fast (already compiled) | Slow (retrieved each time) |

### vs Single-Process Architecture
| Aspect | TRUSTA-AST | Single-process |
|------|-----------|--------|
| Stability | High (isolated) | Low (full crash) |
| Error isolation | Main service unaffected by worker crash | Full service crash |
| Memory management | Precise | Difficult |
| Concurrency capability | High | Limited |

## 📚 Related Documents

- [[Trusta AST Inference Service]] - Detailed service description
- [[Multi-Process Isolation]] - Architecture pattern in detail
- [[OpenAI API Compatibility]] - API compatibility description
- [[Qwen3.5 Inference Optimization]] - Qwen3.5-specific optimizations
- [[Inference Engine Comparison]] - Engine comparison

## 💡 Usage Recommendations

### Choosing an Inference Engine
- **Need maximum compatibility** → Transformers
- **Need the highest throughput** → vLLM
- **Need high performance / low resources** → llama-server (GGUF)

### Best Practices
1. Use the `estimate_memory` endpoint to estimate VRAM requirements
2. Regularly clean up the KV Cache to avoid memory leaks
3. Rely on automatic parameter adjustment for Qwen3.5 models
4. Use a Session ID to maintain conversation context
5. Monitor the health status of the worker process

## 🎓 Learning Resources

- **Official manual**: `wiki/sources/inference_manual.md`
- **API documentation**: OpenAI API standard documentation
- **Engine documentation**:
  - vLLM: https://docs.vllm.ai
  - llama.cpp: https://github.com/ggerganov/llama.cpp
  - Transformers: https://huggingface.co/docs

---

**Last updated**: 2026-05-28
**Knowledge base version**: 1.0
**Related concepts**: [[LLM Inference]], [[Production-Grade Deployment]], [[API Standardization]]
