# What is TRUSTA-AST?

## 📝 Concise Answer

**TRUSTA-AST** is a **production-grade, high-performance LLM inference service backend** that:

1. **Provides a unified OpenAI API interface** - letting clients use it just like the OpenAI API
2. **Supports multiple inference engines** - Transformers, vLLM, llama-server (GGUF)
3. **Ensures service stability** - through a multi-process isolation architecture
4. **Adapts to different hardware needs** - from CPU to multi-GPU configurations

---

## 🎯 Core Value

### For Developers
- ✅ Familiar OpenAI API, low learning cost
- ✅ Supports a variety of models, not tied to a single engine
- ✅ Unified interface, no need to write different code for different engines

### For Operators
- ✅ High availability; a worker failure does not bring down the entire service
- ✅ Precise memory management, avoiding OOM
- ✅ Flexible deployment options, adapting to different hardware

### For End Users
- ✅ Stable service experience
- ✅ Fast response speed
- ✅ Support for multi-turn conversation context

---

## 🏗️ Core Architecture (Illustrated)

```
User request
    │
    ▼
┌─────────────────────────────┐
│  FastAPI server (main proc)  │
│  - Handle HTTP requests        │
│  - Authentication and authz   │
│  - Session management         │
└──────────┬──────────────────┘
           │ IPC Queue
           ▼
┌─────────────────────────────┐
│  Worker process (inference)    │
│  - Load model                 │
│  - Manage VRAM                │
│  - Run inference computation  │
└─────────────────────────────┘
```

**Key design**: the main process and the worker process are fully isolated, so the worker's errors (such as OOM) do not affect the main process.

---

## 🔧 The Three Inference Engines

### 1. Hugging Face Transformers
```yaml
engine: transformers
Use cases:
  - General purpose
  - CPU offload
  - Low-resource environments
Characteristics:
  - Supports int4/int8 quantization
  - Flexible CPU/GPU configuration
  - Default engine
```

### 2. vLLM
```yaml
engine: vllm
Use cases:
  - High-concurrency production environments
  - Multi-GPU configurations
  - Need for ultra-high throughput
Characteristics:
  - PagedAttention technology
  - Native parallel requests
  - Supports AWQ/GPTQ/FP8 quantization
```

### 3. llama-server (GGUF)
```yaml
engine: llama_server
Use cases:
  - Efficient CPU/GPU inference
  - Resource-constrained environments
  - Fast startup requirements
Characteristics:
  - GGUF quantized models
  - Based on llama.cpp
  - Fast startup speed
```

---

## 📊 Main Features

### API Endpoints
```
POST /v1/chat/completions  # Main chat endpoint (OpenAI standard)
POST /inference/load_model # Load a model
POST /inference/unload_model # Unload a model
GET  /inference/status     # Check model status
POST /inference/estimate_memory # Estimate VRAM requirements
```

### Highlighted Features
- **Streaming responses**: SSE real-time token output
- **Session management**: multi-turn conversation context
- **Tool Calling**: function call support
- **Memory management**: automatic KV Cache cleanup
- **Model optimization**: Qwen3.5 automatic parameter adjustment

---

## 💡 Practical Use Cases

### Scenario 1: High-Concurrency Customer Service Chatbot
```
Use vLLM + multi-GPU
- Handle hundreds of requests per second
- Maintain low latency
- Auto-scaling
```

### Scenario 2: Local Development Environment
```
Use llama-server + GGUF
- Runs with CPU inference alone
- Fast startup speed
- Low resource requirements
```

### Scenario 3: Switching Engines as Needed
```
Load a single model at a time (loading a second one returns HTTP 409);
unload as needed, then switch to a different engine:
- Use Transformers for the Qwen3 series
- Use vLLM when high throughput is needed
- Use GGUF + llama-server for CPU/constrained hardware
```

---

## 🆚 Comparison with Other Solutions

### vs Directly Using the OpenAI API
| Aspect | TRUSTA-AST | OpenAI API |
|------|-----------|-----------|
| Cost | Own hardware cost | API call fees |
| Privacy | Data entirely local | Data sent to the cloud |
| Customization | Can customize the model | Fixed model |
| Control | Full control | Limited control |

### vs Single-Process Architecture
| Aspect | TRUSTA-AST | Single-process |
|------|-----------|--------|
| Stability | High (isolated) | Low |
| Error isolation | Main service survives a worker crash | Whole service crashes |
| Concurrency | High | Limited |
| Management | Complex | Simple |

---

## 🚀 Quick Start

### Basic Configuration
```bash
# Install the environment (Linux + CUDA example; on Windows use the scripts under deploy/windows/)
cp .env.example .env
TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh

# Start the service (the service port is controlled by SERVICE_PORT in .env, default 8000)
bash deploy/linux/run_service.sh
```

> Note: the engine and model are **not** specified via environment variables, but rather via the `engine` / `model_name` fields in the body of the `POST /inference/load_model`
> request; there is no `pip install trusta-ast-service` package.

### Basic Invocation
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }
)

for chunk in response.iter_lines():
    print(chunk)
```

---

## 📚 Related Documents

- [[Trusta AST Inference Service]] - Detailed service description
- [[Multi-Process Isolation]] - Architecture pattern in detail
- [[OpenAI API Compatibility]] - API compatibility
- [[Qwen3.5 Inference Optimization]] - Model optimization
- [[Inference Engine Comparison]] - Engine comparison

---

**Simplified summary**: TRUSTA-AST lets you **use the familiar OpenAI API** to **run your own LLM models stably and efficiently**, supporting multiple engines and adapting to different hardware needs.
