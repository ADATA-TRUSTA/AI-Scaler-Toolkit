# Workflow Configuration Guide

繁體中文版本：[WORKFLOW_CONFIG_GUIDE.zh-TW.md](WORKFLOW_CONFIG_GUIDE.zh-TW.md)

## Overview

A workflow config file defines all parameters required for the full load -> chat -> unload flow, including:

- Inference model settings (which engine and model to use)
- Chat settings (prompts, generation params, stream/non-stream)
- Backend URL and timeout

## Config File Structure

```json
{
  "description": "Configuration description",
  "inference_config": "configs/inference/transformers.json",
  "chat_settings": "configs/chat/batch.json",
  "backend_url": "http://your-backend-host",
  "timeout": 600,
  "verbose": false
}
```

### Field Reference

| Field              | Type    | Required | Description                                                      |
| ------------------ | ------- | -------- | ---------------------------------------------------------------- |
| `description`      | string  | No       | Description of the configuration                                 |
| `inference_config` | string  | Yes      | Path to inference model config file                              |
| `chat_settings`    | string  | No       | Path to chat settings file. If omitted, default prompts are used |
| `backend_url`      | string  | No       | Backend URL. Defaults to `BACKEND_URL` in `.env`                      |
| `timeout`          | integer | No       | Request timeout in seconds. Default is 600                       |
| `verbose`          | boolean | No       | Whether to print verbose logs. Default is false                  |

## Default Workflow Configs

### 1. default.json

- Purpose: Standard Transformers inference
- Inference engine: Transformers (full precision / quantization)
- Chat settings: basic chat settings (4 prompts)
- Recommended for: quick tests and general inference

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json
```

### 2. vllm.json

- Purpose: Fast inference (vLLM engine)
- Inference engine: vLLM
- Chat settings: parallel batch settings (8 prompts)
- Recommended for: high-throughput scenarios

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/vllm.json
```

### 3. transformers_offload.json

- Purpose: Large-model inference with GPU/CPU offload
- Inference engine: Transformers with offload
- Chat settings: basic chat settings
- Recommended for: low VRAM environments and larger models

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/transformers_offload.json
```

### 4. llama.json

- Purpose: GGUF format inference (llama.cpp)
- Inference engine: llama.cpp/llama_server profile
- Chat settings: basic chat settings
- Recommended for: quantized models and CPU-focused inference

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/llama.json
```

### 5. parallel_batch_16.json

- Purpose: High-concurrency batch inference
- Inference engine: vLLM
- Chat settings: parallel batch settings (16 prompts)
- Recommended for: large batch workloads and performance tests

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/parallel_batch_16.json
```

## Usage Patterns

### Pattern 1: Use Workflow Config (recommended)

```bash
# default workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json

# vLLM workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/vllm.json
```

### Pattern 2: Override with CLI arguments

Workflow config is used as the base, then selected values are overridden by CLI args:

```bash
# vLLM workflow, but with a different backend URL
uv run examples/workflow/full.py \
    --workflow-config configs/workflow/vllm.json \
    --backend-url http://localhost:8000

# default workflow, but with different chat settings
uv run examples/workflow/full.py \
    --workflow-config configs/workflow/default.json \
    --chat-settings configs/chat/parallel_8.json
```

### Pattern 3: No workflow config (fully explicit)

```bash
uv run examples/workflow/full.py \
    --config configs/inference/vllm.json \
    --chat-settings configs/chat/batch.json \
    --backend-url http://your-backend-host \
    --timeout 900
```

## Chat Settings Format

A chat settings file defines prompt and generation behavior:

```json
{
  "backend_url": "http://your-backend-host",
  "api_key": "your-api-key",
  "model": "trusta-ast/trusta-ast-default",
  "system_prompt": "You are a helpful assistant.",
  "stream": false,
  "temperature": 0.5,
  "top_p": 0.9,
  "max_tokens": 800,
  "continue_on_error": true,
  "extra_body": {
    "top_k": 50,
    "repetition_penalty": 1.1,
    "total_timeout": 300,
    "enable_thinking": false
  },
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"]
}
```

Every field and its default lives on `ChatSettings` in `ai_scaler_console/helpers/chat.py`. Read it there — a table in this file only drifts.

## Workflow Execution Order

1. Load model (PHASE 1)

- Check whether a model is currently loading
- Unload any previously loaded model
- Load the new model from `inference_config`
- Monitor load progress until ready

2. Chat requests (PHASE 2)

- Load `chat_settings`
- Execute each prompt in order
- Support stream and non-stream response modes
- Print token usage statistics

3. Unload model (PHASE 3)

- Release model resources
- Restore clean system state

## Create a Custom Workflow Config

```bash
# copy default config
cp configs/workflow/default.json configs/workflow/custom.json

# edit custom config
# - update inference_config path
# - update chat_settings path
# - tune backend_url and timeout

# run custom workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/custom.json
```

For a workflow that also needs its own prompts, write a chat settings file first and point `chat_settings` at it.

## Troubleshooting

### Problem: Config file not found

Solution: verify path is correct relative to project root.

### Problem: Model loading failed

Solution: verify model path and engine settings in `inference_config`.

### Problem: Chat requests timeout

Solution: increase `timeout` in workflow config.

### Problem: Need more detailed logs

Solution: set `"verbose": true` in workflow config.

## References

- Inference config examples: configs/inference/
- Chat settings examples: configs/chat/
- Full workflow script: examples/workflow/full.py
