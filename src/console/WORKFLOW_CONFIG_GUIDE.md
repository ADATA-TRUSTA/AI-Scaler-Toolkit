# Workflow Configuration Guide

## Overview

A workflow config file defines all parameters required for the full load -> chat -> unload flow, including:

- Inference model settings (which engine and model to use)
- Chat settings (prompts, generation params, stream/non-stream)
- Backend URL and timeout

## Config File Structure

```json
{
  "description": "Configuration description",
  "inference_config": "infer_model_configs/inference-config-transformers.json",
  "chat_settings": "app_settings/app_settings_batch_chat.json",
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
| `backend_url`      | string  | No       | Backend URL. Defaults to `default_backend_settings.json`         |
| `timeout`          | integer | No       | Request timeout in seconds. Default is 600                       |
| `verbose`          | boolean | No       | Whether to print verbose logs. Default is false                  |

## Default Workflow Configs

### 1. workflow-config-default.json

- Purpose: Standard Transformers inference
- Inference engine: Transformers (full precision / quantization)
- Chat settings: basic chat settings (4 prompts)
- Recommended for: quick tests and general inference

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-default.json
```

### 2. workflow-config-vllm.json

- Purpose: Fast inference (vLLM engine)
- Inference engine: vLLM
- Chat settings: parallel batch settings (8 prompts)
- Recommended for: high-throughput scenarios

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-vllm.json
```

### 3. workflow-config-transformers-offload.json

- Purpose: Large-model inference with GPU/CPU offload
- Inference engine: Transformers with offload
- Chat settings: basic chat settings
- Recommended for: low VRAM environments and larger models

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-transformers-offload.json
```

### 4. workflow-config-llama.json

- Purpose: GGUF format inference (llama.cpp)
- Inference engine: llama.cpp/llama_server profile
- Chat settings: basic chat settings
- Recommended for: quantized models and CPU-focused inference

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-llama.json
```

### 5. workflow-config-parallel-batch-16.json

- Purpose: High-concurrency batch inference
- Inference engine: vLLM
- Chat settings: parallel batch settings (16 prompts)
- Recommended for: large batch workloads and performance tests

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-parallel-batch-16.json
```

## Usage Patterns

### Pattern 1: Use Workflow Config (recommended)

```bash
# default workflow
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-default.json

# vLLM workflow
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-vllm.json
```

### Pattern 2: Override with CLI arguments

Workflow config is used as the base, then selected values are overridden by CLI args:

```bash
# vLLM workflow, but with a different backend URL
uv run complete-workflow-example.py \
    --workflow-config app_settings/workflow-config-vllm.json \
    --backend-url http://localhost:8000

# default workflow, but with different chat settings
uv run complete-workflow-example.py \
    --workflow-config app_settings/workflow-config-default.json \
    --chat-settings app_settings/app_settings_parallel_terminal_batch_chat_8.json
```

### Pattern 3: No workflow config (fully explicit)

```bash
uv run complete-workflow-example.py \
    --config infer_model_configs/inference-config-vllm.json \
    --chat-settings app_settings/app_settings_batch_chat.json \
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

### Chat Field Notes

| Field               | Description                          | Default           |
| ------------------- | ------------------------------------ | ----------------- |
| `stream`            | Whether to use streaming output      | false             |
| `temperature`       | Generation diversity (0-2)           | 0.7               |
| `top_p`             | Nucleus sampling parameter           | 0.9               |
| `max_tokens`        | Maximum output tokens                | 512               |
| `continue_on_error` | Continue when one prompt fails       | true              |
| `prompts`           | Prompt list for multi-turn execution | 3 default prompts |

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
cp app_settings/workflow-config-default.json app_settings/workflow-config-custom.json

# edit custom config
# - update inference_config path
# - update chat_settings path
# - tune backend_url and timeout

# run custom workflow
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-custom.json
```

## Common Scenarios

### Scenario 1: Quick test on a new model

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-default.json
```

### Scenario 2: High-performance inference test

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-vllm.json
```

### Scenario 3: Resource-limited inference

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-transformers-offload.json
```

### Scenario 4: Large batch inference

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-parallel-batch-16.json
```

### Scenario 5: Custom complex workflow

1. Create a custom chat settings file (more prompts or special params)
2. Create a custom workflow config that points to that chat settings file
3. Run the workflow

```bash
uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-custom.json
```

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

- Inference config examples: infer_model_configs/
- Chat settings examples: app_settings/
- Full workflow script: complete-workflow-example.py
