# AST Console AI API Client

Headless (no-UI) Python client for the Trusta AST backend.
This project is the API-only counterpart to **Trusta-AST-Frontend**: same backend, same model lifecycle, but driven from a console / script instead of a web UI.

What it covers:

- Loading / unloading models on the backend (`/inference/load_model`, `/inference/unload_model`).
- Chatting with the loaded model via the **OpenAI-compatible** endpoint (`<backend_url>/v1/chat/completions`), text or image.
- Fine-tune training (`/training/start`, `/training/stop`, `/training/status`).

The legacy `/inference/chat` Python wrappers (`AIClient.chat`, `ChatRequest`, `ChatStreamChunk`, etc.) have been removed. All chat now goes through the OpenAI Python SDK against the backend's OpenAI-compatible endpoint, mirroring `runChatCompletions` in the frontend.

The model config files under `infer_model_configs/` are kept aligned with the load-model payload used by Trusta-AST-Frontend.
For more detailed API documentation, see the Trusta-AST-Backend API reference.

## 0. Environment Setup

The project is primarily managed by **uv**:

```bash
# windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
# (optional) remove old .venv / __pycache__ first
uv sync
```

If you are not using uv:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Recommended No-UI Workflow

### 1.1 Load a model

Pick any config under `infer_model_configs/` (transformers / llama_server / vLLM are all supported, fields match the frontend payload).

> **Before running:** edit the chosen config file and set `model_path` to your actual model location. For `llama_server` and `vllm` configs this must be an absolute path on the backend host. For `transformers` configs a HuggingFace model ID works as-is.

```bash
# Transformers
uv run load_model_example.py --config infer_model_configs/inference-config-transformers.json

# Transformers + GPU/CPU offload
uv run load_model_example.py --config infer_model_configs/inference-config-transformers-offload.json

# llama.cpp (GGUF)
uv run load_model_example.py --config infer_model_configs/inference-config-llama.json

# vLLM
uv run load_model_example.py --config infer_model_configs/inference-config-vllm.json

# Override backend URL
uv run load_model_example.py \
    --config infer_model_configs/inference-config-vllm.json \
    --backend-url http://your-backend-host
```

Default backend URL is loaded from `app_settings/default_backend_settings.json`:

```json
{
  "backend_url": "http://your-backend-host"
}
```

Update this file once if you want to change the shared default URL across examples.

### 1.1.1 Get model list

Fetch the unified model list used by the frontend's `/config/models` API:

```bash
uv run get_model_list_example.py

# override backend URL
uv run get_model_list_example.py --backend-url http://your-backend-host

# pretty-print raw JSON only
uv run get_model_list_example.py --json
```

The response is parsed into `ModelListResponse` with three groups:

- `base_models`
- `finetuned_models`
- `llama_gguf_models`

### 1.2 Chat (text)

After the model is loaded:

```bash
uv run openai-compatible-example.py
```

The script uses the OpenAI Python SDK against `<backend_url>/v1/chat/completions`. AST-specific fields (`top_k`, `repetition_penalty`, `total_timeout`, `enable_thinking`, `session_id`, RAG knobs, `request_id`, `tools` ...) are passed via `extra_body`, just like the frontend does.

### 1.2.1 Batch chat (multiple prompts)

Run multiple prompts in one command:

```bash
uv run openai-compatible-batch-example.py

# custom settings path
uv run openai-compatible-batch-example.py \
  --settings app_settings/app_settings_batch_chat.json
```

The script reads prompts from `app_settings/app_settings_batch_chat.json`, sends them to `<backend_url>/v1/chat/completions` one by one, and prints each prompt, response, and per-request usage stats.

### 1.2.2 Parallel terminal batch chat

If you want N prompts to run at the same time in N separate terminal windows, use the platform-specific launcher:

```bash
# Windows (default: 8 prompts)
uv run openai-compatible-parallel-terminal-batch-example.py

# Linux (default: 8 prompts)
uv run openai-compatible-parallel-terminal-batch-linux-example.py

# 8 prompts
uv run openai-compatible-parallel-terminal-batch-example.py \
  --settings app_settings/app_settings_parallel_terminal_batch_chat_8.json

uv run openai-compatible-parallel-terminal-batch-linux-example.py \
  --settings app_settings/app_settings_parallel_terminal_batch_chat_8.json

# 16 prompts
uv run openai-compatible-parallel-terminal-batch-example.py \
  --settings app_settings/app_settings_parallel_terminal_batch_chat_16.json

uv run openai-compatible-parallel-terminal-batch-linux-example.py \
  --settings app_settings/app_settings_parallel_terminal_batch_chat_16.json
```

The Windows launcher opens one Windows terminal window per prompt. The Linux launcher opens one Linux terminal window per prompt using a local terminal emulator. Each worker runs one synchronous chat request and exits. Use these when you want independent terminal windows instead of one sequential batch process.

Set `stream` in the settings file to control response mode:

- `stream: false` -> non-streaming full response per worker
- `stream: true` -> token streaming output per worker terminal

Key bindings:

- `model` is fixed to `trusta-ast/trusta-ast-default` -- this is the OpenAI-compatible binding name on the backend, regardless of which underlying model is loaded.
- `api_key` is required by the SDK but the backend accepts any string when auth is disabled.
- SSL verification is skipped because the dev backend uses a self-signed certificate (`httpx.Client(verify=False)`).

### 1.3 Chat (image / multimodal)

For multimodal models (Gemma 3 / Qwen-VL via transformers, or vLLM with `vllm_mm_image_limit` set in the load config):

```bash
uv run openai-compatible-image-example.py
```

Before running, update the `image_path` variable in the script (or the `images` field in the settings file) to point to your own image.

The script sends OpenAI-compatible `image_url` content parts. It demonstrates:

- local files -> Base64 Data URI (auto-encoded)
- HTTP/HTTPS URLs -> passed through as-is
- multiple images in a single message
- non-stream variant for full-response + usage stats

### 1.3.1 Batch image chat (multiple multimodal tasks)

Run multiple image chat tasks in one command:

```bash
uv run openai-compatible-image-batch-example.py

# custom settings path
uv run openai-compatible-image-batch-example.py \
  --settings app_settings/app_settings_batch_chat_image.json
```

The script reads `tasks` from `app_settings/app_settings_batch_chat_image.json`.
Each task contains one text prompt and one or more images (local path or URL).

### 1.4 Unload the model

```bash
uv run unload_model_example.py
uv run unload_model_example.py --backend-url http://your-backend-host
```

### 1.5 Change backend base URL (all scripts)

If you move to another backend host, update URL settings based on how each script reads config.

#### Option A (one-time shared default for CLI examples)

Edit `app_settings/default_backend_settings.json`:

```json
{
  "backend_url": "https://your-new-backend-host"
}
```

This shared default is used by:

- `load_model_example.py` (unless `--backend-url` is provided)
- `unload_model_example.py` (unless `--backend-url` is provided)
- `get_model_list_example.py` (unless `--backend-url` is provided)
- `openai-compatible-example.py` (single text chat)
- `openai-compatible-image-example.py` (single image chat)
- `complete-workflow-example.py` when workflow/chat config does not override `backend_url`

#### Option B (script-specific settings for batch/parallel chat)

The following scripts read `backend_url` from their own settings JSON first. If present there, it overrides the shared default:

- `openai-compatible-batch-example.py` -> `app_settings/app_settings_batch_chat.json`
- `openai-compatible-image-batch-example.py` -> `app_settings/app_settings_batch_chat_image.json`
- `openai-compatible-parallel-terminal-batch-example.py` -> `app_settings/app_settings_parallel_terminal_batch_chat*.json`
- `openai-compatible-parallel-terminal-batch-linux-example.py` -> `app_settings/app_settings_parallel_terminal_batch_chat*.json`

For easier testing, `backend_url` is currently removed from the default app/workflow settings files in this repo, so they fall back to `app_settings/default_backend_settings.json`.

If you want the config-file override style, keep/add `backend_url` in your own settings JSON. A reference file is provided at `app_settings/app_settings_batch_chat_with_backend_url_example.json`.

#### Option C (per-run temporary override)

For scripts that support CLI override, pass `--backend-url` directly:

```bash
uv run load_model_example.py --config infer_model_configs/inference-config-vllm.json \
  --backend-url https://your-new-backend-host

uv run unload_model_example.py --backend-url https://your-new-backend-host
uv run get_model_list_example.py --backend-url https://your-new-backend-host

uv run complete-workflow-example.py --backend-url https://your-new-backend-host
```

#### Quick checklist when switching backend

1. Update `app_settings/default_backend_settings.json`.
2. (Optional) Add/update `backend_url` inside the specific settings JSON only if you need per-script override.
3. Use `app_settings/app_settings_batch_chat_with_backend_url_example.json` as a reference for the override format.
4. Re-run load model and then chat scripts against the same host.

## 2. Fine-tune

Fine-tuning is unchanged.

```bash
# default settings file: app_settings/app_settings_finetune.json
uv run finetune_example.py

# custom settings file
uv run finetune_example.py --settings app_settings/app_settings_finetune.json

# force interrupt an existing training session and start a new one
uv run finetune_example.py --force

# combined
uv run finetune_example.py \
    --settings app_settings/app_settings_finetune.json \
    --force

# help
uv run finetune_example.py --help
```

## 3. Configuration File Formats

### 3.1 Inference model configs (`infer_model_configs/*.json`)

These mirror the frontend's load-model payload. The `engine` field selects the schema:

```jsonc
// Transformers — model_path can be a HuggingFace model ID (auto-downloaded) or a local directory path
{
  "base_model": "Qwen/Qwen3-4B",
  "model_path": "Qwen/Qwen3-4B",   // ← HuggingFace ID or /absolute/path/to/model
  "engine": "transformers",
  "quantization": "int4",           // ← adjust: "none" | "int4" | "int8"
  "device_map": "cuda:0",           // ← adjust to your GPU index or "auto"
  "max_memory": null
}

// Transformers with GPU/CPU offload
{
  "base_model": "google/gemma-3-12b-it",
  "model_path": "google/gemma-3-12b-it",  // ← HuggingFace ID or /absolute/path/to/model
  "engine": "transformers",
  "quantization": "none",
  "device_map": "auto",
  "max_memory": { "0": "10GiB", "cpu": "25GiB" }  // ← adjust to your hardware
}

// llama.cpp (GGUF) — model_path MUST be the absolute path to the .gguf file on the backend host
{
  "base_model": "unsloth/Qwen3.5-35B-A3B-GGUF",
  "model_path": "/path/to/your/model.gguf",  // ← REQUIRED: absolute path to .gguf file
  "engine": "llama_server",
  "n_gpu_layers": 10,    // ← adjust: number of layers to offload to GPU (0 = CPU only)
  "n_ctx": 262144,
  "n_batch": 512
}

// vLLM — model_path MUST be the absolute path to the model snapshot directory on the backend host
{
  "base_model": "google/gemma-4-E2B-it",
  "model_path": "/path/to/your/model/snapshot",  // ← REQUIRED: absolute path to model directory
  "engine": "vllm",
  "max_memory": null,
  "vllm_gpu_memory_utilization": 0.8,  // ← adjust to your GPU memory budget
  "vllm_dtype": "auto",
  "vllm_kv_cache_dtype": "fp8",
  "vllm_tensor_parallel_size": 1,      // ← set to number of GPUs for tensor parallelism
  "vllm_mm_image_limit": 2             // ← remove or set to null if not using multimodal
}
```

`base_model` is normalized to `model_name` when constructing `InferenceConfig`. Any field defined on `InferenceConfig` (see `config_models.py`) can be added to the JSON.

> **Fields you must fill in before use:**
>
> - `model_path` for `llama_server` and `vllm`: absolute path on the **backend host** (not the client machine). For `transformers`, a HuggingFace model ID works and will be auto-downloaded by the backend.
> - Hardware-dependent limits (`device_map`, `max_memory`, `n_gpu_layers`, `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`): adjust to match your backend hardware.

### 3.2 Fine-tune settings (`app_settings/app_settings_finetune.json`)

```jsonc
{
  "finetune_config_path": "fine_tune_configs/training_qwen3-4B_lora_fasttest.json",
}
```

`backend_url` is optional here. If omitted, fine-tune uses the shared default from `app_settings/default_backend_settings.json`.

#### 3.2.1 Training config (`fine_tune_configs/*.json`)

The `finetune_config_path` points to a training config file. Two examples are provided — edit before use:

```jsonc
{
  "model_name": "Qwen/Qwen3-4B",                        // ← HuggingFace ID or local path of base model
  "method": "lora",
  "dataset_path": "path/to/your/dataset.jsonl",          // ← REQUIRED: path to your training data
  "output_dir": "finetune_output/qwen3-4B_lora_test_fast", // ← where fine-tuned weights are saved
  "offload_folder": "./deepspeed_offload",
  ...
}
```

> **Fields you must fill in before use:**
>
> - `dataset_path`: path to your JSONL training dataset (each line: `{"prompt": "...", "completion": "..."}`).
> - `model_name`: HuggingFace model ID or local absolute path of the base model to fine-tune.
> - `output_dir`: directory where the fine-tuned adapter/model will be written (auto-created).
> - DeepSpeed fields (`use_deepspeed`, `deepspeed_profile`) only apply when the backend supports DeepSpeed; set `use_deepspeed: false` for single-GPU or CPU training.

### 3.3 Batch chat settings (`app_settings/app_settings_batch_chat.json`)

```jsonc
{
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
    "enable_thinking": false,
  },
  "prompts": [
    "Hello, please introduce yourself.",
    "Explain what RAG is in 3 bullet points.",
  ],
}
```

`prompts` must be a non-empty list of strings. Set `continue_on_error` to `false` if you want the batch to stop at the first failed prompt.

`backend_url` is optional. If omitted, the script falls back to `app_settings/default_backend_settings.json`.

If you want a config file that explicitly includes `backend_url`, see `app_settings/app_settings_batch_chat_with_backend_url_example.json`.

### 3.4 Parallel terminal batch chat settings

Three pre-built settings files are provided for different concurrency levels:

| File                                                             | Prompt Count |
| ---------------------------------------------------------------- | ------------ |
| `app_settings/app_settings_parallel_terminal_batch_chat.json`    | 8 (default)  |
| `app_settings/app_settings_parallel_terminal_batch_chat_8.json`  | 8            |
| `app_settings/app_settings_parallel_terminal_batch_chat_16.json` | 16           |

```jsonc
{
  "api_key": "your-api-key",
  "model": "trusta-ast/trusta-ast-default",
  "system_prompt": "You are a helpful assistant.",
  "stream": true,
  "temperature": 0.5,
  "top_p": 0.9,
  "max_tokens": 800,
  "extra_body": {
    "top_k": 50,
    "repetition_penalty": 1.1,
    "total_timeout": 300,
    "enable_thinking": false,
  },
  "prompts": [
    "Hello, please introduce yourself.",
    "Explain what RAG is in 3 bullet points.",
    "...",
  ],
}
```

Each file uses the same `prompts` array format. Both platform-specific launchers read `prompts` and open one terminal window per entry. Increase concurrency by switching to the `_8` or `_16` variant.

`backend_url` is optional. If omitted, the launcher falls back to `app_settings/default_backend_settings.json`.

### 3.5 Batch image chat settings (`app_settings/app_settings_batch_chat_image.json`)

```jsonc
{
  "api_key": "your-api-key",
  "model": "trusta-ast/trusta-ast-default",
  "system_prompt": "You are a helpful assistant that can analyze images.",
  "stream": false,
  "temperature": 0.5,
  "top_p": 0.9,
  "max_tokens": 1024,
  "continue_on_error": true,
  "extra_body": {
    "top_k": 50,
    "repetition_penalty": 1.1,
    "total_timeout": 300,
    "enable_thinking": false,
  },
  "tasks": [
    {
      "prompt": "Describe what you can observe in this image.",
      "images": ["./your-image.jpg"],
    },
  ],
}
```

`backend_url` is optional. If omitted, the script falls back to `app_settings/default_backend_settings.json`.

`tasks` must be a non-empty list. Each task must include:

- `prompt`: non-empty string
- `images`: non-empty list of image sources (`./local/path.jpg` or `https://...`)

## 4. Project Layout

```
ai_client.py                        # Model lifecycle + training endpoints
config_models.py                    # Pydantic models (InferenceConfig, TrainingConfig, ...)
exceptions.py                       # AIClientError

load_model_example.py               # Step 1: load a model
openai-compatible-example.py        # Step 2a: chat (text) via OpenAI SDK
openai-compatible-batch-example.py  # Step 2a-2: chat (batch) via OpenAI SDK
openai-compatible-parallel-terminal-batch-example.py  # Step 2a-3: chat (batch, Windows terminals) via OpenAI SDK
openai-compatible-parallel-terminal-batch-linux-example.py  # Step 2a-4: chat (batch, Linux terminals) via OpenAI SDK
openai-compatible-image-example.py  # Step 2b: chat (image / multimodal) via OpenAI SDK
openai-compatible-image-batch-example.py  # Step 2b-2: chat (image batch) via OpenAI SDK
unload_model_example.py             # Step 3: unload

finetune_example.py                 # Fine-tune entry point (unchanged)

helpers/
|-- config_loader.py                # JSON -> InferenceConfig / TrainingConfig
|-- finetune_initializer.py         # Fine-tune bootstrap
|-- model_loader.py                 # Load/unload orchestration (status polling)
`-- training_handler.py             # Fine-tune progress monitoring

infer_model_configs/                # Frontend-aligned load-model payloads
fine_tune_configs/                  # Training configs
app_settings/                       # Fine-tune app settings
```
