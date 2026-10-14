# AI Scaler Console

繁體中文版本：[README.zh-TW.md](README.zh-TW.md)

Headless (no-UI) Python client for the AI Scaler (Trusta AST) backend.
This project is the API-only counterpart to **Trusta-AST-Frontend**: same backend, same model lifecycle, but driven from a console / script instead of a web UI.

What it covers:

- Loading / unloading models on the backend (`/inference/load_model`, `/inference/unload_model`).
- Chatting with the loaded model via the **OpenAI-compatible** endpoint (`<backend_url>/v1/chat/completions`), text or image.
- Fine-tune training (`/training/start`, `/training/stop`, `/training/status`).

The legacy `/inference/chat` Python wrappers (`AIClient.chat`, `ChatRequest`, `ChatStreamChunk`, etc.) have been removed. All chat now goes through the OpenAI Python SDK against the backend's OpenAI-compatible endpoint, mirroring `runChatCompletions` in the frontend.

The model config files under `configs/inference/` are kept aligned with the load-model payload used by Trusta-AST-Frontend.
For more detailed API documentation, see the Trusta-AST-Backend API reference.

## 0. Environment Setup

The project is managed by **uv**:

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the package and its dependencies. `uv sync` installs `ai_scaler_console` in editable mode, so the examples can be run from any working directory:

```bash
uv sync
```

Then create your local config from the template:

```bash
cp .env.example .env
```

`.env` is not version-controlled. Only `BACKEND_URL` is required:

```ini
BACKEND_URL=http://127.0.0.1:8000
```

The optional variables are documented in `.env.example`:

| Variable      | Default                 | When you need it                                                                                      |
| ------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Always — base URL of the backend, without the `/v1` suffix.                                            |
| `API_KEY`     | `EMPTY`                 | Never, currently. The backend does not authenticate requests; the OpenAI SDK just refuses an empty key. |
| `VERIFY_SSL`  | `true`                  | Set to `false` only when the backend serves https with a self-signed certificate.                       |

## 1. Recommended No-UI Workflow

### 1.1 Load a model

Pick any config under `configs/inference/` (transformers / llama_server / vLLM are all supported, fields match the frontend payload).

> **Before running:** edit the chosen config file and set `model_path` to your actual model location. For `llama_server` and `vllm` configs this must be an absolute path on the backend host. For `transformers` configs a HuggingFace model ID works as-is.

```bash
uv run examples/model/load.py --config configs/inference/transformers.json
```

```bash
uv run examples/model/load.py --config configs/inference/transformers_offload.json
```

```bash
uv run examples/model/load.py --config configs/inference/llama.json
```

```bash
uv run examples/model/load.py --config configs/inference/vllm.json
```

To point at a different backend for one run, override `BACKEND_URL` on the command line:

```bash
uv run examples/model/load.py --config configs/inference/vllm.json --backend-url http://your-backend-host
```

### 1.1.1 Get model list

Fetch the unified model list used by the frontend's `/config/models` API:

```bash
uv run examples/model/list.py
```

```bash
uv run examples/model/list.py --json
```

The response is parsed into `ModelListResponse` with three groups plus a quick reference:

- `base_models`
- `finetuned_models`
- `llama_gguf_models`
- `last_finetuned` — raw contents of the backend's `last_finetuned_model.json`, or `null`

> All three groups come from the backend's model registry, which no longer ships with seeded base models. On a fresh backend the list is empty until models are registered there — that is expected, not a connection failure. Loading and fine-tuning do not depend on the registry: they take a HuggingFace ID or a path directly.

### 1.2 Chat (text)

After the model is loaded:

```bash
uv run examples/chat/basic.py
```

The script uses the OpenAI Python SDK against `<backend_url>/v1/chat/completions`. Fields the SDK knows natively (`messages`, `temperature`, `top_p`, `max_tokens`, `tools`, `tool_choice`, `stream` ...) are passed as normal arguments; AST-specific ones go via `extra_body`, just like the frontend does. The docstring at the top of [`examples/chat/basic.py`](examples/chat/basic.py) lists all twelve.

### 1.2.1 Batch chat (multiple prompts)

Run multiple prompts in one command:

```bash
uv run examples/chat/batch.py
```

```bash
uv run examples/chat/batch.py --settings configs/chat/batch.json
```

The script reads prompts from `configs/chat/batch.json`, sends them to `<backend_url>/v1/chat/completions` one by one, and prints each prompt, response, and per-request usage stats.

### 1.2.2 Parallel chat

To run N prompts concurrently and watch them side by side, use the parallel example. It opens a terminal UI with one pane per prompt, each pane streaming its own worker process:

```bash
uv run examples/chat/parallel.py
```

```bash
uv run examples/chat/parallel.py --settings configs/chat/parallel_16.json
```

```bash
uv run examples/chat/parallel.py --columns 4
```

Press `q` to quit.

This is a single cross-platform script — Windows, Linux, and macOS — and because the UI is a TUI rather than a desktop window, it also works over SSH. One OS process is spawned per prompt, which is what makes the workers genuinely independent clients rather than coroutines sharing one connection pool.

Set `stream` in the settings file to control response mode:

- `stream: false` → non-streaming full response per worker
- `stream: true` → token streaming per pane, with time-to-first-token reported when the worker finishes

Notes:

- `model` is fixed to `trusta-ast/trusta-ast-default` — this is the OpenAI-compatible binding name on the backend, regardless of which underlying model is loaded.
- `api_key` is required by the SDK but the backend does not authenticate requests, so any string works. Leave it out and `.env`'s `API_KEY` default (`EMPTY`) is used.

### 1.3 Chat (image / multimodal)

For multimodal models (Gemma 3 / Qwen-VL via transformers, or vLLM with `vllm_mm_image_limit` set in the load config):

```bash
uv run examples/chat/image.py
```

By default this uses the bundled sample at `examples/assets/test_image.jpg`. Point `SAMPLE_IMAGE` in the script (or the `images` field in the settings file) at your own image to try something else.

The script sends OpenAI-compatible `image_url` content parts. It demonstrates:

- local files → Base64 Data URI (auto-encoded)
- HTTP/HTTPS URLs → passed through as-is
- multiple images in a single message
- non-stream variant for full-response + usage stats

### 1.3.1 Batch image chat (multiple multimodal tasks)

Run multiple image chat tasks in one command:

```bash
uv run examples/chat/image_batch.py
```

```bash
uv run examples/chat/image_batch.py --settings configs/chat/image_batch.json
```

The script reads `tasks` from `configs/chat/image_batch.json`.
Each task contains one text prompt and one or more images (local path or URL).

### 1.4 Unload the model

```bash
uv run examples/model/unload.py
```

```bash
uv run examples/model/unload.py --backend-url http://your-backend-host
```

### 1.5 Complete workflow (load → chat → unload)

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json
```

Available workflow configs: `default.json`, `llama.json`, `vllm.json`, `transformers_offload.json`, `parallel_batch_16.json`. Each one names an inference config and a chat settings file. See [docs/WORKFLOW_CONFIG_GUIDE.md](docs/WORKFLOW_CONFIG_GUIDE.md).

### 1.6 Changing the backend URL

Resolution order, first match wins:

1. `--backend-url` on the command line (`load.py`, `unload.py`, `list.py`, `full.py`).
2. `backend_url` inside the settings JSON being used (the batch, image-batch, parallel, and workflow configs all accept it).
3. `BACKEND_URL` from `.env`.

For the common case — moving every example to a new host — edit `BACKEND_URL` in `.env` and nothing else. The shipped settings files deliberately omit `backend_url` so they inherit it.

## 2. Fine-tune

> **Requires a Linux backend.** Fine-tuning initialises a distributed process group
> even for a single-GPU run, and that rendezvous does not work on Windows. A Windows
> backend refuses with `501` and a message saying so; a WSL2 distro counts as Linux.
> The console itself runs anywhere — what matters is the OS of the backend it points
> at, which `GET /health` reports as `platform` and `training_supported`. Inference,
> chat and GGUF conversion are unaffected.

```bash
uv run examples/finetune/train.py
```

```bash
uv run examples/finetune/train.py --settings configs/finetune/run.json
```

```bash
uv run examples/finetune/train.py --force
```

`--force` interrupts an existing training session before starting a new one.

### 2.1 Following the training log

After starting, the script follows the backend's structured training log over SSE
(`/training/{job_id}/log/stream`), printing each lifecycle phase, metric, and error as
the backend emits it. The job id comes back from `/training/start` as `result.session_id`.

```bash
uv run examples/finetune/train.py --show-resource
```

```bash
uv run examples/finetune/train.py --poll
```

- `--show-resource` also prints the per-second GPU / DRAM / SSD samples, which are
  suppressed by default because they arrive once a second.
- `--poll` falls back to sampling `/training/status` every 3 seconds. That is also what
  happens automatically when the backend returns no job id.

Held-out evaluation metrics only exist on the stream — they are interleaved as `metric`
events with `split: "eval"` and never reach `/training/status`. The console tags them
`[eval]` so they are not mistaken for the training curve. A dropped connection resumes
from the last `seq` via `Last-Event-ID`; a 404 means the job's log directory is gone and
is treated as fatal rather than retried.

To read a finished job's log, or to backfill after a drop, use the polling endpoint
directly via `AIClient.get_training_log(job_id, since=...)`, and
`AIClient.get_training_error_details()` for the full traceback.

## 3. Configuration Files

The shipped files under `configs/` are the reference — open the one you intend to use.
They are plain JSON and cannot carry comments, so what you have to edit before running is listed below.

### 3.1 Inference model configs (`configs/inference/*.json`)

These mirror the frontend's load-model payload; the `engine` field selects the schema.
`base_model` is normalized to `model_name` when constructing `InferenceConfig`. Any field defined on `InferenceConfig` (see `ai_scaler_console/config_models.py`) may be added.

| Config                      | `model_path`                                                | Tune for your hardware                                                |
| --------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `transformers.json`         | HuggingFace ID (backend downloads it) or a local directory   | `quantization`, `device_map`                                          |
| `transformers_offload.json` | same                                                        | `max_memory` — the GPU/CPU split                                      |
| `llama.json`                | **required**: absolute path to the `.gguf` on the backend    | `n_gpu_layers` (`0` = CPU only), `n_ctx`                              |
| `vllm.json`                 | **required**: absolute path to the model dir on the backend  | `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`             |

> Absolute paths are on the **backend host**, not the machine running these scripts.
> `vllm_mm_image_limit` must be set for multimodal input; remove it or set it to `null` otherwise.

### 3.2 Fine-tune settings (`configs/finetune/run.json`)

`run.json` holds one field, `finetune_config_path`, pointing at a training config under `configs/finetune/training_*.json`. `backend_url` is accepted but optional; omitted means `.env`.

Two training configs are provided as examples. Before use, fill in:

- `dataset_path`: a `.json` / `.jsonl` dataset file (each line: `{"prompt": "...", "completion": "..."}`), or a Hugging Face Hub dataset id. Anything that is not a local `.json`/`.jsonl` path is treated as a Hub id — it cannot carry a config/subset or split selector, and skips the local-file format sniffing, so set `prompt_field`/`completion_field`/`text_field` (or `messages_field`) to match the Hub dataset's columns.
- `model_name`: HuggingFace ID or local absolute path of the base model.
- `output_dir`: where the fine-tuned adapter/model is written (auto-created).
- `use_deepspeed`: set `false` for single-GPU or CPU training; the DeepSpeed fields only apply when the backend supports it.

> Training runs on the backend, so `dataset_path` and `output_dir` resolve on the **backend host** — the shipped configs point at the backend's own `dataset/dataset.jsonl`.

### 3.3 Chat settings (`configs/chat/*.json`)

| File               | Used by                        | Shape                     |
| ------------------ | ------------------------------ | ------------------------- |
| `batch.json`       | `examples/chat/batch.py`       | `prompts` (4 prompts)     |
| `parallel_8.json`  | `examples/chat/parallel.py`    | `prompts` (8 prompts)     |
| `parallel_16.json` | `examples/chat/parallel.py`    | `prompts` (16 prompts)    |
| `image_batch.json` | `examples/chat/image_batch.py` | `tasks` (prompt + images) |

Every field and its default lives on `ChatSettings` in [ai_scaler_console/helpers/chat.py](ai_scaler_console/helpers/chat.py) — read it there rather than trusting a copy in this file. Notes that are not obvious from the dataclass:

- `prompts` must be a non-empty list of strings; `tasks` (image variant) a non-empty list of `{prompt, images}`, where each image is a local path or an `https://` URL.
- `continue_on_error: false` stops a batch at the first failed prompt.
- `backend_url` and `api_key` are optional; omitted means the `.env` values.
- AST-specific knobs (`top_k`, `repetition_penalty`, `total_timeout`, `enable_thinking`, ...) go under `extra_body`.

## 4. Project Layout

```
ai_scaler_console/              # The installable package
|-- client.py                   #   AIClient: model lifecycle + training endpoints
|-- config_models.py            #   Pydantic models (InferenceConfig, TrainingConfig, ...)
|-- exceptions.py               #   AIClientError and friends
|-- settings.py                 #   Runtime config resolved from .env
|-- openai_client.py            #   OpenAI-compatible client factories
`-- helpers/
    |-- chat.py                 #   Chat settings loading + single-prompt execution
    |-- config_loader.py        #   JSON -> InferenceConfig / TrainingConfig
    |-- finetune_initializer.py #   Fine-tune bootstrap
    |-- model_loader.py         #   Load/unload orchestration (status polling)
    |-- parallel_ui.py          #   Textual pane grid for the parallel example
    `-- training_handler.py     #   Fine-tune progress monitoring

examples/
|-- chat/                       # basic, batch, image, image_batch, parallel
|-- model/                      # load, unload, list
|-- finetune/                   # train
|-- workflow/                   # full (load -> chat -> unload)
`-- assets/                     # test_image.jpg for the vision example

configs/
|-- chat/                       # Prompt sets for the chat examples
|-- finetune/                   # Fine-tune run settings + training configs
|-- inference/                  # Frontend-aligned load-model payloads
`-- workflow/                   # End-to-end workflow definitions

docs/                           # WORKFLOW_CONFIG_GUIDE.md (+ .zh-TW.md)
internal/                       # Internal benchmarks, not part of the public client
```

## 5. Development

Lint, format, and type-check settings live in `pyproject.toml` and match trust-hmst-backend.

```bash
uv run ruff check
```

```bash
uv run ruff format
```

```bash
uv run pyright
```

To run all of it automatically on every commit:

```bash
uv run pre-commit install
```

`internal/` is excluded from ruff and pyright. It holds the long-prompt KV-offload benchmark and a long-running stability probe — internal tooling that is not part of the client and is not published.

### Tests

```bash
uv run pytest
```

Everything under `tests/unit/` runs offline: HTTP is faked with `httpx2.MockTransport`, and the parallel-chat tests pin their workers at an unreachable address so the result does not depend on whether a backend happens to be running locally.

`tests/integration/` needs a live backend and is deselected by default. Start the backend, load a model, then:

```bash
uv run pytest --integration
```

Integration tests that need inference skip themselves when no model is loaded, so the transport-level ones still run against a bare backend.
