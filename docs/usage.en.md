# Usage & Verification

[← Back to README](../README.md)

## How to Verify After Startup

### Health Check

#### Linux

```bash
curl http://127.0.0.1:8000/health
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### View the Available Model List

#### Linux

```bash
curl http://127.0.0.1:8000/config/models
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/config/models
```

### Load a Model First, Then Send a Chat Request

Before chatting, call `/inference/load_model` to load a model. If the model has not finished loading, chat requests may return a model-not-ready response.

#### Linux

```bash
curl http://127.0.0.1:8000/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen3-4B"
  }'
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/inference/load_model `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model_name":"Qwen/Qwen3-4B"}'
```

After confirming the model is fully loaded, send a chat request:

#### Linux

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trusta-ast-default",
    "messages": [{"role": "user", "content": "Hello, please briefly introduce yourself."}],
    "stream": false
  }'
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"trusta-ast-default","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

---

## Common API Routes

### Basic

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`/v1/models` returns the completion alias `trusta-ast-default` when a model is loaded. Use the same alias in `/v1/chat/completions` requests.

### Inference Management

- `POST /inference/load_model`
- `POST /inference/unload_model`
- `GET /inference/status`
- `POST /inference/stop_generation`
- `GET /inference/error_details`
- `POST /inference/cleanup_generation_memory`
- `POST /inference/force_cleanup_gpu`
- `POST /inference/chat` — **deprecated**, returns HTTP 410; use `/v1/chat/completions`

### Memory Estimation

- `POST /inference/estimate_memory` — parameter-count based, for HF checkpoints
- `GET /inference/estimate_memory/{model_name}`
- `POST /inference/estimate_memory/gguf` — GGUF / llama.cpp sizing
- `POST /inference/estimate_memory/gguf/check`
- `POST /inference/estimate_memory/gguf/plan`
- `POST /inference/estimate_memory/gguf/recommend`
- `POST /inference/estimate_memory/gguf/sweep`

For GGUF models use the `gguf` family; `/inference/estimate_memory` is parameter-count based and is not the right tool for them.

### Training Management

- `POST /training/start`
- `GET /training/status`
- `GET /training/status/{session_id}/history`
- `POST /training/stop`
- `GET /training/error_details`
- `POST /training/force_cleanup_gpu`
- `GET /training/{job_id}/log` — structured event backfill
- `GET /training/{job_id}/log/stream` — SSE live log

### Enumerations and Config Examples

- `GET /config/quantization_types`
- `GET /config/offload_types`
- `GET /config/training_methods`
- `GET /examples/inference`
- `GET /examples/training`
- `GET /examples/conversion`

### Model Configuration and Downloading

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`
- `GET /config/models/downloads` — all download tasks
- `DELETE /config/models/{label}` — accepts `?delete_files=`
- `POST /config/models/refresh_context_lengths`
- `POST /config/models/convert`
- `GET /config/models/convert/{job_id}`

### System Information

- `GET /system/resources`

---

## Frontend Notes

The project already contains static frontend assets:

- `frontend/dist`
- `frontend/dist_client`

The backend currently mounts `frontend/dist` at `/frontend/`, and the root path `/` automatically redirects to `/frontend/` when `index.html` exists.

Therefore:

- Node.js is not required
- No separate frontend build is required
- As long as the service starts successfully, you can directly open `http://127.0.0.1:8000/`

---

## Logging Configuration

```dotenv
LOG_TO_FILE=true
LOG_DIR=<project>/logs
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
```

When enabled, this generates:

- `service.log`
- `service.log.YYYY-MM-DD` (daily rotation)

---

