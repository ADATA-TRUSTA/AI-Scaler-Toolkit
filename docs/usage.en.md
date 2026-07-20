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
- `POST /inference/estimate_memory`
- `POST /inference/stop_generation`

### Training Management

- `POST /training/start`
- `GET /training/status`
- `POST /training/stop`

### Model Configuration and Downloading

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`

### System Information

- `GET /system/resources`

---

## Frontend Notes

The project already contains static frontend assets:

- `src/frontend/dist`
- `src/frontend/dist_client`

The backend currently mounts `src/frontend/dist` at `/frontend/`, and the root path `/` automatically redirects to `/frontend/` when `index.html` exists.

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

