# Usage & Verification

[← Back to README](../README.md)

## The application

The window holds the web interface: chat, the model list (download, register, convert to GGUF) and, on Linux, fine-tuning. The **Show chat** button beside the settings column opens the chat panel.

Closing the window does **not** stop the service: the application keeps running in the tray (the notification area on Windows), because other applications may be using its API. The tray icon's menu:

| Item | Does |
|---|---|
| Open window | Brings the window back |
| Open full interface in browser | Opens the same interface in your browser |
| Settings | The settings window (below) |
| Quit | Stops the service and the application |

Only one copy runs at a time; starting it again brings the existing window forward.

### Settings

| Section | What it is for |
|---|---|
| Startup | Start automatically with Windows / when you log in (on by default), so the service is there for other applications. It does nothing at login until the first setup has been completed |
| Ports | The application port, `8571` by default. Change it if something else uses it |
| Network | **Let another computer use this machine's model**, and the API key other computers need (below) |
| Environment | **Reinstall the environment**: removes the installed Python packages and GPU runtime and sets them up again. Your models are kept. Use it when setup or a load fails in a way that a restart does not fix |
| Updates | Shows the installed version. **Check now** asks GitHub for a newer release; **Update now** downloads it, checks it, installs it and reopens the application |
| Components | Optional third-party tools, installed for your account (below) |

**Let another computer use this machine's model** (Settings → Network): opens the service to your local network so other computers can use this model. Other computers get the API only, and only with the **API key** shown there; the web interface and the API docs stay on this computer. The key is made the first time you turn this on. **Copy** it, **Use my own key…** to set one you choose (at least 12 characters: letters, digits and `. _ ~ + / = -`), or **Make a new key** if it was shared too widely; other computers using the old key then stop working until they get the new one. Anyone with the key can chat with the model and load or unload models, so turn this on only on a network you trust. If a firewall blocks it, **Cannot connect?** there offers to add a rule.

### Components

- **Hermes Agent**: an AI agent that can use this machine's model. **Install**, then **Open**. **Config AST Endpoint** writes this application's endpoint into Hermes's settings so Hermes uses whatever model is loaded here (load a model first). If you already chose another provider in Hermes, it leaves your choice alone and shows the commands to switch by hand. For Hermes on another computer, turn on Settings → Network; the card then shows the commands to run there, with the address and key already in them.
- **ComfyUI**: image and video generation. Pick the card it should run on and press **Use this card** (or **Install and use**); the CPU is listed last as a fallback. The image and video skills start ComfyUI themselves on `127.0.0.1:8188`.

---

## Connecting other applications

The service speaks the OpenAI API. In any app or coding agent that takes an "OpenAI-compatible" provider:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1` |
| Model | `trusta-ast-default` (whatever model is loaded), or the loaded model's own name from `/v1/models` |
| API key | any string on this computer, which needs none; from another computer, the key shown in Settings → Network |

The service answers with the model that is loaded and ignores the name asked for, so `trusta-ast-default` keeps working when you load a different model.

---

## Checking it from the command line

The examples use the application's port, `8571`. A service started from source uses `SERVICE_PORT` from its `.env` (`8000` in `.env.example`).

### Health check

```bash
curl http://127.0.0.1:8571/health
```

```powershell
Invoke-RestMethod http://127.0.0.1:8571/health
```

### Registered models

```bash
curl http://127.0.0.1:8571/config/models
```

### Load a model, then chat

Loading runs in the background: `POST /inference/load_model` returns at once, and `GET /inference/status` reports `"loaded": true` when it is ready. A Hugging Face model:

```bash
curl http://127.0.0.1:8571/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-4B"}'
```

A GGUF file, with llama.cpp:

```bash
curl http://127.0.0.1:8571/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/model.gguf", "model_name": "my-model", "engine": "llama_server", "n_gpu_layers": -1, "n_ctx": 8192}'
```

Then:

```bash
curl http://127.0.0.1:8571/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trusta-ast-default",
    "messages": [{"role": "user", "content": "Hello, please briefly introduce yourself."}],
    "stream": false
  }'
```

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8571/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"trusta-ast-default","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

The full API, with request bodies, is at `http://127.0.0.1:8571/docs`.

---

## API routes

### Basic

- `GET /`
- `GET /health`
- `GET /v1/models` — the loaded model, under its own name
- `GET /v1/models/{model}` — one model; accepts `trusta-ast-default` and the model's other aliases
- `POST /v1/chat/completions`

### Inference management

- `POST /inference/load_model`
- `POST /inference/unload_model`
- `GET /inference/status`
- `POST /inference/stop_generation`
- `GET /inference/error_details`
- `POST /inference/cleanup_generation_memory`
- `POST /inference/force_cleanup_gpu`
- `POST /inference/chat` — **deprecated**, returns HTTP 410; use `/v1/chat/completions`

### Memory estimation

- `POST /inference/estimate_memory` — parameter-count based, for Hugging Face checkpoints
- `GET /inference/estimate_memory/{model_name}`
- `POST /inference/estimate_memory/gguf` — GGUF / llama.cpp sizing
- `POST /inference/estimate_memory/gguf/check`
- `POST /inference/estimate_memory/gguf/plan`
- `POST /inference/estimate_memory/gguf/recommend`
- `POST /inference/estimate_memory/gguf/sweep`

For GGUF models use the `gguf` family; `/inference/estimate_memory` is parameter-count based and is not the right tool for them.

### Training (Linux)

- `POST /training/start`
- `GET /training/status`
- `GET /training/status/{session_id}/history`
- `POST /training/stop`
- `GET /training/error_details`
- `POST /training/force_cleanup_gpu`
- `GET /training/{job_id}/log` — structured event backfill
- `GET /training/{job_id}/log/stream` — SSE live log

### Enumerations and config examples

- `GET /config/quantization_types`
- `GET /config/offload_types`
- `GET /config/training_methods`
- `GET /examples/inference`
- `GET /examples/training`
- `GET /examples/conversion`

### Models: configuration, download, conversion

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`
- `POST /config/models/download/{task_id}/cancel`
- `GET /config/models/downloads` — all download tasks
- `DELETE /config/models/{label}` — accepts `?delete_files=`
- `POST /config/models/refresh_context_lengths`
- `POST /config/models/convert`
- `GET /config/models/convert/{job_id}`

### Documents (RAG)

- `POST /rag/docs`
- `GET /rag/docs`
- `DELETE /rag/docs/{doc_id}`
- `GET /rag/search`

### System information

- `GET /system/resources` — CPU, memory (with each DRAM module), GPUs and disks

---

## Logs

| | Windows | Linux |
|---|---|---|
| Application | `%LOCALAPPDATA%\AIScalerToolkit\logs\` | `~/.local/share/ai-scaler-toolkit/logs/` |

- `launcher.log` — the application: startup, updates, components
- `provision.log` — the last setup run
- `service.log`, rotated daily as `service.log.YYYY-MM-DD` — the service and the engines it runs

A service started from source writes `service.log` to the `LOG_DIR` in its `.env`.
