# Trusta AST Backend

Trusta AST Backend is a FastAPI-based LLM backend service that provides:

- Model loading / unloading
- OpenAI-compatible chat interface
- Inference and streaming responses
- Training job launch and status queries
- Model download and conversion
- RAG document management and retrieval
- Frontend static page mounting

This document covers both **Linux** and **Windows** platforms, with steps split out where they differ.

> **Linux example path**: `/home/test/project/Trusta_AST_Backend`
> **Windows example path**: `C:\Users\<user>\project\Trusta_AST_Backend` (choose your own)

---

## 1. Project Structure

```text
Trusta_AST_Backend/
├─ scripts/
│  ├─ linux/
│  │  ├─ run_service.sh
│  │  ├─ setup_env.sh
│  │  └─ stop_service.sh
│  ├─ windows/
│  │  ├─ run_service.bat
│  │  └─ setup_env.ps1
│  ├─ docker/                   # placeholder; container assets are internal-only
│  └─ gguf_recommend_report.py
├─ dataset/
├─ docs/
│  ├─ inference_manual.html     # Inference service manual (.zh-TW.html = 繁體中文)
│  ├─ finetune_manual.html      # Fine-tune service manual (.zh-TW.html = 繁體中文)
│  ├─ benchmarks.md             # Measured TPS / TTFT per GPU
│  ├─ gguf_memory_estimation.md
│  ├─ training-log-frontend-integration.md
│  └─ site-map.html             # Frontend SPA page architecture
├─ logs/                        # created at runtime
├─ runtime_data/                # created at runtime
├─ Trusta-AST-Frontend/         # gitignored; absent on a fresh clone
│  ├─ dist/                     # Trusta-styled build (the only one served)
│  └─ dist_client/              # ADATA-styled build
├─ service/
│  ├─ app.py
│  └─ settings.py
├─ tests/
│  ├─ README.md                 # tiers, markers, result layout
│  ├─ unit/  smoke/  e2e/  xpu/  stress_tests/
├─ pyproject.toml               # uv project root (deps + ruff + pyright config)
├─ uv.lock
├─ pytest.ini
├─ .pre-commit-config.yaml
├─ .python-version
├─ .venv/                       # created by setup_env at the repo root
├─ .env.example
├─ AGENTS.md                    # instructions for AI coding agents
└─ README.md
```

---

## 2. System Requirements

### Basic requirements (both platforms)

| Tool | Description |
|------|------|
| Git | Version control; also used by `setup_env` to sparse-fetch the GGUF convert scripts |
| Python 3.12+ | Runs the service itself |
| `uv` | Python package / virtual environment management |
| C/C++ toolchain | Optional, only needed on Linux by the training packages built from source (deepspeed on any accelerator; causal-conv1d only with `--extra cuda`); llama is prebuilt now, so nothing to compile |

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y git curl build-essential python3 python3-venv python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, re-login to your shell, or manually load `uv`:

```bash
source "$HOME/.local/bin/env"
```

### Windows

1. **Git for Windows**: <https://git-scm.com/download/win> (or `winget install Git.Git`)
2. **Python 3.12+**: <https://www.python.org/downloads/windows/> (or `winget install Python.Python.3.12`)
   - Check **Add Python to PATH** during installation
3. **uv** (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
> Note: Windows does not need Visual Studio Build Tools / CMake — llama comes from llama.cpp's own release binary and GGUF conversion is pure Python. Linux does need a toolchain, because llama.cpp publishes no Linux CUDA binary and setup_env builds it from source there (see 5.1).

### Optional requirements (both platforms)

- NVIDIA GPU: requires matching drivers and CUDA runtime
- Intel XPU / iGPU (Linux **and** Windows): requires a recent enough Intel graphics driver
- Frontend build: requires [bun](https://bun.sh) (the frontend pins it via `packageManager`)
- Private Hugging Face models: requires `HF_TOKEN`

> **Note**: vLLM only supports **Linux + CUDA**; the vLLM engine is not available on Windows.

> **Intel XPU and the graphics driver.** You do **not** need the oneAPI Base Toolkit: the SYCL
> runtime is installed by `uv sync --extra xpu` as part of the torch wheel. What you do need is a
> graphics driver new enough for *that* runtime — each torch version bundles a different oneAPI
> release, so upgrading torch can raise the driver requirement. When they do not match,
> `torch.xpu.is_available()` still returns `True` and the failure only appears on the first real
> kernel, either as an error (`could not make an engine with allocator`) or as a hang during
> kernel compilation. `setup_env` therefore runs `scripts/xpu_smoke.py` after the sync — a real
> GEMM plus one training step, under a timeout — so a mismatch is caught at setup time instead of
> during your first inference or fine-tune. Bypass it with `-SkipXpuCheck` /
> `TRUSTA_SKIP_XPU_CHECK=1`, or raise the 300s kill timeout with `-XpuCheckTimeoutSec` /
> `TRUSTA_XPU_CHECK_TIMEOUT`.
>
> **Minimum driver — the thing to compare differs per platform.** torch 2.13.0+xpu bundles
> the oneAPI 2026.0.0 runtime; Windows ships one graphics driver, Linux ships a
> compute-runtime package, and their version schemes are unrelated:
>
> | | What to check | Verified working |
> |------|------|------|
> | Windows | Intel Graphics DCH driver — `Get-CimInstance Win32_VideoController` | `32.0.101.8974` (needs ≥ `32.0.101.8801`; `32.0.101.6129` fails) |
> | Linux | `intel-opencl-icd` — `dpkg-query -W intel-opencl-icd` | `26.27.39122.14` from `ppa:kobuk-team/intel-graphics` |
>
> Windows drivers: [Intel® Arc™ Graphics - Windows*](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html).
> On Ubuntu 24.04 the distro's own `intel-opencl-icd` (`23.43.x`) predates this runtime, so
> add the [Intel graphics PPA](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html)
> and install `libze-intel-gpu1 libze1 intel-opencl-icd`. `xpu_smoke.py` prints both the
> bundled runtime and the installed driver, so its output says which side is behind.
> Verified on WSL2 as well, where the GPU is reached through `/dev/dxg`.

---

## 3. Cloning the Project from GitLab

llama.cpp (`service/utils/llama.cpp`, optional, only needed for the GGUF convert tooling) is fetched on demand by `setup_env`, **so a plain clone is enough — `--recursive` is not needed**.

#### Linux

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone <YOUR_GITLAB_REPOSITORY_URL> Trusta_AST_Backend
cd Trusta_AST_Backend
```

#### Windows (PowerShell)

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITLAB_REPOSITORY_URL> Trusta_AST_Backend
cd Trusta_AST_Backend
```

> `setup_env` installs the llama tooling (needed for llama-server inference and GGUF
> conversion) **by default**, but only what is missing — an existing binary or convert
> checkout is left untouched. No compiler or SDK is required. To opt out, or to force a
> reinstall, see 5.1.

---

## 4. Required Setup After First Clone

### 4.1 Create required folders

#### Linux

```bash
mkdir -p logs .cache/huggingface runtime_data/inference_offload runtime_data/finetune_output
```

#### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force logs, .cache\huggingface, runtime_data\inference_offload, runtime_data\finetune_output
```

### 4.2 Create `.env`

At startup the project loads `.env` from the repo root, and **falls back to
`.env.example` when `.env` is missing** (`service/settings.py`). Skipping this
step therefore does not fail loudly — the service quietly runs on the template's
values. Copy the template and edit it:

#### Linux

```bash
cp .env.example .env
```

#### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

Then open `.env` in a text editor. **At minimum, we recommend setting the following**:

#### Linux example

```dotenv
# ===== Basic service settings =====
# SERVICE_HOST: 127.0.0.1 allows local connections only (recommended); 0.0.0.0 allows external access (needs a reverse proxy)
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

# ===== Cache and runtime data =====
HF_HOME=/home/test/project/Trusta_AST_Backend/.cache/huggingface
LOG_DIR=/home/test/project/Trusta_AST_Backend/logs

# ===== Logging =====
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true

# ===== Optional: private Hugging Face models =====
# HF_TOKEN=hf_xxxxxxxxxxxxx

# ===== Optional: llama server (setup_env builds it into the project) =====
# LLAMA_SERVER_BINARY=/home/test/trusta-ast-backend/service/utils/llama-bin/llama-server
# LLAMA_SERVER_URL=http://127.0.0.1:5001

# ===== Optional: Redis =====
# REDIS_HOST=127.0.0.1
# REDIS_PORT=6379
# REDIS_DB=0
```

#### Windows example

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

HF_HOME=C:\Users\<user>\project\Trusta_AST_Backend\.cache\huggingface
LOG_DIR=C:\Users\<user>\project\Trusta_AST_Backend\logs

LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true

# HF_TOKEN=hf_xxxxxxxxxxxxx

# setup_env unpacks llama.cpp's release into the project
# LLAMA_SERVER_BINARY=C:\path\to\trusta-ast-backend\service\utils\llama-bin\llama-server.exe
# LLAMA_SERVER_URL=http://127.0.0.1:5001
```

> Paths can use forward slash `/` or backslash `\` — Python's `pathlib` supports both.

> **Moving `SERVICE_PORT` off 8000 takes the frontend with it.** The built page reads
> its own `config.json` at boot and calls whatever `apiBaseUrl` says; this service only
> serves that file, it does not rewrite it. Leave them disagreeing and the page loads
> normally and reaches nothing. The copy being served is build output —
> `<frontend>/dist/config.json` — so edit that for a local run, and
> `public/<brand>/config.json` in the frontend repository for a change that survives
> the next build. (The desktop installer keeps the two in step by itself.)

### 4.3 Changing the Hugging Face cache path

Just change `HF_HOME` in `.env`:

```dotenv
# Linux
HF_HOME=/data/huggingface-cache

# Windows
HF_HOME=D:\huggingface-cache
```

Restart the service after changing it.

### 4.4 Commonly adjusted path settings

| Variable | Purpose | Default |
|------|------|--------|
| `HF_HOME` | Hugging Face models and cache | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS cache | `<project>/service/caches/tiktoken` |
| `LOG_DIR` | Log output directory | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | Path to the llama executable | `service/utils/llama-bin/llama-server`; `llama-server.exe` on Windows |
| `VLLM_SERVER_PROJECT_DIR` | Project dir whose `.venv` holds vLLM (Linux only); override when the venv is mounted elsewhere | the repo root |

Rule of thumb: **change `.env` first**; only modify `service/settings.py` when you need to change the project-wide defaults.

### 4.5 dmidecode permission setup (Linux, optional)

The backend's `/system/resources` API calls `dmidecode` to get DRAM specs. In non-root environments this requires sudo privileges.

**Recommended approach: add a sudoers rule to run the specific command passwordless**

```bash
sudo visudo -f /etc/sudoers.d/trusta-dmidecode
```

Add the following line (replace `<user>` with the account actually running the service):

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

If you skip this step, the API still works, but the DRAM spec field will be empty.

> This setting is not needed on Windows — the related `dmidecode` functionality is automatically skipped.

---

## 5. Setting Up the Python Environment

The Python virtual environment is created at `.venv` in the repo root.

### Linux

#### Option A: Automatic detection script (recommended)

The script automatically selects `cuda` or `xpu`, and also creates the vLLM isolated environment on CUDA hosts:

> Detection order is `nvidia-smi` → `clinfo`/`/dev/dri` → **fall back to `cuda`**.
> A machine with no GPU at all therefore ends up on `--extra cuda`; there is no
> CPU-only option. Pass `TRUSTA_ACCEL` explicitly if the guess is wrong.

```bash
cd /home/test/project/Trusta_AST_Backend
bash scripts/linux/setup_env.sh
```

Manually specify the accelerator backend:

```bash
TRUSTA_ACCEL=cuda bash scripts/linux/setup_env.sh
# or
TRUSTA_ACCEL=xpu bash scripts/linux/setup_env.sh
```

Skip setting up the vLLM environment:

```bash
TRUSTA_SETUP_VLLM=0 bash scripts/linux/setup_env.sh
```

#### Option B: Manual setup

```bash
cd /home/test/project/Trusta_AST_Backend
uv sync --extra cuda    # NVIDIA GPU
# or
uv sync --extra xpu     # Intel XPU
```

> On Linux, `--extra cuda` additionally pulls `causal-conv1d`, which builds a CUDA
> extension from source and has no Intel equivalent. `--extra xpu` deliberately skips
> it, so an Intel host needs no CUDA toolchain.

#### Activate the virtual environment

```bash
source /home/test/project/Trusta_AST_Backend/.venv/bin/activate
```

---

### Windows

Use the PowerShell setup script, which mirrors `setup_env.sh`:

```powershell
cd C:\Users\<user>\project\Trusta_AST_Backend
.\scripts\windows\setup_env.ps1 -Accel cuda
```

> Without `-Accel`, the script probes for `nvidia-smi` and falls back to `xpu`
> (Intel Arc / iGPU) when it finds none. There is no CPU-only install: `torch`
> is not a base dependency, it only comes from the `cuda` / `xpu` extras.
>
> To drive `uv` yourself instead, always pass an extra — `uv sync --extra cuda`
> or `uv sync --extra xpu`. A bare `uv sync` resolves the environment *without*
> those extras, which downgrades `torch` to the CPU build and uninstalls
> `torchvision` / `torchaudio`.
>
> vLLM does not support Windows, so no extra vLLM environment is needed.

#### Activate the virtual environment (PowerShell)

```powershell
C:\Users\<user>\project\Trusta_AST_Backend\.venv\Scripts\Activate.ps1
```

If you hit an execution policy error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Activate the virtual environment (cmd)

```cmd
C:\Users\<user>\project\Trusta_AST_Backend\.venv\Scripts\activate.bat
```

---

### Enabling the pre-commit checks

Every developer runs this once locally after the environment is ready. From then on `git commit`
automatically runs ruff (lint + format) and the uv-lock check, and blocks the commit on failure:

```bash
pre-commit install
```

- The config lives in `.pre-commit-config.yaml`; the ruff rules are under `[tool.ruff]` in `pyproject.toml`.
- Without the installed hook local commits are unchecked, so do not bypass it with `git commit --no-verify`.

---

### 5.1 If you want to use llama-server (inference) or GGUF conversion

`setup_env` sets the llama tooling up **by default**, installing two things. Where the binary comes from differs by platform: **Windows needs no toolchain**, **Linux builds from source** and so needs `build-essential`, `cmake` and the CUDA Toolkit.

- **Inference / quantization**: `llama-server`, `llama-quantize` and `llama-fit-params` in `service/utils/llama-bin`. The llama build is **decoupled** from the torch accelerator and selected by `-LlamaBackend` (Windows) / `TRUSTA_LLAMA_BACKEND` (Linux): `auto` (default) = CUDA when an NVIDIA card is present, otherwise Vulkan; `vulkan` forces the generic build that "sees every Intel/AMD/NVIDIA card"; `cpu` builds without any GPU backend. There is no probing — the asset (Windows) or the CMake flag (Linux) is chosen outright, so the backend you asked for is the backend you get. Whatever is installed is then checked by `scripts/llama_backend.py` before it is recorded in `.env`; a failure prints four options and stops, and `-AllowLlamaFallback` / `TRUSTA_LLAMA_ALLOW_FALLBACK=1` lets an unattended run accept a degraded binary instead. The version is pinned to `b10107`, overridable with `TRUSTA_LLAMA_VERSION` (Linux) or `-LlamaVersion` (Windows). Install location: `service/utils/llama-bin`; override it with `LLAMA_SERVER_BINARY` in `.env`.
- **GGUF conversion**: a sparse checkout pulls only `convert_hf_to_gguf.py` / `convert_lora_to_gguf.py` / `conversion/` / `gguf-py` into `service/utils/llama.cpp` (~1.8MB of scripts, ~2.7MB on disk with the shallow git metadata; pure Python, no build step — the C++ sources are not fetched).

Each part is installed **only when missing**, so re-running `setup_env` is cheap and
non-destructive:

| Mode | Linux | Windows | Behaviour |
|---|---|---|---|
| auto *(default)* | — | — | Install only what is absent; leave anything already there alone |
| skip | `TRUSTA_INSTALL_LLAMA=0` | `-Llama skip` | Do nothing at all |
| force | `TRUSTA_INSTALL_LLAMA=1` | `-Llama force` | Reinstall / re-fetch even if present |

Both sides accept the same spellings for the environment variable (`1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off`), and the environment variable wins over the Windows parameter. `-InstallLlama` is a deprecated alias for `-Llama force`; passing it together with an explicit `-Llama` is an error rather than a silent override. Every value is validated **before** `uv sync`, so a typo fails in a second instead of after the whole dependency install.

The binary check honours `LLAMA_SERVER_BINARY` — from the environment or from `.env` —
falling back to the default install location and then to `llama` on `PATH`. A server counts
as present only together with `llama-quantize` and `llama-fit-params` — beside it, or wherever
`LLAMA_QUANTIZE_BIN` / `LLAMA_FIT_PARAMS_BIN` point (a unified `llama` needs neither). When
they are missing, an older install in `service/utils/llama-bin` or a prebuilt found on `PATH`
is replaced by the full pinned set in `service/utils/llama-bin` (built on Linux, the release on
Windows; a `PATH` binary is not deleted, just no longer used). Only a server set explicitly
through `LLAMA_SERVER_BINARY` is never replaced: the missing tools go into
`service/utils/llama-tools` instead, and their paths are recorded in `.env`. Either way the
existing server keeps working if that step fails, and the summary's `llama tools` line says
what is still missing. The convert
tooling counts as present only when `convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`
and `gguf-py/` are all there, so a half-finished checkout is repaired rather than
skipped.

> **Already have your own llama?** Point `LLAMA_SERVER_BINARY` in `.env` (or in the
> environment) at your binary and auto mode will detect it and never touch it (missing tools
> are added under `service/utils/llama-tools`, never beside your binary). Only
> `force` overwrites, and the official installer writes to the fixed location above — so
> a copy installed earlier by that same installer is what gets replaced. (A self-built
> `llama-cli` / `llama-server` has a different filename and is unaffected.)

On multi-GPU hosts (e.g. Intel iGPU + NVIDIA), pin the target card with `llama_server_device` (maps to `--device`, e.g. `Vulkan1`); `--list-devices` shows the available Vulkan devices. Note that listing and selecting non-NVIDIA cards (Intel/AMD) requires the **Vulkan build** (`LlamaBackend=vulkan`) — the CUDA build only lists NVIDIA devices such as `CUDA0`.

Why not `ggml-org/llama-install.sh`: <https://github.com/samhong5668/llama-bench-lab>

---

### 5.2 If you want to use the vLLM engine (Linux + CUDA only)

vLLM is an extra of this project (`--extra vllm`), installed into the same `.venv` as the
service, currently at 0.27.1.

Sharing that `.venv` means vLLM's requirements drive three of this project's pins:
`torch` 2.13.0, `fastapi` 0.136.3, and `numpy` 2.3.5. **Upgrading vLLM is therefore a
whole-project change** — check what the new release does to those three first, and expect
to re-run the test suite and the XPU check.

`setup_env.sh` adds the extra automatically on CUDA hosts. To do it manually:

```bash
cd /home/test/project/Trusta_AST_Backend
uv sync --extra cuda --extra vllm
```

---

## 6. Starting the Service

### Linux

```bash
cd /home/test/project/Trusta_AST_Backend
bash scripts/linux/run_service.sh
```

The script checks whether `.venv` exists, then starts the service with `python -m service.app`.

If you need the `/system/resources` API to read full DRAM info via `dmidecode`, you can start with elevated privileges instead:

```bash
cd /home/test/project/Trusta_AST_Backend
sudo bash scripts/linux/run_service.sh
```

> We recommend setting up `sudoers` as described earlier first, granting permission only for the specific `dmidecode` command; running the entire service as root long-term is not recommended unless necessary.


### Windows (PowerShell)

```powershell
cd C:\Users\<user>\project\Trusta_AST_Backend
.\scripts\windows\run_service.bat
```

The batch file resolves `.venv\Scripts\python.exe` at the repo root and runs
`python -m service.app`, so activating the virtual environment first is not
needed. To run it by hand instead:

```powershell
cd C:\Users\<user>\project\Trusta_AST_Backend
.venv\Scripts\Activate.ps1
python -m service.app
```

### Default service address

| Address | Description |
|------|------|
| `http://127.0.0.1:8000` | API Base URL |
| `http://127.0.0.1:8000/health` | Health check |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

## 7. Verifying After Startup

### 7.1 Health check

#### Linux

```bash
curl http://127.0.0.1:8000/health
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 7.2 List available models

#### Linux

```bash
curl http://127.0.0.1:8000/v1/models
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Like vLLM and llama-server, this lists only the model that is loaded right now
(`{"object": "list", "data": []}` when nothing is loaded). Use `GET /config/models`
to browse everything in the registry.

Each entry carries the OpenAI core fields (`id`, `object`, `created`, `owned_by`,
`root`, `parent`, `permission`) plus the serving metadata below:

| Field | Meaning |
| --- | --- |
| `max_context_length` | The context length the model itself was trained for |
| `loaded_context_length` | The window this load actually serves; `context_length` and `max_model_len` (vLLM's name) are aliases of it |
| `engine` | `transformers`, `llama_server` or `vllm` |
| `quantization` | Quantization applied at load time |
| `capabilities` | `chat`, plus `completion` / `multimodal` / `vision` / `audio` where the engine or its config says so |
| `architecture` | Architecture declared by the model's own metadata, e.g. `qwen3moe` |
| `parameter_count` | Parameter count, when the engine reports it (llama-server only) |
| `size_bytes` | On-disk size of the loaded weights, when the engine reports it |
| `meta` | llama-server's own `meta` block, passed through unchanged for clients written against llama.cpp |
| `state` | `loaded` |

The two context fields differ whenever a model is served with a smaller window than it
was trained for. A model trained for 128k but started with `-c 8192` reports
`max_context_length: 131072` and `loaded_context_length: 8192`, so callers can tell
whether a long prompt was rejected by the model or by this deployment.

`loaded_context_length` is the window **one request** gets, and it comes from the
running engine rather than from the load config, because the two routinely disagree:

- llama-server reports it over `/props`, already divided by `-np` and already capped
  to the trained window. A `-c` passed through `llama_server_extra_args` overrides the
  configured one, `-c 0` means "use the trained window", and an oversized `-c` is
  silently capped — none of which the config knows about.
- vLLM reports its effective `max_model_len`, so the window is published even when
  `--max-model-len` was never passed and vLLM sized the window itself.

A field is omitted rather than guessed. If neither the engine nor the config can state
the window, `loaded_context_length` / `context_length` / `max_model_len` are absent, and
a client can fall back knowingly instead of trusting a wrong number.

### 7.3 OpenAI-compatible chat interface

#### Linux

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hi, please briefly introduce yourself"}],
    "stream": false
  }'
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"your-model-name","messages":[{"role":"user","content":"Hi"}],"stream":false}'
```

> If no model has been loaded yet, chat requests may return a "model not ready" response.

`model` is a compatibility field. This service answers with whichever model is
currently loaded and cannot switch per request, so a request naming a different
model is still served — it is not rejected. The mismatch is logged as a warning and
the response reports the model that actually answered, rather than echoing the
request, so an agent can tell it was not routed to the model it selected.

The reported id is the same string `GET /v1/models` advertises. The model path is
also accepted as an alias, so callers holding a path keep working.

---

## 8. Common API Routes

### Basic

- `GET /health`
- `GET /v1/models`
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

- `POST /inference/estimate_memory` — parameter-count based, for HF checkpoints
- `GET /inference/estimate_memory/{model_name}`
- `POST /inference/estimate_memory/gguf` — GGUF / llama.cpp sizing
- `POST /inference/estimate_memory/gguf/check`
- `POST /inference/estimate_memory/gguf/plan`
- `POST /inference/estimate_memory/gguf/recommend`
- `POST /inference/estimate_memory/gguf/sweep`

See [`docs/gguf_memory_estimation.md`](docs/gguf_memory_estimation.md) for the
GGUF family — use it rather than `/inference/estimate_memory` for GGUF models.

### Training management

- `POST /training/start`
- `GET /training/status`
- `GET /training/status/{session_id}/history`
- `POST /training/stop`
- `GET /training/error_details`
- `POST /training/force_cleanup_gpu`
- `GET /training/{job_id}/log` — structured event backfill
- `GET /training/{job_id}/log/stream` — SSE live log

See [`docs/training-log-frontend-integration.md`](docs/training-log-frontend-integration.md)
for the event schema and a reference client.

### Enumerations and config examples

- `GET /config/quantization_types`
- `GET /config/offload_types`
- `GET /config/training_methods`
- `GET /examples/inference`
- `GET /examples/training`
- `GET /examples/conversion`

### RAG

- `GET /rag/docs`
- `POST /rag/docs`
- `DELETE /rag/docs/{doc_id}`
- `GET /rag/search`

### Model configuration and download

- `GET /config/models` — each entry carries a `status`: `ready`, `deleting`, or `delete_failed` (with `delete_error`)
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`
- `GET /config/models/downloads` — all download tasks
- `POST /config/models/download/{task_id}/cancel` — the task then reports `cancelling` until it reports `cancelled`
- `DELETE /config/models/{label}` — accepts `?delete_files=`; returns `202` and removes the files afterwards, so watch `GET /config/models` for the outcome
- `POST /config/models/refresh_context_lengths`
- `POST /config/models/convert`
- `GET /config/models/convert/{job_id}`

### System info

- `GET /system/resources`

---

## 9. Frontend Integration

The backend mounts **one** built frontend. It looks for the first of these that
exists (`service/app.py`), so both a `frontend/` and a `Trusta-AST-Frontend/`
layout work, at the repo root or one level up:

```
<root>/frontend/dist
<root>/Trusta-AST-Frontend/dist
<root>/../frontend/dist
<root>/../Trusta-AST-Frontend/dist
```

Two styled builds are shipped — `dist` (Trusta) and `dist_client` (ADATA) —
but **only `dist` is ever served**; `dist_client` appears nowhere in the backend
code. To run the ADATA styling, copy `dist_client` over `dist`.

Routes provided when a build is found:

- `/frontend/` and `/frontend/<any subpath>` (SPA fallback to `index.html`)
- `/assets/<file>`
- `/` (redirects to `/frontend/`)
- top-level files, when present in the build: `Trusta-logo.svg`, `Trusta-16.ico`,
  `Adata.ico`, `manifest.json`, `robots.txt`, `config.json`

### Building the frontend

The frontend **sources live in a separate repository**. This repo only carries
the built output under `Trusta-AST-Frontend/`, which contains `dist/` and
`dist_client/` and no `package.json` — so `npm run build` cannot be run here.
Build in the frontend repo and copy the resulting `dist/` in, or drop a
prebuilt `dist/` into one of the candidate paths above.

> Note that `Trusta-AST-Frontend/` is gitignored (`*dist*/`), so a fresh clone
> has no frontend at all and the backend starts API-only.

---

## 10. Logging Configuration

```dotenv
LOG_TO_FILE=true
LOG_DIR=<project>/logs
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
```

Once enabled, these files are produced:

- `service.log`
- `service.log.YYYY-MM-DD` (daily rotation, disabled by default on Windows to avoid multi-process conflicts)

Further knobs read by `service/settings.py` but not in `.env.example` — set them
in `.env` when you need them:

| Variable | Purpose | Default |
|---|---|---|
| `LOG_LEVEL` | Root log level (`DEBUG` / `INFO` / …) | `INFO` |
| `LOG_USE_ROTATION` | Enables the daily rotation above | `true`, except `false` on Windows |
| `LOG_FORMAT` / `LOG_DATE_FORMAT` | Formatter strings | see `settings.py` |
| `UVICORN_ACCESS_LOG_LEVEL` | Level for the uvicorn access logger | `WARNING` |
| `RESPONSE_QUEUE_DEBUG` | Verbose worker response-queue tracing | `false` |

`settings.py` also reads a number of undocumented runtime and vLLM variables
(`DEFAULT_GENERATION_TIMEOUT`, `MAX_CONCURRENT_GENERATIONS`, `VLLM_PORT`,
`VLLM_STARTUP_SWEEP`, …). Read `service/settings.py` for the full list before
tuning them.

---

## 11. Testing

The suite is split by what each tier needs: `tests/unit/` runs anywhere,
`tests/smoke/` and `tests/e2e/` need a GPU and auto-skip without one. Fine-tuning
and vLLM are Linux-only features, so their tests carry `linux_only` and skip
themselves on Windows rather than failing. See [`tests/README.md`](tests/README.md)
for the markers and result layout.

### Linux

```bash
source /home/test/project/Trusta_AST_Backend/.venv/bin/activate
cd /home/test/project/Trusta_AST_Backend
pytest tests/unit -q                       # CI-safe unit suite
pytest tests -m inference                  # by marker: inference | finetune | models
pytest tests -m "not linux_only"           # what a Windows host actually runs
```

Run a single test:

```bash
pytest tests/unit/inference/test_runtime_smoke.py
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
cd C:\Users\<user>\project\Trusta_AST_Backend
pytest tests/unit -q
```

The fine-tuning and vLLM tests report as skipped here, naming the feature and this
host — they cover code Windows refuses with a 501, and `deepspeed` / `vllm` are not
installed on this platform. Nothing needs to be deselected by hand.

---

## 12. FAQ

### Q1. Running `run_service.sh` reports that `.venv` cannot be found

The Python environment hasn't been set up yet. Run `scripts/linux/setup_env.sh` (Linux) or `scripts\windows\setup_env.ps1` (Windows) first. The venv is created at the repo root, as `.venv/`.

### Q2. After cloning onto a new machine, the model cache points to the old path

Confirm that `.env` has these set:

```dotenv
HF_HOME=<your cache path>
TIKTOKEN_RS_CACHE_DIR=<your project root>/service/caches/tiktoken
```

### Q3. The API is up, but the homepage shows no frontend

Confirm the frontend has been built and `dist/index.html` exists.

### Q4. I only need the backend API, not the frontend

Just use `/docs`, `/health`, `/v1/chat/completions` directly.

### Q5. Windows reports an execution policy error running `Activate.ps1`

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q6. `uv sync --extra cuda` fails on Windows

The `cuda` extra pulls prebuilt `+cu130` wheels from the PyTorch CUDA index, so
a local CUDA **Toolkit** is not involved — check network access to
`download.pytorch.org` and that the NVIDIA **driver** is recent enough for
CUDA 13.

Without an NVIDIA GPU, use the Intel extra rather than dropping the extra
entirely:

```powershell
.\scripts\windows\setup_env.ps1 -Accel xpu
```

A bare `uv sync` is not a CPU-only fallback — `torch` is not a base dependency,
so omitting the extra downgrades `torch` to the CPU build and uninstalls
`torchvision` / `torchaudio`.

### Q7. Want to deploy with Docker / Podman?

Container assets are maintained internally and are not part of this repository.
Ask the backend team for the `deploy/docker/` bundle.

### Q8. What to do about offload / buffer / swapper errors when fine-tuning with DeepSpeed?

The built-in DeepSpeed profiles for this project live at:

- `service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `service/configs/deepspeed/zero3_offload_cpu_cpu_tight.json`
- `service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `service/configs/deepspeed/zero3_offload_disk_disk.json`

If a fine-tune error message mentions `buffer 0`, `swapper 1`, NVMe offload, aio / offload queue issues, it usually means **there aren't enough offload buffers, disk I/O can't keep up, or the current configuration doesn't fit the data volume**. The most common first step is:

1. Find the DeepSpeed profile you're currently using.
2. Edit the corresponding `service/configs/deepspeed/*.json`.
3. First try raising `zero_optimization.offload_param.buffer_count`.

For example, change from:

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 4
}
```

to:

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 6
}
```

If it still fails, keep raising it gradually up to `8` — don't jump too high at once.

Current shipped values, so you know where you are starting from:

| Profile | `offload_param` | `offload_optimizer` |
|---|---|---|
| `zero3_offload_cpu_cpu.json` | cpu (no buffers) | cpu (no buffers) |
| `zero3_offload_cpu_cpu_tight.json` | cpu (no buffers) | cpu (no buffers) |
| `zero3_offload_cpu_disk.json` | nvme, `buffer_count: 4` | cpu |
| `zero3_offload_disk_cpu.json` | cpu | nvme, `buffer_count: 4` |
| `zero3_offload_disk_disk.json` | nvme, `buffer_count: 6` | nvme, `buffer_count: 6` |

So the 4 → 6 step above applies to `cpu_disk` / `disk_cpu`; `disk_disk` already
ships 6, and there the next step is 8.

If `offload_optimizer.device` is also `nvme`, you can also adjust `zero_optimization.offload_optimizer.buffer_count`.

> In practice, when you see errors like "buffer 0 / swapper 1", starting with `offload_param.buffer_count` is usually the most effective fix.

#### Common tunable DeepSpeed parameters

| Parameter | Location | When to adjust | Common effect / trade-off |
|------|------|----------|------------------|
| `buffer_count` | `zero_optimization.offload_param.buffer_count` | buffer, swapper, or queue errors during NVMe/CPU offload | Increases available buffers, usually improving offload stability, but uses more memory / RAM |
| `buffer_count` | `zero_optimization.offload_optimizer.buffer_count` | Similar errors when offloading the optimizer to NVMe | Similar to above, but affects optimizer state |
| `buffer_size` | `zero_optimization.offload_param.buffer_size` | Disk I/O too fragmented, unstable throughput, frequent offloads | Increasing it reduces the number of chunks, but uses more buffer per transfer |
| `max_in_cpu` | `zero_optimization.offload_param.max_in_cpu` | NVMe too slow, want to keep more parameters in RAM | Raising it reduces disk reads/writes, but increases system RAM usage |
| `pin_memory` | `zero_optimization.offload_param.pin_memory` / `zero_optimization.offload_optimizer.pin_memory` | Enough CPU RAM, want to improve CPU↔GPU transfer | May improve transfer efficiency, but may also increase RAM pressure; disable if memory is tight |
| `stage3_prefetch_bucket_size` | `zero_optimization.stage3_prefetch_bucket_size` | ZeRO-3 prefetch too large causing VRAM / RAM pressure, or I/O stalls | Smaller is more conservative and stable; larger may improve throughput |
| `reduce_bucket_size` | `zero_optimization.reduce_bucket_size` | Communication or reduction uses too much VRAM | Smaller reduces peak memory pressure, but may affect speed |
| `stage3_param_persistence_threshold` | `zero_optimization.stage3_param_persistence_threshold` | Unstable performance from frequent parameter movement | Adjusting changes the residency policy; needs testing against model size |
| `sub_group_size` | `zero_optimization.sub_group_size` | Memory or performance issues when grouping very large models | Smaller is more conservative, larger may improve efficiency |
| `contiguous_gradients` | `zero_optimization.contiguous_gradients` | Gradient fragmentation or memory allocation issues | Usually keep as `true`; test toggling only for specific compatibility issues |
| `overlap_comm` | `zero_optimization.overlap_comm` | Instability, memory spikes, or abnormal performance from overlapping communication | Disabling improves stability, but may be slower |

#### Training parameters often adjusted alongside DeepSpeed config

If adjusting DeepSpeed config alone doesn't stabilize things, you usually also need to reduce training pressure:

| Parameter | Location | Recommended direction |
|------|------|----------|
| `per_device_train_batch_size` | Training request body | Lower this first when VRAM is insufficient |
| `gradient_accumulation_steps` | Training request body or DeepSpeed config | After lowering batch size, raise this to restore the effective batch size |
| `max_seq_length` | Training request body | Lower this first when sequences are too long, e.g. `4096 -> 2048` |
| `gradient_checkpointing` | Training request body | Recommended to keep enabled when memory is insufficient |

#### Recommended adjustment order

1. First confirm which profile is in use (CPU offload / NVMe offload).
2. If the error mentions `buffer` / `swapper`, adjust `offload_param.buffer_count` first.
3. If NVMe offload is still unstable, adjust `offload_optimizer.buffer_count`, `buffer_size`, `max_in_cpu`.
4. If it's a VRAM / RAM shortage, lower `per_device_train_batch_size`, `max_seq_length`.
5. If it's a throughput or stability issue, fine-tune `stage3_prefetch_bucket_size`, `reduce_bucket_size`, `overlap_comm`.

If using NVMe offload, also make sure to check:

- The disk hosting `nvme_path` has enough free space
- Use SSD / NVMe as much as possible, avoid slow HDDs
- Avoid sharing the same disk with other high-I/O workloads

---

## 13. Quick Initialization Flow for First-Time Setup

### Linux (fresh machine)

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone <YOUR_GITLAB_REPOSITORY_URL> Trusta_AST_Backend
cd Trusta_AST_Backend

mkdir -p logs .cache/huggingface runtime_data/inference_offload runtime_data/finetune_output
cp .env.example .env
# Edit .env, set paths like HF_HOME, LOG_DIR, etc.

bash scripts/linux/setup_env.sh
source .venv/bin/activate
bash scripts/linux/run_service.sh
```

> On CUDA hosts, `setup_env.sh` automatically sets up the vLLM isolated environment; to skip it:
> `TRUSTA_SETUP_VLLM=0 bash scripts/linux/setup_env.sh`

---

### Windows (fresh machine, PowerShell)

```powershell
# 1. Install prerequisites (if not already installed)
winget install Git.Git
winget install Python.Python.3.12
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the project
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITLAB_REPOSITORY_URL> Trusta_AST_Backend
cd Trusta_AST_Backend

# 3. Create required folders
New-Item -ItemType Directory -Force logs, .cache\huggingface, runtime_data\inference_offload, runtime_data\finetune_output

# 4. Create .env and edit paths
Copy-Item .env.example .env
notepad .env

# 5. Set up the Python environment
.\scripts\windows\setup_env.ps1 -Accel cuda   # or: -Accel xpu for Intel

# 6. Start the service
.venv\Scripts\Activate.ps1
python -m service.app
```

Once started, open in your browser:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

---

## 14. Additional Notes

### Python version

`pyproject.toml` (repo root) requires Python `>= 3.12`.

### Dependency management

Uses `uv sync` with `pyproject.toml` in the repo root. Always pass an
accelerator extra (`--extra cuda` or `--extra xpu`): `torch` lives only in those
extras, so a bare `uv sync` drops the accelerated build.

### Startup entry point

```
python -m service.app
```

### Engine support by platform

| Engine | Linux | Windows |
|------|-------|---------|
| transformers | ✅ | ✅ |
| llama_server | ✅ (built from source) | ✅ (release binary) |
| vllm | ✅ (CUDA) | ❌ Not supported |

---

## 15. Maintenance Recommendations

After pulling a new version from GitLab each time:

#### Linux

```bash
cd /home/test/project/Trusta_AST_Backend
source .venv/bin/activate
uv sync --extra cuda      # must match the extra used at setup; --extra xpu for Intel
```

> Do not run a bare `uv sync` here. `torch` is not a base dependency, so
> omitting the extra downgrades `torch` to the CPU build and uninstalls
> `torchvision` / `torchaudio`, breaking an already-working install.

vLLM (optional, Linux + CUDA only) — add the extra to the same sync:

```bash
cd /home/test/project/Trusta_AST_Backend
uv sync --extra cuda --extra vllm
```

Switching the accelerator backend:

```bash
cd /home/test/project/Trusta_AST_Backend
TRUSTA_ACCEL=cuda bash scripts/linux/setup_env.sh
```

#### Windows (PowerShell)

```powershell
cd C:\Users\<user>\project\Trusta_AST_Backend
.venv\Scripts\Activate.ps1
uv sync --extra cuda      # must match the extra used at setup; --extra xpu for Intel
```

---

If you need to add Docker, Podman, frontend integration, or API example docs, consider splitting them into separate documents going forward.
