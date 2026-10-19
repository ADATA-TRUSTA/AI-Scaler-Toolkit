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

> This is the `backend/` directory of [AI Scaler Toolkit](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit); commands below run from it.
>
> **Linux example path**: `/home/<user>/AI-Scaler-Toolkit/backend`
> **Windows example path**: `C:\Users\<user>\AI-Scaler-Toolkit\backend` (choose your own)

---

## 1. Project Structure

```text
backend/
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
| C/C++ toolchain, `cmake` | Linux only: llama.cpp is built from source there (Windows downloads the official release), and the training packages built from source need it too (deepspeed on any accelerator; causal-conv1d only with `--extra cuda`) |

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

## 3. Cloning the Project

llama.cpp (`service/utils/llama.cpp`, optional, only needed for the GGUF convert tooling) is fetched on demand by `setup_env`, **so a plain clone is enough — `--recursive` is not needed**.

#### Linux

```bash
cd ~
git clone https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit.git
cd AI-Scaler-Toolkit/backend
```

#### Windows (PowerShell)

```powershell
cd $HOME
git clone https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit.git
cd AI-Scaler-Toolkit/backend
```

> `setup_env` installs the llama tooling (needed for llama-server inference and GGUF
> conversion) **by default**, but only what is missing — an existing binary or convert
> checkout is left untouched. On Linux llama.cpp is compiled (`build-essential`, `cmake`); on
> Windows the official release is downloaded, so no compiler is needed. To opt out, or to force a
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
# SERVICE_HOST: 127.0.0.1 allows local connections only (recommended); 0.0.0.0 opens it to other
# machines -- see "Opening the service to other machines" below before you do
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

# ===== Cache and runtime data =====
HF_HOME=/home/<user>/AI-Scaler-Toolkit/backend/.cache/huggingface
LOG_DIR=/home/<user>/AI-Scaler-Toolkit/backend/logs

# ===== Logging =====
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true

# ===== Optional: private Hugging Face models =====
# HF_TOKEN=hf_xxxxxxxxxxxxx

# ===== Optional: llama server (setup_env builds it into the project) =====
# LLAMA_SERVER_BINARY=/home/<user>/AI-Scaler-Toolkit/backend/service/utils/llama-bin/llama-server
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

HF_HOME=C:\Users\<user>\AI-Scaler-Toolkit\backend\.cache\huggingface
LOG_DIR=C:\Users\<user>\AI-Scaler-Toolkit\backend\logs

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

#### Opening the service to other machines

With `SERVICE_HOST=0.0.0.0` other machines can reach the port, and three settings decide
what they get:

```dotenv
SERVICE_HOST=0.0.0.0
SERVICE_API_KEY=<a long random key>   # other machines send it as "Authorization: Bearer <key>"
SERVICE_REMOTE_ACCESS=inference       # or "all"
```

- Without `SERVICE_API_KEY`, other machines are refused. This machine never needs the key.
  An address that sends 10 wrong keys within a minute is refused (`429`) for 5 minutes.
- `inference` (the default) gives other machines the OpenAI-compatible API under `/v1/`:
  chat, completions, embeddings, the model list. That is all a client such as Hermes, an
  IDE assistant or Open WebUI needs, and it is what the desktop application always uses.
- `all` adds the management API: loading and unloading models, downloads, fine-tuning,
  RAG and system information. Those requests carry file paths and launch arguments, so a
  key holder can reach well beyond the model. Use it only behind protection of your own —
  a VPN such as Tailscale, or a reverse proxy with TLS and authentication, with the
  firewall open to the addresses that need it — and never straight onto the internet.
  The unauthenticated model-management API is what attackers used against exposed Ollama
  servers (CVE-2026-7482).
- The web interface and `/docs` are for this machine only, whatever these are set to.

On this machine, the service also refuses requests a browser marks as coming from
another site, and requests addressed to a host name other than `localhost` (DNS
rebinding), so a web page open here cannot use it either. Scripts and other clients that
are not web pages are unaffected.

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

### 4.5 DRAM module details (Linux)

`/system/resources` lists each DRAM module. On Linux it reads them from udev (what
systemd-udev decoded from SMBIOS at boot), which needs no root, so there is normally
nothing to set up. Only on a system without systemd-udev does it fall back to
`dmidecode`, which needs root. For that case, a sudoers rule lets the service run that one
command without a password:

```bash
sudo visudo -f /etc/sudoers.d/trusta-dmidecode
```

Add the following line (replace `<user>` with the account actually running the service):

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

Without either, the API still works; the module list is just empty.

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
cd /home/<user>/AI-Scaler-Toolkit/backend
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
cd /home/<user>/AI-Scaler-Toolkit/backend
uv sync --extra cuda    # NVIDIA GPU
# or
uv sync --extra xpu     # Intel XPU
```

> On Linux, `--extra cuda` additionally pulls `causal-conv1d`, which builds a CUDA
> extension from source and has no Intel equivalent. `--extra xpu` deliberately skips
> it, so an Intel host needs no CUDA toolchain.

#### Activate the virtual environment

```bash
source /home/<user>/AI-Scaler-Toolkit/backend/.venv/bin/activate
```

---

### Windows

Use the PowerShell setup script, which mirrors `setup_env.sh`:

```powershell
cd C:\Users\<user>\AI-Scaler-Toolkit\backend
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
C:\Users\<user>\AI-Scaler-Toolkit\backend\.venv\Scripts\Activate.ps1
```

If you hit an execution policy error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Activate the virtual environment (cmd)

```cmd
C:\Users\<user>\AI-Scaler-Toolkit\backend\.venv\Scripts\activate.bat
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

Why not `ggml-org/llama-install.sh` or the release archives on Linux: llama.cpp publishes its CUDA builds
for Windows only, so for an NVIDIA card on Linux there is nothing to download; and the CUDA release's CPU
backend is built without AVX/AVX2/FMA. On a model that has to keep part of its weights in RAM, that halves the
speed: measured 42 vs 90 tokens/s (RTX 5060 Ti, a 30B-A3B Q4_0 model, b10107, both builds tuned). A small
model that fits on the GPU loses only about 10%. Linux therefore builds llama.cpp on the machine.

---

### 5.2 If you want to use the vLLM engine (Linux)

vLLM is an extra of this project (`--extra vllm`), installed into the same `.venv` as the
service, currently at 0.27.1.

Sharing that `.venv` means vLLM's requirements drive three of this project's pins:
`torch` 2.13.0, `fastapi` 0.136.3, and `numpy` 2.3.5. **Upgrading vLLM is therefore a
whole-project change** — check what the new release does to those three first, and expect
to re-run the test suite and the XPU check.

`setup_env.sh` adds the extra automatically on CUDA hosts. On an Intel GPU it is opt-in
(`TRUSTA_SETUP_VLLM=1`, which installs the `vllm-xpu` extra). To do it manually on CUDA:

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
uv sync --extra cuda --extra vllm
```

---

## 6. Starting the Service

### Linux

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
bash scripts/linux/run_service.sh
```

The script checks whether `.venv` exists, then starts the service with `python -m service.app`.

If you need the `/system/resources` API to read full DRAM info via `dmidecode`, you can start with elevated privileges instead:

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
sudo bash scripts/linux/run_service.sh
```

> We recommend setting up `sudoers` as described earlier first, granting permission only for the specific `dmidecode` command; running the entire service as root long-term is not recommended unless necessary.


### Windows (PowerShell)

```powershell
cd C:\Users\<user>\AI-Scaler-Toolkit\backend
.\scripts\windows\run_service.bat
```

The batch file resolves `.venv\Scripts\python.exe` at the repo root and runs
`python -m service.app`, so activating the virtual environment first is not
needed. To run it by hand instead:

```powershell
cd C:\Users\<user>\AI-Scaler-Toolkit\backend
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

## 8. API Routes

The route list, with examples, is in AI Scaler Toolkit's [Usage](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/blob/main/docs/usage.en.md#api-routes); the running service documents every route at `/docs`.

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
source /home/<user>/AI-Scaler-Toolkit/backend/.venv/bin/activate
cd /home/<user>/AI-Scaler-Toolkit/backend
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
cd C:\Users\<user>\AI-Scaler-Toolkit\backend
pytest tests/unit -q
```

The fine-tuning and vLLM tests report as skipped here, naming the feature and this
host — they cover code Windows refuses with a 501, and `deepspeed` / `vllm` are not
installed on this platform. Nothing needs to be deselected by hand.

---

## 12. FAQ

### Q1. Running `run_service.sh` reports that `.venv` cannot be found

The Python environment hasn't been set up yet. Run `scripts/linux/setup_env.sh` (Linux) or `scripts\windows\setup_env.ps1` (Windows) first. The venv is created at the repo root, as `.venv/`.

### Q2. Windows reports an execution policy error running `Activate.ps1`

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3. `uv sync --extra cuda` fails on Windows

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

### Q4. Want to deploy with Docker / Podman?

Container assets are maintained internally and are not part of this repository.
Ask the backend team for the `deploy/docker/` bundle.

### Q5. What to do about offload / buffer / swapper errors when fine-tuning with DeepSpeed?

See the tuning guide in AI Scaler Toolkit's [FAQ](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/blob/main/docs/faq.en.md) (the DeepSpeed question). The profiles are under `service/configs/deepspeed/`.

Questions about installing and using the application are answered in the same [FAQ](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/blob/main/docs/faq.en.md).

---

## 13. Additional Notes

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
| vllm | ✅ (CUDA; Intel opt-in) | ❌ Not supported |

---

## 14. Maintenance Recommendations

After pulling a new version each time:

#### Linux

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
source .venv/bin/activate
uv sync --extra cuda      # must match the extra used at setup; --extra xpu for Intel
```

> Do not run a bare `uv sync` here. `torch` is not a base dependency, so
> omitting the extra downgrades `torch` to the CPU build and uninstalls
> `torchvision` / `torchaudio`, breaking an already-working install.

vLLM (optional, Linux; on Intel use the `vllm-xpu` extra) — add the extra to the same sync:

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
uv sync --extra cuda --extra vllm
```

Switching the accelerator backend:

```bash
cd /home/<user>/AI-Scaler-Toolkit/backend
TRUSTA_ACCEL=cuda bash scripts/linux/setup_env.sh
```

#### Windows (PowerShell)

```powershell
cd C:\Users\<user>\AI-Scaler-Toolkit\backend
.venv\Scripts\Activate.ps1
uv sync --extra cuda      # must match the extra used at setup; --extra xpu for Intel
```

---

If you need to add Docker, Podman, frontend integration, or API example docs, consider splitting them into separate documents going forward.
