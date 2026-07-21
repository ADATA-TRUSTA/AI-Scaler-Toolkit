# Installation & Setup

[← Back to README](../README.md)

## System Requirements

### Basic Requirements (Both Platforms)

| Tool | Description |
|------|-------------|
| Git | For cloning and the setup-time llama.cpp fetch |
| Python 3.12+ | Runs the service |
| `uv` | Python package / environment management |
| C/C++ toolchain | Required when compiling llama.cpp |
| cmake | Required when compiling llama.cpp |

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake python3 python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, log in to a new shell session or load `uv` manually:

```bash
source "$HOME/.local/bin/env"
```

### Windows

1. **Git for Windows**: <https://git-scm.com/download/win>
2. **Python 3.12+**: <https://www.python.org/downloads/windows/>
   - Enable **Add Python to PATH** during installation
3. **uv** (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. **Visual Studio Build Tools 2022** (only required when compiling llama.cpp)
5. If using Intel XPU / iGPU, Intel oneAPI Runtime / Toolkit is recommended

### Optional Requirements

- NVIDIA GPU: install the matching driver and CUDA runtime
- Intel XPU / iGPU: supported on both Linux and Windows with the required driver/runtime
- Private Hugging Face models: prepare `HF_TOKEN`

> **Note**: vLLM supports **Linux + CUDA** only. Windows cannot use the vLLM engine.
>
> **Note**: Fine-tuning also currently supports **Linux + CUDA** only; Windows, Linux + XPU, and CPU-only environments are not supported.
>
> Static frontend assets are already included in the repository, so you do not need to install Node.js or rebuild the frontend.

---

## Download the Project from GitHub

A plain `git clone` is enough — there are no Git submodules. (`llama.cpp` is fetched at setup time; see below.)

#### Linux

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

#### Windows (PowerShell)

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

---

## Required Setup After First Download

### Create Required Directories

#### Linux

```bash
mkdir -p logs .cache/huggingface
```

#### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force logs, .cache\huggingface
```

### Create `.env`

When the project starts, `backend/service/settings.py` loads `.env` from `backend/` (the directory holding the `service/` package) first. If `.env` does not exist, it falls back to `.env.example`.

**Rule: update `.env` first; do not modify `backend/service/settings.py` directly unless necessary.**

#### Linux

```bash
cp backend/.env.example backend/.env
```

#### Windows (PowerShell)

```powershell
Copy-Item backend\.env.example backend\.env
```

Then open `.env` in a text editor and at least verify the following values:

#### Linux Example

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

HF_HOME=/home/test/project/AI-Scaler-Toolkit/.cache/huggingface
TIKTOKEN_RS_CACHE_DIR=/home/test/project/AI-Scaler-Toolkit
LOG_DIR=/home/test/project/AI-Scaler-Toolkit/logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true
UVICORN_USE_COLORS=true
```

#### Windows Example

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

HF_HOME=C:\Users\<user>\project\AI-Scaler-Toolkit\.cache\huggingface
TIKTOKEN_RS_CACHE_DIR=C:\Users\<user>\project\AI-Scaler-Toolkit
LOG_DIR=C:\Users\<user>\project\AI-Scaler-Toolkit\logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true
UVICORN_USE_COLORS=true
```

To enable optional settings that are commented out in `.env.example`, remove the leading `#` and fill in the real values. For example:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxx
LLAMA_SERVER_URL=http://127.0.0.1:5001
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
```

If you already compiled llama.cpp, you can also add this to `.env`:

```dotenv
LLAMA_SERVER_BINARY=./backend/service/utils/llama.cpp/build/bin/llama-server
```

Windows example:

```dotenv
LLAMA_SERVER_BINARY=./backend/service/utils/llama.cpp/build/bin/Release/llama-server.exe
```

> Paths can use either `/` or `\`; Python `pathlib` supports both.

### Common Path Settings You May Need to Adjust

| Variable | Purpose | Default |
|------|------|--------|
| `HF_HOME` | Hugging Face models and cache | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS cache | Project root |
| `LOG_DIR` | Log output directory | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | Path to the `llama-server` executable | Auto-detected from build output |
| `VLLM_SERVER_PROJECT_DIR` | vLLM isolated environment directory | `backend/service/inference/engines/vllm_server` |

### `dmidecode` Permission Setup (Linux, Optional)

The backend `/system/resources` API calls `dmidecode` to read DRAM specifications. In a non-root environment, you can add a `sudoers` rule to expose the full details:

```bash
sudo visudo -f /etc/sudoers.d/ast-dmidecode
```

Add this line and replace `<user>` with the actual service account:

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

If you skip this step, the API still works, but DRAM specification fields may be empty.

## Create the Python Environment

### Linux

#### Recommended: use the automated setup script

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
```

If the machine does not have NVIDIA CUDA, switch to XPU:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash backend/scripts/linux/setup_env.sh
```

Skip vLLM environment setup:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda TRUSTA_SETUP_VLLM=0 bash backend/scripts/linux/setup_env.sh
```

The script creates the environment in `backend/service/.venv`; **you do not need to activate the virtual environment manually**.

> **Reminder**: If you need fine-tuning, use the **Linux + CUDA** installation path.

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel xpu
```

To use CUDA instead:

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel cuda
```

The script creates the environment in `backend\service\.venv`; **you do not need to activate the virtual environment manually**.

### If You Want to Use `llama-server`

The `llama.cpp` source is fetched at setup time (`TRUSTA_SETUP_LLAMA=1 bash backend/scripts/linux/setup_env.sh`, or `.\backend\scripts\windows\setup_env.ps1 -SetupLlama` on Windows) into `backend/service/utils/llama.cpp`, then compiled manually:

#### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit/backend/service/utils/llama.cpp
cmake -B build
cmake --build build -j
```

If CUDA is available:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

#### Windows (PowerShell, requires Visual Studio Build Tools 2022)

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit\backend\service\utils\llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
```

If CUDA is available:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build build --config Release -j
```

After compilation, set `LLAMA_SERVER_BINARY` in `.env` to the actual output path.

Official build instructions: <https://github.com/ggml-org/llama.cpp#build>

### If You Want to Use the vLLM Engine (Linux + CUDA only)

On CUDA-capable Linux hosts, `setup_env.sh` automatically creates an isolated vLLM environment. Windows does not support the vLLM engine.

## Start the Service

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
bash backend/scripts/linux/run_service.sh
```

If you need `/system/resources` to read full DRAM information through `dmidecode`, you can start the service with sufficient permissions:

```bash
cd /home/test/project/AI-Scaler-Toolkit
sudo bash backend/scripts/linux/run_service.sh
```

> It is recommended to use the `sudoers` setup described earlier to grant only the required `dmidecode` command instead of running the whole service as root long-term.

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\run_service.bat
```

The startup script directly uses Python from `backend/service/.venv`; **you do not need to activate the virtual environment manually**.

### Default Service URLs

| URL | Description |
|------|------|
| `http://127.0.0.1:8000/` | Frontend home page (if frontend files exist, it redirects to `/frontend/`) |
| `http://127.0.0.1:8000/health` | Health check |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

