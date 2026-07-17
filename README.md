# AI Scaler Toolkit

<img src="src/frontend/dist/Trusta-16.ico" alt="Trusta Icon" width="24" height="24" style="vertical-align: middle; margin-right: 8px;">
<img src="src/frontend/dist_client/Adata.ico" alt="Adata Icon" width="24" height="24" style="vertical-align: middle; margin-left: 8px;">

## 🌐 Language / 語言

[🇬🇧 English](README.md) | [🇹🇼 繁體中文](README.zh-TW.md)

---

AI Scaler Toolkit is a FastAPI-based LLM backend service that provides:

- Model loading / unloading
- OpenAI-compatible chat interface
- Inference and streaming responses
- Training job startup and status queries
- Model downloading
- Static frontend mounting

This document covers both Linux and Windows. The quick-start initialization flow is placed first; more details are provided in later sections.

> Example Linux path: `/home/test/project/AI-Scaler-Toolkit`
>
> Example Windows path: `C:\Users\<user>\project\AI-Scaler-Toolkit`

---

## 1. First-Time Quick Start

### Linux (CUDA is recommended by default)

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

mkdir -p logs .cache/huggingface
cp .env.example .env
# Edit .env and update HF_HOME, LOG_DIR, SERVICE_HOST, and SERVICE_PORT first

TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
bash deploy/linux/run_service.sh
```

After the service starts successfully, open:

- `http://127.0.0.1:8000/`

If the machine does not have an NVIDIA CUDA environment, use `TRUSTA_ACCEL=xpu` instead.

> **Note**: Fine-tuning currently supports **Linux + CUDA** only.

### Windows (XPU is recommended first)

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

New-Item -ItemType Directory -Force logs, .cache\huggingface
Copy-Item .env.example .env
notepad .env

.\deploy\windows\setup_env.ps1 -Accel xpu
.\deploy\windows\run_service.bat
```

After the service starts successfully, open:

- `http://127.0.0.1:8000/`

To use NVIDIA CUDA instead, change `-Accel xpu` to `-Accel cuda`.

> For detailed installation requirements, `.env` settings, model loading, and validation steps, see the later sections.

---

## 2. Project Structure

```text
AI-Scaler-Toolkit/
├─ deploy/
│  ├─ linux/
│  │  ├─ run_service.sh
│  │  ├─ setup_env.sh
│  │  └─ stop_service.sh
│  ├─ windows/
│  │  ├─ run_service.bat
│  │  └─ setup_env.ps1
│  └─ docker/
├─ docs/
├─ examples/
│  └─ datasets/
├─ src/
│  ├─ service/
│  │  ├─ app.py
│  │  ├─ settings.py
│  │  ├─ pyproject.toml
│  │  └─ configs/
│  ├─ frontend/
│  │  ├─ dist/
│  │  └─ dist_client/
│  └─ console/
├─ tests/
├─ wiki/
├─ logs/
├─ .github/
├─ .env.example
├─ pytest.ini
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

---

## 3. System Requirements

### Basic Requirements (Both Platforms)

| Tool | Description |
|------|-------------|
| Git | With submodule support |
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

## 📊 Performance Benchmarks (TPS / TTFT)

Measured with the **Llama Engine** (`llama-server`). Common test conditions for every run: `n_ctx = 4096`, input `50–108` tokens, output `50` tokens. **TPS** = output tokens / sec (higher is better); **TTFT** = time to first token (lower is better). Numbers come from internal testing and may vary with driver, build, and prompt.

### Test Machines

| ID | GPU | OS | CPU | VRAM | DRAM |
|----|-----|----|-----|------|------|
| **A** | RTX 5060 Ti | Ubuntu 24.04 | Core Ultra 5 225 | 16 GB | DDR5-4800 / 128 GB |
| **B** | RTX 6000 Ada | Ubuntu 24.04 | Xeon w5-3535X | 48 GB | DDR5-4800 / 512 GB |
| **C** | RTX 5080 | Windows 11 | Core Ultra 9 275HX | 16 GB | DDR5-6400 / 32·64 GB |
| **D** | Intel iGPU | Windows 11 | Core Ultra 7 355 | shared | DDR5-6400 / 32·64 GB |

### Summary — Best result per model

Each cell shows the **best TPS** reached on that machine and its TTFT. **Bold** = fastest machine for that model. `–` = not tested.

| Model (size) | A · 5060 Ti | B · 6000 Ada | C · 5080 | D · iGPU |
|--------------|-------------|--------------|----------|----------|
| GPT-OSS-20B-FP16 (13.8 GB) | **90 / 0.2s** | **155 / 0.1s** | 133 / 0.5s | 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M (11.6 GB) | – | – | **179 / 0.3s** | 20 / 0.9s |
| Qwen3-8B-Q8_0 (8.7 GB) | – | – | **73 / 0.05s** | 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M (22 GB) | 57.5 / 0.4s | **162 / 0.1s** | 80 / 0.5s | 20 / 1s |
| Gemma4-26B-A4B-Q8_0 (26.9 GB) | 29 / 1s | **114 / 0.8s** | 45 / 0.7s | 15 / 1s |
| Gemma4-31B-Q4_K_M (18.3 GB) | 3 / 2.5s | **7.5 / 0.8s** | 6 / 1.5s | 4 / 2s |
| Gemma4-31B-Q8_0 (32.6 GB) | 1.9 / 3.5s | **24.5 / 0.2s** | 3.5 / 2s | 2 / 2s |
| GPT-OSS-120B&sup1; (~63–65 GB) | 15.5 / 1.5s | **48 / 0.5s** | 35 / 3.5s | 9.5 / 15s |

&sup1; On **A/B** the 120B model is **FP16 (65.4 GB)**; on **C/D** it is **Q8 / Q4_K_M (~63 GB)**. Different quantization — not directly comparable.

### Detailed results per machine

<details>
<summary><b>A · RTX 5060 Ti 16 GB</b> — full results (128 GB DRAM)</summary>

| Model | Config (VRAM / DRAM) | TPS | TTFT |
|-------|----------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU all layers (13 GB) | 90 | 0.2s |
| GPT-OSS-120B-FP16 | auto (14.5 / 50.9 GB) | 15.5 | 1.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (12 / 53.4 GB) | 14 | 1.7s |
| Qwen3.5-35B-Q4_K_M | auto (15.7 / 7.5 GB) | 57.5 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (14.1 / 9.1 GB) | 53 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 38 | 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=17 (15.7 / 13 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6 / 22.7 GB) | 20 | 1s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 3 | 2.5s |
| Gemma4-31B-Q8_0 | auto (15.8 / 19.4 GB) | 1.9 | 3.5s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.8 / 20.4 GB) | 1.8 | 3.7s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.7 / 25.5 GB) | 1.5 | 4.5s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (5 / 30.2 GB) | 1.2 | 5.5s |

</details>

<details>
<summary><b>B · RTX 6000 Ada 48 GB</b> — full results (512 GB DRAM)</summary>

| Model | Config (VRAM / DRAM) | TPS | TTFT |
|-------|----------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU all layers (13 GB) | 155 | 0.1s |
| GPT-OSS-120B-FP16 | auto (46 / 19.4 GB) | 48 | 0.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=27 (19 / 46.4 GB) | 24 | 1s |
| GPT-OSS-120B-FP16 | n-cpu-moe=29 (16.3 / 49.1 GB) | 20 | 1.3s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (11.6 / 53.8 GB) | 18 | 1.3s |
| Qwen3.5-35B-Q4_K_M | auto (23.2 GB) | 162 | 0.1s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (13.7 / 9.5 GB) | 67 | 0.3s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 45 | 0.5s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 114 | 0.8s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=16 (16.7 / 12 GB) | 42 | 0.4s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6.1 / 22.6 GB) | 24 | 0.6s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 7.5 | 0.8s |
| Gemma4-31B-Q8_0 | auto (35.2 GB) | 24.5 | 0.2s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.5 / 20.7 GB) | 4.5 | 1.3s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.4 / 25.8 GB) | 4 | 1.7s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (4.7 / 30.5 GB) | 3 | 2s |

</details>

<details>
<summary><b>C · RTX 5080 16 GB</b> — full results (Windows 11, by DRAM size)</summary>

TPS / TTFT per DRAM configuration. The **64 GB + moe** column uses `--n-cpu-moe=N`.

| Model | Config | 32 GB | 64 GB | 64 GB + moe |
|-------|--------|-------|-------|-------------|
| GPT-OSS-20B-FP16 | GPU all (13 GB) | 133 / 0.5s | 133 / 0.5s | – |
| GPT-OSS-20B-FP16 | 0 GPU / 25 CPU | 15 / 3s | 19 / 2s | – |
| GPT-OSS-20B-Q4_K_M | GPU all (12 GB) | 179 / 0.3s | 179 / 0.3s | – |
| GPT-OSS-120B-Q8 | auto (14.6 GB) | 5 / 60s | 33 / 4s | – |
| GPT-OSS-120B-Q8 | 10 GPU / 27 CPU | 2.5 / 77s | 5 / 7s | 10 / 7s |
| GPT-OSS-120B-Q8 | 8 GPU / 29 CPU | 3 / 70s | 23 / 5s | 34 / 4s |
| GPT-OSS-120B-Q8 | 5 GPU / 32 CPU | 2.5 / 80s | 20 / 6s | 32 / 4s |
| GPT-OSS-120B-Q4_K_M | 8 GPU / 29 CPU | 3.3 / 70s | 23.5 / 5s | 35 / 3.5s |
| Qwen3-8B-Q8_0 | GPU all (9.5 GB) | 73 / 0.05s | 73 / 0.05s | – |
| Qwen3-8B-Q8_0 | 15 GPU / 22 CPU | 16 / 1s | 16 / 1s | – |
| Qwen3-8B-Q8_0 | 0 GPU / 37 CPU | 10 / 1.3s | 10 / 1.3s | – |
| Qwen3.5-35B-Q4_K_M | auto (15.5 GB) | 70 / 1s | 80 / 0.5s | – |
| Qwen3.5-35B-Q4_K_M | 20 GPU / 21 CPU | 33 / 1.5s | 33 / 1s | 71 / 0.5s |
| Qwen3.5-35B-Q4_K_M | 0 GPU / 41 CPU | 19 / 2s | 19 / 2s | 52 / 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (15.7 GB) | 43 / 1.5s | 45 / 0.7s | – |
| Gemma4-26B-A4B-Q8_0 | 15 GPU / 16 CPU | 30 / 2s | 30 / 1.5s | 44 / 0.5s |
| Gemma4-26B-A4B-Q8_0 | 0 GPU / 31 CPU | 18 / 3.5s | 18 / 2.5s | 33 / 0.5s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU | 6 / 2s | 6 / 1.5s | – |
| Gemma4-31B-Q8_0 | auto (15.8 GB) | 3.5 / 2s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU | 3.5 / 3s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU | 3 / 60s | 3 / 3s | – |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU | NA | 2.5 / 3s | – |

</details>

<details>
<summary><b>D · Intel Core Ultra 7 355 (iGPU)</b> — full results (Windows 11, by DRAM size)</summary>

Cell format: `iGPU / CPU layer split — TPS / TTFT`.

| Model | 32 GB DRAM | 64 GB DRAM |
|-------|------------|------------|
| GPT-OSS-20B-FP16 | iGPU all (13 GB) — 15 / 0.9s | iGPU all (13 GB) — 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M | iGPU all (12 GB) — 20 / 0.9s | iGPU all (12 GB) — 20 / 0.9s |
| GPT-OSS-120B-Q8 | NA | iGPU 20L (33 GB) / CPU 17 — 7.5 / 17s |
| GPT-OSS-120B-Q4_K_M | iGPU 10 / CPU 27 — 1.9 / 60s | iGPU 20L (33 GB) / CPU 17 — 9.5 / 15s |
| Qwen3-8B-Q8_0 | iGPU all (10 GB) — 9.5 / 1s | iGPU all (10 GB) — 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M | iGPU 30L (20 GB) / CPU 11 — 13 / 3s | iGPU all (23 GB) — 20 / 1s |
| Gemma4-26B-A4B-Q8_0 | iGPU 20L (20 GB) / CPU 11 — 10.5 / 6s | iGPU all (29 GB) — 15 / 1s |
| Gemma4-31B-Q4_K_M | iGPU all (22 GB) — 3.6 / 6s | iGPU all (22 GB) — 4 / 2s |
| Gemma4-31B-Q8_0 | iGPU 30L (18 GB) / CPU 31 — 0.05 / 60s | iGPU all (33 GB) — 2 / 2s |

</details>

---

## 4. Download the Project from GitHub

This project includes Git submodules. **Use `--recursive` when cloning.**

#### Linux

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

#### Windows (PowerShell)

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

> If you already cloned without it, run:
> ```bash
> git submodule update --init --recursive
> ```

---

## 5. Required Setup After First Download

### 5.1 Create Required Directories

#### Linux

```bash
mkdir -p logs .cache/huggingface
```

#### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force logs, .cache\huggingface
```

### 5.2 Create `.env`

When the project starts, `src/service/settings.py` loads `.env` from the project root first. If `.env` does not exist, it falls back to `.env.example`.

**Rule: update `.env` first; do not modify `src/service/settings.py` directly unless necessary.**

#### Linux

```bash
cp .env.example .env
```

#### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
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
LLAMA_SERVER_BINARY=./src/service/utils/llama.cpp/build/bin/llama-server
```

Windows example:

```dotenv
LLAMA_SERVER_BINARY=./src/service/utils/llama.cpp/build/bin/Release/llama-server.exe
```

> Paths can use either `/` or `\`; Python `pathlib` supports both.

### 5.3 Common Path Settings You May Need to Adjust

| Variable | Purpose | Default |
|------|------|--------|
| `HF_HOME` | Hugging Face models and cache | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS cache | Project root |
| `LOG_DIR` | Log output directory | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | Path to the `llama-server` executable | Auto-detected from build output |
| `VLLM_SERVER_PROJECT_DIR` | vLLM isolated environment directory | `src/service/inference/engines/vllm_server` |

### 5.4 `dmidecode` Permission Setup (Linux, Optional)

The backend `/system/resources` API calls `dmidecode` to read DRAM specifications. In a non-root environment, you can add a `sudoers` rule to expose the full details:

```bash
sudo visudo -f /etc/sudoers.d/ast-dmidecode
```

Add this line and replace `<user>` with the actual service account:

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

If you skip this step, the API still works, but DRAM specification fields may be empty.

## 6. Create the Python Environment

### Linux

#### Recommended: use the automated setup script

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
```

If the machine does not have NVIDIA CUDA, switch to XPU:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh
```

Skip vLLM environment setup:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda TRUSTA_SETUP_VLLM=0 bash deploy/linux/setup_env.sh
```

The script creates the environment in `src/service/.venv`; **you do not need to activate the virtual environment manually**.

> **Reminder**: If you need fine-tuning, use the **Linux + CUDA** installation path.

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel xpu
```

To use CUDA instead:

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel cuda
```

The script creates the environment in `service\.venv`; **you do not need to activate the virtual environment manually**.

### 6.1 If You Want to Use `llama-server`

The `llama.cpp` source is included as the `src/service/utils/llama.cpp` submodule and must be compiled manually.

#### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit/src/service/utils/llama.cpp
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
cd C:\Users\<user>\project\AI-Scaler-Toolkit\service\utils\llama.cpp
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

### 6.2 If You Want to Use the vLLM Engine (Linux + CUDA only)

On CUDA-capable Linux hosts, `setup_env.sh` automatically creates an isolated vLLM environment. Windows does not support the vLLM engine.

## 7. Start the Service

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
bash deploy/linux/run_service.sh
```

If you need `/system/resources` to read full DRAM information through `dmidecode`, you can start the service with sufficient permissions:

```bash
cd /home/test/project/AI-Scaler-Toolkit
sudo bash deploy/linux/run_service.sh
```

> It is recommended to use the `sudoers` setup described earlier to grant only the required `dmidecode` command instead of running the whole service as root long-term.

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\run_service.bat
```

The startup script directly uses Python from `src/service/.venv`; **you do not need to activate the virtual environment manually**.

### Default Service URLs

| URL | Description |
|------|------|
| `http://127.0.0.1:8000/` | Frontend home page (if frontend files exist, it redirects to `/frontend/`) |
| `http://127.0.0.1:8000/health` | Health check |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

## 8. How to Verify After Startup

### 8.1 Health Check

#### Linux

```bash
curl http://127.0.0.1:8000/health
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 8.2 View the Available Model List

#### Linux

```bash
curl http://127.0.0.1:8000/config/models
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/config/models
```

### 8.3 Load a Model First, Then Send a Chat Request

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

## 9. Common API Routes

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

## 10. Frontend Notes

The project already contains static frontend assets:

- `src/frontend/dist`
- `src/frontend/dist_client`

The backend currently mounts `src/frontend/dist` at `/frontend/`, and the root path `/` automatically redirects to `/frontend/` when `index.html` exists.

Therefore:

- Node.js is not required
- No separate frontend build is required
- As long as the service starts successfully, you can directly open `http://127.0.0.1:8000/`

---

## 11. Logging Configuration

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

## 12. Maintenance Updates

Each time you pull a new version from GitHub, it is recommended to resync dependencies.

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh
```

If switching to CUDA:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
```

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel xpu
```

If switching to CUDA:

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel cuda
```

---

## 13. FAQ

### Q1. The startup script says `.venv` cannot be found

This means the Python environment has not been created yet. Run:

- Linux: `TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh`
- Windows: `.\deploy\windows\setup_env.ps1 -Accel xpu`

### Q2. After moving to a new machine, the model cache still points to the old path

Check `.env` first:

```dotenv
HF_HOME=<your cache path>
TIKTOKEN_RS_CACHE_DIR=<your project root>
```

### Q3. The API is running, but the home page does not show the frontend

First verify that the frontend files exist in the project:

- `src/frontend/dist/index.html`
- `src/frontend/dist/assets/`

If files already exist under `dist`, simply reopen `http://127.0.0.1:8000/`.

### Q4. I only need the backend API, not the frontend

Use these routes directly:

- `/docs`
- `/health`
- `/v1/chat/completions`

### Q5. `setup_env.ps1` fails on Windows

Verify that:

- Python 3.12+ is installed
- `uv` is installed
- If using XPU, Intel oneAPI is installed

If needed, first update the PowerShell execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q6. Building the environment with `-Accel cuda` fails on Windows

Make sure CUDA Toolkit is installed and the driver version is compatible. If you are not using an NVIDIA GPU, use:

```powershell
.\deploy\windows\setup_env.ps1 -Accel xpu
```

### Q7. What should I do if fine-tuning with DeepSpeed hits offload / buffer / swapper errors?

The built-in DeepSpeed profiles are located at:

- `src/service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `src/service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `src/service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `src/service/configs/deepspeed/zero3_offload_disk_disk.json`

If the error mentions `buffer`, `swapper`, NVMe offload, aio, or offload queue problems, the most common first step is to increase `zero_optimization.offload_param.buffer_count`.

For example:

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 6
}
```

If it is still unstable, increase it gradually to `8` instead of jumping too far at once.

If `offload_optimizer.device` is also `nvme`, you can also adjust `zero_optimization.offload_optimizer.buffer_count`.

> In practice, when you see errors like “buffer 0 / swapper 1”, adjusting `offload_param.buffer_count` first is usually the most effective change.

#### Common DeepSpeed Parameters to Tune

| Parameter | Location | When to Adjust | Typical Effect / Trade-off |
|------|------|----------|------------------|
| `buffer_count` | `zero_optimization.offload_param.buffer_count` | Buffer, swapper, or queue errors during NVMe/CPU offload | Increases available buffers and often improves offload stability, but uses more memory / RAM |
| `buffer_count` | `zero_optimization.offload_optimizer.buffer_count` | Similar errors when optimizer state is offloaded to NVMe | Similar to above, but affects optimizer state |
| `buffer_size` | `zero_optimization.offload_param.buffer_size` | Disk I/O is too fragmented, throughput is unstable, or offload happens too often | Larger values reduce chunking overhead, but increase single-buffer usage |
| `max_in_cpu` | `zero_optimization.offload_param.max_in_cpu` | NVMe is too slow and you want to keep more parameters in RAM | Higher values reduce disk reads/writes, but increase system RAM usage |
| `pin_memory` | `zero_optimization.offload_param.pin_memory` / `zero_optimization.offload_optimizer.pin_memory` | RAM is sufficient and you want better CPU↔GPU transfer | May improve transfer efficiency, but can increase RAM pressure; disable if memory is tight |
| `stage3_prefetch_bucket_size` | `zero_optimization.stage3_prefetch_bucket_size` | ZeRO-3 prefetch is too large and causes VRAM / RAM pressure, or I/O instability | Smaller is more conservative and stable; larger may improve throughput |
| `reduce_bucket_size` | `zero_optimization.reduce_bucket_size` | Communication or aggregation uses too much VRAM | Smaller values reduce peak memory pressure, but may affect speed |
| `stage3_param_persistence_threshold` | `zero_optimization.stage3_param_persistence_threshold` | Frequent parameter movement causes unstable performance | Adjusts persistence behavior and should be tested against model size |
| `sub_group_size` | `zero_optimization.sub_group_size` | Memory or performance is poor with very large model partitioning | Smaller is safer; larger may improve efficiency |
| `contiguous_gradients` | `zero_optimization.contiguous_gradients` | Gradient fragmentation or memory allocation issues | Usually keep `true`; toggle only for specific compatibility issues |
| `overlap_comm` | `zero_optimization.overlap_comm` | Communication overlap causes instability, memory spikes, or abnormal performance | Disabling may improve stability, but can reduce speed |

#### Training Parameters Often Tuned Together

If changing DeepSpeed config alone is not enough, it is also common to reduce training pressure:

| Parameter | Location | Suggested Direction |
|------|------|----------|
| `per_device_train_batch_size` | Training request body | Reduce first when VRAM is insufficient |
| `gradient_accumulation_steps` | Training request body or DeepSpeed config | Increase to recover effective batch size after lowering batch size |
| `max_seq_length` | Training request body | Reduce first if the sequence is too long, for example `4096 -> 2048` |
| `gradient_checkpointing` | Training request body | Keep enabled when memory is insufficient |

#### Recommended Tuning Order

1. Confirm which profile you are using first (CPU offload or NVMe offload).
2. If the error includes `buffer` / `swapper`, adjust `offload_param.buffer_count` first.
3. If NVMe offload is still unstable, adjust `offload_optimizer.buffer_count`, `buffer_size`, and `max_in_cpu`.
4. If VRAM / RAM is insufficient, reduce `per_device_train_batch_size` and `max_seq_length`.
5. For throughput or stability issues, fine-tune `stage3_prefetch_bucket_size`, `reduce_bucket_size`, and `overlap_comm`.

If you use NVMe offload, also verify:

- The disk at `nvme_path` has enough free space
- Prefer SSD / NVMe instead of a slow HDD
- Avoid sharing the same disk with other heavy I/O workloads
