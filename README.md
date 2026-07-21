# AI Scaler Toolkit

<p align="center">
  <img src="frontend/dist/Trusta-16.ico" alt="Trusta" width="32" height="32">
  &nbsp;&nbsp;
  <img src="frontend/dist_client/Adata.ico" alt="ADATA" width="32" height="32">
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="README.zh-TW.md">繁體中文</a>
</p>

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
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

mkdir -p logs .cache/huggingface
cp backend/.env.example backend/.env
# Edit .env and update HF_HOME, LOG_DIR, SERVICE_HOST, and SERVICE_PORT first

TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
bash backend/scripts/linux/run_service.sh
```

After the service starts successfully, open:

- `http://127.0.0.1:8000/`

If the machine does not have an NVIDIA CUDA environment, use `TRUSTA_ACCEL=xpu` instead.

> **Note**: Fine-tuning currently supports **Linux + CUDA** only.

### Windows (XPU is recommended first)

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

New-Item -ItemType Directory -Force logs, .cache\huggingface
Copy-Item backend\.env.example backend\.env
notepad .env

.\backend\scripts\windows\setup_env.ps1 -Accel xpu
.\backend\scripts\windows\run_service.bat
```

After the service starts successfully, open:

- `http://127.0.0.1:8000/`

To use NVIDIA CUDA instead, change `-Accel xpu` to `-Accel cuda`.

> For detailed installation requirements, `.env` settings, model loading, and validation steps, see the later sections.

---

## 2. Project Structure

```text
AI-Scaler-Toolkit/
├─ backend/                 # Backend service, synced from the upstream backend repo
│  ├─ service/
│  │  ├─ app.py
│  │  ├─ settings.py
│  │  ├─ pyproject.toml
│  │  └─ configs/
│  ├─ .env.example          # service reads .env from here (PROJECT_ROOT = backend/)
│  ├─ tests/
│  ├─ pytest.ini
│  └─ scripts/
│     ├─ linux/
│     │  ├─ run_service.sh
│     │  ├─ setup_env.sh
│     │  └─ stop_service.sh
│     ├─ windows/
│     │  ├─ run_service.bat
│     │  └─ setup_env.ps1
│     └─ docker/
├─ console/                 # Headless Python client
├─ frontend/                # Prebuilt web UI (served by the backend)
│  ├─ dist/
│  └─ dist_client/
├─ docs/
├─ examples/
│  └─ datasets/
├─ wiki/
├─ .github/
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

---

## 📚 Documentation

- **[Installation & Setup](docs/installation.en.md)** — system requirements, download, `.env`, Python environment, starting the service
- **[Usage & Verification](docs/usage.en.md)** — health check, loading a model, common API routes, frontend & logging notes
- **[Performance Benchmarks](docs/benchmarks.en.md)** — TPS / TTFT across GPUs and quantizations
- **[Maintenance & FAQ](docs/faq.en.md)** — updates and troubleshooting

> 中文文件：[安裝](docs/installation.md) · [使用](docs/usage.md) · [效能](docs/benchmarks.md) · [FAQ](docs/faq.md)
