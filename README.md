# AI Scaler Toolkit

<p align="center">
  <img src="assets/trusta-icon.png" alt="Trusta" width="32" height="32">
  &nbsp;&nbsp;
  <img src="assets/adata-icon.png" alt="ADATA" width="32" height="32">
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="README.zh-TW.md">繁體中文</a>
</p>

---

AI Scaler Toolkit runs large language models on your own PC, on an NVIDIA or Intel GPU. It is a desktop application for Windows and Linux:

- Chat with models in its window, with streaming and pictures for models that read them
- Download models from Hugging Face and convert them to GGUF
- Three engines: Transformers, llama.cpp, and vLLM (Linux)
- An OpenAI-compatible API on `http://127.0.0.1:8571/v1`, so your other apps and coding agents can use the same model
- Fine-tuning (Linux) and document search (RAG)
- Optional components installed for you: Hermes Agent and ComfyUI

Nothing leaves your PC.

---

## 1. Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Install AI Scaler Toolkit on this PC for me: https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit - follow docs/AI_SETUP.md in that repository.
```

It checks your GPU, driver and disk, downloads the right file from the latest release, checks it, installs it and tells you how to connect your apps.

### Or do it yourself

Download the file for your platform from the [latest release](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/releases/latest):

| Platform | File | Then |
|---|---|---|
| Windows 10/11 (x64) | `AIScalerToolkit-<version>-x64-Setup.exe` | Run it. No administrator rights needed |
| Linux (x86_64, Ubuntu 22.04 or newer) | `AIScalerToolkit-<version>-x86_64.AppImage` | `chmod +x` it and run it |

**You need**: an NVIDIA GPU with compute capability 7.5 or newer (RTX 20 / GTX 16 series and later) and driver 580 or newer, or an Intel Arc GPU or Intel iGPU; and about 20 GB of free disk. On Linux with a recent Intel GPU, add Intel's graphics package source first ([details](docs/installation.en.md#linux)).

**The first launch** sets everything up: it installs the Python packages and the GPU runtime for your card (about 6-10 GB, 5-15 minutes) and a small starter model, then opens the chat. On Linux it asks once for your password to install system packages. Later starts take seconds.

**Updates**: Settings → Updates → Check now → Update now.

Everything else, including uninstalling and Linux servers without a desktop: [Installation](docs/installation.en.md).

---

## 2. Use it from other apps

Point any app that accepts an "OpenAI-compatible" provider at:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1` |
| Model | `trusta-ast-default` (whatever model is loaded) |
| API key | any string |

Hermes Agent, once installed from Settings → Components, is set up with one button: **Config AST Endpoint**. More in [Usage](docs/usage.en.md).

---

## 3. Project structure

```text
AI-Scaler-Toolkit/
├─ backend/                 # The service (FastAPI), synced from the backend repository
│  ├─ service/              # app.py, settings.py, inference engines, training, configs
│  ├─ scripts/
│  │  ├─ linux/             # setup_env.sh, run_service.sh, stop_service.sh
│  │  ├─ windows/           # setup_env.ps1, run_service.bat
│  │  └─ xpu_smoke.py       # Intel GPU check, run by setup_env
│  ├─ tests/
│  ├─ pyproject.toml
│  └─ .env.example          # the service reads backend/.env
├─ console/                 # Headless Python client
├─ frontend/                # Prebuilt web interface (served by the backend)
│  ├─ dist/
│  └─ dist_client/
├─ deploy/                  # The desktop application, the Windows installer, the AppImage
├─ docs/
├─ wiki/
├─ .github/
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

Working on the code: [Installation → Installing from source](docs/installation.en.md#installing-from-source-developers) and [CONTRIBUTING](.github/CONTRIBUTING.md).

---

## 📚 Documentation

- **[Installation & Setup](docs/installation.en.md)** — requirements, first launch, updating, uninstalling, Linux servers, installing from source
- **[Usage & Verification](docs/usage.en.md)** — the window, settings, components, connecting other apps, the API
- **[Maintenance & FAQ](docs/faq.en.md)** — updates and troubleshooting
- **[AI_SETUP.md](docs/AI_SETUP.md)** — installation instructions for AI coding assistants

> 中文文件：[安裝](docs/installation.md) · [使用](docs/usage.md) · [FAQ](docs/faq.md)
