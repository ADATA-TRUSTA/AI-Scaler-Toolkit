# Installation & Setup

[← Back to README](../README.md)

> Using an AI coding assistant? It can do all of this for you: see [AI_SETUP.md](AI_SETUP.md).

## Installing a release (most users)

Download the file for your platform from the GitHub [Releases](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/releases) page. Every version has two, one per architecture:

| Platform | File | Requires |
|----------|------|----------|
| Windows | `AIScalerToolkit-<version>-x64-Setup.exe` | Windows 10/11, x64. No administrator rights needed |
| Linux | `AIScalerToolkit-<version>-x86_64.AppImage` | x86_64, glibc 2.35 or newer (Ubuntu 22.04 or newer, for example) |

**GPU**: NVIDIA needs compute capability 7.5 or newer (RTX 20, GTX 16 series and later) and an R580 or newer driver; Intel Arc and iGPUs work too. Only x64/x86_64 builds exist; there is no ARM build.

**Disk**: the first launch downloads about 6-10 GB (Python packages and the GPU runtime); keep at least 20 GB free, plus room for the models you download.

Each release page lists the SHA-256 of both files. The Windows installer is signed by ADATA Technology Co., Ltd.

### Windows

Run `Setup.exe`. It installs for your account only, into `%LOCALAPPDATA%\Programs\AIScalerToolkit`, and adds a Start menu entry (a desktop shortcut is optional). Start **AI Scaler Toolkit** from either.

### Linux

**Before the first launch**: make the file executable (`chmod +x`, or tick "Allow executing file as program" in its properties), or double-clicking it does nothing. Keep it somewhere it will stay (for example `~/Applications/`): the menu entry and the autostart entry point at that file. The AppImage mounts itself through FUSE using `fuse3`, which Ubuntu installs by default; `libfuse2` is not needed. If it reports `fusermount: not found`, run `sudo apt install fuse3`.

**First launch**: the AppImage installs the system packages it is missing in one go, asking once for your password:

- WebKitGTK (to draw its window) and the tray icon
- the build tools `build-essential` and `cmake`: llama.cpp is compiled on your machine, tuned to its CPU
- **with an Intel GPU**, also: Level Zero and OpenCL (`libze1`, `libze-intel-gpu1`, `intel-opencl-icd`, through which PyTorch uses the GPU), `libze-dev` (fine-tuning needs it), `intel-ocloc`, the Vulkan driver and tools (`mesa-vulkan-drivers`, `vulkan-tools`), and `libvulkan-dev`, `glslc` and `spirv-headers`, which llama.cpp's Vulkan build compiles against. On Windows these come with Intel's driver; a stock Ubuntu has none of them
- **a recent Intel GPU (Lunar Lake, Panther Lake, Arc B-series) needs Intel's graphics package source first**: the version Ubuntu 24.04 ships (from 2023) is too old to see the GPU at all. Once, before the first launch:

  ```bash
  sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
  ```

  The AppImage does not add a package source for you, since it changes what every later system update installs. Ubuntu 22.04's archive has no Level Zero, so for an Intel GPU use 24.04 or newer
- an NVIDIA driver is proprietary and yours to install beforehand (`nvidia-smi` has to run)

### What the first launch does

On both platforms the first launch shows a setup window and then the application:

1. checks the PC (disk space, the window runtime, the GPU)
2. installs the Python packages for your GPU (CUDA for NVIDIA, XPU for Intel) and the llama.cpp runtime. With both an NVIDIA and an Intel GPU it uses NVIDIA; see [FAQ Q14](faq.en.md#q14-i-have-an-nvidia-and-an-intel-gpu-which-one-is-used) to use the Intel one instead
3. starts the service and fetches a small starter model (MiniCPM5-1B), so the model list is not empty

This takes 5-15 minutes, longer on a slow connection. If it is interrupted, start the application again: it continues where it stopped and installs only what is missing. Later starts take seconds.

The service listens on `http://127.0.0.1:8571` (this computer only). See [Usage](usage.en.md) for the window, the settings and the API.

### Updating

Settings → Updates → **Check now**, then **Update now**. The application downloads the new version, checks it against the SHA-256 published with the release (and on Windows its signature), installs it and reopens. When a new version needs different Python packages, its first start brings the environment up to date (seconds to a few minutes).

You can also install a newer release over an existing one: download it and run it as above. Your models and settings are kept.

### Uninstalling

**Windows**: Settings → Apps → AI Scaler Toolkit → Uninstall. It removes the program and the environment it installed, and asks whether to remove the downloaded models too.

**Linux**: there is no uninstaller. Quit the application (tray icon → Quit), then delete:

- the AppImage
- `~/.local/share/ai-scaler-toolkit` (the environment) and `~/.config/ai-scaler-toolkit` (settings)
- `~/.local/share/applications/ai-scaler-toolkit*.desktop`, `~/.local/share/icons/hicolor/256x256/apps/ai-scaler-toolkit*.png`, `~/.config/autostart/ai-scaler-toolkit.desktop` and `~/Desktop/ai-scaler-toolkit.desktop`

Downloaded models stay in `~/.cache/huggingface` until you delete them.

### Where things are

| | Windows | Linux |
|---|---|---|
| Program | `%LOCALAPPDATA%\Programs\AIScalerToolkit` | the AppImage |
| Environment, logs | `%LOCALAPPDATA%\AIScalerToolkit` (`logs\launcher.log`, `provision.log`, `service.log`) | `~/.local/share/ai-scaler-toolkit` (`logs/…`) |
| Settings | `%LOCALAPPDATA%\AIScalerToolkit\state.json` | `~/.config/ai-scaler-toolkit/state.json` |
| Downloaded models | `%USERPROFILE%\.cache\huggingface` | `~/.cache/huggingface` |

The model cache is the standard Hugging Face location, so it is shared with other Hugging Face tools on the machine.

### No desktop (Linux servers)

`--headless` sets up and runs the service with no window. It cannot ask for a password, so install the system packages first:

```bash
sudo apt install build-essential cmake
# and with an Intel GPU (a recent one: add the PPA above first):
sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd libze-dev intel-ocloc mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc spirv-headers
chmod +x AIScalerToolkit-*-x86_64.AppImage
./AIScalerToolkit-*-x86_64.AppImage --headless
```

It provisions on the first run, then keeps the service running on `127.0.0.1:8571` until you stop it with Ctrl+C. On a machine without FUSE, set `APPIMAGE_EXTRACT_AND_RUN=1`.

---

## Installing from source (developers)

The rest of this page is for working on the code. A source checkout runs the backend directly: there is no setup window, the `.env` is yours to write, and the service port comes from it (`.env.example` uses `8000`).

### Requirements

| Tool | Description |
|------|-------------|
| Git | Cloning, and fetching llama.cpp's conversion scripts at setup |
| `uv` | Python and environment management; it installs Python 3.12 itself |
| Linux: `build-essential`, `cmake` | llama.cpp is compiled from source on Linux |

Linux (Ubuntu / Debian):

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Windows (PowerShell): install [Git for Windows](https://git-scm.com/download/win), then uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On Windows llama.cpp is not compiled: setup downloads the official prebuilt release.

> **Feature support**: vLLM runs on **Linux** only, installed by default with NVIDIA (CUDA) and opt-in on Intel (`TRUSTA_SETUP_VLLM=1`). Fine-tuning runs on **Linux** only (CUDA or Intel XPU). On Windows the engines are Transformers and llama.cpp. To use vLLM or fine-tuning on a Windows PC with an NVIDIA GPU, run the Linux version inside WSL2 (an Ubuntu distribution): for example the AppImage with `--headless`, as above.
>
> Static frontend assets are already included in the repository, so you do not need Node.js or a frontend build.

### Get the code

A plain `git clone` is enough; there are no submodules.

```bash
git clone https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit.git
cd AI-Scaler-Toolkit
```

### Create `.env`

When the service starts, `backend/service/settings.py` loads `backend/.env`. If it does not exist, it falls back to `backend/.env.example`, which means the example's values, including a model cache inside the project. Copy it and set at least the paths:

```bash
cp backend/.env.example backend/.env          # Windows: Copy-Item backend\.env.example backend\.env
```

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000

HF_HOME=/home/<user>/.cache/huggingface      # Windows: C:\Users\<user>\.cache\huggingface
LOG_DIR=/home/<user>/AI-Scaler-Toolkit/logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
```

Optional settings are commented out in `.env.example`; remove the `#` and fill in a value to use one, for example `HF_TOKEN=hf_xxx` for private Hugging Face models. Paths can use `/` or `\`.

| Variable | Purpose | Default |
|------|------|--------|
| `HF_HOME` | Hugging Face models and cache | `<project>/.cache/huggingface` |
| `LOG_DIR` | Log output directory | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | The `llama-server` executable | written by `setup_env`; `backend/service/utils/llama-bin/` |
| `VLLM_SERVER_PROJECT_DIR` | Project whose `.venv` holds vLLM | the backend project (vLLM is installed through the `vllm` extra) |

### Create the Python environment

Linux, NVIDIA:

```bash
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
```

Linux, Intel: `TRUSTA_ACCEL=xpu`. Skip vLLM with `TRUSTA_SETUP_VLLM=0`.

Windows:

```powershell
.\backend\scripts\windows\setup_env.ps1 -Accel cuda      # or -Accel xpu for Intel
```

The script creates `backend/.venv` (you do not need to activate it), installs llama.cpp into `backend/service/utils/llama-bin/` and records it in `.env`, fetches llama.cpp's GGUF conversion scripts into `backend/service/utils/llama.cpp/`, and with an Intel GPU runs a check that the GPU really computes. Each part is installed only when missing, so running it again is cheap.

**llama.cpp**: version `b10107`, overridable with `TRUSTA_LLAMA_VERSION` (Linux) or `-LlamaVersion` (Windows). The build is chosen separately from the torch accelerator with `TRUSTA_LLAMA_BACKEND` (Linux) or `-LlamaBackend` (Windows): `auto` (default) uses CUDA when an NVIDIA card is present and Vulkan otherwise. To reinstall it, `TRUSTA_INSTALL_LLAMA=1` or `-Llama force`; to skip it, `TRUSTA_INSTALL_LLAMA=0` or `-Llama skip`. To use a llama of your own, point `LLAMA_SERVER_BINARY` at it and setup leaves it alone.

### Start the service

```bash
bash backend/scripts/linux/run_service.sh          # Windows: .\backend\scripts\windows\run_service.bat
```

| URL (with `SERVICE_PORT=8000`) | Description |
|------|------|
| `http://127.0.0.1:8000/` | Web interface (redirects to `/frontend/`) |
| `http://127.0.0.1:8000/health` | Health check |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

To run the desktop launcher from a checkout instead, see [deploy/README.md](../deploy/README.md).
