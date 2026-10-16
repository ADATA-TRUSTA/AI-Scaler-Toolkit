# Installing AI Scaler Toolkit for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked
to install AI Scaler Toolkit on their PC. AI Scaler Toolkit is a desktop application that runs LLMs on the user's own
NVIDIA or Intel GPU and serves an OpenAI-compatible API on `http://127.0.0.1:8571`. The human-oriented version of
everything below is [installation.en.md](installation.en.md).

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Never type a password for the user.** On Linux the first launch asks once for the user's password to install
  system packages, in a dialog on their screen. Ask the user to answer it. Do not run `sudo` yourself unless the user
  explicitly asks you to and types the password themselves.
- **Do not install or change GPU drivers** without asking. A driver update can break other software on the PC.
- **Do not open the service to the network.** It listens on `127.0.0.1` only. Leave "Let another computer use this
  machine's model" off unless the user asks for it. If they do, tell them that other computers then need the API key
  shown on the Hermes card, and that anyone with that key can use the model and load or unload models.
- **Tell the user about the download first.** The installer is small (about 30 MB on Windows, 160 MB on Linux), but
  the first launch downloads the GPU runtime and Python packages, about 6-10 GB, plus a small starter model. Confirm
  that is fine, especially on a metered connection.
- First launch is long-running (5-15 minutes, longer on a slow connection). Do not kill it because it is quiet; the
  window shows progress.

## 1. Check the PC

| What | Windows (PowerShell) | Linux |
| --- | --- | --- |
| OS and architecture | `[Environment]::OSVersion; $env:PROCESSOR_ARCHITECTURE` (must be AMD64) | `cat /etc/os-release; uname -m` (must be x86_64); `ldd --version` (glibc 2.35 or newer) |
| NVIDIA GPU and driver | `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv` | the same |
| Intel GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name` | `lspci \| grep -i -E 'vga\|display'` |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` |
| Free disk | `Get-PSDrive C` | `df -h ~` |

Requirements:

- **NVIDIA**: compute capability 7.5 or newer (RTX 20 / GTX 16 series and later) and a driver of version 580 or newer.
  If `nvidia-smi` is missing or older, tell the user they need to install or update the NVIDIA driver first.
- **Intel**: Arc graphics cards and Intel integrated GPUs. On Linux use Ubuntu 24.04 or newer. A recent Intel GPU
  (Lunar Lake, Panther Lake, Arc B-series) also needs Intel's graphics package source, added once by the user:
  `sudo add-apt-repository -y ppa:kobuk-team/intel-graphics`.
- **Disk**: at least 20 GB free, more for the models the user will download.
- **No GPU at all, AMD, ARM or macOS**: not supported. Tell the user rather than installing.

## 2. Download the latest release

Get the newest release from the GitHub API and pick the file for the platform:

| Platform | Asset name |
| --- | --- |
| Windows | `AIScalerToolkit-<version>-x64-Setup.exe` |
| Linux | `AIScalerToolkit-<version>-x86_64.AppImage` |

```text
https://api.github.com/repos/ADATA-TRUSTA/AI-Scaler-Toolkit/releases/latest
```

Each asset in that response has a `browser_download_url` and a `digest` (`sha256:<hex>`). Download the file and
**check its SHA-256 against the `digest`** before running it. If they differ, delete the file and stop.

- Windows: `(Get-FileHash <file>).Hash`
- Linux: `sha256sum <file>`

## 3. Install

### Windows

Run the downloaded `Setup.exe`. It needs no administrator rights and installs for the current user. The user can
click through it, or run it unattended:

```powershell
Start-Process .\AIScalerToolkit-<version>-x64-Setup.exe -ArgumentList '/SILENT','/SUPPRESSMSGBOXES','/NORESTART' -Wait
```

An unattended install refuses to replace a newer version with an older one; that is intentional.

Then start the application from the Start menu or the desktop shortcut ("AI Scaler Toolkit"). If you start it
yourself, start it the way a double-click does, in the user's desktop session:

```powershell
Start-Process "$env:WINDIR\explorer.exe" -ArgumentList "`"$env:LOCALAPPDATA\Programs\AIScalerToolkit\AIScalerToolkit.exe`""
```

### Linux

Keep the AppImage somewhere it will stay (for example `~/Applications/` or the desktop), make it executable and run
it **from the user's desktop session** (not over SSH: it opens a window and the password dialog appears on the
screen):

```bash
chmod +x AIScalerToolkit-<version>-x86_64.AppImage
./AIScalerToolkit-<version>-x86_64.AppImage
```

On the first launch it asks once for the user's password to install what it needs (WebKitGTK, `build-essential`,
`cmake`, and with an Intel GPU its Level Zero, OpenCL and Vulkan packages): llama.cpp is compiled on the machine.
The AppImage needs `fuse3`, which Ubuntu installs by default; if it says `fusermount: not found`, the user needs
`sudo apt install fuse3`.

For a machine with no desktop, see "No desktop" in [installation.en.md](installation.en.md).

## 4. Wait for the first launch to finish

The window shows each step: checking the PC, installing Python packages, downloading the GPU runtime, and fetching a
small starter model (MiniCPM5-1B). When it is done the chat interface opens. You can follow it from outside:

```text
GET http://127.0.0.1:8571/health        -> {"status": "healthy", ...} once the service is up
```

Logs, if something goes wrong:

| | Windows | Linux |
| --- | --- | --- |
| Launcher and setup | `%LOCALAPPDATA%\AIScalerToolkit\logs\launcher.log`, `provision.log` | `~/.local/share/ai-scaler-toolkit/logs/launcher.log`, `provision.log` |
| Service | `...\logs\service.log` | `.../logs/service.log` |

If setup fails, read the last lines of `provision.log` and tell the user what they say. Running the application again
continues where it stopped; it installs only what is missing.

## 5. Tell the user how to use it

- The application window holds the chat and the model list. Closing the window keeps it running in the tray (the
  service stays up for other apps); quit from the tray icon.
- **Other apps and coding agents** can use it as an OpenAI-compatible provider:
  - base URL: `http://127.0.0.1:8571/v1`
  - model: `trusta-ast-default` (whatever model is loaded) or the loaded model's name
  - API key: any string on this computer; another computer needs the key shown on the Hermes card
- **Hermes Agent and ComfyUI** can be installed from Settings → Components. "Config AST Endpoint" points Hermes at
  this machine's model.
- **Updates**: Settings → Updates → Check now, then Update now.

## 6. Uninstalling

- Windows: Settings → Apps → AI Scaler Toolkit → Uninstall. It asks whether to also remove the downloaded models
  (they live in `~/.cache/huggingface`, shared with other Hugging Face tools).
- Linux: there is no uninstaller. Quit the application (tray icon → Quit), then delete:
  - the AppImage itself
  - `~/.local/share/ai-scaler-toolkit` (the environment, about 6-10 GB) and `~/.config/ai-scaler-toolkit`
  - its menu entries and icons: `~/.local/share/applications/ai-scaler-toolkit*.desktop` and
    `~/.local/share/icons/hicolor/256x256/apps/ai-scaler-toolkit*.png`
  - `~/.config/autostart/ai-scaler-toolkit.desktop` and the desktop shortcut `~/Desktop/ai-scaler-toolkit.desktop`

  Downloaded models are in `~/.cache/huggingface`; delete them only if the user wants them gone.
