"""Where everything lives, per platform.

Every path is per-user. Nothing here needs administrator rights, which is what keeps
the installer per-user too -- the same bargain Ollama makes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AIScalerToolkit"
APP_ID = "ai-scaler-toolkit"
APP_TITLE = "AI Scaler Toolkit"

IS_WINDOWS = sys.platform == "win32"

# Service port. The backend's own default is 8000; the desktop build moves off it so a
# hand-run backend and the packaged one can coexist on one machine.
DEFAULT_SERVICE_PORT = 8571
SERVICE_HOST = "127.0.0.1"

# Left alone on purpose. Both are part of an inference configuration that the
# load-model payload carries, so anything written here is overwritten in normal use.
LLAMA_SERVER_PORT = 5001
VLLM_PORT = 5000

# Provisioning downloads a venv (~3.7 GB), the GPU build of torch (~2.7 GB) and the
# llama binaries. Checked before starting rather than discovered halfway through.
MIN_FREE_DISK_GB = 10


def _win_local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))


def app_dir() -> Path:
    """Installed program directory.

    Frozen: the directory the executable sits in. From source: the repo's deploy/.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """Root for files bundled with the launcher (UI pages, the component list).

    PyInstaller unpacks bundled data under sys._MEIPASS, which is not where the
    modules appear to live, so resolving these from __file__ works from a checkout and
    breaks in the shipped build -- the failure being a missing UI page at first launch.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def payload_dir() -> Path:
    """Root of the shipped payload (backend/, frontend/, console/).

    Installed builds put the payload beside the launcher; running from a source
    checkout it is the repo root one level up.
    """
    here = app_dir()
    if (here / "backend").is_dir():
        return here
    return here.parent


def backend_dir() -> Path:
    return payload_dir() / "backend"


def frontend_config() -> Path:
    """The runtime config the WebUI fetches. Holds apiBaseUrl and hideChatPanel."""
    return payload_dir() / "frontend" / "dist" / "config.json"


def data_dir() -> Path:
    if IS_WINDOWS:
        return _win_local_appdata() / APP_NAME
    return Path.home() / ".local" / "share" / APP_ID


def config_dir() -> Path:
    if IS_WINDOWS:
        return _win_local_appdata() / APP_NAME
    return Path.home() / ".config" / APP_ID


def state_file() -> Path:
    return config_dir() / "state.json"


def log_dir() -> Path:
    return data_dir() / "logs"


def venv_dir() -> Path:
    """Where the backend's virtual environment goes.

    Deliberately outside the program directory: an uninstall-reinstall cycle should not
    have to re-download 6+ GB, and a program directory that grows by gigabytes after
    install surprises people.

    uv would otherwise create .venv inside the backend project, so provision.py passes
    this through UV_PROJECT_ENVIRONMENT.
    """
    return data_dir() / "service" / ".venv"


def hf_home() -> Path:
    """Model cache. Left in the conventional location so it is shared with any other
    Hugging Face tooling the user already has, and so uninstalling can offer to keep
    it -- it reaches tens of gigabytes."""
    return Path.home() / ".cache" / "huggingface"


def llama_binary() -> Path:
    """Where the official llama installer puts the binary, which is also the backend's
    default for LLAMA_SERVER_BINARY. setup_env records the resolved path in .env; this
    is only used for detection."""
    if IS_WINDOWS:
        return _win_local_appdata() / "Microsoft" / "WindowsApps" / "llama.exe"
    return Path.home() / ".local" / "bin" / "llama"


def llama_staging_dir() -> Path:
    """Scratch directory the llama installer stages downloads in. Uninstall clears it."""
    if IS_WINDOWS:
        return _win_local_appdata() / "llama-app"
    return Path.home() / ".llama-app"


def gguf_tools_dir() -> Path:
    """Sparse checkout of the llama.cpp conversion scripts that setup_env creates."""
    return backend_dir() / "service" / "utils" / "llama.cpp"


def setup_env_script() -> Path:
    """The backend's own provisioning script.

    Not reimplemented here. It already detects the accelerator, runs uv sync with the
    right extras, installs llama, fetches the GGUF conversion tooling and writes its
    part of .env. A second copy of that logic would drift from this one within a
    release or two.
    """
    scripts = backend_dir() / "scripts"
    return scripts / "windows" / "setup_env.ps1" if IS_WINDOWS else scripts / "linux" / "setup_env.sh"


def autostart_target() -> Path:
    """Per-user autostart entry. Windows uses a registry value instead of a file, so
    this is only meaningful on Linux."""
    return Path.home() / ".config" / "autostart" / f"{APP_ID}.desktop"


def service_url(port: int) -> str:
    return f"http://{SERVICE_HOST}:{port}"
