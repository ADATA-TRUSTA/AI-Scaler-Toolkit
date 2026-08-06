"""
Centralized Settings for LLM Service
Contains all configurable parameters for logging, debugging, and service behavior.
"""

import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv

SETTINGS_FILE = Path(__file__).resolve()
SERVICE_DIR = SETTINGS_FILE.parent
PROJECT_ROOT = SERVICE_DIR.parent
# The uv project root is the repo root, so the venv is created beside it —
# see scripts/{linux/run_service.sh,windows/run_service.bat}, which launch from
# this same path. _prepend_cuda_library_paths() globs here for the bundled
# CUDA shared libraries.
VENV_DIR = PROJECT_ROOT / ".venv"
UTILS_DIR = SERVICE_DIR / "utils"
LLAMA_CPP_DIR = UTILS_DIR / "llama.cpp"


def _load_project_env() -> None:
    """Load project environment from `.env`, or fall back to `.env.example`."""
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"):
        if env_path.is_file():
            load_dotenv(env_path)
            break


# Load .env from the project root; fall back to .env.example if it is missing.
_load_project_env()


def _resolve_project_path(raw_path: str | Path, *, base_dir: Path | None = None) -> str:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir or PROJECT_ROOT) / path
    return str(path.resolve())


def _get_env_path(
    env_name: str,
    default_path: str | Path,
    *,
    base_dir: Path | None = None,
) -> str:
    """Read path from env and normalize relative paths for cross-platform deployment."""
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return _resolve_project_path(default_path, base_dir=base_dir)
    return _resolve_project_path(raw_value.strip(), base_dir=base_dir)


def _parse_bool_env(raw_value: str | None, default: bool = False) -> bool:
    """Convert an environment variable string to a boolean."""
    if raw_value is None:
        return default
    return raw_value.strip().lower() in ("true", "1", "yes", "on")


def _prepend_cuda_library_paths() -> None:
    """Best-effort: expose CUDA shared libraries bundled in the venv to subprocesses."""
    site_packages = VENV_DIR / ("Lib" if os.name == "nt" else "lib")
    candidates = []

    if site_packages.exists():
        for pattern in (
            "python*/site-packages/nvidia/cu13/lib",
            "python*/site-packages/nvidia/nvjitlink/lib",
            "python*/site-packages/torch/lib",
            "site-packages/torch/lib",
        ):
            candidates.extend(str(match) for match in site_packages.glob(pattern) if match.is_dir())

    env_var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    existing = [item for item in os.getenv(env_var, "").split(os.pathsep) if item]
    merged: list[str] = []
    for path in candidates + existing:
        if path and path not in merged:
            merged.append(path)

    if merged:
        os.environ[env_var] = os.pathsep.join(merged)


_prepend_cuda_library_paths()

# ==================== Logging Configuration ====================

# Global logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# 💡 Edit here to change the default, or override with the LOG_LEVEL env var
_DEFAULT_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = cast(
    'Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
    os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper(),
)

# Convert string to logging level constant
LOG_LEVEL_INT = getattr(logging, LOG_LEVEL, logging.INFO)

# Enable debug output for response queue messages in model_inference_process
# 💡 Edit here to enable/disable debug output, or override with the RESPONSE_QUEUE_DEBUG env var
_DEFAULT_RESPONSE_QUEUE_DEBUG: bool = False

RESPONSE_QUEUE_DEBUG: bool = os.getenv(
    "RESPONSE_QUEUE_DEBUG", str(_DEFAULT_RESPONSE_QUEUE_DEBUG)
).lower() in ("true", "1", "yes")

# ==================== Logging Format Configuration ====================

# Log format - can be customized for different environments
# 💡 Edit here to change the log format
_DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

LOG_FORMAT = os.getenv("LOG_FORMAT", _DEFAULT_LOG_FORMAT)

# Date format for logs
# 💡 Edit here to change the timestamp format
_DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", _DEFAULT_LOG_DATE_FORMAT)

# ==================== File Logging Configuration ====================

# Enable file logging
_DEFAULT_LOG_TO_FILE: bool = True
LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", str(_DEFAULT_LOG_TO_FILE)).lower() in (
    "true",
    "1",
    "yes",
)

# Log directory and filename
_DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR = _get_env_path("LOG_DIR", _DEFAULT_LOG_DIR)

_DEFAULT_LOG_FILE_NAME = "service.log"
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", _DEFAULT_LOG_FILE_NAME)

# Retention count for rotated logs
_DEFAULT_LOG_BACKUP_COUNT = 14
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", str(_DEFAULT_LOG_BACKUP_COUNT)))

# Whether to enable daily rotation (off by default on Windows to avoid multi-process rename conflicts, WinError 32)
_DEFAULT_LOG_USE_ROTATION: bool = os.name != "nt"
LOG_USE_ROTATION: bool = os.getenv("LOG_USE_ROTATION", str(_DEFAULT_LOG_USE_ROTATION)).lower() in (
    "true",
    "1",
    "yes",
)

# ==================== Service Configuration ====================

# Uvicorn host/port/reload
_DEFAULT_SERVICE_HOST: str = "127.0.0.1"
SERVICE_HOST: str = os.getenv("SERVICE_HOST", _DEFAULT_SERVICE_HOST)

_DEFAULT_SERVICE_PORT: int = 8000
SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", str(_DEFAULT_SERVICE_PORT)))

_DEFAULT_UVICORN_RELOAD: bool = False
UVICORN_RELOAD: bool = os.getenv("UVICORN_RELOAD", str(_DEFAULT_UVICORN_RELOAD)).lower() in (
    "true",
    "1",
    "yes",
)

_DEFAULT_UVICORN_RELOAD_EXCLUDES: tuple[str, ...] = ("logs/*", "logs/**")
UVICORN_RELOAD_EXCLUDES: list[str] = [
    pattern.strip()
    for pattern in os.getenv(
        "UVICORN_RELOAD_EXCLUDES", ",".join(_DEFAULT_UVICORN_RELOAD_EXCLUDES)
    ).split(",")
    if pattern.strip()
]

# Uvicorn access log (prints: "GET /path 200 OK" etc.)
# Note: this is uvicorn's access log, not the app logger.
_DEFAULT_UVICORN_ACCESS_LOG: bool = True
UVICORN_ACCESS_LOG: bool = os.getenv(
    "UVICORN_ACCESS_LOG", str(_DEFAULT_UVICORN_ACCESS_LOG)
).lower() in ("true", "1", "yes")

_DEFAULT_UVICORN_ACCESS_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
UVICORN_ACCESS_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv(
    "UVICORN_ACCESS_LOG_LEVEL", _DEFAULT_UVICORN_ACCESS_LOG_LEVEL
).upper()  # type: ignore[assignment]

_DEFAULT_UVICORN_USE_COLORS: bool = True
UVICORN_USE_COLORS: bool = os.getenv(
    "UVICORN_USE_COLORS", str(_DEFAULT_UVICORN_USE_COLORS)
).lower() in ("true", "1", "yes")

_DEFAULT_WATCHFILES_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
WATCHFILES_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv(
    "WATCHFILES_LOG_LEVEL", _DEFAULT_WATCHFILES_LOG_LEVEL
).upper()  # type: ignore[assignment]

# Maximum timeout for model generation (seconds)
DEFAULT_GENERATION_TIMEOUT: int = int(os.getenv("DEFAULT_GENERATION_TIMEOUT", "300"))

# Maximum new tokens for generation
DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))

# llama-server (OpenAI-compatible endpoint) configuration
_DEFAULT_LLAMA_SERVER_URL: str = "http://127.0.0.1:5001"
LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", _DEFAULT_LLAMA_SERVER_URL)


def _default_llama_server_binary() -> str:
    # The official prebuilt unified `llama` from ggml-org/llama-install.sh
    # (install.ps1 → WindowsApps, install.sh → ~/.local/bin). Point
    # LLAMA_SERVER_BINARY at a custom/source-built llama-server to override.
    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        return str(Path(local_app) / "Microsoft" / "WindowsApps" / "llama.exe")
    return str(Path.home() / ".local" / "bin" / "llama")


# Defaults to the prebuilt llama from install.sh/install.ps1; overridable via env var
_DEFAULT_LLAMA_SERVER_BINARY: str = _default_llama_server_binary()
LLAMA_SERVER_BINARY: str = _get_env_path(
    "LLAMA_SERVER_BINARY",
    _DEFAULT_LLAMA_SERVER_BINARY,
)

LLAMA_SERVER_API_KEY: str | None = os.getenv("LLAMA_SERVER_API_KEY", None)

_DEFAULT_LLAMA_SERVER_TIMEOUT: int = 300
LLAMA_SERVER_TIMEOUT: int = int(
    os.getenv("LLAMA_SERVER_TIMEOUT", str(_DEFAULT_LLAMA_SERVER_TIMEOUT))
)

_DEFAULT_MAX_CONCURRENT_GENERATIONS: int = 8
MAX_CONCURRENT_GENERATIONS: int = int(
    os.getenv("MAX_CONCURRENT_GENERATIONS", str(_DEFAULT_MAX_CONCURRENT_GENERATIONS))
)

# Worker process cleanup timeout (seconds)
WORKER_CLEANUP_TIMEOUT: int = int(os.getenv("WORKER_CLEANUP_TIMEOUT", "5"))

# ==================== vLLM Configuration ====================
# 💡 Central home for vLLM engine env vars


# Whether to sweep leftover vLLM serve processes when the service starts
# Env var: VLLM_STARTUP_SWEEP
_DEFAULT_VLLM_STARTUP_SWEEP: bool = True
VLLM_STARTUP_SWEEP: bool = _parse_bool_env(
    os.getenv("VLLM_STARTUP_SWEEP"), _DEFAULT_VLLM_STARTUP_SWEEP
)

# Ports to check during the startup sweep (comma-separated string)
# Env var: VLLM_SWEEP_PORTS (defaults to "5000", the VLLM_PORT default)
_DEFAULT_VLLM_SWEEP_PORTS: str = "5000"
VLLM_SWEEP_PORTS_RAW: str = os.getenv("VLLM_SWEEP_PORTS", _DEFAULT_VLLM_SWEEP_PORTS)

# Ports actually used by the startup sweep (invalid and out-of-range values filtered out)
VLLM_SWEEP_PORTS: list[int] = []
for item in VLLM_SWEEP_PORTS_RAW.split(","):
    value = item.strip()
    if not value:
        continue
    try:
        port = int(value)
        if 1 <= port <= 65535:
            VLLM_SWEEP_PORTS.append(port)
    except ValueError:
        continue

# OpenAI-compatible API key (used when the engine connects to the local vLLM server)
# Env var: VLLM_OPENAI_API_KEY
_DEFAULT_VLLM_OPENAI_API_KEY: str = "EMPTY"
VLLM_OPENAI_API_KEY: str = os.getenv("VLLM_OPENAI_API_KEY", _DEFAULT_VLLM_OPENAI_API_KEY)

# Seconds to wait on the vLLM server health check (during load_model)
# Env var: VLLM_HEALTH_TIMEOUT
_DEFAULT_VLLM_HEALTH_TIMEOUT: float = 300.0
VLLM_HEALTH_TIMEOUT: float = float(
    os.getenv("VLLM_HEALTH_TIMEOUT", str(_DEFAULT_VLLM_HEALTH_TIMEOUT))
)

# Host the API client uses to reach the vLLM server (usually 127.0.0.1)
# Env var: VLLM_CLIENT_HOST
_DEFAULT_VLLM_CLIENT_HOST: str = "127.0.0.1"
VLLM_CLIENT_HOST: str = os.getenv("VLLM_CLIENT_HOST", _DEFAULT_VLLM_CLIENT_HOST)

# Port the vLLM server binds to and is reached on
# Env var: VLLM_PORT
_DEFAULT_VLLM_PORT: int = 5000
VLLM_PORT: int = int(os.getenv("VLLM_PORT", str(_DEFAULT_VLLM_PORT)))

# Whether to enable vLLM request logging (controls --no-enable-log-requests)
# Env var: VLLM_ENABLE_LOG_REQUESTS
_DEFAULT_VLLM_ENABLE_LOG_REQUESTS: bool = False
VLLM_ENABLE_LOG_REQUESTS: bool = _parse_bool_env(
    os.getenv("VLLM_ENABLE_LOG_REQUESTS"), _DEFAULT_VLLM_ENABLE_LOG_REQUESTS
)

# Host the vLLM server binds to (vllm serve --host)
# Env var: VLLM_SERVER_HOST
_DEFAULT_VLLM_SERVER_HOST: str = "0.0.0.0"  # noqa: S104 - the service is intentionally exposed; binding all interfaces is the default
VLLM_SERVER_HOST: str = os.getenv("VLLM_SERVER_HOST", _DEFAULT_VLLM_SERVER_HOST)

# The served model name vLLM reports; derived from the model source when unset
# Env var: VLLM_SERVED_MODEL_NAME
VLLM_SERVED_MODEL_NAME: str | None = os.getenv("VLLM_SERVED_MODEL_NAME", None)

# Log level for the vLLM server process (written to the subprocess env var VLLM_LOGGING_LEVEL)
# Env var: VLLM_LOGGING_LEVEL
_DEFAULT_VLLM_LOGGING_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
VLLM_LOGGING_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv(
    "VLLM_LOGGING_LEVEL", _DEFAULT_VLLM_LOGGING_LEVEL
).upper()  # type: ignore[assignment]

# Project directory of the isolated vllm_server environment (holds its own .venv and vllm deps)
# Env var: VLLM_SERVER_PROJECT_DIR
_DEFAULT_VLLM_SERVER_PROJECT_DIR: str = str(SERVICE_DIR / "inference" / "engines" / "vllm_server")
VLLM_SERVER_PROJECT_DIR: str = _get_env_path(
    "VLLM_SERVER_PROJECT_DIR", _DEFAULT_VLLM_SERVER_PROJECT_DIR
)

# ==================== Redis Configuration ====================
# 💡 Edit here to change the Redis connection settings, or override with env vars

# Redis Host
_DEFAULT_REDIS_HOST: str = "localhost"
REDIS_HOST: str = os.getenv("REDIS_HOST", _DEFAULT_REDIS_HOST)

# Redis Port
_DEFAULT_REDIS_PORT: int = 6379
REDIS_PORT: int = int(os.getenv("REDIS_PORT", str(_DEFAULT_REDIS_PORT)))

# Redis DB
_DEFAULT_REDIS_DB: int = 0
REDIS_DB: int = int(os.getenv("REDIS_DB", str(_DEFAULT_REDIS_DB)))

# ==================== Path Configuration ====================

# Hugging Face cache directory
# 💡 Edit here to change the HF_HOME path, or override with the HF_HOME env var
_DEFAULT_HF_HOME: str = str(PROJECT_ROOT / ".cache" / "huggingface")

HF_HOME: str = _get_env_path("HF_HOME", _DEFAULT_HF_HOME)

# TikToken cache directory (for OpenAI GPT-OSS models)
# 💡 Edit here to change the TIKTOKEN_RS_CACHE_DIR path, or override with the TIKTOKEN_RS_CACHE_DIR env var
# The cache travels with the backend package (service/ → the synced src/backend/); offline blobs sit at this directory's root
_DEFAULT_TIKTOKEN_CACHE_DIR: str = str(SERVICE_DIR / "caches" / "tiktoken")

TIKTOKEN_CACHE_DIR: str = _get_env_path("TIKTOKEN_RS_CACHE_DIR", _DEFAULT_TIKTOKEN_CACHE_DIR)

# Apply environment variables (must be set before importing transformers/tiktoken)
os.environ["HF_HOME"] = HF_HOME
os.environ["TIKTOKEN_RS_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

# ==================== Helper Functions ====================

_LOGGING_CONFIGURED = False


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Avoid raising PermissionError on failed rotation under Windows/multi-process setups."""

    def doRollover(self) -> None:
        """Roll the log file over, swallowing PermissionError on Windows/multi-process to stay alive."""
        try:
            super().doRollover()
        except PermissionError:
            # Typical case: WinError 32, the target file is held by another process
            # Degraded path: skip this rotation, reopen the stream, set the next rollover time
            try:
                if self.stream:
                    self.stream.close()
            except Exception:
                pass

            try:
                self.stream = self._open()
            except Exception:
                self.stream = None

            current_time = int(time.time())
            next_rollover = self.computeRollover(current_time)
            while next_rollover <= current_time:
                next_rollover += self.interval
            self.rolloverAt = next_rollover


def configure_logging(name: str | None = None) -> logging.Logger:
    """
    Configure and return a logger with centralized settings.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance
    """
    global _LOGGING_CONFIGURED

    # Initialize once per process so every module import does not force a reconfigure
    if not _LOGGING_CONFIGURED:
        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if LOG_TO_FILE:
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                log_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

                if LOG_USE_ROTATION:
                    file_handler = SafeTimedRotatingFileHandler(
                        log_path,
                        when="midnight",
                        interval=1,
                        backupCount=LOG_BACKUP_COUNT,
                        encoding="utf-8",
                        delay=True,
                    )
                    file_handler.suffix = "%Y-%m-%d"
                else:
                    file_handler = logging.FileHandler(
                        log_path,
                        mode="a",
                        encoding="utf-8",
                        delay=True,
                    )

                handlers.append(file_handler)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to set up file logging: {e}")

        logging.basicConfig(
            level=LOG_LEVEL_INT,
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
            handlers=handlers,
            force=True,
        )
        _LOGGING_CONFIGURED = True

    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()

    logger.setLevel(LOG_LEVEL_INT)
    return logger


def get_uvicorn_log_config() -> dict[str, Any]:
    """
    Build a uvicorn-compatible logging config derived from this module settings.

    Why: when starting via `python -m uvicorn ...`, uvicorn applies its own dictConfig
    and may ignore/override `logging.basicConfig`. Providing an explicit log_config
    ensures uvicorn + app logs follow the same level/format/handlers.
    """
    # NOTE: uvicorn will mutate `formatters.default/access.use_colors` inside
    # Config.configure_logging(), so we must provide these keys.
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access_console": {
            "class": "logging.StreamHandler",
            "level": UVICORN_ACCESS_LOG_LEVEL,
            "formatter": "access",
            "stream": "ext://sys.stderr",
        },
    }

    root_handlers = ["console"]

    if LOG_TO_FILE:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_path = os.path.join(LOG_DIR, LOG_FILE_NAME)
            if LOG_USE_ROTATION:
                handlers["file"] = {
                    "class": "service.settings.SafeTimedRotatingFileHandler",
                    "level": LOG_LEVEL,
                    "formatter": "file",
                    "filename": log_path,
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": LOG_BACKUP_COUNT,
                    "encoding": "utf-8",
                    "delay": True,
                }
            else:
                handlers["file"] = {
                    "class": "logging.FileHandler",
                    "level": LOG_LEVEL,
                    "formatter": "file",
                    "filename": log_path,
                    "mode": "a",
                    "encoding": "utf-8",
                    "delay": True,
                }
            root_handlers.append("file")
        except Exception:
            # If folder permission fails, still keep console logging.
            pass

    access_level = UVICORN_ACCESS_LOG_LEVEL if UVICORN_ACCESS_LOG else "CRITICAL"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            # Keep uvicorn's expected formatter keys: default/access
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(message)s",
                "datefmt": LOG_DATE_FORMAT,
                "use_colors": UVICORN_USE_COLORS,
            },
            # Access log formatter (GET /path 200)
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": UVICORN_USE_COLORS,
            },
            # File formatter keeps user-defined LOG_FORMAT (no colors).
            "file": {
                "format": LOG_FORMAT,
                "datefmt": LOG_DATE_FORMAT,
            },
        },
        "handlers": handlers,
        "root": {
            "level": LOG_LEVEL,
            "handlers": root_handlers,
        },
        "loggers": {
            # uvicorn loggers
            "uvicorn": {"level": LOG_LEVEL, "handlers": root_handlers, "propagate": False},
            "uvicorn.error": {"level": LOG_LEVEL, "handlers": root_handlers, "propagate": False},
            "uvicorn.access": {
                "level": access_level,
                "handlers": (["access_console"] + (["file"] if "file" in root_handlers else [])),
                "propagate": False,
            },
            "watchfiles.main": {
                "level": WATCHFILES_LOG_LEVEL,
                "handlers": root_handlers,
                "propagate": False,
            },
        },
    }


def get_response_queue_debug() -> bool:
    """
    Get the current debug setting for response queue logging.

    Returns:
        True if response queue debug output is enabled
    """
    return RESPONSE_QUEUE_DEBUG


# ==================== Example Usage ====================
"""
# In your module:
from .settings import configure_logging, get_response_queue_debug

logger = configure_logging(__name__)

# Use logger as normal:
logger.info("This is an info message")
logger.debug("This is a debug message (only shown if LOG_LEVEL=DEBUG)")

# Check debug flag:
if get_response_queue_debug():
    logger.debug("Detailed response queue info...")
"""
