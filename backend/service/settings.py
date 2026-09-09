"""
Centralized Settings for LLM Service
Contains all configurable parameters for logging, debugging, and service behavior.
"""

import logging
import os
import shutil
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


# Hard ceiling on generated tokens, applied to every request that reaches an
# OpenAI-compatible engine. Unset means the only limit is the served context
# window -- the OpenAI contract for an absent max_tokens, and what llama.cpp
# does with no n_predict. A number here is an operator's blanket limit, not a
# default: it is clamped onto whatever the caller asked for, never substituted
# for a caller that asked for nothing. The name says CAP for that reason -- the
# previous DEFAULT_MAX_NEW_TOKENS was read by nothing, and its 512 is the same
# figure the request model used to substitute, which truncated every
# reasoning-model answer before it left the think block.
def _read_output_token_cap() -> int | None:
    raw = os.getenv("MAX_OUTPUT_TOKENS_CAP", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


MAX_OUTPUT_TOKENS_CAP: int | None = _read_output_token_cap()

# llama-server (OpenAI-compatible endpoint) configuration
_DEFAULT_LLAMA_SERVER_URL: str = "http://127.0.0.1:5001"
LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", _DEFAULT_LLAMA_SERVER_URL)


# Where LLAMA_SERVER_BINARY came from, so a mismatch between what setup_env resolved and
# what the service resolved is visible in the startup log instead of only surfacing later
# as "binary not found". setup_env records its answer in .env precisely so that the common
# case is "env", and the service never has to guess from a possibly different PATH.
LLAMA_SERVER_BINARY_SOURCE: str = "default"

# Where setup_env installs llama: llama.cpp's release zip on Windows, its own build output on
# Linux. This is the single source of truth for that location - scripts/windows/setup_env.ps1
# and scripts/linux/setup_env.sh cannot import Python, so they repeat the literal and
# tests/unit/test_llama_bin_dir_agrees.py asserts all three still agree. Changing it in one
# place only would otherwise install into A while the service looks in B, with no error beyond
# "binary not found".
LLAMA_BIN_DIR: Path = SERVICE_DIR / "utils" / "llama-bin"

# Executable suffix for the current platform, so callers stop repeating the os.name check.
EXE_SUFFIX: str = ".exe" if os.name == "nt" else ""


def _default_llama_server_binary() -> str:
    # Point LLAMA_SERVER_BINARY at another build to override any of this.
    global LLAMA_SERVER_BINARY_SOURCE
    exe = EXE_SUFFIX
    project_bin = LLAMA_BIN_DIR / f"llama-server{exe}"
    if project_bin.is_file():
        LLAMA_SERVER_BINARY_SOURCE = "project (setup_env)"
        return str(project_bin)
    # Fall back to PATH, matching what setup_env accepts as "already installed": it treats a
    # binary on PATH as present and skips the install, so resolving only the default location
    # here would make setup report success while the engine reports a missing binary.
    #
    # ggml-org/llama-install.sh's locations (%LOCALAPPDATA%\Microsoft\WindowsApps\llama.exe and
    # ~/.local/bin/llama) are deliberately not searched: its CUDA builds ship a CPU backend with
    # no vector ISA, which costs 2.1x once any weight is computed on the CPU. Anyone who still
    # wants one has to say so through LLAMA_SERVER_BINARY. See scripts/llama_backend.py.
    on_path = shutil.which(f"llama-server{exe}") or shutil.which("llama-server")
    if on_path:
        LLAMA_SERVER_BINARY_SOURCE = "PATH"
        return on_path
    LLAMA_SERVER_BINARY_SOURCE = "project default (absent)"
    return str(project_bin)


# Defaults to the prebuilt llama from install.sh/install.ps1; overridable via env var
_DEFAULT_LLAMA_SERVER_BINARY: str = _default_llama_server_binary()
LLAMA_SERVER_BINARY: str = _get_env_path(
    "LLAMA_SERVER_BINARY",
    _DEFAULT_LLAMA_SERVER_BINARY,
)
if os.getenv("LLAMA_SERVER_BINARY", "").strip():
    # Set explicitly, or written into .env by setup_env; either way it is a recorded decision.
    LLAMA_SERVER_BINARY_SOURCE = "env/.env"


def effective_llama_binary(configured: str | None) -> str:
    """
    The binary a load actually runs: the per-request override, else the default.

    The engine that spawns it and the status endpoint that reports it must give
    the same answer, because a co-resident tool cannot guess the process name:
    the official installer ships the unified `llama`, a source build is
    `llama-server`.
    """
    return (configured or "").strip() or LLAMA_SERVER_BINARY


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

# Where DeepSpeed writes NVMe offload files when a profile asks for device="nvme".
# Env var: DEEPSPEED_NVME_DIR
#
# The checked-in profiles carry the sentinel "AUTO" rather than a real path:
# they used to hardcode one developer's mount points, which then went stale
# (a UUID that is no longer mounted, and a renamed project directory), so every
# nvme profile crashed at zero.Init on any other machine. A per-job
# `offload_folder` still wins over this default.
_DEFAULT_DEEPSPEED_NVME_DIR: str = str(
    PROJECT_ROOT / ".deepspeed_offload"
)  # portable fallback; set DEEPSPEED_NVME_DIR to a fast, roomy disk
DEEPSPEED_NVME_DIR: str = os.getenv("DEEPSPEED_NVME_DIR", _DEFAULT_DEEPSPEED_NVME_DIR)

# The value profiles use to mean "resolve this at run time".
DEEPSPEED_NVME_PATH_SENTINEL: str = "AUTO"

# Ceiling on page-locked DRAM for DeepSpeed's NVMe swap buffers.
# Env var: DEEPSPEED_NVME_PINNED_BUDGET_GB
#
# Those buffers must each fit the largest single parameter, and DeepSpeed pins
# them regardless of `pin_memory`. Sizing them from the model without a ceiling
# would reserve buffer_count x buffer_size x 4 bytes -- tens of GB. The resolver
# lowers buffer_count to stay under this instead.
_DEFAULT_DEEPSPEED_NVME_PINNED_BUDGET_GB: float = 24.0
DEEPSPEED_NVME_PINNED_BUDGET_BYTES: int = int(
    float(
        os.getenv("DEEPSPEED_NVME_PINNED_BUDGET_GB", str(_DEFAULT_DEEPSPEED_NVME_PINNED_BUDGET_GB))
    )
    * 1024**3
)

# System prompt injected when a chat request carries none of its own.
# Env var: DEFAULT_SYSTEM_PROMPT -- set it to an EMPTY string to inject nothing.
#
# Fine-tuned models need that empty setting. SFT renders only the roles present
# in the training data, so a fine-tune trained without system turns has never
# seen this prefix; injecting it at serve time shifts the prompt (measured on
# gemma-4-E4B: 282 -> 322 tokens) and a small, heavily-overfitted adapter keyed
# on the exact prefix then fails to fire at all.
_DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant."
    "Do not repeat yourself. Do not generate multiple versions of the same answer. "
    "Respond in the same language as the user's question."
)
DEFAULT_SYSTEM_PROMPT: str = os.getenv("DEFAULT_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

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
# Loopback by default: the vLLM server is an internal component that only this
# backend talks to (always via VLLM_CLIENT_HOST), so it has no reason to accept
# off-host connections. Keeping it on loopback is also what lets the engine turn on
# VLLM_SERVER_DEV_MODE, which is the only way vLLM 0.20.1 mounts its admin endpoints
# (/reset_prefix_cache and friends, but also /update_weights and /collective_rpc --
# hence the pairing: admin surface on, reachable from this host only).
# Env var: VLLM_SERVER_HOST
_DEFAULT_VLLM_SERVER_HOST: str = "127.0.0.1"
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

# Project directory whose .venv holds vllm. vllm now comes from this project's own
# `vllm` extra (`uv sync --extra cuda --extra vllm`), so the default is the repo root.
# Kept overridable: a container may mount the venv somewhere else.
# Env var: VLLM_SERVER_PROJECT_DIR
_DEFAULT_VLLM_SERVER_PROJECT_DIR: str = str(PROJECT_ROOT)
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

# How long to wait for a Redis that may not be there. Without a limit this inherits
# the operating system's, which on Windows is tens of seconds -- and the connection is
# attempted during startup, before the server binds, so the whole service is late by
# however long the socket takes to give up. Redis is optional in every path that uses
# it; waiting a long time to establish that it is absent buys nothing.
_DEFAULT_REDIS_CONNECT_TIMEOUT: float = 2.0
REDIS_CONNECT_TIMEOUT: float = float(
    os.getenv("REDIS_CONNECT_TIMEOUT", str(_DEFAULT_REDIS_CONNECT_TIMEOUT))
)

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
