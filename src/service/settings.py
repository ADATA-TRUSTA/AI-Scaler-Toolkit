"""
Centralized Settings for LLM Service
Contains all configurable parameters for logging, debugging, and service behavior
"""
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import time
from typing import Any, Dict, Literal, Optional
from dotenv import load_dotenv


SETTINGS_FILE = Path(__file__).resolve()
SERVICE_DIR = SETTINGS_FILE.parent
PROJECT_ROOT = SERVICE_DIR.parent
VENV_DIR = SERVICE_DIR / ".venv"
UTILS_DIR = SERVICE_DIR / "utils"
LLAMA_CPP_DIR = UTILS_DIR / "llama.cpp"


def _load_project_env() -> None:
    """Load project environment from `.env`, or fall back to `.env.example`."""
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"):
        if env_path.is_file():
            load_dotenv(env_path)
            break


# 先載入專案根目錄 .env；若不存在則退回 .env.example。
_load_project_env()


def _resolve_project_path(raw_path: str | Path, *, base_dir: Optional[Path] = None) -> str:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir or PROJECT_ROOT) / path
    return str(path.resolve())


def _get_env_path(
    env_name: str,
    default_path: str | Path,
    *,
    base_dir: Optional[Path] = None,
) -> str:
    """Read path from env and normalize relative paths for cross-platform deployment."""
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return _resolve_project_path(default_path, base_dir=base_dir)
    return _resolve_project_path(raw_value.strip(), base_dir=base_dir)


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
            for match in site_packages.glob(pattern):
                if match.is_dir():
                    candidates.append(str(match))

    llama_build_bin = LLAMA_CPP_DIR / "build" / "bin"
    if llama_build_bin.is_dir():
        candidates.append(str(llama_build_bin))

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
# 💡 直接修改這裡來改變預設值，或使用環境變數 LOG_LEVEL 覆蓋
_DEFAULT_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv(
    "LOG_LEVEL", _DEFAULT_LOG_LEVEL
).upper()

# Convert string to logging level constant
LOG_LEVEL_INT = getattr(logging, LOG_LEVEL, logging.INFO)

# Enable debug output for response queue messages in model_inference_process
# 💡 直接修改這裡來啟用/停用 debug 輸出，或使用環境變數 RESPONSE_QUEUE_DEBUG 覆蓋
_DEFAULT_RESPONSE_QUEUE_DEBUG: bool = False

RESPONSE_QUEUE_DEBUG: bool = (
    os.getenv("RESPONSE_QUEUE_DEBUG", str(_DEFAULT_RESPONSE_QUEUE_DEBUG)).lower() 
    in ("true", "1", "yes")
)

# ==================== Logging Format Configuration ====================

# Log format - can be customized for different environments
# 💡 直接修改這裡來改變日誌格式
_DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

LOG_FORMAT = os.getenv("LOG_FORMAT", _DEFAULT_LOG_FORMAT)

# Date format for logs
# 💡 直接修改這裡來改變時間格式
_DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", _DEFAULT_LOG_DATE_FORMAT)

# ==================== File Logging Configuration ====================

# Enable file logging
_DEFAULT_LOG_TO_FILE: bool = True
LOG_TO_FILE: bool = (
    os.getenv("LOG_TO_FILE", str(_DEFAULT_LOG_TO_FILE)).lower() in ("true", "1", "yes")
)

# Log directory and filename
_DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR = _get_env_path("LOG_DIR", _DEFAULT_LOG_DIR)

_DEFAULT_LOG_FILE_NAME = "service.log"
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", _DEFAULT_LOG_FILE_NAME)

# Retention count for rotated logs
_DEFAULT_LOG_BACKUP_COUNT = 14
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", str(_DEFAULT_LOG_BACKUP_COUNT)))

# 是否啟用日切輪替（Windows 預設關閉，避免多進程 rename 衝突 WinError 32）
_DEFAULT_LOG_USE_ROTATION: bool = (os.name != "nt")
LOG_USE_ROTATION: bool = (
    os.getenv("LOG_USE_ROTATION", str(_DEFAULT_LOG_USE_ROTATION)).lower() in ("true", "1", "yes")
)

# ==================== Service Configuration ====================

# Uvicorn host/port/reload
_DEFAULT_SERVICE_HOST: str = "127.0.0.1"
SERVICE_HOST: str = os.getenv("SERVICE_HOST", _DEFAULT_SERVICE_HOST)

_DEFAULT_SERVICE_PORT: int = 8000
SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", str(_DEFAULT_SERVICE_PORT)))

_DEFAULT_UVICORN_RELOAD: bool = False
UVICORN_RELOAD: bool = (
    os.getenv("UVICORN_RELOAD", str(_DEFAULT_UVICORN_RELOAD)).lower() in ("true", "1", "yes")
)

# Uvicorn access log (prints: "GET /path 200 OK" etc.)
# 注意：這是 uvicorn 的「存取日誌」，不是 app logger。
_DEFAULT_UVICORN_ACCESS_LOG: bool = True
UVICORN_ACCESS_LOG: bool = (
    os.getenv("UVICORN_ACCESS_LOG", str(_DEFAULT_UVICORN_ACCESS_LOG)).lower() in ("true", "1", "yes")
)

_DEFAULT_UVICORN_ACCESS_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
UVICORN_ACCESS_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv(
    "UVICORN_ACCESS_LOG_LEVEL", _DEFAULT_UVICORN_ACCESS_LOG_LEVEL
).upper()  # type: ignore[assignment]

_DEFAULT_UVICORN_USE_COLORS: bool = True
UVICORN_USE_COLORS: bool = (
    os.getenv("UVICORN_USE_COLORS", str(_DEFAULT_UVICORN_USE_COLORS)).lower() in ("true", "1", "yes")
)

# Maximum timeout for model generation (seconds)
DEFAULT_GENERATION_TIMEOUT: int = int(os.getenv("DEFAULT_GENERATION_TIMEOUT", "300"))

# Maximum new tokens for generation
DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))

# llama-server (OpenAI-compatible endpoint) configuration
_DEFAULT_LLAMA_SERVER_URL: str = "http://127.0.0.1:5001"
LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", _DEFAULT_LLAMA_SERVER_URL)


def _default_llama_server_binary() -> str:
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-server",
    ]

    if os.name == "nt":
        candidates = [
            LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-server.exe",
            LLAMA_CPP_DIR / "build" / "bin" / "llama-server.exe",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str(candidates[0])


# 預設使用專案內 llama.cpp 編譯輸出路徑；可由環境變數覆蓋
_DEFAULT_LLAMA_SERVER_BINARY: str = _default_llama_server_binary()
LLAMA_SERVER_BINARY: str = _get_env_path(
    "LLAMA_SERVER_BINARY",
    _DEFAULT_LLAMA_SERVER_BINARY,
)

LLAMA_SERVER_API_KEY: Optional[str] = os.getenv("LLAMA_SERVER_API_KEY", None)

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
# 💡 vLLM 引擎相關環境變數集中管理


def _parse_bool_env(raw_value: Optional[str], default: bool = False) -> bool:
    """將環境變數字串轉為布林值。"""
    if raw_value is None:
        return default
    return raw_value.strip().lower() in ("true", "1", "yes", "on")


# 啟動服務時是否先清理殘留的 vLLM serve 進程
# 對應環境變數：VLLM_STARTUP_SWEEP
_DEFAULT_VLLM_STARTUP_SWEEP: bool = True
VLLM_STARTUP_SWEEP: bool = _parse_bool_env(
    os.getenv("VLLM_STARTUP_SWEEP"), _DEFAULT_VLLM_STARTUP_SWEEP
)

# 啟動清理要檢查的 port 清單（逗號分隔字串）
# 對應環境變數：VLLM_SWEEP_PORTS （預設 "5000"，即 VLLM_PORT 的預設值）
_DEFAULT_VLLM_SWEEP_PORTS: str = "5000"
VLLM_SWEEP_PORTS_RAW: str = os.getenv("VLLM_SWEEP_PORTS", _DEFAULT_VLLM_SWEEP_PORTS)

# 啟動清理時實際使用的 port 清單（已過濾無效值與範圍）
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

# OpenAI-compatible API key（供 engine 連本地 vLLM server 時使用）
# 對應環境變數：VLLM_OPENAI_API_KEY
_DEFAULT_VLLM_OPENAI_API_KEY: str = "EMPTY"
VLLM_OPENAI_API_KEY: str = os.getenv(
    "VLLM_OPENAI_API_KEY", _DEFAULT_VLLM_OPENAI_API_KEY
)

# vLLM server 健康檢查等待秒數（load_model 階段）
# 對應環境變數：VLLM_HEALTH_TIMEOUT
_DEFAULT_VLLM_HEALTH_TIMEOUT: float = 300.0
VLLM_HEALTH_TIMEOUT: float = float(
    os.getenv("VLLM_HEALTH_TIMEOUT", str(_DEFAULT_VLLM_HEALTH_TIMEOUT))
)

# API client 連線 vLLM server 的 host（通常為 127.0.0.1）
# 對應環境變數：VLLM_CLIENT_HOST
_DEFAULT_VLLM_CLIENT_HOST: str = "127.0.0.1"
VLLM_CLIENT_HOST: str = os.getenv("VLLM_CLIENT_HOST", _DEFAULT_VLLM_CLIENT_HOST)

# vLLM server 綁定/連線的 port
# 對應環境變數：VLLM_PORT
_DEFAULT_VLLM_PORT: int = 5000
VLLM_PORT: int = int(os.getenv("VLLM_PORT", str(_DEFAULT_VLLM_PORT)))

# 是否啟用 vLLM request logging（控制 --no-enable-log-requests）
# 對應環境變數：VLLM_ENABLE_LOG_REQUESTS
_DEFAULT_VLLM_ENABLE_LOG_REQUESTS: bool = False
VLLM_ENABLE_LOG_REQUESTS: bool = _parse_bool_env(
    os.getenv("VLLM_ENABLE_LOG_REQUESTS"), _DEFAULT_VLLM_ENABLE_LOG_REQUESTS
)

# vLLM server 綁定的 host（vllm serve --host）
# 對應環境變數：VLLM_SERVER_HOST
_DEFAULT_VLLM_SERVER_HOST: str = "0.0.0.0"
VLLM_SERVER_HOST: str = os.getenv("VLLM_SERVER_HOST", _DEFAULT_VLLM_SERVER_HOST)

# 指定 vLLM 對外回報的 served model name；未設定時由模型來源自動推導
# 對應環境變數：VLLM_SERVED_MODEL_NAME
VLLM_SERVED_MODEL_NAME: Optional[str] = os.getenv("VLLM_SERVED_MODEL_NAME", None)

# vLLM server 進程日誌等級（寫入子進程環境變數 VLLM_LOGGING_LEVEL）
# 對應環境變數：VLLM_LOGGING_LEVEL
_DEFAULT_VLLM_LOGGING_LEVEL: Literal[
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
] = "ERROR"
VLLM_LOGGING_LEVEL: Literal[
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
] = os.getenv(
    "VLLM_LOGGING_LEVEL", _DEFAULT_VLLM_LOGGING_LEVEL
).upper()  # type: ignore[assignment]

# vllm_server 隔離環境專案目錄（內含獨立 .venv 與 vllm 依賴）
# 對應環境變數：VLLM_SERVER_PROJECT_DIR
_DEFAULT_VLLM_SERVER_PROJECT_DIR: str = str(
    SERVICE_DIR / "inference" / "engines" / "vllm_server"
)
VLLM_SERVER_PROJECT_DIR: str = _get_env_path(
    "VLLM_SERVER_PROJECT_DIR", _DEFAULT_VLLM_SERVER_PROJECT_DIR
)

# ==================== Redis Configuration ====================
# 💡 直接修改這裡來改變 Redis 連線設定，或使用環境變數覆蓋

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
# 💡 直接修改這裡來改變 HF_HOME 路徑，或使用環境變數 HF_HOME 覆蓋
_DEFAULT_HF_HOME: str = str(PROJECT_ROOT / ".cache" / "huggingface")

HF_HOME: str = _get_env_path("HF_HOME", _DEFAULT_HF_HOME)

# TikToken cache directory (for OpenAI GPT-OSS models)
# 💡 直接修改這裡來改變 TIKTOKEN_RS_CACHE_DIR 路徑，或使用環境變數 TIKTOKEN_RS_CACHE_DIR 覆蓋
_DEFAULT_TIKTOKEN_CACHE_DIR: str = str(PROJECT_ROOT)

TIKTOKEN_CACHE_DIR: str = _get_env_path(
    "TIKTOKEN_RS_CACHE_DIR", _DEFAULT_TIKTOKEN_CACHE_DIR
)

# Apply environment variables (must be set before importing transformers/tiktoken)
os.environ["HF_HOME"] = HF_HOME
os.environ["TIKTOKEN_RS_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

# ==================== Helper Functions ====================

_LOGGING_CONFIGURED = False


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """在 Windows/多進程場景下，輪替失敗時避免拋出 PermissionError 中斷日誌流程。"""

    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            # 典型案例：WinError 32, 目標檔案正被另一個 process 持有
            # 退化策略：略過本次輪替、重開 stream、更新下次 rollover 時間
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

def configure_logging(name: str = None) -> logging.Logger:
    """
    Configure and return a logger with centralized settings.
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Configured logger instance
    """
    global _LOGGING_CONFIGURED

    # 僅在 process 內初始化一次，避免每個 module import 時重複 force reconfigure
    if not _LOGGING_CONFIGURED:
        handlers = [logging.StreamHandler()]

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
            force=True
        )
        _LOGGING_CONFIGURED = True
    
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(LOG_LEVEL_INT)
    return logger


def get_uvicorn_log_config() -> Dict[str, Any]:
    """Build a uvicorn-compatible logging config derived from this module settings.

    Why: when starting via `python -m uvicorn ...`, uvicorn applies its own dictConfig
    and may ignore/override `logging.basicConfig`. Providing an explicit log_config
    ensures uvicorn + app logs follow the same level/format/handlers.
    """
    # NOTE: uvicorn will mutate `formatters.default/access.use_colors` inside
    # Config.configure_logging(), so we must provide these keys.
    handlers: Dict[str, Any] = {
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
                "fmt": "%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
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
