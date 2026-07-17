"""
FastAPI Service for LLM Inference and Fine-tuning
Supports streaming inference with Accelerate, quantization, and model offload
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Request
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    FileResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
import json
import signal
import sys
import asyncio
import uuid
import threading
import time

from .config_models import (
    InferenceConfig,
    InferenceEngine,
    ChatRequest,
    OpenAIChatCompletionRequest,
    StopGenerationRequest,
    CleanupGenerationMemoryRequest,
    TrainingConfig,
    ModelStatus,
    TrainingStatus,
    TrainingHistoryResponse,
    SystemResourceHistoryResponse,
    MemoryEstimateRequest,
    MemoryEstimateResponse,
    SystemResourcesResponse,
    ModelConversionRequest,
    ConversionResponse,
)
from .model_manager import model_manager
from .training_manager import training_manager
from .inference.memory_estimator import memory_estimator
from .session_manager import session_manager
from .model_registry import model_registry
from .rag_manager import rag_manager
from .download_manager import download_manager
from .inference.engines.vllm_engine import sweep_stale_vllm_processes
from .utils.conversion_manager import conversion_manager
from .utils.openai_request_parser import (
    parse_openai_chat_request_payload,
    sanitize_openai_request_for_logging,
)
from .utils.system_monitor import system_monitor
from .settings import (
    configure_logging,
    LOG_LEVEL,
    SERVICE_HOST,
    SERVICE_PORT,
    UVICORN_RELOAD,
    UVICORN_ACCESS_LOG,
    UVICORN_USE_COLORS,
    MAX_CONCURRENT_GENERATIONS,
    get_uvicorn_log_config,
)
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pydantic import ValidationError
import config_examples


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parent.parent
    for env_path in (project_root / ".env", project_root / ".env.example"):
        if env_path.is_file():
            load_dotenv(env_path)
            break


# 載入 .env 環境變數；若不存在則退回 .env.example
_load_project_env()

logger = configure_logging(__name__)
generation_semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_GENERATIONS))

# 將「前端會話」與「worker 生成任務」分離：
# - session_id: 前端對話會話識別（可長期不變）
# - worker_request_id: 單次生成任務識別（每次 chat 重新產生）
_session_active_worker_request: Dict[str, str] = {}
_worker_request_session: Dict[str, str] = {}
_request_map_lock = threading.Lock()


def _bind_worker_request(session_id: Optional[str], worker_request_id: str) -> None:
    if not session_id:
        return
    sid = str(session_id).strip()
    if not sid:
        return
    with _request_map_lock:
        prev = _session_active_worker_request.get(sid)
        if prev and prev != worker_request_id:
            _worker_request_session.pop(prev, None)
        _session_active_worker_request[sid] = worker_request_id
        _worker_request_session[worker_request_id] = sid


def _unbind_worker_request(worker_request_id: Optional[str]) -> None:
    if not worker_request_id:
        return
    with _request_map_lock:
        sid = _worker_request_session.pop(worker_request_id, None)
        if sid and _session_active_worker_request.get(sid) == worker_request_id:
            _session_active_worker_request.pop(sid, None)


def _resolve_worker_request_id(
    request_id: Optional[str], session_id: Optional[str]
) -> Optional[str]:
    # request_id（若有）優先，保留相容舊行為
    if request_id:
        rid = str(request_id).strip()
        if rid:
            return rid
    if session_id:
        sid = str(session_id).strip()
        if sid:
            with _request_map_lock:
                return _session_active_worker_request.get(sid)
    return None


def _normalize_openai_role(role: Optional[str]) -> str:
    return str(role or "").strip().lower()


def _normalize_request_id(request_id: Optional[str], prefix: str = "req") -> str:
    value = str(request_id or "").strip()
    return value or f"{prefix}-{uuid.uuid4().hex}"


def _resolve_loaded_model_name() -> str:
    config = model_manager.config
    if not config:
        return ""
    return str(config.model_path or config.model_name or "")


def _is_qwen35_model(model_name: Optional[str]) -> bool:
    normalized = str(model_name or "").strip().lower()
    return "qwen3.5" in normalized


def _resolve_model_aware_generation_options(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    enable_thinking: Optional[bool],
) -> Dict[str, Any]:
    """Resolve generation options with model-specific safe defaults."""
    model_name = _resolve_loaded_model_name()

    resolved = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "enable_thinking": enable_thinking,
    }

    if not _is_qwen35_model(model_name):
        return resolved

    # --- Qwen3.5 MoE 模型專用安全參數 ---
    # Qwen3.5 (尤其 MoE 架構如 35B-A3B) 在 repetition_penalty > 1.0 時
    # 會產生多語言混合亂碼 (gibberish)。必須 **無條件** 強制設為 1.0。
    if resolved["repetition_penalty"] > 1.0:
        logger.warning(
            "Qwen3.5 detected: forcing repetition_penalty from %.2f → 1.0 "
            "(values > 1.0 cause mixed-language gibberish for MoE models)",
            resolved["repetition_penalty"],
        )
        resolved["repetition_penalty"] = 1.0

    # 對齊 Qwen3.5 官方 generation_config：
    # temperature=1.0, top_p=0.95, top_k=20。
    # 先前使用 0.7 / 0.8 雖然較保守，但在這個 MoE 模型上容易讓取樣退化，
    # 反而產生跨語言亂碼與不自然輸出。
    if resolved["temperature"] != 1.0:
        logger.warning(
            "Qwen3.5 detected: forcing temperature from %s -> 1.0 for official sampling behavior",
            resolved["temperature"],
        )
        resolved["temperature"] = 1.0

    if resolved["top_p"] != 0.95:
        logger.warning(
            "Qwen3.5 detected: forcing top_p from %s -> 0.95 for official sampling behavior",
            resolved["top_p"],
        )
        resolved["top_p"] = 0.95

    if resolved["top_k"] != 20:
        logger.warning(
            "Qwen3.5 detected: forcing top_k from %s -> 20 for official sampling behavior",
            resolved["top_k"],
        )
        resolved["top_k"] = 20

    # 對 Qwen3.5 一律關閉 thinking。
    # 你提供的日誌已顯示目前仍以 enable_thinking=True 進行生成；
    # 官方範例未使用 thinking mode，且該模式在多模態 MoE 架構下
    # 容易讓模板/停止條件失配，最終退化成多語言亂碼。
    if resolved["enable_thinking"] is not False:
        logger.warning(
            "Qwen3.5 detected: forcing enable_thinking from %s -> False "
            "for generation stability",
            resolved["enable_thinking"],
        )
        resolved["enable_thinking"] = False

    logger.info(
        "Applied model-aware generation options for %s: temp=%s, top_p=%s, top_k=%s, rep_penalty=%s, enable_thinking=%s",
        model_name,
        resolved["temperature"],
        resolved["top_p"],
        resolved["top_k"],
        resolved["repetition_penalty"],
        resolved["enable_thinking"],
    )
    return resolved


_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant."
    "Do not repeat yourself. Do not generate multiple versions of the same answer. "
    "Respond in the same language as the user's question."
)


def _normalize_content_parts(content: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI-style content parts for downstream generation."""
    if not isinstance(content, list):
        return []

    normalized_parts: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = str(part.get("type", "")).strip().lower()
        if part_type in {"text", "input_text"}:
            text = part.get("text") or part.get("content")
            if isinstance(text, str) and text.strip():
                normalized_parts.append({"type": "text", "text": text})
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str) and url.strip():
                normalized_parts.append(
                    {"type": "image_url", "image_url": {"url": url.strip()}}
                )

    return normalized_parts


def _normalize_prompt_content(content: Any) -> Any:
    """Normalize message content while preserving multimodal parts."""
    if isinstance(content, list):
        return _normalize_content_parts(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _extract_text_from_content(content: Any) -> str:
    """Flatten textual content for session persistence and system prompt merge."""
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts: List[str] = []
        for part in _normalize_content_parts(content):
            if part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _content_has_prompt_payload(content: Any) -> bool:
    """Check whether content still contains text or image payload."""
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for part in _normalize_content_parts(content):
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return True
        elif part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.strip():
                    return True
    return False


def _coerce_prompt_message(role: str, content: Any) -> Optional[Dict[str, Any]]:
    """Create a normalized prompt message if payload exists."""
    normalized_content = _normalize_prompt_content(content)
    if not _content_has_prompt_payload(normalized_content):
        return None
    return {"role": role, "content": normalized_content}


def _session_history_to_prompt_messages(history: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    """Convert persisted text-only session history into prompt messages."""
    system_messages: List[str] = []
    prompt_messages: List[Dict[str, Any]] = []

    for item in history:
        if not isinstance(item, dict):
            continue
        role = _normalize_openai_role(item.get("role"))
        text = _extract_text_from_content(item.get("content"))
        if not text:
            continue

        if role == "system":
            system_messages.append(text)
        elif role in {"user", "assistant"}:
            prompt_messages.append({"role": role, "content": text})

    return system_messages, prompt_messages


def _resolve_rag_context_text(request: OpenAIChatCompletionRequest) -> Optional[str]:
    """Resolve RAG context for OpenAI-compatible requests."""
    if not getattr(request, "use_rag", False):
        return None

    try:
        fallback_query = ""
        for msg in reversed(request.messages):
            if _normalize_openai_role(msg.role) != "user":
                continue
            fallback_query = _extract_text_from_content(msg.content)
            if fallback_query:
                break

        rag_query = request.rag_query or fallback_query
        results = rag_manager.search(rag_query, k=getattr(request, "rag_top_k", 3))
        if not results:
            return None

        lines = []
        for i, result in enumerate(results, 1):
            snippet = result.get("snippet") or (result.get("content") or "")[:200]
            doc_id = result.get("doc_id")
            if getattr(request, "rag_include_sources", True):
                lines.append(f"[Source {i} | id={doc_id}]\n{snippet}\n")
            else:
                lines.append(snippet)
        return "\n".join(lines)
    except Exception as exc:
        logger.warning(f"RAG retrieval failed: {exc}")
        return None


def _build_openai_prompt_messages(
    request: OpenAIChatCompletionRequest,
) -> tuple[List[Dict[str, Any]], Optional[str], str]:
    """Build final generation messages directly from OpenAI request payload."""
    session_id = request.session_id or request.user
    if request.reset_history and session_id:
        session_manager.reset(session_id)

    last_user_idx = -1
    for idx, msg in enumerate(request.messages):
        if _normalize_openai_role(msg.role) == "user":
            last_user_idx = idx

    if last_user_idx < 0:
        raise HTTPException(status_code=400, detail="messages must include at least one user message")

    explicit_history = request.messages[:last_user_idx]
    current_user = request.messages[last_user_idx]
    current_user_prompt = _coerce_prompt_message("user", current_user.content)
    current_user_text = _extract_text_from_content(current_user.content)
    if current_user_prompt is None:
        raise HTTPException(status_code=400, detail="last user message content is empty")

    explicit_system_texts: List[str] = []
    explicit_history_prompt: List[Dict[str, Any]] = []
    session_history_snapshot: List[Dict[str, str]] = []

    for msg in explicit_history:
        role = _normalize_openai_role(msg.role)
        text = _extract_text_from_content(msg.content)
        if role == "system":
            if text:
                explicit_system_texts.append(text)
                session_history_snapshot.append({"role": role, "content": text})
            continue

        if role not in {"user", "assistant"}:
            continue

        prompt_message = _coerce_prompt_message(role, msg.content)
        if prompt_message is not None:
            explicit_history_prompt.append(prompt_message)
        if text:
            session_history_snapshot.append({"role": role, "content": text})

    effective_system_texts: List[str]
    effective_history_prompt: List[Dict[str, Any]]

    if explicit_history:
        effective_system_texts = explicit_system_texts
        effective_history_prompt = explicit_history_prompt
        if session_id:
            session_manager.set_history(session_id, session_history_snapshot)
    elif session_id:
        stored_history = session_manager.get_history(session_id)
        effective_system_texts, effective_history_prompt = _session_history_to_prompt_messages(
            stored_history
        )
    else:
        effective_system_texts = []
        effective_history_prompt = []

    system_instruction = "\n\n".join([text for text in effective_system_texts if text]).strip()
    if not system_instruction:
        system_instruction = _DEFAULT_SYSTEM_PROMPT

    rag_context_text = _resolve_rag_context_text(request)
    if rag_context_text:
        system_instruction += (
            "\n\nReference information (use only if helpful):\n"
            + rag_context_text.strip()
        )

    recent_history = effective_history_prompt[-6:] if len(effective_history_prompt) > 6 else effective_history_prompt
    prompt_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_instruction}
    ]
    prompt_messages.extend(recent_history)
    prompt_messages.append(current_user_prompt)
    return prompt_messages, session_id, current_user_text


# 自定义 asyncio 异常处理，抑制无害的 socket.send() 警告
def custom_exception_handler(loop, context):
    """处理 asyncio 中的异常，抑制客户端断开连接时的警告"""
    exception = context.get("exception")
    message = context.get("message", "")

    # 忽略客户端断开连接时的 socket 错误
    if exception and isinstance(exception, (ConnectionError, BrokenPipeError, OSError)):
        # 这些是正常的客户端断开，不记录警告
        return

    # 检查是否是 socket.send() 相关的错误
    if "socket.send()" in message or "Connection" in message:
        # 记录为 debug 级别，不作为 warning
        logger.debug(f"Client connection closed: {message}")
        return

    # 其他异常正常处理
    if exception:
        logger.error(f"Asyncio exception: {exception}", exc_info=exception)
    else:
        logger.error(f"Asyncio error: {context}")


def signal_handler(signum, frame):
    """處理終止信號"""
    logger.info(f"Received signal {signum}, cleaning up...")
    try:
        model_manager.cleanup()
        logger.info("✅ Model manager cleanup completed")
    except Exception as e:
        logger.error(f"Error during signal cleanup: {e}")

    try:
        session_manager.close()
        logger.info("✅ Session manager cleanup completed")
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")

    sys.exit(0)


# 註冊信號處理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill 命令


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    logger.info("🚀 Starting LLM Service...")

    # 啟動階段先清理可能殘留的 vLLM serve 進程，避免後續 load_model 發生 port 衝突。
    try:
        sweep_result = sweep_stale_vllm_processes()
        if sweep_result.get("enabled"):
            logger.info(
                "vLLM startup sweep finished: ports=%s killed=%s pids=%s",
                sweep_result.get("ports"),
                sweep_result.get("killed"),
                sweep_result.get("pids"),
            )
    except Exception as e:
        logger.warning(f"vLLM startup sweep failed (non-blocking): {e}")

    # 确保 asyncio 异常处理器已设置
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_exception_handler)
        logger.info("✅ Asyncio exception handler configured")
    except Exception as e:
        logger.warning(f"Failed to set asyncio exception handler: {e}")

    yield

    logger.info("🛑 Shutting down LLM Service...")
    # Cleanup - 使用新的清理方法
    try:
        model_manager.cleanup()
        logger.info("✅ Model manager cleanup completed")
    except Exception as e:
        logger.error(f"Error during model_manager cleanup: {e}")
    # Close session manager resources
    try:
        session_manager.close()
        logger.info("✅ Session manager cleanup completed")
    except Exception as e:
        logger.error(f"Error during session_manager cleanup: {e}")


# Create FastAPI app
app = FastAPI(
    title="LLM Inference & Training Service",
    description="FastAPI service for LLM inference with streaming and fine-tuning support",
    version="1.0.0",
    lifespan=lifespan,
)

# ==================== Frontend Static Files ====================
# React build (e.g. created by `npm run build`) expected at: Trusta-AST-Frontend/dist
# We mount it at /frontend for direct file access, and provide a SPA fallback route
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "Trusta-AST-Frontend" / "dist"
if FRONTEND_DIST.exists():
    # html=True enables index.html serving on /frontend/ requests
    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIST), html=True),
        name="frontend",
    )
    logger.info(f"✅ Frontend static mounted: {FRONTEND_DIST}")
    # Also mount assets under root /assets if React build references absolute /assets/... paths
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir), html=False),
            name="frontend-assets",
        )
        logger.info(f"✅ Frontend assets mounted at /assets from: {assets_dir}")
    # Serve common top-level build artifacts that Vite may reference with absolute paths
    # e.g. /vite.svg, /favicon.ico. Add lightweight explicit routes to avoid broad catch-all.
    from fastapi import APIRouter

    _frontend_router = APIRouter(include_in_schema=False)

    _TOP_LEVEL_FILES = [
        "Trusta-logo.svg",
        "Trusta-16.ico",
        "Adata.ico",
        "manifest.json",
        "robots.txt",
        "config.json",
    ]
    for fname in _TOP_LEVEL_FILES:
        file_path = FRONTEND_DIST / fname
        if file_path.is_file():

            @_frontend_router.get(f"/{fname}")  # type: ignore
            async def _serve_file(fp=str(file_path)):
                return FileResponse(fp)

            logger.info(f"🔗 Frontend top-level asset route added: /{fname}")
        else:
            logger.debug(f"Top-level asset not found (skip): {file_path}")
    app.include_router(_frontend_router)
else:
    logger.warning(
        f"⚠️ Frontend dist directory not found, skipping mount: {FRONTEND_DIST}"
    )


@app.get("/frontend/{full_path:path}")
async def frontend_spa_fallback(full_path: str):
    """SPA fallback: if a deep route isn't an existing file, serve index.html.
    This allows React Router (or similar) client-side routing to work when refreshing.
    """
    if not FRONTEND_DIST.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    candidate = FRONTEND_DIST / full_path
    if candidate.is_file():
        return FileResponse(str(candidate))
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="index.html not found in dist")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================


@app.get("/")
async def root():
    """根路徑：若前端存在則 302 轉址至 /frontend/，否則返回服務資訊"""
    try:
        if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").is_file():
            return RedirectResponse(url="/frontend/")
    except Exception as e:
        logger.warning(f"Root redirect check failed: {e}")
    return {
        "message": "LLM Inference & Training Service",
        "version": "1.0.0",
        "frontend": "/frontend/" if FRONTEND_DIST.exists() else None,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "model_loaded_config": (
            model_manager.config.model_dump() if model_manager.is_loaded() else None
        ),
        "training_active": training_manager.get_status().is_training,
    }

@app.get("/v1/models")
def list_models():
    if model_manager.is_loaded():
        return {
            "object": "list",
            "data": [{"id": "trusta-ast-default", "object": "model"}]
        }
    else:
        return {
            "object": "list",
            "data": []
        }
# ==================== Inference Endpoints ====================


@app.post("/inference/load_model")
async def load_model(
    config: InferenceConfig = Body(
        ..., openapi_examples=config_examples.INFERENCE_CONFIG_EXAMPLES
    )
):
    """
    加載推理模型
    前端傳送模型名稱、量化類型和 offload 設定

    注意：模型加載在後台執行，立即返回。使用 /inference/status 檢查載入狀態。
    """
    try:
        logger.info(f"Loading model request: {config.model_name}")

        # 若已經有模型處於載入完成狀態，根據規則回應
        if model_manager.is_loaded():
            current = model_manager.config

            # 判斷是否與現有配置相同（同一模型 + 關鍵配置一致）
            def _norm(v):
                # 將 dict/其他型別正規化為可比較的穩定字符串
                try:
                    if isinstance(v, dict):
                        return json.dumps(v, sort_keys=True)
                    return json.dumps(v, sort_keys=True)
                except Exception:
                    return str(v)

            def _same_inference_config(a, b):
                if a is None or b is None:
                    return False
                # 模型識別：優先使用 model_path，其次 model_name
                a_id = a.model_path or a.model_name
                b_id = b.model_path or b.model_name
                if a_id != b_id:
                    return False
                # 比較關鍵欄位
                fields = [
                    "quantization",
                    "device_map",
                    "max_memory",
                    "offload_folder",
                    "torch_dtype",
                    "model_total_memory",
                ]
                for f in fields:
                    if _norm(getattr(a, f, None)) != _norm(getattr(b, f, None)):
                        return False
                return True

            if _same_inference_config(current, config):
                return {
                    "status": "already_loaded",
                    "message": f"Model {config.model_name} is already loaded with the same configuration.",
                    "config": config.model_dump(),
                }
            # 不同模型或不同配置 → 409 請先卸載
            raise HTTPException(
                status_code=409,
                detail="A model is already loaded. Please unload the model before loading a new model.",
            )

        # 檢查是否已經有模型正在加載
        status = model_manager.get_status()
        if status.get("is_loading"):
            raise HTTPException(
                status_code=409,
                detail="A model is already being loaded. Please wait or check status.",
            )

        # 分階段載入：立即設定 config，後台進程執行實際權重載入
        try:
            model_manager.start_loading(config)
        except ValueError as ve:
            # 配置檢查錯誤（例如量化限制）→ 400
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            # 其它當前狀態衝突（理論上前面已處理）→ 409
            raise HTTPException(status_code=409, detail=str(re))

        return {
            "status": "loading",
            "message": f"Model {config.model_name} is loading in background. Check status for progress.",
            "config": config.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model loading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/unload_model")
async def unload_model():
    """卸載模型"""
    try:
        return model_manager.unload_model()

    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/stop_generation")
async def stop_generation(
    request: StopGenerationRequest = Body(default_factory=StopGenerationRequest),
):
    """
    停止当前正在进行的生成
    适用于流式和非流式生成
    """
    try:
        target_request_id = _resolve_worker_request_id(
            request.request_id, request.session_id
        )
        result = model_manager.stop_generation(request_id=target_request_id)
        if target_request_id:
            _unbind_worker_request(target_request_id)
        if request.session_id and target_request_id:
            result["session_id"] = request.session_id
        if target_request_id:
            result["request_id"] = target_request_id
        return result
    except Exception as e:
        logger.error(f"Failed to stop generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/cleanup_generation_memory")
async def cleanup_generation_memory(
    request: CleanupGenerationMemoryRequest = Body(
        default_factory=CleanupGenerationMemoryRequest
    ),
):
    """
    清理生成過程中累積的記憶體（KV cache 和中間激活）
    不卸載模型，適用於：
    - 長對話後釋放 KV cache
    - 切換 session 時清理記憶體
    - OOM 錯誤後的恢復（在重試前調用）
    """
    try:
        result = model_manager.cleanup_generation_memory(slot=request.slot)
        return result
    except Exception as e:
        logger.error(f"Failed to cleanup generation memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/force_cleanup_gpu")
async def force_cleanup_gpu():
    """
    強制清理 GPU 記憶體
    當模型加載因 OOM 失敗且 VRAM 未釋放時使用
    會強制終止 worker 進程並重啟一個乾淨的進程
    """
    try:
        logger.warning("Force GPU cleanup requested")
        result = model_manager.force_cleanup_gpu()
        return result
    except Exception as e:
        logger.error(f"Failed to force cleanup GPU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/status", response_model=ModelStatus)
async def get_model_status():
    """獲取模型狀態"""
    try:
        status = model_manager.get_status()
        return ModelStatus(**status)
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/error_details")
async def get_error_details():
    """
    獲取詳細的錯誤信息（包括完整的 traceback）
    當模型加載失敗時，可以通過此端點獲取完整的錯誤堆棧信息
    """
    try:
        error_details = model_manager.get_error_details()
        if error_details is None:
            return {"has_error": False, "message": "No error occurred"}

        return {
            "has_error": True,
            "error": error_details.get("error"),
            "error_type": error_details.get("error_type"),
            "is_oom": error_details.get("is_oom", False),
            "error_traceback": error_details.get("error_traceback"),
            "process_alive": error_details.get("process_alive", False),
        }
    except Exception as e:
        logger.error(f"Failed to get error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Memory Estimation Endpoints ====================


@app.post("/inference/estimate_memory", response_model=MemoryEstimateResponse)
async def estimate_memory_requirements(
    request: MemoryEstimateRequest = Body(
        ..., openapi_examples=config_examples.MEMORY_ESTIMATE_EXAMPLES
    )
):
    """
    估計模型在 offload 混合模式下的 GPU 記憶體需求

    此端點會根據模型名稱和量化配置，估算：
    - 完整 GPU 模式所需的記憶體
    - CPU offload 混合模式的最低 GPU 記憶體需求
    - Disk offload 模式的最低 GPU 記憶體需求
    - 不同 offload 策略的建議

    Args:
        request: 包含 model_name, quantization 等參數的請求

    Returns:
        詳細的記憶體需求估計和 offload 策略建議
    """
    try:
        logger.info(
            f"Estimating memory for model: {request.model_name}, quantization: {request.quantization}"
        )

        result = memory_estimator.estimate_memory_requirements(
            model_name=request.model_name,
            quantization=request.quantization.value,
            include_activations=request.include_activations,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
        )

        # 如果估計失敗（無法識別模型）
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)

        return MemoryEstimateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/estimate_memory/{model_name}")
async def estimate_memory_by_name(
    model_name: str,
    quantization: str = "none",
    batch_size: int = 1,
    sequence_length: int = 2048,
):
    """
    通過 URL 參數快速估計模型記憶體需求（簡化版）

    範例: /inference/estimate_memory/llama-2-7b?quantization=int8
    """
    try:
        # 處理 URL 編碼的模型名稱
        from urllib.parse import unquote

        decoded_model_name = unquote(model_name)

        result = memory_estimator.estimate_memory_requirements(
            model_name=decoded_model_name,
            quantization=quantization,
            include_activations=True,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/chat", deprecated=True)
async def chat(request: ChatRequest):
    """已棄用；請改用 OpenAI 相容 API。"""
    raise HTTPException(
        status_code=410,
        detail={
            "error": "deprecated_endpoint",
            "message": "'/inference/chat' 已棄用，請改用 '/v1/chat/completions'。",
        },
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(http_request: Request):
    """OpenAI 相容聊天端點。"""

    raw_payload = await parse_openai_chat_request_payload(http_request)
    logger.info(
        "Received /v1/chat/completions request: content_type=%s payload=%s",
        http_request.headers.get("content-type", ""),
        json.dumps(
            sanitize_openai_request_for_logging(raw_payload),
            ensure_ascii=False,
        ),
    )

    try:
        request = OpenAIChatCompletionRequest.model_validate(raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    backend_request_id = _normalize_request_id(request.request_id, prefix="openai")

    def _stop_backend_generation() -> None:
        try:
            model_manager.stop_generation(request_id=backend_request_id)
        except Exception as exc:
            logger.debug(
                "Failed to stop backend generation for %s: %s",
                backend_request_id,
                exc,
            )

    def _resolve_enable_thinking() -> Optional[bool]:
        if "enable_thinking" in request.model_fields_set:
            return request.enable_thinking

        chat_template_kwargs = request.chat_template_kwargs
        if isinstance(chat_template_kwargs, dict):
            value = chat_template_kwargs.get("enable_thinking")
            if isinstance(value, bool):
                return value

        return None

    def _resolve_repetition_penalty() -> float:
        if request.presence_penalty is None:
            return request.repetition_penalty
        return max(1.0, float(request.presence_penalty))

    def _build_openai_stream_error(
        message: Any, error_type: str = "server_error"
    ) -> Dict[str, Any]:
        detail = message
        if isinstance(detail, dict):
            error_message = (
                detail.get("message")
                or detail.get("error")
                or json.dumps(detail, ensure_ascii=False)
            )
            error_code = detail.get("code")
        else:
            error_message = str(detail)
            error_code = None

        payload: Dict[str, Any] = {
            "error": {
                "message": error_message,
                "type": error_type,
            }
        }
        if error_code is not None:
            payload["error"]["code"] = error_code
        return payload

    def _should_include_stream_usage() -> bool:
        stream_options = request.stream_options
        if not isinstance(stream_options, dict):
            return False
        return bool(stream_options.get("include_usage"))

    def _build_openai_usage(
        prompt_tokens: Any,
        completion_tokens: Any,
        total_tokens: Any,
        payload: Any = None,
    ) -> Optional[Dict[str, Any]]:
        payload_dict = payload if isinstance(payload, dict) else {}

        if all(v is None for v in [prompt_tokens, completion_tokens, total_tokens]) and all(
            payload_dict.get(key) is None
            for key in ["gen_tokens", "gen_tps", "prompt_tps"]
        ):
            return None

        pt = int(
            prompt_tokens
            if prompt_tokens is not None
            else (payload_dict.get("prompt_tokens") or 0)
        )
        ct = int(
            completion_tokens
            if completion_tokens is not None
            else (payload_dict.get("gen_tokens") or 0)
        )
        tt = int(
            total_tokens
            if total_tokens is not None
            else (payload_dict.get("total_tokens") or (pt + ct))
        )

        usage: Dict[str, Any] = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
        }
        if payload_dict.get("gen_tokens") is not None:
            usage["gen_tokens"] = payload_dict.get("gen_tokens")
        if payload_dict.get("gen_tps") is not None:
            usage["gen_tps"] = payload_dict.get("gen_tps")
        if payload_dict.get("prompt_tps") is not None:
            usage["prompt_tps"] = payload_dict.get("prompt_tps")
        return usage

    def _normalize_tool_choice_for_backend(tool_choice: Any) -> Any:
        if not isinstance(tool_choice, dict):
            return tool_choice

        choice_type = str(tool_choice.get("type", "")).strip().lower()
        if choice_type in {"auto", "none", "required"}:
            return choice_type

        if choice_type in {"function", "tool"}:
            function_obj = tool_choice.get("function")
            if not isinstance(function_obj, dict):
                function_obj = {}

            name = function_obj.get("name") or tool_choice.get("name")
            if isinstance(name, str) and name.strip():
                return {
                    "type": "function",
                    "function": {"name": name.strip()},
                }

        return tool_choice

    def _has_tooling_payload() -> bool:
        if request.tools or request.tool_choice is not None:
            return True
        for msg in request.messages:
            role = _normalize_openai_role(msg.role)
            if role == "tool":
                return True
            if getattr(msg, "tool_calls", None) or getattr(msg, "tool_call_id", None):
                return True
        return False

    def _should_fallback_tool_passthrough_stream() -> bool:
        return False

    def _is_known_tool_stream_mismatch_error(error: Any) -> bool:
        message = str(error or "").strip().lower()
        if not message:
            return False
        known_patterns = [
            "invalid diff",
            "less tool calls",
            "more tool calls",
            "tool call mismatch",
            "tool_calls",
        ]
        return any(pattern in message for pattern in known_patterns)

    def _iter_text_stream_chunks(text: str, chunk_size: int = 32) -> List[str]:
        """Split finalized text into small SSE-friendly chunks for smoother UI updates."""
        normalized = str(text or "")
        if not normalized:
            return []

        chunks: List[str] = []
        current = ""
        for char in normalized:
            current += char
            if char == "\n" or len(current) >= chunk_size:
                chunks.append(current)
                current = ""

        if current:
            chunks.append(current)

        return chunks

    def _build_passthrough_messages() -> List[Dict[str, Any]]:
        passthrough_messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            role = _normalize_openai_role(msg.role)
            item: Dict[str, Any] = {"role": role}

            if getattr(msg, "name", None):
                item["name"] = msg.name
            if getattr(msg, "tool_calls", None) is not None:
                item["tool_calls"] = msg.tool_calls
            if getattr(msg, "tool_call_id", None):
                item["tool_call_id"] = msg.tool_call_id

            normalized_content = _normalize_prompt_content(msg.content)
            if _content_has_prompt_payload(normalized_content):
                item["content"] = normalized_content

            if (
                item.get("content") is None
                and item.get("tool_calls") is None
                and item.get("tool_call_id") is None
            ):
                item["content"] = ""

            passthrough_messages.append(item)

        return passthrough_messages

    async def _handle_tool_passthrough() -> JSONResponse | StreamingResponse:
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=400,
                detail="Model not loaded. Please load a model first using /inference/load_model",
            )

        generation_options = _resolve_model_aware_generation_options(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=_resolve_repetition_penalty(),
            enable_thinking=_resolve_enable_thinking(),
        )
        passthrough_messages = _build_passthrough_messages()
        model_name = (
            request.model
            or (
                model_manager.config.model_path
                if model_manager.config and model_manager.config.model_path
                else None
            )
            or (model_manager.config.model_name if model_manager.config else None)
            or "unknown"
        )
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        if not request.stream:
            async with generation_semaphore:
                loop = asyncio.get_running_loop()
                result_queue: asyncio.Queue = asyncio.Queue()

                def _non_stream_worker():
                    try:
                        internal_resp = model_manager.generate(
                            prompt=passthrough_messages,
                            max_new_tokens=request.max_tokens,
                            temperature=generation_options["temperature"],
                            top_p=generation_options["top_p"],
                            top_k=generation_options["top_k"],
                            repetition_penalty=generation_options["repetition_penalty"],
                            system_prompt=None,
                            total_timeout=request.total_timeout,
                            enable_thinking=generation_options["enable_thinking"],
                            tools=request.tools,
                            tool_choice=_normalize_tool_choice_for_backend(request.tool_choice),
                            request_id=backend_request_id,
                        )
                        loop.call_soon_threadsafe(
                            result_queue.put_nowait, ("ok", internal_resp)
                        )
                    except Exception as exc:
                        loop.call_soon_threadsafe(result_queue.put_nowait, ("error", exc))

                threading.Thread(target=_non_stream_worker, daemon=True).start()
                kind, payload = await result_queue.get()

            if kind == "error":
                raise HTTPException(status_code=500, detail=str(payload))

            internal_resp = payload
            if not isinstance(internal_resp, dict):
                raise HTTPException(
                    status_code=500, detail="Unexpected non-stream response type"
                )

            content = internal_resp.get("response", internal_resp.get("result", ""))
            tool_calls = internal_resp.get("tool_calls")
            finish_reason = internal_resp.get("finish_reason") or (
                "tool_calls" if tool_calls else "stop"
            )
            message: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
                if not content:
                    message["content"] = None

            output: Dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            usage_payload = _build_openai_usage(
                internal_resp.get("prompt_tokens"),
                internal_resp.get("gen_tokens"),
                internal_resp.get("total_tokens"),
                internal_resp,
            )
            if usage_payload is not None:
                output["usage"] = usage_payload
            return JSONResponse(content=output)

        async def _passthrough_stream_to_openai():
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            async with generation_semaphore:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                use_non_stream_fallback = _should_fallback_tool_passthrough_stream()

                def _producer():
                    try:
                        backend_kwargs = {
                            "prompt": passthrough_messages,
                            "max_new_tokens": request.max_tokens,
                            "temperature": generation_options["temperature"],
                            "top_p": generation_options["top_p"],
                            "top_k": generation_options["top_k"],
                            "repetition_penalty": generation_options["repetition_penalty"],
                            "system_prompt": None,
                            "total_timeout": request.total_timeout,
                            "enable_thinking": generation_options["enable_thinking"],
                            "tools": request.tools,
                            "tool_choice": _normalize_tool_choice_for_backend(request.tool_choice),
                            "request_id": backend_request_id,
                        }

                        if use_non_stream_fallback:
                            logger.info(
                                "OpenAI tool passthrough stream uses non-stream backend planning to avoid tool diff mismatch during streaming"
                            )
                            internal_resp = model_manager.generate(**backend_kwargs)
                            internal_error = None
                            if isinstance(internal_resp, dict):
                                internal_error = internal_resp.get("error")
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                (
                                    "data",
                                    {
                                        "error": internal_error,
                                        "chunk": internal_resp.get("response", internal_resp.get("result", "")),
                                        "tool_calls": internal_resp.get("tool_calls"),
                                        "done": True,
                                        "finish_reason": internal_resp.get("finish_reason"),
                                        "prompt_tokens": internal_resp.get("prompt_tokens"),
                                        "gen_tokens": internal_resp.get("gen_tokens"),
                                        "total_tokens": internal_resp.get("total_tokens"),
                                    },
                                ),
                            )
                        else:
                            text_stream_emitted = False
                            try:
                                stream_iterator = model_manager.generate_stream(**backend_kwargs)
                                for item in stream_iterator:
                                    if isinstance(item, dict) and item.get("chunk"):
                                        text_stream_emitted = True
                                    loop.call_soon_threadsafe(queue.put_nowait, ("data", item))
                            except Exception as exc:
                                if (
                                    not text_stream_emitted
                                    and _is_known_tool_stream_mismatch_error(exc)
                                ):
                                    logger.warning(
                                        "OpenAI tool passthrough stream fallback triggered after tool diff mismatch: %s",
                                        exc,
                                    )
                                    internal_resp = model_manager.generate(**backend_kwargs)
                                    internal_error = None
                                    if isinstance(internal_resp, dict):
                                        internal_error = internal_resp.get("error")
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait,
                                        (
                                            "data",
                                            {
                                                "error": internal_error,
                                                "chunk": internal_resp.get("response", internal_resp.get("result", "")),
                                                "tool_calls": internal_resp.get("tool_calls"),
                                                "done": True,
                                                "finish_reason": internal_resp.get("finish_reason"),
                                                "prompt_tokens": internal_resp.get("prompt_tokens"),
                                                "gen_tokens": internal_resp.get("gen_tokens"),
                                                "total_tokens": internal_resp.get("total_tokens"),
                                            },
                                        ),
                                    )
                                else:
                                    raise
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

                threading.Thread(target=_producer, daemon=True).start()

                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == "error":
                            raise payload
                        if kind == "done":
                            break

                        item = payload
                        if not isinstance(item, dict):
                            continue

                        if item.get("error"):
                            logger.error(
                                "OpenAI tool passthrough stream error: %s",
                                item.get("error"),
                            )
                            yield f"data: {json.dumps(_build_openai_stream_error(item.get('error')))}\n\n"
                            return

                        tool_calls = item.get("tool_calls")
                        chunk_text = str(item.get("chunk", "") or "")
                        is_done = item.get("done") is True

                        # Stream text content live; tool_calls are deferred to the done
                        # event to avoid incremental delta inconsistencies (e.g. Hermes format)
                        if chunk_text:
                            for text_chunk in _iter_text_stream_chunks(chunk_text):
                                chunk_payload = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text_chunk},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk_payload)}\n\n"

                        if is_done:
                            # Emit complete accumulated tool_calls as a single chunk
                            if tool_calls:
                                chunk_payload = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"tool_calls": tool_calls},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk_payload)}\n\n"

                            finish_reason = item.get("finish_reason") or (
                                "tool_calls" if tool_calls else "stop"
                            )
                            done_payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(done_payload)}\n\n"

                            if _should_include_stream_usage():
                                usage_payload = _build_openai_usage(
                                    item.get("prompt_tokens"),
                                    item.get("gen_tokens"),
                                    item.get("total_tokens"),
                                    item,
                                )
                                if usage_payload is not None:
                                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [], 'usage': usage_payload})}\n\n"

                            yield "data: [DONE]\n\n"
                            return
                except (asyncio.CancelledError, ConnectionError, BrokenPipeError) as exc:
                    logger.warning(
                        "OpenAI tool passthrough stream disconnected: %s", exc
                    )
                    _stop_backend_generation()
                    raise
                except RuntimeError as exc:
                    logger.warning(
                        "OpenAI tool passthrough stream backend error: %s", exc
                    )
                    _stop_backend_generation()
                    yield f"data: {json.dumps(_build_openai_stream_error(str(exc)))}\n\n"
                    return
                except Exception as exc:
                    logger.error(
                        "OpenAI tool passthrough stream exception: %s", exc
                    )
                    yield f"data: {json.dumps(_build_openai_stream_error(str(exc)))}\n\n"
                    return

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _passthrough_stream_to_openai(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if _has_tooling_payload():
        return await _handle_tool_passthrough()

    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Please load a model first using /inference/load_model",
        )

    generation_options = _resolve_model_aware_generation_options(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=_resolve_repetition_penalty(),
        enable_thinking=_resolve_enable_thinking(),
    )
    prompt_messages, session_id, current_user_text = _build_openai_prompt_messages(
        request
    )
    model_name = (
        request.model
        or (
            model_manager.config.model_path
            if model_manager.config and model_manager.config.model_path
            else None
        )
        or (model_manager.config.model_name if model_manager.config else None)
        or "unknown"
    )
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if not request.stream:
        async with generation_semaphore:
            loop = asyncio.get_running_loop()
            result_queue: asyncio.Queue = asyncio.Queue()

            def _non_stream_worker():
                try:
                    response_payload = model_manager.generate(
                        prompt=prompt_messages,
                        max_new_tokens=request.max_tokens,
                        temperature=generation_options["temperature"],
                        top_p=generation_options["top_p"],
                        top_k=generation_options["top_k"],
                        repetition_penalty=generation_options["repetition_penalty"],
                        system_prompt=None,
                        total_timeout=request.total_timeout,
                        enable_thinking=generation_options["enable_thinking"],
                        request_id=backend_request_id,
                    )
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait, ("ok", response_payload)
                    )
                except Exception as exc:
                    loop.call_soon_threadsafe(result_queue.put_nowait, ("error", exc))

            threading.Thread(target=_non_stream_worker, daemon=True).start()
            kind, payload = await result_queue.get()

        if kind == "error":
            error_str = str(payload)
            is_oom = (
                "out of memory" in error_str.lower()
                or "oom" in error_str.lower()
                or "OutOfMemoryError" in type(payload).__name__
            )
            if is_oom:
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": "Out of Memory (OOM)",
                        "message": error_str,
                        "is_oom": True,
                        "suggestions": [
                            "Use POST /inference/force_cleanup_gpu to clean GPU memory",
                            "Reload model with more CPU/disk offload",
                            "Use smaller max_new_tokens",
                            "Try a smaller model or higher quantization (int4/int8)",
                        ],
                    },
                )
            raise HTTPException(status_code=500, detail=error_str)

        internal_resp = payload if isinstance(payload, dict) else {"result": payload}
        content = internal_resp.get("response", internal_resp.get("result", ""))

        if session_id:
            try:
                if current_user_text:
                    session_manager.append_message(
                        session_id, {"role": "user", "content": current_user_text}
                    )
                session_manager.append_message(
                    session_id, {"role": "assistant", "content": content}
                )
            except Exception as exc:
                logger.error(f"Failed to save session history (openai non-stream): {exc}")

        output: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        usage_payload = _build_openai_usage(
            internal_resp.get("prompt_tokens"),
            internal_resp.get("gen_tokens"),
            internal_resp.get("total_tokens"),
            internal_resp,
        )
        if usage_payload is not None:
            output["usage"] = usage_payload
        return JSONResponse(content=output)

    async def _stream_openai_generation():
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        assistant_text = ""
        async with generation_semaphore:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _producer():
                try:
                    stream_iterator = model_manager.generate_stream(
                        prompt=prompt_messages,
                        max_new_tokens=request.max_tokens,
                        temperature=generation_options["temperature"],
                        top_p=generation_options["top_p"],
                        top_k=generation_options["top_k"],
                        repetition_penalty=generation_options["repetition_penalty"],
                        system_prompt=None,
                        total_timeout=request.total_timeout,
                        enable_thinking=generation_options["enable_thinking"],
                        request_id=backend_request_id,
                    )
                    for item in stream_iterator:
                        loop.call_soon_threadsafe(queue.put_nowait, ("data", item))
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

            threading.Thread(target=_producer, daemon=True).start()

            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "error":
                        raise payload
                    if kind == "done":
                        break

                    item = payload if isinstance(payload, dict) else {"chunk": str(payload)}
                    if item.get("error"):
                        logger.error(
                            "OpenAI compatibility stream error: %s",
                            item.get("error"),
                        )
                        yield f"data: {json.dumps(_build_openai_stream_error(item.get('error')))}\n\n"
                        return

                    chunk_text = item.get("chunk", "")
                    if chunk_text:
                        assistant_text += chunk_text
                        chunk_payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"

                    if item.get("done") is True:
                        done_payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                        yield f"data: {json.dumps(done_payload)}\n\n"

                        if _should_include_stream_usage():
                            usage_payload = _build_openai_usage(
                                item.get("prompt_tokens"),
                                item.get("gen_tokens"),
                                item.get("total_tokens"),
                                item,
                            )
                            if usage_payload is not None:
                                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [], 'usage': usage_payload})}\n\n"

                        if session_id:
                            try:
                                if current_user_text:
                                    session_manager.append_message(
                                        session_id,
                                        {"role": "user", "content": current_user_text},
                                    )
                                session_manager.append_message(
                                    session_id,
                                    {"role": "assistant", "content": assistant_text},
                                )
                            except Exception as exc:
                                logger.error(
                                    f"Failed to save session history (openai stream): {exc}"
                                )

                        yield "data: [DONE]\n\n"
                        return
            except (asyncio.CancelledError, ConnectionError, BrokenPipeError, RuntimeError) as exc:
                logger.warning("OpenAI compatibility stream disconnected: %s", exc)
                _stop_backend_generation()
                raise
            except Exception as exc:
                logger.error("OpenAI compatibility stream exception: %s", exc)
                yield f"data: {json.dumps(_build_openai_stream_error(str(exc)))}\n\n"
                return

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream_openai_generation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== RAG Endpoints ====================


@app.get("/rag/docs")
async def rag_list_docs():
    """列出目前既有文件"""
    try:
        return JSONResponse(content={"documents": rag_manager.list_documents()})
    except Exception as e:
        logger.error(f"Failed to list RAG docs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/docs")
async def rag_add_doc(payload: dict):
    """新增或更新文件與對應資料庫內容
    請求格式: {"doc_id": Optional[str], "content": str}
    """
    try:
        doc_id = payload.get("doc_id")
        content = payload.get("content")
        if not content or not isinstance(content, str):
            raise HTTPException(
                status_code=400, detail="content is required and must be a string"
            )
        result = rag_manager.add_document(content=content, doc_id=doc_id)
        return JSONResponse(content={"status": "ok", "result": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add RAG doc: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/docs/{doc_id}")
async def rag_delete_doc(doc_id: str):
    """刪除指定文件與資料庫內容"""
    try:
        result = rag_manager.delete_document(doc_id)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        logger.error(f"Failed to delete RAG doc: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search")
async def rag_search(q: str, k: int = 3):
    """搜尋 RAG 文檔，回傳前 k 筆"""
    try:
        results = rag_manager.search(q, k=k)
        return JSONResponse(content={"query": q, "k": k, "results": results})
    except Exception as e:
        logger.error(f"Failed to search RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Training Endpoints ====================


@app.post("/training/start")
async def start_training(
    config: TrainingConfig = Body(
        ..., openapi_examples=config_examples.TRAINING_CONFIG_EXAMPLES
    )
):
    """
    開始訓練
    支持 LoRA/QLoRA/Full Parameter Training
    """
    try:
        logger.info(
            f"Starting training request: {config.model_name} with method {config.method}"
        )

        # Check if model is loaded for inference
        if model_manager.is_loaded():
            logger.warning(
                "Inference model is loaded. Consider unloading it before training."
            )

        # Start training
        result = training_manager.start_training(config)

        return {
            "status": "success",
            "message": "Training started",
            "config": config.model_dump(),
            "result": result,
        }

    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """獲取訓練狀態"""
    try:
        status = training_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/training/status/{session_id}/history", response_model=TrainingHistoryResponse
)
async def get_training_history(session_id: str):
    """獲取訓練歷史紀錄（包含 Loss, Learning Rate 等）"""
    try:
        history = training_manager.get_history(session_id)
        if not history or "training_logs" not in history:
            raise HTTPException(
                status_code=404,
                detail="Training session not found or no history available",
            )

        return TrainingHistoryResponse(
            session_id=session_id, logs=history["training_logs"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/system/resource/{session_id}/history",
    response_model=SystemResourceHistoryResponse,
    response_model_exclude_none=True,
)
async def get_system_resource_history(session_id: str):
    """獲取訓練時的系統資源歷史紀錄"""
    try:
        history = training_manager.get_history(session_id)
        if not history or "resource_logs" not in history:
            raise HTTPException(
                status_code=404,
                detail="Training session not found or no history available",
            )

        return SystemResourceHistoryResponse(
            session_id=session_id, resources=history["resource_logs"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system resource history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/error_details")
async def get_training_error_details():
    """
    獲取詳細的訓練錯誤信息（包括完整的 traceback）
    當訓練失敗時，可以通過此端點獲取完整的錯誤堆棧信息
    格式與 /inference/error_details 一致
    """
    try:
        error_details = training_manager.get_error_details()
        if error_details is None:
            return {"has_error": False, "message": "No error occurred"}

        return {
            "has_error": True,
            "error": error_details.get("error"),
            "error_type": error_details.get("error_type"),
            "is_oom": error_details.get("is_oom", False),
            "error_traceback": error_details.get("error_traceback"),
            "process_alive": error_details.get("process_alive", False),
        }
    except Exception as e:
        logger.error(f"Failed to get training error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/stop")
async def stop_training():
    """停止訓練"""
    try:
        result = training_manager.stop_training()
        return {
            "status": "success",
            "message": "Training stop requested",
            "result": result,
        }
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/force_cleanup_gpu")
async def force_cleanup_training_gpu():
    """強制清理 Training 相關 GPU 進程與記憶體。

    行為與 /inference/force_cleanup_gpu 類似：
    - 終止當前 training worker process（若存在）。
    - 重置 TrainingProcessManager 狀態，確保下次訓練在乾淨進程上啟動。
    注意：這不會影響 inference 相關的 worker。
    """
    try:
        logger.warning("Force training GPU cleanup requested")
        # 直接呼叫 training_process_manager.cleanup() 終止當前 worker 進程並重置狀態。
        from .training.training_process import training_process_manager

        training_process_manager.cleanup()
        return {
            "status": "success",
            "message": "Training worker process terminated and state reset.",
        }
    except Exception as e:
        logger.error(f"Failed to force cleanup training GPU: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Configuration Endpoints ====================


@app.get("/config/quantization_types")
async def get_quantization_types():
    """獲取支持的量化類型"""
    return {
        "quantization_types": [
            {
                "value": "none",
                "label": "No Quantization",
                "description": "FP16 Full precision",
            },
            {"value": "int8", "label": "INT8", "description": "8-bit quantization"},
            {"value": "int4", "label": "INT4", "description": "4-bit quantization"},
        ]
    }


@app.get("/config/offload_types")
async def get_offload_types():
    """獲取支持的 offload 類型"""
    return {
        "offload_types": [
            {"value": "none", "label": "No Offload", "description": "Keep all in GPU"},
            {
                "value": "cpu",
                "label": "CPU Offload",
                "description": "Offload to CPU RAM",
            },
            {
                "value": "auto",
                "label": "Auto",
                "description": "Automatically decide based on available max_memory",
            },
        ]
    }


@app.get("/config/training_methods")
async def get_training_methods():
    """獲取支持的訓練方法"""
    return {
        "training_methods": [
            {
                "value": "full",
                "label": "Full Parameter",
                "description": "Train all parameters",
            },
            {"value": "lora", "label": "LoRA", "description": "Low-Rank Adaptation"},
            {
                "value": "qlora",
                "label": "QLoRA",
                "description": "Quantized LoRA (4-bit)",
            },
        ]
    }


# ==================== Models Listing Endpoint ====================


@app.get("/config/models")
async def list_available_models():
    """列出可用的推理模型清單（基礎模型 + 本地微調模型）"""
    try:
        data = model_registry.list_models()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/models/refresh_context_lengths")
async def refresh_model_context_lengths():
    """
    刷新所有 Hugging Face 模型的 max context length
    從 HF API 獲取最新的 max_position_embeddings 資訊
    """
    try:
        # 嘗試讀取 HF token
        from .utils.token_utils import load_hf_token

        hf_token = load_hf_token()

        model_registry.refresh_all_context_lengths(hf_token)

        return {
            "status": "success",
            "message": "Model context lengths refreshed successfully",
        }
    except Exception as e:
        logger.error(f"Failed to refresh context lengths: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/models/download")
async def download_and_register_model(
    model_id: str = Body(
        ..., description="Hugging Face 模型 ID，例如: 'Qwen/Qwen2.5-0.5B-Instruct'"
    ),
    label: Optional[str] = Body(
        None,
        description="自訂標籤名稱，若不提供則使用模型 ID 的最後部分。若是 GGUF 檔案，請在此填入檔案完整名稱 (如 'model.gguf')",
    ),
    cache_dir: Optional[str] = Body(
        None, description="自訂 cache 根目錄，若不提供則使用 HF_HOME 或預設路徑"
    ),
    force_download: bool = Body(
        False, description="是否強制重新下載（即使本地已存在）"
    ),
    filename: Optional[str] = Body(
        None, description="[GGUF專用] 若指定此欄位，則僅下載該檔案並註冊為 GGUF 模型"
    ),
):
    """
    從 Hugging Face 下載模型並註冊到 model registry

    此端點會：
    1. 啟動背景下載任務（使用 DownloadManager）
    2. 返回 task_id 以供查詢進度

    查詢進度: GET /config/models/download/{task_id}
    """
    try:
        # 優先使用明確傳入的 filename
        target_filename = filename

        # 生成標籤（若未提供）
        if not label:
            label = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        else:
            # 便利功能：如果 label 看起來像是 GGUF 檔名且 user 未指定 filename，則將其視為 filename
            if not target_filename and (label.endswith(".gguf") or ".gguf" in label):
                target_filename = label

        # 檢查標籤是否已存在
        existing_models = model_registry.list_models()
        if target_filename:
            gguf_models = existing_models.get("llama_gguf_models", [])
            # GGUF 模型的註冊特例：label 可能是路徑，也可能是顯示名稱。
            # 這裡我們只做簡單檢查，download manager 會處理路徑覆蓋
            pass
        else:
            base_models = existing_models.get("base_models", [])
            # 若強制下載，則允許重複（覆蓋）; 但通常標籤唯一性會保留
            if not force_download and any(m.get("label") == label for m in base_models):
                raise HTTPException(
                    status_code=409,
                    detail=f"Model with label '{label}' already exists in registry. Use force_download=true to re-download.",
                )

        # 啟動下載任務
        task_id = download_manager.start_download(
            model_id=model_id,
            label=label,
            cache_dir=cache_dir,
            force_download=force_download,
            filename=target_filename,
        )

        return {
            "status": "started",
            "message": "Model download started"
            + (f" (File: {target_filename})" if target_filename else ""),
            "task_id": task_id,
            "model_id": model_id,
            "label": label,
            "check_status_url": f"/config/models/download/{task_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/models/download/{task_id}")
async def get_download_status(task_id: str):
    """查詢下載任務狀態"""
    task = download_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/config/models/downloads")
async def list_download_tasks():
    """列出所有下載任務"""
    return {"tasks": download_manager.list_tasks()}


@app.post("/config/models/convert", response_model=ConversionResponse)
async def convert_model_to_gguf(
    request: ModelConversionRequest = Body(
        ..., openapi_examples=config_examples.CONVERSION_CONFIG_EXAMPLES
    ),
):
    """
    將 Hugging Face 模型轉換為 GGUF 格式
    """
    try:
        job_id = conversion_manager.start_conversion(
            model_path=request.model_path,
            output_path=request.output_path,
            outtype=request.outtype,
        )
        return ConversionResponse(
            job_id=job_id, status="pending", message="Conversion job started"
        )
    except Exception as e:
        logger.error(f"Failed to start conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/models/convert/{job_id}", response_model=ConversionResponse)
async def get_conversion_status(job_id: str):
    """
    查詢模型轉換狀態
    """
    status = conversion_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return ConversionResponse(
        job_id=status["job_id"], status=status["status"], message=status["message"]
    )


@app.delete("/config/models/{label:path}")
async def delete_model(label: str, delete_files: bool = False):
    """
    刪除已註冊的模型

    Args:
        label: 模型標籤
        delete_files: 是否同時刪除本地檔案 (預設 False)
    """
    try:
        # 1. Remove from registry
        model_info = model_registry.delete_model(label)

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model with label '{label}' not found"
            )

        result = {
            "status": "deleted",
            "label": label,
            "registry_removed": True,
            "files_removed": False,
            "path": None,
        }

        # 2. Delete local files if requested
        if delete_files:
            import shutil
            from pathlib import Path

            path_to_delete = None

            if model_info.get("type") == "base":
                # For base models, check 'local_path'
                raw_path = model_info.get("local_path")
                if raw_path:
                    path_to_delete = raw_path
                    # 優化：如果是 Hugging Face Cache 結構，嘗試刪除整個模型目錄以釋放 blobs 空間
                    # 標準結構: .../models--Org--Repo/snapshots/HASH
                    try:
                        p = Path(raw_path)
                        if "snapshots" in p.parts:
                            # 往上找直到找到 models-- 開頭的目錄
                            current = p
                            while len(current.parts) > 1:
                                if current.name.startswith("models--"):
                                    path_to_delete = str(current)
                                    logger.info(
                                        f"Detected HF cache, expanding deletion path to model root: {path_to_delete}"
                                    )
                                    break
                                current = current.parent
                    except Exception as e:
                        logger.warning(f"Error parsing HF cache path: {e}")
                else:
                    # Fallback: 如果沒有 local_path，嘗試從 HF_HOME 推斷預設路徑
                    hf_id = model_info.get("base_model_name")
                    if hf_id and "/" in hf_id:
                        try:
                            hf_home = os.getenv("HF_HOME") or os.path.expanduser(
                                "~/.cache/huggingface"
                            )
                            # HF cache directory naming convention: models--Org--Repo
                            dir_name = "models--" + hf_id.replace("/", "--")
                            potential_path = Path(hf_home) / dir_name

                            if potential_path.exists():
                                path_to_delete = str(potential_path)
                                logger.info(
                                    f"No local_path found, but detected default HF cache path: {path_to_delete}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Error inferring default HF cache path: {e}"
                            )

            elif model_info.get("type") == "finetuned":
                # For finetuned models, check 'output_dir'
                path_to_delete = model_info.get("output_dir")

            elif model_info.get("type") == "llama_gguf":
                # For GGUF models, check 'local_path' or 'filename'
                path_to_delete = model_info.get("local_path")
                # If no local_path, check filename (might be absolute path in some legacy entries)
                if not path_to_delete and model_info.get("filename"):
                    path_to_delete = model_info.get("filename")

                # If path detected, check if it is part of HF cache (to delete entire model folder)
                if path_to_delete:
                    try:
                        p = Path(path_to_delete)
                        if "snapshots" in p.parts:
                            # 往上找直到找到 models-- 開頭的目錄
                            current = p
                            while len(current.parts) > 1:
                                if current.name.startswith("models--"):
                                    path_to_delete = str(current)
                                    logger.info(
                                        f"Detected HF cache for GGUF model, expanding deletion path to model root: {path_to_delete}"
                                    )
                                    break
                                current = current.parent
                    except Exception as e:
                        logger.warning(
                            f"Error parsing GGUF path for HF cache detection: {e}"
                        )

            if path_to_delete:
                # Guard against deleting OS system directories via a poisoned
                # registry entry (e.g. output_dir="/etc" from a malicious config).
                _BLOCKED_DELETE_PREFIXES = (
                    "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
                    "/lib", "/lib64", "/boot", "/sys", "/proc", "/dev", "/run",
                    "/root", "/home",
                )
                _resolved_delete = os.path.realpath(path_to_delete)
                if not os.path.lexists(path_to_delete):
                    result["files_removed"] = True
                    result["path"] = path_to_delete
                    result["file_deletion_note"] = (
                        "Path already missing; treated as deleted"
                    )
                    logger.info(
                        f"Model files for {label} already missing at {path_to_delete}; treated as deleted"
                    )
                elif any(
                    _resolved_delete == p or _resolved_delete.startswith(p + "/")
                    for p in _BLOCKED_DELETE_PREFIXES
                ):
                    logger.error(
                        f"Blocked attempt to delete protected system path: {path_to_delete}"
                    )
                    result["file_deletion_error"] = "Deletion blocked: path resolves to a protected system directory"
                else:
                    try:
                        if os.path.isdir(path_to_delete):
                            shutil.rmtree(path_to_delete)
                        else:
                            os.remove(path_to_delete)
                        result["files_removed"] = True
                        result["path"] = path_to_delete
                        logger.info(
                            f"Deleted local files for model {label} at {path_to_delete}"
                        )
                    except FileNotFoundError:
                        result["files_removed"] = True
                        result["path"] = path_to_delete
                        result["file_deletion_note"] = (
                            "Path already missing during deletion; treated as deleted"
                        )
                        logger.info(
                            f"Model files for {label} disappeared during deletion at {path_to_delete}; treated as deleted"
                        )
                    except Exception as e:
                        logger.error(f"Failed to delete files at {path_to_delete}: {e}")
                        result["file_deletion_error"] = str(e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System Resource Endpoints ====================


@app.get("/system/resources", response_model=SystemResourcesResponse)
async def get_system_resources(
    mode: str = "spec", disk_path: str = "/", calc_size: bool = False
):
    """
    查詢系統資源（CPU/RAM/GPU/DISK）。
    - mode=spec: 回傳硬體規格（總量、型號等）
    - mode=usage: 回傳當前使用量（負載、已用記憶體等）

    可選參數：
    - disk_path: 指定磁碟查詢路徑（預設 "/"）
    """
    try:
        # Validate mode
        if mode not in ["spec", "usage"]:
            raise HTTPException(
                status_code=400, detail="mode must be 'spec' or 'usage'"
            )

        # calc_size triggers a recursive os.walk; block OS system directories
        # that are never valid data/model paths to prevent filesystem enumeration.
        if calc_size:
            _BLOCKED_CALC_PREFIXES = (
                "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
                "/lib", "/lib64", "/boot", "/sys", "/proc", "/dev", "/run",
                "/root", "/home",
            )
            _resolved = os.path.realpath(disk_path)
            if any(
                _resolved == p or _resolved.startswith(p + "/")
                for p in _BLOCKED_CALC_PREFIXES
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"calc_size is not allowed for system directory: {disk_path}",
                )

        cpu_res = system_monitor.get_cpu_resource(mode)
        gpu_res = system_monitor.get_gpu_resource(mode)
        disk_res = system_monitor.get_disk_resource(
            path=disk_path, calc_size=calc_size, mode=mode
        )

        return SystemResourcesResponse(
            mode=mode,
            timestamp=datetime.now().isoformat() + "Z",
            cpu=cpu_res,
            gpu=gpu_res,
            disk=disk_res,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Example Usage ====================


@app.get("/examples/inference")
async def get_inference_example():
    """獲取推理配置範例"""
    return {
        "load_model_example": {
            "model_name": "Qwen/Qwen3-4B",
            "quantization": "int4",
            "offload": "auto",
            "device_map": "auto",
            "torch_dtype": "auto",
        },
        "chat_example": {
            "message": "What is machine learning?",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stream": True,
            "system_prompt": "You are a helpful AI assistant.",
        },
    }


@app.get("/examples/training")
async def get_training_example():
    """獲取訓練配置範例"""
    return {
        "qlora_example": {
            "model_name": "Qwen/Qwen3-4B",
            "method": "qlora",
            "dataset_path": "dataset/mydataset.jsonl",
            "output_dir": "./output/qlora_training",
            "offload": "none",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
        },
        "lora_example": {
            "model_name": "Qwen/Qwen3-8B",
            "method": "lora",
            "dataset_path": "dataset/mydataset.jsonl",
            "output_dir": "./output/lora_training",
            "offload": "cpu",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "num_train_epochs": 3,
        },
    }


@app.get("/examples/conversion")
async def get_conversion_example():
    """獲取 GGUF 轉換範例"""
    return {
        "recommended_outtypes": ["auto", "f16", "bf16", "q8_0", "f32"],
        "notes": {
            "auto": "推薦，讓 llama.cpp 依權重自動選擇常見 16-bit 類型",
            "f16": "最常見的半精度完整模型格式",
            "bf16": "常用於原始模型本身是 bfloat16 的情況",
            "q8_0": "常見高品質量化格式，體積較小",
            "f32": "全精度，體積最大，通常只用於驗證或除錯",
        },
        "examples": config_examples.CONVERSION_CONFIG_EXAMPLES,
    }


def run() -> None:
    """Start uvicorn using settings.py.

    讓 `python -m service.app` 與 shell script 啟動時，都能共用同一套 settings。
    """
    import uvicorn

    uvicorn.run(
        "service.app:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=UVICORN_RELOAD,
        log_level=LOG_LEVEL.lower(),
        log_config=get_uvicorn_log_config(),
        access_log=UVICORN_ACCESS_LOG,
        use_colors=UVICORN_USE_COLORS,
    )


if __name__ == "__main__":
    run()
