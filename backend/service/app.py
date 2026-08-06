"""
FastAPI Service for LLM Inference and Fine-tuning
Supports streaming inference with Accelerate, quantization, and model offload.
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import asyncio
import json
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Annotated, Any, cast

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    ORJSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import config_examples
from .config_models import (
    ChatRequest,
    CleanupGenerationMemoryRequest,
    ConversionResponse,
    GgufConfigCheckResponse,
    GgufEstimateRequest,
    GgufEstimateResponse,
    GgufPlanRequest,
    GgufPlanResponse,
    GgufRecommendRequest,
    GgufRecommendResponse,
    GgufSweepRequest,
    GgufSweepResponse,
    InferenceConfig,
    InferenceEngine,
    MemoryEstimateRequest,
    MemoryEstimateResponse,
    ModelConversionRequest,
    ModelStatus,
    OpenAIChatCompletionRequest,
    StopGenerationRequest,
    SystemResourceHistoryResponse,
    SystemResourcesResponse,
    TrainingConfig,
    TrainingHistoryResponse,
    TrainingLogEventsResponse,
    TrainingStatus,
)
from .download_manager import DownloadTask, download_manager
from .inference.engines.llama_server_engine import resolve_local_model_path
from .inference.engines.vllm_engine import sweep_stale_vllm_processes
from .inference.gguf_estimator import (
    RECOMMEND_SOLVED_KEYS,
    SWEEP_SOLVED_KEYS,
    apply_cli_args,
    gguf_memory_estimator,
)
from .inference.memory_estimator import memory_estimator
from .model_manager import model_manager
from .model_registry import model_registry
from .rag_manager import rag_manager
from .session_manager import session_manager
from .settings import (
    LOG_LEVEL,
    MAX_CONCURRENT_GENERATIONS,
    SERVICE_HOST,
    SERVICE_PORT,
    UVICORN_ACCESS_LOG,
    UVICORN_RELOAD,
    UVICORN_RELOAD_EXCLUDES,
    UVICORN_USE_COLORS,
    configure_logging,
    get_uvicorn_log_config,
)
from .training_manager import training_manager
from .utils.conversion_manager import conversion_manager
from .utils.openai_request_parser import (
    parse_openai_chat_request_payload,
    sanitize_openai_request_for_logging,
)
from .utils.system_monitor import system_monitor


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parent.parent
    for env_path in (project_root / ".env", project_root / ".env.example"):
        if env_path.is_file():
            load_dotenv(env_path)
            break


# Load .env; fall back to .env.example when it does not exist
_load_project_env()

logger = configure_logging(__name__)
generation_semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_GENERATIONS))

# Keep the "frontend session" and the "worker generation task" separate:
# - session_id: identifies a frontend chat session (can stay stable long-term)
# - worker_request_id: identifies a single generation task (regenerated on every chat)
_session_active_worker_request: dict[str, str] = {}
_worker_request_session: dict[str, str] = {}
_request_map_lock = threading.Lock()


def _bind_worker_request(session_id: str | None, worker_request_id: str) -> None:
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


def _unbind_worker_request(worker_request_id: str | None) -> None:
    if not worker_request_id:
        return
    with _request_map_lock:
        sid = _worker_request_session.pop(worker_request_id, None)
        if sid and _session_active_worker_request.get(sid) == worker_request_id:
            _session_active_worker_request.pop(sid, None)


def _resolve_worker_request_id(request_id: str | None, session_id: str | None) -> str | None:
    # request_id takes precedence when present, preserving legacy behavior
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


def _normalize_openai_role(role: str | None) -> str:
    return str(role or "").strip().lower()


def _normalize_request_id(request_id: str | None, prefix: str = "req") -> str:
    value = str(request_id or "").strip()
    return value or f"{prefix}-{uuid.uuid4().hex}"


def _resolve_loaded_model_name() -> str:
    config = model_manager.config
    if not config:
        return ""
    return str(config.model_path or config.model_name or "")


def _is_qwen35_model(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return "qwen3.5" in normalized


def _resolve_model_aware_generation_options(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    enable_thinking: bool | None,
) -> dict[str, Any]:
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

    # --- Safe parameters specific to Qwen3.5 MoE models ---
    # With repetition_penalty > 1.0, Qwen3.5 (especially MoE architectures such as
    # 35B-A3B) produces mixed-language gibberish. It must be forced to 1.0 **unconditionally**.
    if resolved["repetition_penalty"] > 1.0:
        logger.warning(
            "Qwen3.5 detected: forcing repetition_penalty from %.2f → 1.0 "
            "(values > 1.0 cause mixed-language gibberish for MoE models)",
            resolved["repetition_penalty"],
        )
        resolved["repetition_penalty"] = 1.0

    # Match the official Qwen3.5 generation_config:
    # temperature=1.0, top_p=0.95, top_k=20.
    # The previous 0.7 / 0.8 was more conservative, but on this MoE model it easily
    # degrades sampling and instead yields cross-language gibberish and unnatural output.
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

    # Qwen3.5 thinking policy: honor an explicit request value, but default to
    # OFF when unspecified — the GGUF chat template enables thinking by default
    # (prefills '<think>\n'), which silently burns the whole max_tokens budget
    # on reasoning for callers that never asked for it. Explicit toggling is
    # verified working end-to-end against llama-server (--jinja) via
    # chat_template_kwargs.enable_thinking.
    if resolved["enable_thinking"] is None:
        logger.info("Qwen3.5 detected: enable_thinking unspecified; defaulting to False")
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


def _normalize_content_parts(content: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - OpenAI content is dynamic parsed JSON (str/list/dict)
    """Normalize OpenAI-style content parts for downstream generation."""
    if not isinstance(content, list):
        return []

    normalized_parts: list[dict[str, Any]] = []
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
                normalized_parts.append({"type": "image_url", "image_url": {"url": url.strip()}})

    return normalized_parts


def _normalize_prompt_content(content: Any) -> Any:  # noqa: ANN401 - dynamic parsed content in, str-or-list out
    """Normalize message content while preserving multimodal parts."""
    if isinstance(content, list):
        return _normalize_content_parts(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _extract_text_from_content(content: Any) -> str:  # noqa: ANN401 - OpenAI content is dynamic parsed JSON (str/list/dict)
    """Flatten textual content for session persistence and system prompt merge."""
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in _normalize_content_parts(content):
            if part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _content_has_prompt_payload(content: Any) -> bool:  # noqa: ANN401 - OpenAI content is dynamic parsed JSON (str/list/dict)
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


def _coerce_prompt_message(role: str, content: Any) -> dict[str, Any] | None:  # noqa: ANN401 - OpenAI content is dynamic parsed JSON (str/list/dict)
    """Create a normalized prompt message if payload exists."""
    normalized_content = _normalize_prompt_content(content)
    if not _content_has_prompt_payload(normalized_content):
        return None
    return {"role": role, "content": normalized_content}


def _session_history_to_prompt_messages(
    history: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert persisted text-only session history into prompt messages."""
    system_messages: list[str] = []
    prompt_messages: list[dict[str, Any]] = []

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


def _resolve_rag_context_text(request: OpenAIChatCompletionRequest) -> str | None:
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
) -> tuple[list[dict[str, Any]], str | None, str]:
    """Build final generation messages directly from OpenAI request payload."""
    session_id = request.session_id or request.user
    if request.reset_history and session_id:
        session_manager.reset(session_id)

    last_user_idx = -1
    for idx, msg in enumerate(request.messages):
        if _normalize_openai_role(msg.role) == "user":
            last_user_idx = idx

    if last_user_idx < 0:
        raise HTTPException(
            status_code=400, detail="messages must include at least one user message"
        )

    explicit_history = request.messages[:last_user_idx]
    current_user = request.messages[last_user_idx]
    current_user_prompt = _coerce_prompt_message("user", current_user.content)
    current_user_text = _extract_text_from_content(current_user.content)
    if current_user_prompt is None:
        raise HTTPException(status_code=400, detail="last user message content is empty")

    explicit_system_texts: list[str] = []
    explicit_history_prompt: list[dict[str, Any]] = []
    session_history_snapshot: list[dict[str, str]] = []

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

    effective_system_texts: list[str]
    effective_history_prompt: list[dict[str, Any]]

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
            "\n\nReference information (use only if helpful):\n" + rag_context_text.strip()
        )

    recent_history = (
        effective_history_prompt[-6:]
        if len(effective_history_prompt) > 6
        else effective_history_prompt
    )
    prompt_messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    prompt_messages.extend(recent_history)
    prompt_messages.append(current_user_prompt)
    return prompt_messages, session_id, current_user_text


# Custom asyncio exception handling that suppresses harmless socket.send() warnings
def custom_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Handle asyncio exceptions, suppressing warnings when a client disconnects."""
    exception = context.get("exception")
    message = context.get("message", "")

    # Ignore socket errors caused by client disconnects
    if exception and isinstance(exception, (ConnectionError, BrokenPipeError, OSError)):
        # These are normal client disconnects; do not log a warning
        return

    # Check whether the error is socket.send() related
    if "socket.send()" in message or "Connection" in message:
        # Log at debug level rather than as a warning
        logger.debug(f"Client connection closed: {message}")
        return

    # Handle all other exceptions normally
    if exception:
        logger.error(f"Asyncio exception: {exception}", exc_info=exception)
    else:
        logger.error(f"Asyncio error: {context}")


def signal_handler(signum: int, frame: FrameType | None) -> None:
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, cleaning up...")
    try:
        model_manager.cleanup()
        logger.info("✅ Model manager cleanup completed")
    except Exception as e:
        logger.exception(f"Error during signal cleanup: {e}")

    try:
        session_manager.close()
        logger.info("✅ Session manager cleanup completed")
    except Exception as e:
        logger.exception(f"Error during session cleanup: {e}")

    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle management."""
    logger.info("🚀 Starting LLM Service...")

    # Sweep leftover vLLM serve processes at startup so later load_model calls do not hit port conflicts.
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

    # Make sure the asyncio exception handler is installed
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_exception_handler)
        logger.info("✅ Asyncio exception handler configured")
    except Exception as e:
        logger.warning(f"Failed to set asyncio exception handler: {e}")

    yield

    logger.info("🛑 Shutting down LLM Service...")
    # Cleanup - use the new cleanup methods
    try:
        model_manager.cleanup()
        logger.info("✅ Model manager cleanup completed")
    except Exception as e:
        logger.exception(f"Error during model_manager cleanup: {e}")
    # Close session manager resources
    try:
        session_manager.close()
        logger.info("✅ Session manager cleanup completed")
    except Exception as e:
        logger.exception(f"Error during session_manager cleanup: {e}")


# Create FastAPI app
app = FastAPI(
    title="LLM Inference & Training Service",
    description="FastAPI service for LLM inference with streaming and fine-tuning support",
    version="1.0.0",
    lifespan=lifespan,
    # orjson serializes responses faster and natively handles numpy / datetime;
    # covers routes returning plain dict/model. Explicit responses below use
    # ORJSONResponse too so the whole API goes through the same encoder.
    default_response_class=ORJSONResponse,
)

# ==================== Frontend Static Files ====================
# React build (e.g. created by `npm run build`). Located by trying known
# candidate locations so the package stays relocatable across layouts
# (e.g. <base>/frontend, <base>/Trusta-AST-Frontend, or repo-root/frontend one
# level up); the first existing one wins.
_frontend_base = Path(__file__).resolve().parent.parent
_frontend_candidates = [
    _frontend_base / "frontend" / "dist",
    _frontend_base / "Trusta-AST-Frontend" / "dist",
    _frontend_base.parent / "frontend" / "dist",
    _frontend_base.parent / "Trusta-AST-Frontend" / "dist",
]
FRONTEND_DIST = next(
    (p for p in _frontend_candidates if p and p.exists()),
    _frontend_base / "frontend" / "dist",
)
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
            async def _serve_file(fp: str = str(file_path)) -> FileResponse:
                return FileResponse(fp)

            logger.info(f"🔗 Frontend top-level asset route added: /{fname}")
        else:
            logger.debug(f"Top-level asset not found (skip): {file_path}")
    app.include_router(_frontend_router)
else:
    logger.warning(f"⚠️ Frontend dist directory not found, skipping mount: {FRONTEND_DIST}")


@app.get("/frontend/{full_path:path}", response_model=None)
async def frontend_spa_fallback(full_path: str) -> FileResponse:
    """
    SPA fallback: if a deep route isn't an existing file, serve index.html.
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
    allow_origins=["*"],  # Should be restricted to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================


@app.get("/", response_model=None)
async def root() -> dict[str, Any] | RedirectResponse:
    """Root path: 302 redirect to /frontend/ when the frontend exists, otherwise return service info."""
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
async def health_check() -> dict[str, Any]:
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "model_loaded_config": (
            model_manager.config.model_dump()
            if model_manager.is_loaded() and model_manager.config is not None
            else None
        ),
        "training_active": training_manager.get_status().is_training,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """List loaded models in OpenAI-compatible format."""
    if model_manager.is_loaded():
        return {"object": "list", "data": [{"id": "trusta-ast-default", "object": "model"}]}
    else:
        return {"object": "list", "data": []}


# ==================== Inference Endpoints ====================


@app.post("/inference/load_model")
async def load_model(
    config: Annotated[
        InferenceConfig,
        Body(openapi_examples=cast(dict[str, Any], config_examples.INFERENCE_CONFIG_EXAMPLES)),
    ],
) -> dict[str, Any]:
    """
    Load an inference model
    The frontend sends the model name, quantization type and offload settings.

    Note: loading runs in the background and returns immediately. Use /inference/status
    to check the loading state.
    """
    try:
        logger.info(f"Loading model request: {config.model_name}")

        # A model is already fully loaded: respond according to the rules below
        if model_manager.is_loaded():
            current = model_manager.config

            # Decide whether it matches the current config (same model + same key settings)
            def _norm(v: Any) -> str:  # noqa: ANN401 - normalizes arbitrary config field values
                # Normalize dicts and other types into a stable, comparable string
                try:
                    if isinstance(v, dict):
                        return json.dumps(v, sort_keys=True)
                    return json.dumps(v, sort_keys=True)
                except Exception:
                    return str(v)

            def _same_inference_config(
                a: InferenceConfig | None, b: InferenceConfig | None
            ) -> bool:
                if a is None or b is None:
                    return False
                # Model identity: prefer model_path, then model_name
                a_id = a.model_path or a.model_name
                b_id = b.model_path or b.model_name
                if a_id != b_id:
                    return False
                # Compare the key fields
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
            # Different model or different config -> 409, unload first
            raise HTTPException(
                status_code=409,
                detail="A model is already loaded. Please unload the model before loading a new model.",
            )

        # Check whether a model is already being loaded
        status = model_manager.get_status()
        if status.get("is_loading"):
            raise HTTPException(
                status_code=409,
                detail="A model is already being loaded. Please wait or check status.",
            )

        # Staged loading: set the config immediately, load the actual weights in a background process
        try:
            model_manager.start_loading(config)
        except ValueError as ve:
            # Config validation error (e.g. quantization limits) -> 400
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except RuntimeError as re:
            # Any other state conflict (should already be handled above) -> 409
            raise HTTPException(status_code=409, detail=str(re)) from re

        return {
            "status": "loading",
            "message": f"Model {config.model_name} is loading in background. Check status for progress.",
            "config": config.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to start model loading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/unload_model")
async def unload_model() -> dict[str, Any]:
    """Unload the model."""
    try:
        return model_manager.unload_model()

    except Exception as e:
        logger.exception(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/stop_generation")
async def stop_generation(
    request: Annotated[StopGenerationRequest, Body(default_factory=StopGenerationRequest)],
) -> dict[str, Any]:
    """
    Stop the generation currently in progress
    Works for both streaming and non-streaming generation.
    """
    try:
        target_request_id = _resolve_worker_request_id(request.request_id, request.session_id)
        result = model_manager.stop_generation(request_id=target_request_id)
        if target_request_id:
            _unbind_worker_request(target_request_id)
        if request.session_id and target_request_id:
            result["session_id"] = request.session_id
        if target_request_id:
            result["request_id"] = target_request_id
        return result
    except Exception as e:
        logger.exception(f"Failed to stop generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/cleanup_generation_memory")
async def cleanup_generation_memory(
    request: Annotated[
        CleanupGenerationMemoryRequest, Body(default_factory=CleanupGenerationMemoryRequest)
    ],
) -> dict[str, Any]:
    """
    Free the memory accumulated during generation (KV cache and intermediate activations)
    Does not unload the model. Useful for:
    - releasing the KV cache after a long conversation
    - freeing memory when switching sessions
    - recovering from an OOM error (call before retrying).
    """
    try:
        result = model_manager.cleanup_generation_memory(slot=request.slot)
        return result
    except Exception as e:
        logger.exception(f"Failed to cleanup generation memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/force_cleanup_gpu")
async def force_cleanup_gpu() -> dict[str, Any]:
    """
    Force GPU memory cleanup
    Use this when model loading failed with OOM and VRAM was not released.
    Kills the worker process and restarts a clean one.
    """
    try:
        logger.warning("Force GPU cleanup requested")
        result = model_manager.force_cleanup_gpu()
        return result
    except Exception as e:
        logger.exception(f"Failed to force cleanup GPU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/inference/status")
async def get_model_status() -> ModelStatus:
    """Get the model status."""
    try:
        status = model_manager.get_status()
        return ModelStatus(**status)
    except Exception as e:
        logger.exception(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/inference/error_details")
async def get_error_details() -> dict[str, Any]:
    """
    Get detailed error information (including the full traceback)
    When model loading fails, this endpoint returns the complete error stack.
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
        logger.exception(f"Failed to get error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Memory Estimation Endpoints ====================


def _dominant_cost(estimate: dict[str, Any]) -> str:
    """Name the largest GPU memory term, so a failed fit points at the right knob."""
    gpu = estimate["memory_breakdown_mib"]["gpu_total"]
    settings = estimate["settings"]
    largest = max(("model", "context", "compute"), key=lambda k: gpu[k])
    if largest == "compute":
        return (
            f"The GPU compute buffer dominates at {gpu['compute']:.0f} MiB, driven by "
            f"-ub {settings['n_ubatch']} (and -b {settings['n_batch']}). Lower the ubatch "
            f"size first; it scales this buffer almost linearly."
        )
    if largest == "context":
        return (
            f"The KV cache dominates at {gpu['context']:.0f} MiB for n_ctx "
            f"{settings['n_ctx']}. Shorten the context or use -ctk q8_0 -ctv q8_0, which "
            f"roughly halves it."
        )
    return (
        f"Model weights dominate at {gpu['model']:.0f} MiB. Lower -ngl, or for a MoE model "
        f"use -ncmoe to push expert weights to RAM while keeping the KV cache on the GPU."
    )


def _require_gguf_path(model_path: str | None) -> str:
    """Resolve the GGUF path, which may come from `model_path` or from `-m` inside `args`."""
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="A GGUF path is required: set model_path, or pass -m/--model inside args.",
        )
    return model_path


@app.post("/inference/estimate_memory")
async def estimate_memory_requirements(
    request: Annotated[
        MemoryEstimateRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.MEMORY_ESTIMATE_EXAMPLES)),
    ],
) -> MemoryEstimateResponse:
    """
    Estimate a model's GPU memory requirements under mixed offload modes.

    From the model name and quantization config, this endpoint estimates:
    - the memory needed for full GPU mode
    - the minimum GPU memory needed for mixed CPU offload
    - the minimum GPU memory needed for disk offload
    - recommendations for the different offload strategies

    Args:
        request: the request carrying model_name, quantization and other parameters

    Returns:
        A detailed memory requirement estimate and offload strategy recommendations
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

        # Estimation failed (model could not be identified)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)

        return MemoryEstimateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Memory estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/estimate_memory/gguf")
async def estimate_gguf_memory(
    request: Annotated[
        GgufEstimateRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.GGUF_ESTIMATE_EXAMPLES)),
    ],
) -> GgufEstimateResponse:
    """
    Estimate how much memory a GGUF model needs under llama.cpp.

    Reads the GGUF tensor table directly, so the weight and KV cache figures are computed
    from real tensor sizes rather than inferred from a parameter count. Returns the
    model / context / compute split for every GPU and for host RAM, plus a per-layer
    breakdown into attention, dense FFN, MoE experts and shared expert.

    Args:
        request: llama.cpp runtime settings (-ngl / -c / -ctk / -ctv / -ncmoe and friends)

    Returns:
        Per-device memory breakdown and the per-layer detail table
    """
    try:
        kwargs, parsed_args = apply_cli_args(
            {
                "n_gpu_layers": request.n_gpu_layers,
                "n_ctx": request.n_ctx,
                "n_batch": request.n_batch,
                "n_ubatch": request.n_ubatch,
                "cache_type_k": request.cache_type_k.value,
                "cache_type_v": request.cache_type_v.value,
                "n_cpu_moe": request.n_cpu_moe,
                "cpu_moe": request.cpu_moe,
                "flash_attn": request.flash_attn,
                "n_parallel": request.n_parallel,
                "no_kv_offload": request.no_kv_offload,
                "swa_full": request.swa_full,
                "n_gpu": request.n_gpu,
                "tensor_split": request.tensor_split,
                "model_path": request.model_path,
            },
            request.args,
        )
        model_path = _require_gguf_path(kwargs.pop("model_path"))

        result = gguf_memory_estimator.estimate(
            model_path, include_per_layer=request.include_per_layer, **kwargs
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        if parsed_args:
            result["parsed_args"] = parsed_args
            result["notes"].extend(parsed_args["warnings"])

        if request.verify:
            exact = gguf_memory_estimator.probe_exact(model_path, **kwargs)
            if exact is None:
                result["notes"].append(
                    "Exact verification was requested but the fit-params probe is unavailable; "
                    "install the prebuilt llama (setup_env with TRUSTA_INSTALL_LLAMA=1 / "
                    "-InstallLlama) or set LLAMA_FIT_PARAMS_BIN to enable it."
                )
            else:
                result["verification"] = exact

        return GgufEstimateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"GGUF memory estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/estimate_memory/gguf/check")
async def check_gguf_config(
    config: Annotated[
        InferenceConfig,
        Body(openapi_examples=cast(dict[str, Any], config_examples.INFERENCE_CONFIG_EXAMPLES)),
    ],
    margin_mib: float = 1024,
    verify: bool = False,
    include_per_layer: bool = False,
    gpu_budget_mib: float | None = None,
    host_budget_mib: float | None = None,
) -> GgufConfigCheckResponse:
    """
    Pre-flight check whether a llama-server configuration will run out of memory.

    Accepts the very same InferenceConfig the frontend sends to `/inference/load_model`, so
    no separate payload has to be assembled. Reads `n_gpu_layers` / `n_ctx` / `n_batch` /
    `llama_server_np` and parses the llama.cpp flags inside `llama_server_extra_args`
    (`-ncmoe`, `-ctk`, `-ctv`, `-ub`, `-fa`, `-ot` and so on). The extra arguments take
    precedence, matching how llama-server appends them last on the command line.

    When the configuration does not fit, a `suggestion` with a workable set of settings and
    a ready-to-use argument string is returned alongside the verdict.

    Args:
        config: the same inference configuration accepted by load_model
        margin_mib: memory to leave free on each GPU
        verify: also cross-check with llama-fit-params (requires that binary)
        include_per_layer: whether to return the per-layer detail table
        gpu_budget_mib: override the GPU budget; defaults to currently free VRAM
        host_budget_mib: override the host budget; defaults to currently free DRAM

    Returns:
        Whether it fits, the memory breakdown, and a suggested configuration if it does not
    """
    try:
        # Resolve exactly as the engine will at launch. Note this follows symlinks, so a
        # HuggingFace-cached GGUF lands on an extension-less blob file; the file is
        # identified by its GGUF magic rather than by its name.
        model_path = resolve_local_model_path(config.model_path or config.model_name)

        kwargs, parsed_args = apply_cli_args(
            {
                "n_gpu_layers": config.n_gpu_layers,
                "n_ctx": config.n_ctx,
                "n_batch": config.n_batch,
                "n_parallel": config.llama_server_np,
            },
            config.llama_server_extra_args,
        )
        kwargs.pop("model_path", None)  # -m inside extra args must not retarget the check

        result = gguf_memory_estimator.estimate(
            model_path, include_per_layer=include_per_layer, **kwargs
        )
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail={
                    **result,
                    "hint": (
                        "This endpoint only checks GGUF models served by llama.cpp. Point "
                        "model_path (or model_name) at a GGUF file; a HuggingFace "
                        "transformers directory cannot be checked here."
                    ),
                    "resolved_path": model_path,
                },
            )

        notes = result["notes"]
        if config.engine != InferenceEngine.LLAMA_SERVER:
            notes.append(
                f"Config engine is '{config.engine}', not 'llama_server'; these figures "
                f"describe what llama.cpp would use."
            )
        if parsed_args:
            notes.extend(parsed_args["warnings"])
            if parsed_args["ignored"]:
                notes.append(
                    f"Flags with no effect on memory were skipped: "
                    f"{', '.join(parsed_args['ignored'])}."
                )

        budgets = gguf_memory_estimator.resolve_budgets(gpu_budget_mib, host_budget_mib)
        gpu_used = result["memory_breakdown_mib"]["gpu_total"]["total"]
        host_used = result["memory_breakdown_mib"]["host"]["total"]
        gpu_budget = budgets["gpu_budget_mib"]
        host_budget = budgets["host_budget_mib"]

        gpu_ok = gpu_budget is None or gpu_used + margin_mib <= gpu_budget
        host_ok = host_budget is None or host_used <= host_budget
        if gpu_budget is None:
            verdict = "unknown_budget"
            summary = (
                f"Needs {gpu_used:.0f} MiB of VRAM, but no GPU budget could be read. "
                f"Pass gpu_budget_mib to get a verdict."
            )
        elif not gpu_ok:
            verdict = "gpu_short"
            summary = (
                f"Does not fit: needs {gpu_used:.0f} MiB plus a {margin_mib:.0f} MiB margin, "
                f"but only {gpu_budget:.0f} MiB of VRAM is free."
            )
        elif not host_ok:
            verdict = "host_short"
            summary = (
                f"Fits in VRAM ({gpu_used:.0f} of {gpu_budget:.0f} MiB) but needs "
                f"{host_used:.0f} MiB of host RAM with only {host_budget:.0f} MiB free. "
                f"llama.cpp mmaps weights, so this may still run from page cache."
            )
        else:
            verdict = "ok"
            summary = (
                f"Fits: {gpu_used:.0f} MiB of {gpu_budget:.0f} MiB free VRAM, "
                f"leaving {gpu_budget - gpu_used:.0f} MiB headroom."
            )

        suggestion = None
        if verdict == "gpu_short":
            advice = gguf_memory_estimator.recommend(
                model_path,
                n_ctx=kwargs.get("n_ctx", 0),
                gpu_budget_mib=gpu_budget,
                host_budget_mib=host_budget,
                margin_mib=margin_mib,
                verify=False,
                n_batch=kwargs.get("n_batch", 2048),
                n_ubatch=kwargs.get("n_ubatch", 512),
                flash_attn=kwargs.get("flash_attn", True),
                n_parallel=kwargs.get("n_parallel", 1),
            )
            if "error" in advice:
                # recommend() only tunes -ngl / -ncmoe / -c / KV quant, so when it cannot
                # find anything the binding term is usually one it does not touch.
                suggestion = {"error": advice["error"], "dominant_cost": _dominant_cost(result)}
                notes.append(suggestion["dominant_cost"])
            else:
                suggestion = {
                    "recommended": advice["recommended"],
                    "llama_server_args": advice["llama_server_args"],
                    "memory_breakdown_mib": advice["memory_breakdown_mib"],
                }

        verification = None
        if verify:
            exact = gguf_memory_estimator.probe_exact(model_path, **kwargs)
            if exact is None:
                notes.append(
                    "Exact verification was requested but the fit-params probe is unavailable; "
                    "install the prebuilt llama (setup_env with TRUSTA_INSTALL_LLAMA=1 / "
                    "-InstallLlama) or set LLAMA_FIT_PARAMS_BIN to enable it."
                )
            else:
                verification = exact

        return GgufConfigCheckResponse(
            fits=(verdict == "ok"),
            verdict=verdict,
            summary=summary,
            model_path=model_path,
            resolved_settings=result["settings"],
            parsed_args=parsed_args or None,
            model_info=result["model_info"],
            budgets_mib=budgets,
            memory_breakdown_mib=result["memory_breakdown_mib"],
            placement=result["placement"],
            per_layer_mib=result.get("per_layer_mib"),
            suggestion=suggestion,
            verification=verification,
            notes=notes,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"GGUF config check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/estimate_memory/gguf/sweep")
async def sweep_gguf_memory(
    request: Annotated[
        GgufSweepRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.GGUF_SWEEP_EXAMPLES)),
    ],
) -> GgufSweepResponse:
    """
    Sweep -ngl x context x KV quantization and flag which combinations fit in VRAM.

    Purely analytic: no llama.cpp process is started, so a whole grid costs almost nothing.
    Budgets default to the currently free VRAM/DRAM when not supplied.

    Args:
        request: the grid to sweep and the memory budgets to test against

    Returns:
        Memory usage and feasibility for every combination in the grid
    """
    try:
        kwargs, parsed_args = apply_cli_args(
            {
                "n_batch": request.n_batch,
                "n_ubatch": request.n_ubatch,
                "flash_attn": request.flash_attn,
                "n_parallel": request.n_parallel,
                "n_gpu": request.n_gpu,
                "tensor_split": request.tensor_split,
                "model_path": request.model_path,
            },
            request.args,
            solved_keys=SWEEP_SOLVED_KEYS,
        )
        model_path = _require_gguf_path(kwargs.pop("model_path"))

        result = gguf_memory_estimator.sweep(
            model_path,
            n_gpu_layers_grid=request.n_gpu_layers_grid,
            n_ctx_grid=request.n_ctx_grid,
            kv_quant_grid=[k.value for k in request.kv_quant_grid]
            if request.kv_quant_grid
            else None,
            gpu_budget_mib=request.gpu_budget_mib,
            host_budget_mib=request.host_budget_mib,
            margin_mib=request.margin_mib,
            **kwargs,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        if parsed_args:
            result["parsed_args"] = parsed_args
        return GgufSweepResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"GGUF memory sweep error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/estimate_memory/gguf/recommend")
async def recommend_gguf_settings(
    request: Annotated[
        GgufRecommendRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.GGUF_RECOMMEND_EXAMPLES)),
    ],
) -> GgufRecommendResponse:
    """
    Solve for the best llama-server settings the current hardware can run.

    Maximises -ngl first. For MoE models it prefers --n-cpu-moe to push expert weights into
    DRAM, because that keeps attention and the KV cache on the GPU, and only then shortens
    the context or quantizes the KV cache. Returns an argument string that can be pasted
    straight into `llama_server_extra_args`.

    Args:
        request: target context, memory budgets, and which knobs may be adjusted

    Returns:
        The chosen settings, the memory breakdown, and a llama-server argument string
    """
    try:
        kwargs, parsed_args = apply_cli_args(
            {
                "n_batch": request.n_batch,
                "n_ubatch": request.n_ubatch,
                "flash_attn": request.flash_attn,
                "n_parallel": request.n_parallel,
                "n_gpu": request.n_gpu,
                "tensor_split": request.tensor_split,
                "n_ctx": request.n_ctx,
                "model_path": request.model_path,
            },
            request.args,
            solved_keys=RECOMMEND_SOLVED_KEYS,
        )
        model_path = _require_gguf_path(kwargs.pop("model_path"))
        n_ctx = kwargs.pop("n_ctx")

        result = gguf_memory_estimator.recommend(
            model_path,
            n_ctx=n_ctx,
            n_ctx_min=request.n_ctx_min,
            n_ctx_max=request.n_ctx_max,
            gpu_budget_mib=request.gpu_budget_mib,
            host_budget_mib=request.host_budget_mib,
            margin_mib=request.margin_mib,
            allow_kv_quant=request.allow_kv_quant,
            allow_ctx_reduction=request.allow_ctx_reduction,
            target_utilization=request.target_utilization,
            verify=request.verify,
            **kwargs,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        if parsed_args:
            result["parsed_args"] = parsed_args
            result["notes"].extend(parsed_args["warnings"])
        return GgufRecommendResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"GGUF recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/estimate_memory/gguf/plan")
async def plan_gguf_settings(
    request: Annotated[
        GgufPlanRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.GGUF_PLAN_EXAMPLES)),
    ],
) -> GgufPlanResponse:
    """
    Offer several usable configurations per GPU budget, one per KV cache type.

    Where /recommend returns a single answer and only quantizes the KV cache as a last
    resort, this solves every requested quantization against every requested budget and
    ranks the results. That exposes the trade-off: on a fixed budget a q8_0 or q4_0 cache
    buys context or offloaded layers, and a candidate that gains neither is flagged
    `dominated`. Each candidate carries its own pasteable argument string.

    Costs roughly a second per candidate when `verify` is on, so keep budgets x KV types
    small; the limit is 32 combinations.

    Args:
        request: the GPU budgets, the KV cache types to offer, and the context bounds

    Returns:
        One plan per budget, each holding the ranked candidates and their memory figures
    """
    try:
        kwargs, parsed_args = apply_cli_args(
            {
                "n_batch": request.n_batch,
                "n_ubatch": request.n_ubatch,
                "flash_attn": request.flash_attn,
                "n_parallel": request.n_parallel,
                "n_gpu": request.n_gpu,
                "tensor_split": request.tensor_split,
                "n_ctx": request.n_ctx,
                "model_path": request.model_path,
            },
            request.args,
            solved_keys=RECOMMEND_SOLVED_KEYS,
        )
        model_path = _require_gguf_path(kwargs.pop("model_path"))
        n_ctx = kwargs.pop("n_ctx")

        result = gguf_memory_estimator.plan(
            model_path,
            gpu_budgets_mib=request.gpu_budgets_mib,
            kv_cache_types=[k.value for k in request.kv_cache_types]
            if request.kv_cache_types
            else None,
            host_budget_mib=request.host_budget_mib,
            margin_mib=request.margin_mib,
            n_ctx=n_ctx,
            n_ctx_min=request.n_ctx_min,
            n_ctx_max=request.n_ctx_max,
            target_utilization=request.target_utilization,
            verify=request.verify,
            **kwargs,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        if parsed_args:
            result["parsed_args"] = parsed_args
            result["notes"].extend(parsed_args["warnings"])
        return GgufPlanResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"GGUF plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/inference/estimate_memory/{model_name}")
async def estimate_memory_by_name(
    model_name: str,
    quantization: str = "none",
    batch_size: int = 1,
    sequence_length: int = 2048,
) -> dict[str, Any]:
    """
    Quickly estimate a model's memory requirements from URL parameters (simplified).

    Example: /inference/estimate_memory/llama-2-7b?quantization=int8
    """
    try:
        # Handle the URL-encoded model name
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
        logger.exception(f"Memory estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference/chat", deprecated=True)
async def chat(request: ChatRequest) -> None:
    """Deprecated; use the OpenAI-compatible API instead."""
    raise HTTPException(
        status_code=410,
        detail={
            "error": "deprecated_endpoint",
            "message": "'/inference/chat' is deprecated; use '/v1/chat/completions' instead.",
        },
    )


@app.post("/v1/chat/completions", response_model=None)
async def openai_chat_completions(http_request: Request) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat endpoint."""

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

    async def _backend_generate(**kwargs: Any) -> Any:  # noqa: ANN401 - kwargs/return are dynamic backend generation passthrough
        """
        Non-stream generation: direct async HTTP for server engines, else the
        sync worker path run off the event loop."""
        if model_manager.uses_async_http():
            return await model_manager.agenerate(**kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: model_manager.generate(**kwargs))

    async def _backend_generate_stream(**kwargs: Any) -> AsyncIterator[Any]:  # noqa: ANN401 - kwargs passthrough, items are dynamic backend chunks
        """
        Stream items: direct async HTTP for server engines, else the sync
        worker generator bridged onto the event loop via a producer thread."""
        if model_manager.uses_async_http():
            async for item in model_manager.agenerate_stream(**kwargs):
                yield item
            return

        loop = asyncio.get_running_loop()
        bridge: asyncio.Queue = asyncio.Queue()

        def _producer() -> None:
            try:
                for item in model_manager.generate_stream(**kwargs):
                    loop.call_soon_threadsafe(bridge.put_nowait, ("data", item))
                loop.call_soon_threadsafe(bridge.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(bridge.put_nowait, ("error", exc))

        threading.Thread(target=_producer, daemon=True).start()
        while True:
            kind, payload = await bridge.get()
            if kind == "error":
                raise payload
            if kind == "done":
                return
            yield payload

    def _resolve_enable_thinking() -> bool | None:
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
        message: Any,  # noqa: ANN401 - backend error payload is dynamic (str or dict)
        error_type: str = "server_error",
    ) -> dict[str, Any]:
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

        payload: dict[str, Any] = {
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
        prompt_tokens: Any,  # noqa: ANN401 - token counts are dynamic backend values
        completion_tokens: Any,  # noqa: ANN401 - token counts are dynamic backend values
        total_tokens: Any,  # noqa: ANN401 - token counts are dynamic backend values
        payload: Any = None,  # noqa: ANN401 - dynamic backend response payload
    ) -> dict[str, Any] | None:
        payload_dict = payload if isinstance(payload, dict) else {}

        if all(v is None for v in [prompt_tokens, completion_tokens, total_tokens]) and all(
            payload_dict.get(key) is None for key in ["gen_tokens", "gen_tps", "prompt_tps"]
        ):
            return None

        pt = int(
            prompt_tokens if prompt_tokens is not None else (payload_dict.get("prompt_tokens") or 0)
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

        usage: dict[str, Any] = {
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

    def _normalize_tool_choice_for_backend(tool_choice: Any) -> Any:  # noqa: ANN401 - tool_choice is dynamic OpenAI payload (str or dict), passed through
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

    def _is_known_tool_stream_mismatch_error(error: Any) -> bool:  # noqa: ANN401 - error is a dynamic backend exception/message
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

    def _iter_text_stream_chunks(text: str, chunk_size: int = 32) -> list[str]:
        """Split finalized text into small SSE-friendly chunks for smoother UI updates."""
        normalized = str(text or "")
        if not normalized:
            return []

        chunks: list[str] = []
        current = ""
        for char in normalized:
            current += char
            if char == "\n" or len(current) >= chunk_size:
                chunks.append(current)
                current = ""

        if current:
            chunks.append(current)

        return chunks

    def _build_passthrough_messages() -> list[dict[str, Any]]:
        passthrough_messages: list[dict[str, Any]] = []
        for msg in request.messages:
            role = _normalize_openai_role(msg.role)
            item: dict[str, Any] = {"role": role}

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

        tool_kwargs = {
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

        if not request.stream:
            async with generation_semaphore:
                try:
                    payload = await _backend_generate(**tool_kwargs)
                    kind = "ok"
                except Exception as exc:
                    kind, payload = "error", exc

            if kind == "error":
                raise HTTPException(status_code=500, detail=str(payload))

            internal_resp = payload
            if not isinstance(internal_resp, dict):
                raise HTTPException(status_code=500, detail="Unexpected non-stream response type")

            content = internal_resp.get("response", internal_resp.get("result", ""))
            tool_calls = internal_resp.get("tool_calls")
            finish_reason = internal_resp.get("finish_reason") or (
                "tool_calls" if tool_calls else "stop"
            )
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
                if not content:
                    message["content"] = None

            output: dict[str, Any] = {
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
            return ORJSONResponse(content=output)

        async def _passthrough_stream_to_openai() -> AsyncIterator[str]:
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            async with generation_semaphore:
                use_non_stream_fallback = _should_fallback_tool_passthrough_stream()

                def _fallback_done_payload(internal_resp: Any) -> dict[str, Any]:  # noqa: ANN401 - dynamic backend response payload
                    internal_error = (
                        internal_resp.get("error") if isinstance(internal_resp, dict) else None
                    )
                    return {
                        "error": internal_error,
                        "chunk": internal_resp.get("response", internal_resp.get("result", "")),
                        "tool_calls": internal_resp.get("tool_calls"),
                        "done": True,
                        "finish_reason": internal_resp.get("finish_reason"),
                        "prompt_tokens": internal_resp.get("prompt_tokens"),
                        "gen_tokens": internal_resp.get("gen_tokens"),
                        "total_tokens": internal_resp.get("total_tokens"),
                    }

                async def _iter_items() -> AsyncIterator[Any]:  # noqa: ANN401 - yields dynamic backend chunk dicts
                    # Both branches route through _backend_generate[_stream], so
                    # server engines go over async HTTP and transformers over the
                    # worker path — while preserving the tool-diff-mismatch
                    # fallback to non-stream planning.
                    if use_non_stream_fallback:
                        logger.info(
                            "OpenAI tool passthrough stream uses non-stream backend planning to avoid tool diff mismatch during streaming"
                        )
                        internal_resp = await _backend_generate(**tool_kwargs)
                        yield _fallback_done_payload(internal_resp)
                        return

                    text_stream_emitted = False
                    try:
                        async for item in _backend_generate_stream(**tool_kwargs):
                            if isinstance(item, dict) and item.get("chunk"):
                                text_stream_emitted = True
                            yield item
                    except Exception as exc:
                        if not text_stream_emitted and _is_known_tool_stream_mismatch_error(exc):
                            logger.warning(
                                "OpenAI tool passthrough stream fallback triggered after tool diff mismatch: %s",
                                exc,
                            )
                            internal_resp = await _backend_generate(**tool_kwargs)
                            yield _fallback_done_payload(internal_resp)
                        else:
                            raise

                try:
                    async for item in _iter_items():
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
                    logger.warning("OpenAI tool passthrough stream disconnected: %s", exc)
                    _stop_backend_generation()
                    raise
                except RuntimeError as exc:
                    logger.warning("OpenAI tool passthrough stream backend error: %s", exc)
                    _stop_backend_generation()
                    yield f"data: {json.dumps(_build_openai_stream_error(str(exc)))}\n\n"
                    return
                except Exception as exc:
                    logger.exception("OpenAI tool passthrough stream exception: %s", exc)
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
    prompt_messages, session_id, current_user_text = _build_openai_prompt_messages(request)
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
        # Bind this generation to the session so /inference/stop_generation
        # can resolve it by session_id alone.
        _bind_worker_request(session_id, backend_request_id)
        async with generation_semaphore:
            gen_kwargs = {
                "prompt": prompt_messages,
                "max_new_tokens": request.max_tokens,
                "temperature": generation_options["temperature"],
                "top_p": generation_options["top_p"],
                "top_k": generation_options["top_k"],
                "repetition_penalty": generation_options["repetition_penalty"],
                "system_prompt": None,
                "total_timeout": request.total_timeout,
                "enable_thinking": generation_options["enable_thinking"],
                "request_id": backend_request_id,
            }
            try:
                try:
                    payload = await _backend_generate(**gen_kwargs)
                    kind = "ok"
                except asyncio.CancelledError:
                    # Client disconnected mid-generation; stop the backend so it
                    # doesn't keep generating after we've gone away.
                    _stop_backend_generation()
                    raise
                except Exception as exc:
                    kind, payload = "error", exc
            finally:
                _unbind_worker_request(backend_request_id)

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
                    session_id, {"role": "assistant", "content": cast(str, content)}
                )
            except Exception as exc:
                logger.exception(f"Failed to save session history (openai non-stream): {exc}")

        output: dict[str, Any] = {
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
        return ORJSONResponse(content=output)

    async def _stream_openai_generation() -> AsyncIterator[str]:
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        assistant_text = ""
        # Bind this generation to the session so /inference/stop_generation
        # can resolve it by session_id alone.
        _bind_worker_request(session_id, backend_request_id)
        async with generation_semaphore:
            gen_kwargs = {
                "prompt": prompt_messages,
                "max_new_tokens": request.max_tokens,
                "temperature": generation_options["temperature"],
                "top_p": generation_options["top_p"],
                "top_k": generation_options["top_k"],
                "repetition_penalty": generation_options["repetition_penalty"],
                "system_prompt": None,
                "total_timeout": request.total_timeout,
                "enable_thinking": generation_options["enable_thinking"],
                "request_id": backend_request_id,
            }

            try:
                async for payload in _backend_generate_stream(**gen_kwargs):
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
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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
                                logger.exception(
                                    f"Failed to save session history (openai stream): {exc}"
                                )

                        yield "data: [DONE]\n\n"
                        return
            except (asyncio.CancelledError, ConnectionError, BrokenPipeError) as exc:
                # Genuine client disconnects only. Backend generation failures
                # are wrapped in RuntimeError by both backend paths and must
                # fall through to the generic handler below so the client
                # receives a structured SSE error payload instead of a
                # silently dropped connection.
                logger.warning("OpenAI compatibility stream disconnected: %s", exc)
                _stop_backend_generation()
                raise
            except Exception as exc:
                logger.exception("OpenAI compatibility stream exception: %s", exc)
                yield f"data: {json.dumps(_build_openai_stream_error(str(exc)))}\n\n"
                return
            finally:
                _unbind_worker_request(backend_request_id)

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


@app.get("/rag/docs", response_model=None)
async def rag_list_docs() -> JSONResponse:
    """List the existing documents."""
    try:
        return ORJSONResponse(content={"documents": rag_manager.list_documents()})
    except Exception as e:
        logger.exception(f"Failed to list RAG docs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/docs", response_model=None)
async def rag_add_doc(payload: dict) -> JSONResponse:
    """
    Add or update a document and its database content
    Request format: {"doc_id": Optional[str], "content": str}.
    """
    try:
        doc_id = payload.get("doc_id")
        content = payload.get("content")
        if not content or not isinstance(content, str):
            raise HTTPException(status_code=400, detail="content is required and must be a string")
        result = rag_manager.add_document(content=content, doc_id=doc_id)
        return ORJSONResponse(content={"status": "ok", "result": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to add RAG doc: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/rag/docs/{doc_id}", response_model=None)
async def rag_delete_doc(doc_id: str) -> JSONResponse:
    """Delete the given document and its database content."""
    try:
        result = rag_manager.delete_document(doc_id)
        return ORJSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        logger.exception(f"Failed to delete RAG doc: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/rag/search", response_model=None)
async def rag_search(q: str, k: int = 3) -> JSONResponse:
    """Search the RAG documents and return the top k results."""
    try:
        results = rag_manager.search(q, k=k)
        return ORJSONResponse(content={"query": q, "k": k, "results": results})
    except Exception as e:
        logger.exception(f"Failed to search RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Training Endpoints ====================


@app.post("/training/start")
async def start_training(
    config: Annotated[
        TrainingConfig,
        Body(openapi_examples=cast(dict[str, Any], config_examples.TRAINING_CONFIG_EXAMPLES)),
    ],
) -> dict[str, Any]:
    """
    Start training
    Supports LoRA/QLoRA/Full Parameter Training.
    """
    try:
        logger.info(f"Starting training request: {config.model_name} with method {config.method}")

        # Check if model is loaded for inference
        if model_manager.is_loaded():
            logger.warning("Inference model is loaded. Consider unloading it before training.")

        # Start training
        result = training_manager.start_training(config)

        return {
            "status": "success",
            "message": "Training started",
            "config": config.model_dump(),
            "result": result,
        }

    except Exception as e:
        logger.exception(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/training/status")
async def get_training_status() -> TrainingStatus:
    """Get the training status."""
    try:
        status = training_manager.get_status()
        return status
    except Exception as e:
        logger.exception(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/training/status/{session_id}/history")
async def get_training_history(session_id: str) -> TrainingHistoryResponse:
    """Get the training history (Loss, Learning Rate and so on)."""
    try:
        history = training_manager.get_history(session_id)
        if not history or "training_logs" not in history:
            raise HTTPException(
                status_code=404,
                detail="Training session not found or no history available",
            )

        return TrainingHistoryResponse(
            session_id=session_id,
            logs=history["training_logs"],
            eval_logs=history.get("eval_logs", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get training history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/system/resource/{session_id}/history",
    response_model_exclude_none=True,
)
async def get_system_resource_history(session_id: str) -> SystemResourceHistoryResponse:
    """Get the system resource history recorded during training."""
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
        logger.exception(f"Failed to get system resource history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _validate_job_id(job_id: str) -> str:
    # Guard against path traversal — job ids are uuid-like tokens.
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    return job_id


@app.get("/training/{job_id}/log")
async def get_training_log(job_id: str, since: int = 0) -> TrainingLogEventsResponse:
    """
    Return structured training-log events for a job (incremental via `since`).

    Reads the durable events.jsonl on disk, so it works even when Redis is down
    and lets clients browse past sessions / backfill after an SSE reconnect.
    """
    from .training.core.job_logger import job_log_dir, read_events

    _validate_job_id(job_id)
    events = await asyncio.to_thread(read_events, job_id, since)
    if not events and not job_log_dir(job_id).exists():
        raise HTTPException(status_code=404, detail="Training job log not found")
    cursor = events[-1]["seq"] if events else since
    return TrainingLogEventsResponse(job_id=job_id, cursor=cursor, events=events)


@app.get("/training/{job_id}/log/stream", response_model=None)
async def stream_training_log(job_id: str, request: Request, since: int = 0) -> StreamingResponse:
    """
    Server-Sent Events stream of a job's training log.

    Emits the backlog first, then tails new events as they are appended. Each
    event carries its `seq` as the SSE id so a client can resume with
    `Last-Event-ID` / `?since=`. Closes on a terminal event or client disconnect.
    """
    from .training.core.job_logger import is_terminal_event, job_log_dir, read_events

    _validate_job_id(job_id)
    if not job_log_dir(job_id).exists():
        raise HTTPException(status_code=404, detail="Training job log not found")

    # Resume support: Last-Event-ID header wins over the query param.
    last_event_id = request.headers.get("last-event-id")
    if last_event_id:
        try:
            since = int(last_event_id)
        except ValueError:
            pass

    async def event_gen() -> AsyncIterator[str]:
        last_seq = since
        ticks_since_data = 0
        while True:
            if await request.is_disconnected():
                break
            events = await asyncio.to_thread(read_events, job_id, last_seq)
            if events:
                ticks_since_data = 0
                for ev in events:
                    last_seq = int(ev.get("seq", last_seq))
                    yield (
                        f"id: {last_seq}\n"
                        f"event: {ev.get('type', 'message')}\n"
                        f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    )
                    if is_terminal_event(ev):
                        return
            else:
                ticks_since_data += 1
                if ticks_since_data % 15 == 0:
                    # keep-alive comment so proxies don't drop an idle stream
                    yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable nginx proxy buffering for SSE
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.get("/training/error_details")
async def get_training_error_details() -> dict[str, Any]:
    """
    Get detailed training error information (including the full traceback)
    When training fails, this endpoint returns the complete error stack,
    in the same format as /inference/error_details.
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
        logger.exception(f"Failed to get training error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/training/stop")
async def stop_training() -> dict[str, Any]:
    """Stop training."""
    try:
        result = training_manager.stop_training()
        return {
            "status": "success",
            "message": "Training stop requested",
            "result": result,
        }
    except Exception as e:
        logger.exception(f"Failed to stop training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/training/force_cleanup_gpu")
async def force_cleanup_training_gpu() -> dict[str, Any]:
    """
    Force cleanup of the training-related GPU processes and memory.

    Behaves like /inference/force_cleanup_gpu:
    - Terminates the current training worker process (if any).
    - Resets the TrainingProcessManager state so the next run starts on a clean process.
    Note: this does not affect the inference workers.
    """
    try:
        logger.warning("Force training GPU cleanup requested")
        # Call training_process_manager.cleanup() directly to kill the current worker and reset state.
        from .training.training_process import training_process_manager

        training_process_manager.cleanup()
        return {
            "status": "success",
            "message": "Training worker process terminated and state reset.",
        }
    except Exception as e:
        logger.exception(f"Failed to force cleanup training GPU: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Configuration Endpoints ====================


@app.get("/config/quantization_types")
async def get_quantization_types() -> dict[str, Any]:
    """Get the supported quantization types."""
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
async def get_offload_types() -> dict[str, Any]:
    """Get the supported offload types."""
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
async def get_training_methods() -> dict[str, Any]:
    """Get the supported training methods."""
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


@app.get("/config/models", response_model=None)
async def list_available_models() -> JSONResponse:
    """List the available inference models (base models + local fine-tuned models)."""
    try:
        data = model_registry.list_models()
        return ORJSONResponse(content=data)
    except Exception as e:
        logger.exception(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/config/models/refresh_context_lengths")
async def refresh_model_context_lengths() -> dict[str, Any]:
    """
    Refresh the max context length of every Hugging Face model
    Fetches the latest max_position_embeddings from the HF API.
    """
    try:
        # Try to read the HF token
        from .utils.token_utils import load_hf_token

        hf_token = load_hf_token()

        model_registry.refresh_all_context_lengths(hf_token)

        return {
            "status": "success",
            "message": "Model context lengths refreshed successfully",
        }
    except Exception as e:
        logger.exception(f"Failed to refresh context lengths: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/config/models/download")
async def download_and_register_model(
    model_id: Annotated[
        str, Body(description="Hugging Face model ID, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'")
    ],
    label: Annotated[
        str | None,
        Body(
            description="Custom label; defaults to the last segment of the model ID. For a GGUF file, put the full file name here (e.g. 'model.gguf')"
        ),
    ] = None,
    cache_dir: Annotated[
        str | None, Body(description="Custom cache root; defaults to HF_HOME or the default path")
    ] = None,
    force_download: Annotated[bool, Body(description="Force re-download even if present")] = False,
    filename: Annotated[
        str | None, Body(description="[GGUF only] Download this file only and register as GGUF")
    ] = None,
) -> dict[str, Any]:
    """
    Download a model from Hugging Face and register it in the model registry.

    This endpoint:
    1. starts a background download task (via DownloadManager)
    2. returns a task_id for progress queries

    Check progress: GET /config/models/download/{task_id}
    """
    try:
        # Prefer the explicitly supplied filename
        target_filename = filename

        # Generate a label when none was supplied
        if not label:
            label = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        else:
            # Convenience: if the label looks like a GGUF file name and no filename was given, treat it as the filename
            if not target_filename and (label.endswith(".gguf") or ".gguf" in label):
                target_filename = label

        # Check whether the label already exists
        existing_models = model_registry.list_models()
        if target_filename:
            # Special case for GGUF registration: the label may be a path or a display name.
            # Only a light check here; the download manager handles path overrides
            pass
        else:
            base_models = existing_models.get("base_models", [])
            # Force download allows duplicates (overwrite); normally label uniqueness is kept
            if not force_download and any(m.get("label") == label for m in base_models):
                raise HTTPException(
                    status_code=409,
                    detail=f"Model with label '{label}' already exists in registry. Use force_download=true to re-download.",
                )

        # Start the download task
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
        logger.exception(f"Failed to start model download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/config/models/download/{task_id}")
async def get_download_status(task_id: str) -> DownloadTask:
    """Get the status of a download task."""
    task = download_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/config/models/downloads")
async def list_download_tasks() -> dict[str, Any]:
    """List all download tasks."""
    return {"tasks": download_manager.list_tasks()}


@app.post("/config/models/convert")
async def convert_model_to_gguf(
    request: Annotated[
        ModelConversionRequest,
        Body(openapi_examples=cast(dict[str, Any], config_examples.CONVERSION_CONFIG_EXAMPLES)),
    ],
) -> ConversionResponse:
    """
    Convert a Hugging Face model to GGUF format.
    """
    try:
        job_id = conversion_manager.start_conversion(
            model_path=request.model_path,
            output_path=request.output_path,
            outtype=request.outtype,
        )
        return ConversionResponse(job_id=job_id, status="pending", message="Conversion job started")
    except Exception as e:
        logger.exception(f"Failed to start conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/config/models/convert/{job_id}")
async def get_conversion_status(job_id: str) -> ConversionResponse:
    """
    Get the status of a model conversion.
    """
    status = conversion_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return ConversionResponse(
        job_id=status["job_id"], status=status["status"], message=status["message"]
    )


@app.delete("/config/models/{label:path}")
async def delete_model(label: str, delete_files: bool = False) -> dict[str, Any]:
    """
    Delete a registered model.

    Args:
        label: the model label
        delete_files: also delete the local files (default False)
    """
    try:
        # 1. Remove from registry
        model_info = model_registry.delete_model(label)

        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model with label '{label}' not found")

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
                    # Optimization: for a Hugging Face cache layout, delete the whole model
                    # directory to reclaim the blobs. Standard layout: .../models--Org--Repo/snapshots/HASH
                    try:
                        p = Path(raw_path)
                        if "snapshots" in p.parts:
                            # Walk up until a directory starting with models-- is found
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
                    # Fallback: without a local_path, infer the default path from HF_HOME
                    hf_id = model_info.get("base_model_name")
                    if hf_id and "/" in hf_id:
                        try:
                            hf_home = os.getenv("HF_HOME") or os.path.expanduser(  # noqa: ASYNC240 - cheap string expansion, not blocking I/O
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
                            logger.warning(f"Error inferring default HF cache path: {e}")

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
                            # Walk up until a directory starting with models-- is found
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
                        logger.warning(f"Error parsing GGUF path for HF cache detection: {e}")

            if path_to_delete:
                # Guard against deleting OS system directories via a poisoned
                # registry entry (e.g. output_dir="/etc" from a malicious config).
                _BLOCKED_DELETE_PREFIXES = (
                    "/etc",
                    "/bin",
                    "/sbin",
                    "/usr/bin",
                    "/usr/sbin",
                    "/usr/lib",
                    "/lib",
                    "/lib64",
                    "/boot",
                    "/sys",
                    "/proc",
                    "/dev",
                    "/run",
                    "/root",
                    "/home",
                )
                _resolved_delete = os.path.realpath(path_to_delete)  # noqa: ASYNC240 - cheap stat; the actual delete already runs in to_thread
                if not os.path.lexists(path_to_delete):  # noqa: ASYNC240 - cheap stat
                    result["files_removed"] = True
                    result["path"] = path_to_delete
                    result["file_deletion_note"] = "Path already missing; treated as deleted"
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
                    result["file_deletion_error"] = (
                        "Deletion blocked: path resolves to a protected system directory"
                    )
                else:
                    try:
                        if os.path.isdir(path_to_delete):  # noqa: ASYNC240 - cheap stat
                            # Deleting a large model directory can take seconds; move it off the event loop
                            await asyncio.to_thread(shutil.rmtree, path_to_delete)
                        else:
                            await asyncio.to_thread(os.remove, path_to_delete)
                        result["files_removed"] = True
                        result["path"] = path_to_delete
                        logger.info(f"Deleted local files for model {label} at {path_to_delete}")
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
                        logger.exception(f"Failed to delete files at {path_to_delete}: {e}")
                        result["file_deletion_error"] = str(e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== System Resource Endpoints ====================


@app.get("/system/resources")
async def get_system_resources(
    mode: str = "spec", disk_path: str = "/", calc_size: bool = False
) -> SystemResourcesResponse:
    """
    Query system resources (CPU/RAM/GPU/DISK).
    - mode=spec: return hardware specs (totals, models and so on)
    - mode=usage: return current usage (load, used memory and so on).

    Optional parameters:
    - disk_path: the disk path to query (default "/")
    """
    try:
        # Validate mode
        if mode not in ["spec", "usage"]:
            raise HTTPException(status_code=400, detail="mode must be 'spec' or 'usage'")

        # calc_size triggers a recursive os.walk; block OS system directories
        # that are never valid data/model paths to prevent filesystem enumeration.
        if calc_size:
            _BLOCKED_CALC_PREFIXES = (
                "/etc",
                "/bin",
                "/sbin",
                "/usr/bin",
                "/usr/sbin",
                "/usr/lib",
                "/lib",
                "/lib64",
                "/boot",
                "/sys",
                "/proc",
                "/dev",
                "/run",
                "/root",
                "/home",
            )
            _resolved = os.path.realpath(disk_path)  # noqa: ASYNC240 - cheap stat
            if any(_resolved == p or _resolved.startswith(p + "/") for p in _BLOCKED_CALC_PREFIXES):
                raise HTTPException(
                    status_code=400,
                    detail=f"calc_size is not allowed for system directory: {disk_path}",
                )

        cpu_res = system_monitor.get_cpu_resource(mode)
        gpu_res = system_monitor.get_gpu_resource(mode)
        # calc_size triggers a recursive os.walk; move it off the event loop to avoid blocking
        disk_res = await asyncio.to_thread(
            system_monitor.get_disk_resource, path=disk_path, calc_size=calc_size, mode=mode
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
        logger.exception(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Example Usage ====================


@app.get("/examples/inference")
async def get_inference_example() -> dict[str, Any]:
    """Get the inference config examples."""
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
async def get_training_example() -> dict[str, Any]:
    """Get the training config examples."""
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
async def get_conversion_example() -> dict[str, Any]:
    """Get the GGUF conversion examples."""
    return {
        "recommended_outtypes": ["auto", "f16", "bf16", "q8_0", "f32"],
        "notes": {
            "auto": "Recommended; lets llama.cpp pick a common 16-bit type from the weights",
            "f16": "The most common half-precision full-model format",
            "bf16": "Commonly used when the original model is already bfloat16",
            "q8_0": "A common high-quality quantization format, smaller in size",
            "f32": "Full precision, largest in size, normally only for verification or debugging",
        },
        "examples": config_examples.CONVERSION_CONFIG_EXAMPLES,
    }


def run() -> None:
    """
    Start uvicorn using settings.py.

    Lets `python -m service.app` and the shell scripts share the same settings.
    """
    import uvicorn

    uvicorn.run(
        "service.app:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=UVICORN_RELOAD,
        reload_excludes=UVICORN_RELOAD_EXCLUDES,
        log_level=LOG_LEVEL.lower(),
        log_config=get_uvicorn_log_config(),
        access_log=UVICORN_ACCESS_LOG,
        use_colors=UVICORN_USE_COLORS,
    )


if __name__ == "__main__":
    run()
