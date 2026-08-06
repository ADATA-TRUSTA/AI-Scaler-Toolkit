"""
Async HTTP client for server-based inference engines (llama-server / vLLM).

Both server engines expose an OpenAI-compatible HTTP API. Historically the
main (FastAPI) process reached them the long way round: through a
multiprocessing worker + a shared ``data_queue`` + a per-request thread bridge.
This module lets the main process talk to those managed servers **directly and
asynchronously**, so a request is just an ``await`` on the event loop.

The payloads emitted here intentionally match the manager-level shape that the
app layer already consumes from ``ModelInferenceProcess`` (``chunk`` / ``done``
for streaming; ``result`` / ``tool_calls`` / token stats for non-stream), so the
call sites need no new payload handling.

NOTE: this path only serves the server engines. The transformers engine runs
in-process and still uses the worker + queue + dispatcher demux path.
"""

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, AsyncOpenAI

from ..config_models import InferenceConfig, InferenceEngine
from ..settings import (
    LLAMA_SERVER_API_KEY,
    VLLM_CLIENT_HOST,
    VLLM_OPENAI_API_KEY,
    VLLM_PORT,
    VLLM_SERVED_MODEL_NAME,
    configure_logging,
)

logger = configure_logging(__name__)

# Engines that expose an OpenAI-compatible HTTP server and can therefore be
# reached directly from the main process (bypassing the worker + queue).
SERVER_ENGINES = (InferenceEngine.LLAMA_SERVER, InferenceEngine.VLLM)


def is_server_engine(config: InferenceConfig | None) -> bool:
    """Return True if the config targets an OpenAI-compatible server engine."""
    return bool(config) and getattr(config, "engine", None) in SERVER_ENGINES


def resolve_server_endpoint(config: InferenceConfig) -> dict[str, Any]:
    """
    Return {base_url, api_key} for the active server engine's managed server.

    Mirrors ``LlamaServerEngine.load_model`` exactly so the main process talks
    to the same server the engine targets: an explicit ``llama_server_url``
    wins regardless of ``llama_server_auto_start``; otherwise the engine binds
    ``config.llama_server_host/port``. ``base_url`` already includes the
    ``/v1`` suffix expected by the OpenAI client.
    """
    engine = getattr(config, "engine", None)
    if engine == InferenceEngine.LLAMA_SERVER:
        url = (getattr(config, "llama_server_url", None) or "").strip()
        base = url or f"http://{config.llama_server_host}:{config.llama_server_port}"
        api_key = getattr(config, "llama_server_api_key", None) or LLAMA_SERVER_API_KEY or "EMPTY"
        return {
            "base_url": f"{base.rstrip('/')}/v1",
            "api_key": api_key,
        }
    if engine == InferenceEngine.VLLM:
        return {
            "base_url": f"http://{VLLM_CLIENT_HOST}:{VLLM_PORT}/v1",
            "api_key": VLLM_OPENAI_API_KEY or "EMPTY",
        }
    raise ValueError(f"Engine {engine} is not an OpenAI-compatible server engine")


def client_for_config(config: InferenceConfig, *, model: str | None = None) -> "AsyncServerClient":
    """Build an AsyncServerClient pointed at the active server engine."""
    endpoint = resolve_server_endpoint(config)
    engine = getattr(config, "engine", None)
    if model is None:
        if engine == InferenceEngine.VLLM:
            # vLLM validates the payload's model field against its
            # --served-model-name, which VllmEngine sets to
            # VLLM_SERVED_MODEL_NAME or model_path or model_name (see
            # VllmEngine._resolve_served_model_name) — model_name alone would
            # 404 whenever model_path is set.
            model = (
                VLLM_SERVED_MODEL_NAME
                or getattr(config, "model_path", None)
                or getattr(config, "model_name", None)
            )
        else:
            # llama.cpp's server ignores the model field; keep model_name for
            # Qwen detection in the thinking-tag / reasoning logic.
            model = getattr(config, "model_name", None)
    return AsyncServerClient(
        endpoint["base_url"],
        api_key=endpoint["api_key"],
        model=model,
        engine=engine,
    )


_MEDIA_PART_TYPES = {
    "image_url",
    "image",
    "audio_url",
    "audio",
    "input_audio",
    "video_url",
    "video",
}
_TEXT_PART_TYPES = {"text"}


def normalize_messages(prompt: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - arbitrary client-supplied prompt payload
    """
    Coerce a prompt into OpenAI chat messages, preserving multi-part content.

    Ported from ``VllmEngine._normalize_messages``: list content (multimodal
    multi-part) must be kept as-is — never ``str()``-flattened, or images are
    lost.
    """
    if isinstance(prompt, list):
        normalized: list[dict[str, Any]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                normalized.append({"role": role, "content": content})
            elif isinstance(content, dict):
                normalized.append({"role": role, "content": [content]})
            else:
                normalized.append({"role": role, "content": str(content)})
        if normalized:
            return normalized

    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": str(prompt)}]


def reorder_multimodal_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Order image/audio/video parts before text parts within multi-part content.

    Ported from the former ``VllmEngine._reorder_multimodal_content`` (removed
    with the worker generate path; this is now the only implementation).
    Gemma best practice: media before text. Only reorders when both media and
    text are present; unknown parts kept last.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        media_parts: list[Any] = []
        text_parts: list[Any] = []
        other_parts: list[Any] = []
        for p in content:
            if not isinstance(p, dict):
                other_parts.append(p)
                continue
            ptype = p.get("type")
            if ptype in _MEDIA_PART_TYPES:
                media_parts.append(p)
            elif ptype in _TEXT_PART_TYPES:
                text_parts.append(p)
            else:
                other_parts.append(p)
        if media_parts and text_parts:
            msg["content"] = media_parts + text_parts + other_parts
    return messages


def _is_qwen_model(model_name: Any) -> bool:  # noqa: ANN401 - accepts model id of any incoming type
    return "qwen" in str(model_name or "").lower()


def _is_qwen35_model(model_name: Any) -> bool:  # noqa: ANN401 - accepts model id of any incoming type
    return "qwen3.5" in str(model_name or "").lower()


def _apply_thinking_tag(
    messages: list[dict[str, Any]],
    enable_thinking: bool | None,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Append ``/think`` or ``/no_think`` to the last user message for Qwen.

    Ported from the dev llama_server_runner: alongside
    ``chat_template_kwargs.enable_thinking`` this is Qwen's belt-and-braces way
    of toggling reasoning. No-op unless the model is a Qwen model and
    ``enable_thinking`` is explicitly set. Copies before mutating so the caller's
    messages are not modified in place.

    Qwen3.5 is excluded: its chat template has no /think soft switch (thinking
    is toggled solely by ``chat_template_kwargs.enable_thinking``), and the
    literal tag leaks into the prompt — verified against Qwen3.5-35B-A3B, whose
    reasoning then treats the tag as part of the user's request.
    """
    if (
        enable_thinking is None
        or not _is_qwen_model(model_name)
        or _is_qwen35_model(model_name)
        or not messages
    ):
        return messages

    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break
    if last_user_idx < 0:
        return messages

    tag = " /think" if enable_thinking else " /no_think"
    msg = dict(messages[last_user_idx])
    content = msg.get("content")

    if isinstance(content, str):
        stripped = content.rstrip()
        if not (stripped.endswith("/think") or stripped.endswith("/no_think")):
            msg["content"] = content + tag
            messages = list(messages)
            messages[last_user_idx] = msg
            logger.info("[AsyncServerClient] Applied thinking tag '%s'", tag.strip())
        return messages

    if isinstance(content, list):
        last_text_idx = -1
        for idx in range(len(content) - 1, -1, -1):
            part = content[idx]
            if isinstance(part, dict) and str(part.get("type", "")).strip().lower() == "text":
                last_text_idx = idx
                break
        if last_text_idx >= 0:
            new_content = list(content)
            part = dict(new_content[last_text_idx])
            text = str(part.get("text", ""))
            stripped = text.rstrip()
            if not (stripped.endswith("/think") or stripped.endswith("/no_think")):
                part["text"] = text + tag
                new_content[last_text_idx] = part
                msg["content"] = new_content
                messages = list(messages)
                messages[last_user_idx] = msg
                logger.info("[AsyncServerClient] Applied thinking tag '%s'", tag.strip())
    return messages


def _extract_reasoning(obj: Any) -> str | None:  # noqa: ANN401 - OpenAI SDK delta/message object
    """
    Pull reasoning text out of an SDK delta/message.

    Servers disagree on the field name (llama-server / gpt-oss use
    ``reasoning_content``; vLLM's reasoning parsers use ``reasoning``) and the
    OpenAI SDK stashes unknown fields in ``model_extra`` rather than exposing
    them as attributes, so check both names in both places.
    """
    if obj is None:
        return None
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _wrap_reasoning(text: Any) -> str:  # noqa: ANN401 - reasoning value may be str or other SDK type
    """Collapse whitespace and wrap reasoning text in a single think block."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    if "<think>" in normalized or "</think>" in normalized:
        return normalized
    return f"<think>{normalized}</think>"


def build_chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any] | None,
    stream: bool,
    engine: InferenceEngine | None = None,
) -> dict[str, Any]:
    """
    Map internal generation params to an OpenAI chat-completions payload.

    Mirrors the dev engine payload builders so the async path is behaviourally
    identical to the old queue path: same normalization, multimodal reorder,
    Qwen thinking tag, and engine-specific repetition-penalty naming
    (llama.cpp wants ``repeat_penalty``; vLLM wants ``repetition_penalty``).
    """
    params = params or {}
    enable_thinking = params.get("enable_thinking")

    messages = reorder_multimodal_content(normalize_messages(messages))
    messages = _apply_thinking_tag(messages, enable_thinking, model)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max(1, int(params.get("max_new_tokens", 512))),
        "temperature": max(0.0, min(float(params.get("temperature", 0.7)), 2.0)),
        "top_p": max(0.0, min(float(params.get("top_p", 0.9)), 1.0)),
        "stream": stream,
    }

    tools = params.get("tools")
    if tools:
        payload["tools"] = tools

    tool_choice = params.get("tool_choice")
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    extra_body: dict[str, Any] = {}
    repetition_penalty = params.get("repetition_penalty")
    if engine == InferenceEngine.LLAMA_SERVER:
        # llama.cpp's OpenAI server uses the native name and silently ignores
        # ``repetition_penalty``; dev always sends ``repeat_penalty`` (default 1.1).
        extra_body["repeat_penalty"] = (
            float(repetition_penalty) if repetition_penalty is not None else 1.1
        )
    elif repetition_penalty is not None:
        extra_body["repetition_penalty"] = float(repetition_penalty)

    top_k = params.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = int(top_k)

    if enable_thinking is not None:
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)

    if extra_body:
        payload["extra_body"] = extra_body

    return payload


def _accumulate_tool_call_deltas(acc: dict[int, dict[str, Any]], deltas: Any) -> None:  # noqa: ANN401 - OpenAI SDK streaming tool_call delta objects
    """
    Merge streaming tool_call fragments (index-keyed) into an accumulator.

    OpenAI streams tool calls as partial fragments: the name arrives once and
    arguments come in pieces, keyed by ``index``. We reassemble them so the
    complete tool_calls can be emitted on the final (done) event, matching what
    the tool-passthrough consumer expects.
    """
    if not deltas:
        return
    for tc in deltas:
        idx = getattr(tc, "index", 0) or 0
        slot = acc.setdefault(
            idx,
            {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
        )
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        if getattr(tc, "type", None):
            slot["type"] = tc.type
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["function"]["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["function"]["arguments"] += fn.arguments


def _assemble_tool_calls(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not acc:
        return None
    return [acc[i] for i in sorted(acc.keys())]


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:  # noqa: ANN401 - OpenAI SDK tool_call objects
    """Convert OpenAI SDK tool_call objects into plain JSON-able dicts."""
    if not tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        try:
            fn = getattr(tc, "function", None)
            out.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                }
            )
        except Exception:  # pragma: no cover - defensive
            continue
    return out or None


class AsyncServerClient:
    """Thin async wrapper around an OpenAI-compatible managed server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        model: str | None = None,
        engine: InferenceEngine | None = None,
    ) -> None:
        # ``base_url`` is expected to already include the ``/v1`` suffix.
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self._model = model
        self.engine = engine
        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client, ignoring shutdown errors."""
        try:
            await self.client.close()
        except Exception:  # pragma: no cover - defensive
            pass

    async def resolve_model(self, timeout_s: float = 5.0) -> str:
        """Return the served model id, querying ``/v1/models`` if unknown."""
        if self._model:
            return self._model
        models = await self.client.models.list(timeout=timeout_s)
        data = getattr(models, "data", None) or []
        if not data:
            raise RuntimeError("Server reported no served models")
        self._model = data[0].id
        return self._model

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream chat completions, yielding manager-level payload dicts.

        Yields ``{"chunk": str, "done": False, ...}`` per token batch and a final
        ``{"chunk": "", "done": True, <token stats>}``. Raises ``RuntimeError`` on
        a server/transport error (mirrors the queue path so callers can reuse
        their existing OOM/error handling).
        """
        params = params or {}
        model = await self.resolve_model()
        payload = build_chat_payload(model, messages, params, stream=True, engine=self.engine)
        timeout_s = float(params.get("total_timeout", 300))

        # Reasoning models (e.g. gpt-oss) emit reasoning_content while thinking
        # and no content yet; forward it wrapped in a single <think> block so the
        # frontend shows progress instead of a silent gap. Mirrors dev's
        # llama_server_runner stream logic.
        include_reasoning = params.get("enable_thinking") is not None
        reasoning_fallback = _is_qwen_model(model)
        in_thinking_block = False

        start = time.perf_counter()
        prompt_tokens: int | None = None
        gen_tokens = 0
        total_tokens: int | None = None
        stopped = False
        tool_call_acc: dict[int, dict[str, Any]] = {}

        try:
            stream = await self.client.chat.completions.create(
                timeout=timeout_s,
                stream_options={"include_usage": True},
                **payload,
            )
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.exception("[AsyncServerClient] generate_stream error: %s", e)
            raise RuntimeError(str(e)) from e

        try:
            async for event in stream:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break

                out_parts: list[str] = []
                if event.choices:
                    delta = event.choices[0].delta
                    _accumulate_tool_call_deltas(tool_call_acc, getattr(delta, "tool_calls", None))
                    reasoning_chunk = _extract_reasoning(delta)
                    content_chunk = delta.content

                    if include_reasoning and isinstance(reasoning_chunk, str) and reasoning_chunk:
                        if not in_thinking_block:
                            out_parts.append("<think>\n")
                            in_thinking_block = True
                        out_parts.append(reasoning_chunk)
                    elif (
                        reasoning_fallback
                        and isinstance(reasoning_chunk, str)
                        and reasoning_chunk
                        and not content_chunk
                    ):
                        # Qwen sometimes returns only reasoning_content even with
                        # thinking off; forward it so the stream is not empty.
                        if not in_thinking_block:
                            out_parts.append("<think>\n")
                            in_thinking_block = True
                        out_parts.append(reasoning_chunk)

                    if isinstance(content_chunk, str) and content_chunk:
                        if in_thinking_block:
                            out_parts.append("\n</think>\n")
                            in_thinking_block = False
                        out_parts.append(content_chunk)

                usage = getattr(event, "usage", None)
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        prompt_tokens = int(usage.prompt_tokens)
                    if getattr(usage, "completion_tokens", None) is not None:
                        gen_tokens = int(usage.completion_tokens)
                    if getattr(usage, "total_tokens", None) is not None:
                        total_tokens = int(usage.total_tokens)

                chunk_text = "".join(out_parts)
                if chunk_text:
                    # Stream text (and reasoning) live; tool_calls are reassembled
                    # and delivered on the done event (fragments are useless here).
                    yield {"chunk": chunk_text, "done": False}

            # Close a dangling think block (reasoning-only or stopped mid-think).
            if in_thinking_block:
                in_thinking_block = False
                yield {"chunk": "\n</think>", "done": False}

            elapsed = max(1e-6, time.perf_counter() - start)
            gen_tps = float(gen_tokens) / elapsed if gen_tokens else 0.0
            prompt_tps = float(prompt_tokens) / elapsed if prompt_tokens else 0.0
            done_payload: dict[str, Any] = {
                "chunk": "",
                "done": True,
                "gen_tokens": gen_tokens,
                "gen_tps": gen_tps,
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
                "prompt_tps": prompt_tps,
            }
            if total_tokens is not None:
                done_payload["total_tokens"] = total_tokens
            elif prompt_tokens is not None:
                done_payload["total_tokens"] = int(prompt_tokens) + int(gen_tokens)
            else:
                done_payload["total_tokens"] = gen_tokens
            assembled_tool_calls = _assemble_tool_calls(tool_call_acc)
            if assembled_tool_calls is not None:
                done_payload["tool_calls"] = assembled_tool_calls
                if not done_payload.get("finish_reason"):
                    done_payload["finish_reason"] = "tool_calls"
            if stopped:
                done_payload["stopped"] = True
            yield done_payload
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.exception("[AsyncServerClient] generate_stream error: %s", e)
            raise RuntimeError(str(e)) from e
        finally:
            # Ensure the underlying httpx stream is released on any exit path:
            # normal completion, stop_event break, an error, or the consumer
            # abandoning iteration (GeneratorExit on client disconnect).
            try:
                await stream.close()
            except Exception:
                pass

    async def generate(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Non-stream chat completion, returning a manager-level result dict."""
        params = params or {}
        model = await self.resolve_model()
        payload = build_chat_payload(model, messages, params, stream=False, engine=self.engine)
        timeout_s = float(params.get("total_timeout", 300))

        start = time.perf_counter()
        try:
            completion = await self.client.chat.completions.create(timeout=timeout_s, **payload)
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.exception("[AsyncServerClient] generate error: %s", e)
            raise RuntimeError(str(e)) from e

        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None) if choice else None
        content = getattr(message, "content", "") if message else ""
        # Merge reasoning_content (gpt-oss / Qwen) into the returned text, wrapped
        # in a think block, mirroring dev's non-stream behaviour.
        reasoning = _extract_reasoning(message) if message else None
        if isinstance(reasoning, str) and reasoning:
            include_reasoning = params.get("enable_thinking") is not None
            if include_reasoning or _is_qwen_model(model):
                content = f"{_wrap_reasoning(reasoning)}{content or ''}"
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        tool_calls = _serialize_tool_calls(
            getattr(message, "tool_calls", None) if message else None
        )

        elapsed = max(1e-6, time.perf_counter() - start)
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        gen_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = (
            int(getattr(usage, "total_tokens", 0) or 0) if usage else prompt_tokens + gen_tokens
        )

        result: dict[str, Any] = {
            "result": content or "",
            "gen_tokens": gen_tokens,
            "gen_tps": float(gen_tokens) / elapsed if gen_tokens else 0.0,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": float(prompt_tokens) / elapsed if prompt_tokens else 0.0,
            "total_tokens": total_tokens,
        }
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        if tool_calls is not None:
            result["tool_calls"] = tool_calls
        return result
