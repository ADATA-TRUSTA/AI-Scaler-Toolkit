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
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, AsyncOpenAI

from ..config_models import InferenceConfig, InferenceEngine
from ..settings import (
    LLAMA_SERVER_API_KEY,
    MAX_OUTPUT_TOKENS_CAP,
    VLLM_CLIENT_HOST,
    VLLM_OPENAI_API_KEY,
    VLLM_PORT,
    VLLM_SERVED_MODEL_NAME,
    configure_logging,
)
from .model_family import is_qwen35_family

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
        model_path=getattr(config, "model_path", None),
    )


# OpenAI's "developer" role carries the same meaning as "system" -- it exists
# because reasoning models on the real API reject a literal "system" message, not
# because the instructions are semantically different. Clients built against that
# convention (pi-agent's coding-agent harness included) send "developer" as the
# very first message. Local chat templates were never taught that role: Qwen's
# (confirmed against QuantTrio/Qwen3.5-4B-AWQ under vLLM 0.27.1, but this is a
# generic Jinja pattern, not Qwen-specific) walks an explicit
# system/user/assistant/tool chain and raises "Unexpected message role." on
# anything else, which the OpenAI-compatible server surfaces as a 400 the caller
# cannot work around. Remapping here means every server engine gets the fix once,
# rather than each chat template needing to special-case a role it will never see
# from a real model-authored conversation.
_ROLE_ALIASES = {"developer": "system"}


def normalize_message_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remap client-side role names a local chat template would reject (see above)."""
    out = messages
    for idx, msg in enumerate(messages):
        # A missing or non-string role has no alias to look up, and dict.get would reject
        # None as a key on a str-keyed mapping.
        role = msg.get("role")
        if not isinstance(role, str):
            continue
        alias = _ROLE_ALIASES.get(role)
        if alias is None:
            continue
        if out is messages:
            out = list(messages)
        out[idx] = {**msg, "role": alias}
    return out


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


def reorder_multimodal_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Order image/audio/video parts before text parts within multi-part content.

    Ported from the former ``VllmEngine._reorder_multimodal_content`` (removed
    with the worker generate path; this is now the only implementation).
    Gemma best practice: media before text. Only reorders when both media and
    text are present; unknown parts kept last.

    Copy-on-write: callers reuse their message list (app.py's passthrough
    messages also feed session history), so a reordered message is replaced with
    a copy instead of being mutated in place. This used to be masked by a
    normalization pass that handed over freshly built dicts; that pass is gone
    because it silently dropped tool_calls / tool_call_id / name, which made the
    model imitate a broken history and stop after a preamble.
    """
    out = messages
    for idx, msg in enumerate(messages):
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
            if out is messages:
                out = list(messages)
            out[idx] = {**msg, "content": media_parts + text_parts + other_parts}
    return out


def _merge_images(messages: list[dict[str, Any]], images: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401 - caller-supplied image list of unknown shape
    """
    Attach ``images`` to the last user message as OpenAI multi-part content.

    agenerate/agenerate_stream take an ``images`` argument and pass it down in params,
    but nothing downstream ever read it, so every caller that sent an image got a
    text-only answer with no indication the image had been dropped. A vision model
    describing a picture it was never shown is worse than an error.

    Images belong on the last *user* turn — the one being answered. Appending a
    separate message would break the alternation llama.cpp's chat templates expect, and
    attaching to the tail regardless of role could hang them off an assistant turn.

    Entries may be data: URIs or plain URLs; both are passed through unchanged, since
    that is exactly what the OpenAI image_url part carries. A message whose content is
    already multi-part keeps its parts and gains the images. Ordering relative to text
    is left to reorder_multimodal_content, which the caller runs next.
    """
    if not images:
        return messages

    urls = [str(u) for u in images if u]
    if not urls:
        return messages

    index = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None
    )
    if index is None:
        # No user turn to attach to. Dropping the images silently is what this fix
        # exists to stop, so carry them in one of their own.
        messages = [*messages, {"role": "user", "content": []}]
        index = len(messages) - 1

    target = dict(messages[index])
    content = target.get("content")
    parts: list[Any] = []
    if isinstance(content, list):
        parts = list(content)
    elif content:
        parts = [{"type": "text", "text": str(content)}]

    parts.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
    target["content"] = parts

    merged = list(messages)
    merged[index] = target
    return merged


def _is_qwen_model(model_name: Any) -> bool:  # noqa: ANN401 - accepts model id of any incoming type
    return "qwen" in str(model_name or "").lower()


def _is_qwen35_model(model_name: Any, model_path: str | None = None) -> bool:  # noqa: ANN401 - accepts model id of any incoming type
    """Detect the Qwen3.5-generation family from declared arch, name as fallback."""
    return is_qwen35_family(model_path=model_path, model_name=model_name)


def _apply_thinking_tag(
    messages: list[dict[str, Any]],
    enable_thinking: bool | None,
    model_name: str,
    model_path: str | None = None,
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
        or _is_qwen35_model(model_name, model_path)
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
    model_path: str | None = None,
) -> dict[str, Any]:
    """
    Map internal generation params to an OpenAI chat-completions payload.

    Mirrors the dev engine payload builders so the async path is behaviourally
    identical to the old queue path: role normalization, image merge, multimodal
    reorder, Qwen thinking tag, and engine-specific repetition-penalty naming
    (llama.cpp wants ``repeat_penalty``; vLLM wants ``repetition_penalty``).
    """
    params = params or {}
    enable_thinking = params.get("enable_thinking")

    messages = normalize_message_roles(messages)
    messages = _merge_images(messages, params.get("images"))
    messages = reorder_multimodal_content(messages)
    messages = _apply_thinking_tag(messages, enable_thinking, model, model_path)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": max(0.0, min(float(params.get("temperature", 0.7)), 2.0)),
        "top_p": max(0.0, min(float(params.get("top_p", 0.9)), 1.0)),
        "stream": stream,
    }
    # No cap asked for means "until EOS or the window is full" -- the OpenAI
    # default, and what llama.cpp does with no n_predict. Substituting a number
    # truncated every caller that did not name one: at 512 tokens a reasoning
    # model can spend the whole budget thinking and return nothing at all.
    #
    # MAX_OUTPUT_TOKENS_CAP is the operator's blanket ceiling and is clamped on
    # top, including for a caller that named nothing -- that is a limit someone
    # deliberately configured, not a figure invented on their behalf. Unset (the
    # default) leaves the served context window as the only bound.
    cap = params.get("max_new_tokens")
    cap = cap if isinstance(cap, int) and cap > 0 else None
    if MAX_OUTPUT_TOKENS_CAP is not None:
        cap = MAX_OUTPUT_TOKENS_CAP if cap is None else min(cap, MAX_OUTPUT_TOKENS_CAP)
    if cap is not None:
        payload["max_tokens"] = cap

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
        # Reuse the KV cache for the shared prefix of a multi-turn conversation.
        # Without it llama-server reprocesses the whole history on every turn, which
        # grows with the conversation and shows up as seconds of added TTFT — the
        # queue path forced it on and the async path silently did not.
        #
        # dev also pinned id_slot so a conversation kept landing on the same slot.
        # That needs the engine's slot bookkeeping, which this client has no access
        # to; without it llama-server still matches the longest cached prefix across
        # slots, so this recovers most of the benefit and none of the risk.
        extra_body["cache_prompt"] = True
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


# Some models do not reliably follow the exact tag their own chat template
# documents for a tool call. Observed from Qwen2.5-Coder under vLLM 0.27.1 with
# --enable-auto-tool-choice --tool-call-parser hermes (both AWQ and fp8, so this
# is not a quantization artifact): the template says <tool_call>...</tool_call>,
# but generation instead produced <tools>{...}</tools> (confusing it with the
# *listing* tag from the same system prompt), a fenced ```xml <response>{...}
# </response>``` block, or -- seen live against a real pi-agent request -- a
# plain prose explanation ("You can use the bash function...") followed by a
# bare ```json {...} ``` fence with no wrapper tag at all. hermes (and every
# other bundled vLLM tool parser) then finds nothing and leaves the whole thing
# in `content` as plain text, which a client that only understands tool_calls
# displays as prose instead of acting on it. These patterns recognise a JSON
# object with "name"/"arguments" in any of those shapes and recover it into a
# normal tool_calls entry; the last one is deliberately the most permissive, so
# it is tried last and only wins when nothing more specific matched.
_FALLBACK_TOOL_CALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<tools>\s*(\{.*?\})\s*</tools>", re.DOTALL),
    re.compile(r"<response>\s*(\{.*?\})\s*</response>", re.DOTALL),
    re.compile(r"```[a-zA-Z]*\s*\n?(\{.*?\})\s*\n?```", re.DOTALL),
)
# Opening markers for the same shapes, used by the streaming path to decide
# whether to keep withholding content chunks while the wrapper is still being
# written. The fence is included because ```xml normally precedes <response>,
# and now also stands on its own for the bare ```json fence.
_FALLBACK_OPEN_MARKERS = ("<tools>", "<response>", "```")
_FALLBACK_MAX_MARKER_LEN = max(len(marker) for marker in _FALLBACK_OPEN_MARKERS)
# Give up waiting for a close tag past this many buffered characters -- a real
# tool-call JSON payload is short, so this is generous without risking an
# unbounded stall on a genuine long-form answer that happens to start the same way.
_FALLBACK_BUFFER_LIMIT = 2000


def _fallback_tool_names(payload: dict[str, Any]) -> frozenset[str]:
    """Function names the request actually declared, for validating a recovered match."""
    names: set[str] = set()
    for tool in payload.get("tools") or ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def _fallback_match_tool_call(
    content: str, tool_names: frozenset[str] | None = None
) -> tuple[re.Match[str], dict[str, Any]] | None:
    """
    Find the first fallback-wrapped tool call in `content`, anywhere in it.

    A model that wraps a tool call in prose (see the module comment above) puts
    it after an explanation rather than at the start, so this searches the
    whole string rather than anchoring to the front. ``tool_names``, when
    given, rejects a match whose "name" is not one of the request's actual
    declared tools -- without it the last (bare-fence) pattern would also catch
    an unrelated JSON example the model was legitimately asked to produce.
    """
    if not content:
        return None
    for pattern in _FALLBACK_TOOL_CALL_PATTERNS:
        for match in pattern.finditer(content):
            try:
                parsed = json.loads(match.group(1))
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            if not isinstance(name, str) or not name:
                continue
            if tool_names and name not in tool_names:
                continue
            return match, parsed
    return None


def _fallback_tool_call_dict(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a normal tool_calls entry from a recovered {"name", "arguments"} object."""
    arguments = parsed.get("arguments")
    if isinstance(arguments, dict):
        arguments_json = json.dumps(arguments)
    elif isinstance(arguments, str):
        try:
            json.loads(arguments)  # already a JSON-encoded arguments string
            arguments_json = arguments
        except (ValueError, TypeError):
            arguments_json = "{}"
    else:
        arguments_json = "{}"
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": parsed["name"], "arguments": arguments_json},
        }
    ]


def _fallback_parse_tool_call(
    content: str, tool_names: frozenset[str] | None = None
) -> list[dict[str, Any]] | None:
    """Recover a tool call the configured server-side parser did not recognise."""
    found = _fallback_match_tool_call(content, tool_names)
    return _fallback_tool_call_dict(found[1]) if found else None


def _looks_like_fallback_prefix(buffer: str) -> bool:
    """True while `buffer` could still grow into one of the fallback wrapper shapes."""
    probe = re.sub(r"^```[a-zA-Z]*\n?", "", buffer).lstrip()
    if not probe:
        return True
    if probe.startswith("{"):
        # The bare ```json {...}``` shape: the object starts right after the
        # fence, with no inner <tools>/<response> tag to match against below.
        return True
    return any(
        marker.startswith(probe) or probe.startswith(marker) for marker in _FALLBACK_OPEN_MARKERS
    )


def _find_fallback_marker_start(buffer: str) -> int | None:
    """Earliest index where a fallback wrapper marker fully appears in `buffer`, if any."""
    positions = [buffer.index(marker) for marker in _FALLBACK_OPEN_MARKERS if marker in buffer]
    return min(positions) if positions else None


def _fallback_marker_tail_hold(buffer: str) -> int:
    """
    How many trailing characters of `buffer` might still grow into a marker.

    A chunk boundary can split a marker across two deltas (a lone "`" now, "``
    json" next), so while scanning plain content for where a wrapper might
    start, the last few characters are held back rather than forwarded until
    it is clear whether they are about to become one. Only called once
    `_find_fallback_marker_start` has confirmed no marker is fully present yet.
    """
    limit = min(len(buffer), _FALLBACK_MAX_MARKER_LEN - 1)
    for length in range(limit, 0, -1):
        tail = buffer[-length:]
        if any(marker.startswith(tail) for marker in _FALLBACK_OPEN_MARKERS):
            return length
    return 0


_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "failed to allocate",
    "cannot allocate",
    "insufficient memory",
    "not enough memory",
    "cuda error: out of memory",
    "ggml_backend_alloc",
)

# "OOM" as a word of its own. It used to sit in the tuple above and be matched as a
# bare substring, which also fired on "room", "zoom", and any model id, path or
# hostname carrying those three letters. A misfire is not cosmetic here: it sets
# recoverable=False, so the caller tells the user to shrink the context instead of
# retrying what was really a dropped connection.
_OOM_WORD = re.compile(r"\boom\b")

_DISCONNECT_MARKERS = (
    "server disconnected without sending a response",
    "remoteprotocolerror",
    "connection reset",
    "broken pipe",
    "connection aborted",
    "incomplete chunked read",
)


def classify_server_error(exc: Exception) -> dict[str, Any]:
    """
    Structured payload for a server-side generation failure.

    The async path raised ``RuntimeError(str(e))`` and nothing else, so an OOM and a
    dropped socket reached the UI as the same opaque string. Callers could not tell a
    user to shrink the context rather than retry, and /inference/error_details had
    nothing to serve.

    LlamaServerEngine.build_runtime_error_payload already does this properly, but it
    reads the managed process's exit code and stderr and probes the port — state this
    client has no handle on. So this classifies on the exception text alone and reports
    what it can actually establish: ``process_alive`` and ``fatal`` are left out rather
    than guessed, since claiming a live server that has in fact died would be worse
    than saying nothing. When the engine's richer payload is available it should win.

    The shape matches build_runtime_error_payload's so /inference/error_details can
    serve either without branching.
    """
    raw = str(exc).strip() or exc.__class__.__name__
    low = raw.lower()

    is_oom = bool(_OOM_WORD.search(low)) or any(marker in low for marker in _OOM_MARKERS)
    disconnected = any(marker in low for marker in _DISCONNECT_MARKERS)

    if is_oom:
        error_type = "LlamaServerOOM"
    elif disconnected:
        # Cannot distinguish "the process died" from "the connection dropped" without
        # the process handle; the engine's version splits these into ProcessExited and
        # Disconnected. Disconnected is the one that does not overclaim.
        error_type = "LlamaServerDisconnected"
    else:
        error_type = exc.__class__.__name__ or "LlamaServerRuntimeError"

    return {
        "error": raw,
        "error_type": error_type,
        "is_oom": is_oom,
        "recoverable": not is_oom,
    }


def _server_timings(obj: Any) -> tuple[float, float] | None:  # noqa: ANN401 - OpenAI SDK response object
    """
    (prompt_seconds, decode_seconds) as reported by llama-server, or None.

    llama.cpp attaches a ``timings`` object to its OpenAI-compatible responses with
    the two phases already separated and measured server-side — no network or client
    scheduling in the numbers. The OpenAI SDK keeps unknown fields, so it survives.

    Preferred over anything measured here whenever it is present; vLLM does not send
    it, and then the caller falls back to its own clock.
    """
    timings = getattr(obj, "timings", None)
    if timings is None and hasattr(obj, "model_extra"):
        timings = (obj.model_extra or {}).get("timings")
    if timings is None:
        return None

    def _ms(name: str) -> float | None:
        value = timings.get(name) if isinstance(timings, dict) else getattr(timings, name, None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    prompt_ms, predicted_ms = _ms("prompt_ms"), _ms("predicted_ms")
    if prompt_ms is None or predicted_ms is None:
        return None
    return prompt_ms / 1000.0, predicted_ms / 1000.0


def _as_runtime_error(exc: Exception) -> RuntimeError:
    """Wrap a transport/server error, carrying the classification alongside the text."""
    error = RuntimeError(str(exc))
    error.error_payload = classify_server_error(exc)  # type: ignore[attr-defined]
    return error


# llama.cpp counts prompt + max_tokens against n_ctx and refuses the whole
# request, naming both figures.
_CTX_OVERFLOW_RE = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)"
)
# Slack between the count in the refusal and the retry: the chat template can
# add a token or two, and a retry that overflows again is worse than one that
# leaves a little output on the table.
_CTX_REFIT_MARGIN = 64
# Below this there is nothing useful to generate, so the prompt itself is what
# does not fit. Let that error through -- the caller has to shorten its history,
# and silently returning a two-token answer would hide the real problem.
_CTX_REFIT_MIN = 256


def _refit_max_tokens(payload: dict[str, Any], exc: Exception) -> int | None:
    """
    An output cap that fits, or None if this is not an overflow we can fix.

    A client that sizes max_tokens from the advertised context window overflows
    by the length of its own prompt, every time, however short that prompt is --
    and reads the refusal as "my history is too long", which compressing cannot
    fix. The numbers in the refusal are the only exact prompt length available
    here; the alternative is tokenising every request ourselves to re-derive
    what the engine just told us.
    """
    match = _CTX_OVERFLOW_RE.search(str(exc))
    if not match:
        return None
    requested, window = int(match.group(1)), int(match.group(2))
    asked = payload.get("max_tokens")
    if not isinstance(asked, int) or asked <= 0 or asked > requested:
        return None
    room = window - (requested - asked) - _CTX_REFIT_MARGIN
    return room if room >= _CTX_REFIT_MIN else None


class AsyncServerClient:
    """Thin async wrapper around an OpenAI-compatible managed server."""

    async def _create_completion(
        self,
        payload: dict[str, Any],
        timeout_s: float,
        label: str,
        **extra: Any,  # noqa: ANN401 - passthrough to the OpenAI client
    ) -> Any:  # noqa: ANN401 - the client returns a stream or a completion
        """create(), retried once with an output cap that fits the context."""
        for attempt in (1, 2):
            try:
                return await self.client.chat.completions.create(
                    timeout=timeout_s, **extra, **payload
                )
            except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
                fitted = _refit_max_tokens(payload, e) if attempt == 1 else None
                if fitted is None:
                    logger.exception("[AsyncServerClient] %s error: %s", label, e)
                    raise _as_runtime_error(e) from e
                logger.warning(
                    "[AsyncServerClient] max_tokens=%s left no room for the prompt; "
                    "retrying %s with %s",
                    payload.get("max_tokens"),
                    label,
                    fitted,
                )
                payload = {**payload, "max_tokens": fitted}
        raise AssertionError("unreachable")  # pragma: no cover

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        model: str | None = None,
        engine: InferenceEngine | None = None,
        model_path: str | None = None,
    ) -> None:
        # ``base_url`` is expected to already include the ``/v1`` suffix.
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self._model = model
        self.engine = engine
        # Local path to the weights, when known: model-family detection reads the
        # declared architecture from the file instead of guessing from the name.
        self.model_path = model_path
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
        payload = build_chat_payload(
            model, messages, params, stream=True, engine=self.engine, model_path=self.model_path
        )
        timeout_s = float(params.get("total_timeout", 300))

        # Reasoning models (e.g. gpt-oss) emit reasoning_content while thinking
        # and no content yet; forward it wrapped in a single <think> block so the
        # frontend shows progress instead of a silent gap. Mirrors dev's
        # llama_server_runner stream logic.
        include_reasoning = params.get("enable_thinking") is not None
        reasoning_fallback = _is_qwen_model(model)
        in_thinking_block = False

        start = time.perf_counter()
        # When the first token reached us — the boundary between prefill and decode.
        first_token_at: float | None = None
        prompt_tokens: int | None = None
        gen_tokens = 0
        total_tokens: int | None = None
        stopped = False
        tool_call_acc: dict[int, dict[str, Any]] = {}
        server_finish_reason: str | None = None

        # See _fallback_parse_tool_call: some models wrap a tool call in a shape
        # none of vLLM's parsers recognise, sometimes after a prose lead-in
        # ("You can use the bash function...") rather than at the very start of
        # the response. So this scans continuously for where a wrapper marker
        # begins, forwarding everything before it live, then withholds only the
        # suspected wrapper itself until it is resolved one way or the other.
        # Only armed when the request actually offered tools -- a plain chat
        # completion is never buffered or delayed by this.
        #   scanning -- no wrapper marker seen (yet); forwarding content live,
        #               holding back only the last few characters in case they
        #               are the start of one split across a chunk boundary
        #   watching -- a marker started; buffering it instead of yielding it
        #   resolved -- a tool call was recovered; drop any further content
        #   off      -- request declared no tools; plain passthrough throughout
        fallback_state = "scanning" if payload.get("tools") else "off"
        fallback_scan = ""
        fallback_buffer = ""
        fallback_tool_names = _fallback_tool_names(payload) if payload.get("tools") else frozenset()

        stream = await self._create_completion(
            payload,
            timeout_s,
            "generate_stream",
            stream_options={"include_usage": True},
        )

        try:
            async for event in stream:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break

                out_parts: list[str] = []
                if event.choices:
                    choice = event.choices[0]
                    # The server reports why it stopped only on the final choice
                    # delta; keep the last non-null value so the done payload can
                    # distinguish a real EOS from "length" (max_tokens truncation)
                    # or a tool-call stop. Without this every stop looks like "stop".
                    chunk_finish_reason = getattr(choice, "finish_reason", None)
                    if isinstance(chunk_finish_reason, str) and chunk_finish_reason:
                        server_finish_reason = chunk_finish_reason

                    delta = choice.delta
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
                        if fallback_state == "off":
                            out_parts.append(content_chunk)
                        elif fallback_state == "scanning":
                            fallback_scan += content_chunk
                            marker_idx = _find_fallback_marker_start(fallback_scan)
                            if marker_idx is None:
                                hold = _fallback_marker_tail_hold(fallback_scan)
                                safe_len = len(fallback_scan) - hold
                                if safe_len:
                                    out_parts.append(fallback_scan[:safe_len])
                                    fallback_scan = fallback_scan[safe_len:]
                                # else: the whole tail might still become a marker
                            else:
                                # A wrapper marker just started -- forward whatever
                                # came before it live, then start withholding from
                                # the marker itself instead of the whole response.
                                if marker_idx:
                                    out_parts.append(fallback_scan[:marker_idx])
                                fallback_buffer = fallback_scan[marker_idx:]
                                fallback_scan = ""
                                if (
                                    _fallback_parse_tool_call(fallback_buffer, fallback_tool_names)
                                    is not None
                                ):
                                    fallback_state = "resolved"
                                else:
                                    fallback_state = "watching"
                        elif fallback_state == "watching":
                            fallback_buffer += content_chunk
                            still_plausible = len(
                                fallback_buffer
                            ) <= _FALLBACK_BUFFER_LIMIT and _looks_like_fallback_prefix(
                                fallback_buffer
                            )
                            if (
                                _fallback_parse_tool_call(fallback_buffer, fallback_tool_names)
                                is not None
                            ):
                                # Fully closed and parses: this is the tool call. Do
                                # not emit any of it as content, and stop watching --
                                # anything the model generates after this is not
                                # shown either, matching a real tool-call turn.
                                fallback_state = "resolved"
                            elif not still_plausible:
                                # Ruled out -- either it stopped matching, or matched
                                # for too long without a close tag. Release what was
                                # withheld and resume scanning: this one marker was a
                                # false alarm, but a later one may still be genuine.
                                fallback_state = "scanning"
                                out_parts.append(fallback_buffer)
                                fallback_buffer = ""
                            # else: still ambiguous -- keep withholding
                        # resolved: drop the chunk, nothing more is shown

                usage = getattr(event, "usage", None)
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        prompt_tokens = int(usage.prompt_tokens)
                    if getattr(usage, "completion_tokens", None) is not None:
                        gen_tokens = int(usage.completion_tokens)
                    if getattr(usage, "total_tokens", None) is not None:
                        total_tokens = int(usage.total_tokens)

                chunk_text = "".join(out_parts)
                if chunk_text and first_token_at is None:
                    # Reasoning counts: for a thinking model the prefill ends when it
                    # starts emitting, whether or not that text is the final answer.
                    first_token_at = time.perf_counter()
                if chunk_text:
                    # Stream text (and reasoning) live; tool_calls are reassembled
                    # and delivered on the done event (fragments are useless here).
                    yield {"chunk": chunk_text, "done": False}

            # Close a dangling think block (reasoning-only or stopped mid-think).
            if in_thinking_block:
                in_thinking_block = False
                yield {"chunk": "\n</think>", "done": False}

            fallback_tool_calls: list[dict[str, Any]] | None = None
            if fallback_state == "scanning" and fallback_scan:
                # Stream ended with only a held-back partial marker tail
                # outstanding (e.g. a trailing "`"); it never grew into one, so
                # it was just ordinary text and was never actually withheld.
                yield {"chunk": fallback_scan, "done": False}
            elif fallback_state == "watching" and fallback_buffer:
                # The stream ended while still ambiguous (e.g. a short response
                # that finished before hitting the buffer cap). Resolve it now,
                # one way or the other, rather than losing the withheld text.
                fallback_tool_calls = _fallback_parse_tool_call(
                    fallback_buffer, fallback_tool_names
                )
                if fallback_tool_calls is None:
                    yield {"chunk": fallback_buffer, "done": False}
            elif fallback_state == "resolved":
                fallback_tool_calls = _fallback_parse_tool_call(
                    fallback_buffer, fallback_tool_names
                )
            if fallback_tool_calls is not None:
                logger.warning(
                    "[AsyncServerClient] recovered tool_calls via fallback parsing "
                    "mid-stream (server-side parser produced none); model=%s",
                    model,
                )

            # Split the two phases at the first token. Dividing both counts by the
            # total elapsed time made each metric a function of the other's duration:
            # a long prompt dragged gen_tps down, and a long generation dragged
            # prompt_tps down, so neither number measured what its name claims and
            # the two moved together instead of independently.
            #
            # prompt_tps is prompt_tokens / time-to-first-token: the prefill is what
            # the wait before the first token consists of. gen_tps is
            # gen_tokens / (total - TTFT): decode only.
            now = time.perf_counter()
            prefill_s = max(1e-6, (first_token_at if first_token_at is not None else now) - start)
            decode_s = max(1e-6, now - (first_token_at if first_token_at is not None else start))
            gen_tps = float(gen_tokens) / decode_s if gen_tokens else 0.0
            prompt_tps = float(prompt_tokens) / prefill_s if prompt_tokens else 0.0
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
            if server_finish_reason is not None:
                done_payload["finish_reason"] = server_finish_reason
            assembled_tool_calls = _assemble_tool_calls(tool_call_acc)
            if assembled_tool_calls is not None:
                done_payload["tool_calls"] = assembled_tool_calls
                if not done_payload.get("finish_reason"):
                    done_payload["finish_reason"] = "tool_calls"
            elif fallback_tool_calls is not None:
                # Unlike the native reassembly above, this only ever fires on a
                # fully closed, successfully parsed call (see
                # _fallback_parse_tool_call), so there is no truncation case to
                # defer to -- the server's own "stop" is what a model that thinks
                # it just answered normally reports, and must not override this.
                done_payload["tool_calls"] = fallback_tool_calls
                done_payload["finish_reason"] = "tool_calls"
            if stopped:
                done_payload["stopped"] = True
            logger.info(
                "[AsyncServerClient] stream finished: finish_reason=%s (server=%s) "
                "gen_tokens=%s tool_calls=%s client_stopped=%s",
                done_payload.get("finish_reason"),
                server_finish_reason,
                gen_tokens,
                len(done_payload.get("tool_calls") or []),
                stopped,
            )
            yield done_payload
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.exception("[AsyncServerClient] generate_stream error: %s", e)
            raise _as_runtime_error(e) from e
        finally:
            # Ensure the underlying httpx2 stream is released on any exit path:
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
        payload = build_chat_payload(
            model, messages, params, stream=False, engine=self.engine, model_path=self.model_path
        )
        timeout_s = float(params.get("total_timeout", 300))

        start = time.perf_counter()
        completion = await self._create_completion(payload, timeout_s, "generate")

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
        if tool_calls is None and payload.get("tools"):
            found = _fallback_match_tool_call(content, _fallback_tool_names(payload))
            if found is not None:
                match, parsed = found
                logger.warning(
                    "[AsyncServerClient] recovered tool_calls via fallback parsing "
                    "(server-side parser produced none); model=%s",
                    model,
                )
                tool_calls = _fallback_tool_call_dict(parsed)
                # Keep whatever the model said before the wrapper (e.g. "You can use
                # the bash function...") instead of discarding it -- the streaming
                # path already forwarded that prefix live and cannot take it back,
                # so this stays consistent with what a streaming caller sees.
                content = content[: match.start()].rstrip()
                finish_reason = "tool_calls"

        elapsed = max(1e-6, time.perf_counter() - start)
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        gen_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = (
            int(getattr(usage, "total_tokens", 0) or 0) if usage else prompt_tokens + gen_tokens
        )

        # A non-stream response arrives in one piece, so there is no first token to
        # split the phases at. llama-server reports them itself; when it does, use
        # that. vLLM does not, and then both rates fall back to total elapsed — the
        # old behaviour, which is wrong in the same way as before but is the only
        # thing a single timestamp can support. The streaming path, which is what the
        # UI uses, does not have this limitation.
        timings = _server_timings(completion)
        prefill_s = max(1e-6, timings[0]) if timings else elapsed
        decode_s = max(1e-6, timings[1]) if timings else elapsed

        result: dict[str, Any] = {
            "result": content or "",
            "gen_tokens": gen_tokens,
            "gen_tps": float(gen_tokens) / decode_s if gen_tokens else 0.0,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": float(prompt_tokens) / prefill_s if prompt_tokens else 0.0,
            "total_tokens": total_tokens,
        }
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        if tool_calls is not None:
            result["tool_calls"] = tool_calls
        logger.info(
            "[AsyncServerClient] completion finished: finish_reason=%s gen_tokens=%s tool_calls=%s",
            finish_reason,
            gen_tokens,
            len(tool_calls) if tool_calls else 0,
        )
        return result
