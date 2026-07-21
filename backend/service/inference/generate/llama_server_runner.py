"""
Llama Server Runner - Handles inference using llama-server (OpenAI-compatible API)
"""
import copy
import re
import time
from threading import Event as ThreadEvent
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from openai import APIError

from ...settings import configure_logging

logger = configure_logging(__name__)


def register_pending_request(engine: Any, request_id: Optional[str]) -> None:
    if not request_id:
        return
    with engine._trace_lock:
        engine._request_trace.setdefault(request_id, {"task_id": None, "slot": None, "updated_at": time.time()})
        engine._pending_request_ids.append(request_id)


def bind_task_slot_from_log(engine: Any, task_id: int, slot: Optional[int]) -> None:
    if task_id <= 0:
        return
    with engine._trace_lock:
        req_id = engine._task_to_request.get(task_id)
        if req_id is None:
            while engine._pending_request_ids:
                candidate = engine._pending_request_ids.popleft()
                trace = engine._request_trace.get(candidate)
                if trace is None:
                    continue
                if trace.get("task_id") is None:
                    req_id = candidate
                    break
        if req_id is None:
            return

        engine._task_to_request[task_id] = req_id
        trace = engine._request_trace.setdefault(req_id, {})
        trace["task_id"] = task_id
        if isinstance(slot, int) and slot >= 0:
            trace["slot"] = slot
        trace["updated_at"] = time.time()


def get_request_slot_from_trace(engine: Any, request_id: Optional[str]) -> Optional[int]:
    if not request_id:
        return None
    with engine._trace_lock:
        trace = engine._request_trace.get(request_id) or {}
        slot = trace.get("slot")
        if isinstance(slot, int):
            return slot
        return None


def finalize_request_trace(engine: Any, request_id: Optional[str], slot: Optional[int] = None) -> None:
    if not request_id:
        return
    with engine._trace_lock:
        trace = engine._request_trace.get(request_id)
        if trace is None:
            return
        if isinstance(slot, int):
            trace["slot"] = slot
        task_id = trace.get("task_id")
        if isinstance(task_id, int):
            engine._task_to_request.pop(task_id, None)
        trace["updated_at"] = time.time()


def handle_server_log_line(engine: Any, line: str) -> None:
    """解析 llama-server stderr，維護 task/slot 與 timing。"""
    if not line:
        return

    m_task_slot = re.search(r"\bslot\s+[^:]+:\s+id\s+(\d+)\s+\|\s+task\s+(-?\d+)\s+\|", line)
    if m_task_slot:
        slot_id = int(m_task_slot.group(1))
        task_id = int(m_task_slot.group(2))
        bind_task_slot_from_log(engine, task_id=task_id, slot=slot_id)

    m_slot = re.search(r"slot\s+print_timing:\s+id\s+(\d+)", line)
    if m_slot:
        slot = int(m_slot.group(1))
        with engine._timing_lock:
            engine._active_timing_slot = slot
            entry = engine._slot_timings.setdefault(slot, {})
            entry["updated_at"] = time.time()
        return

    m_prompt = re.search(r"prompt\s+eval\s+time\s*=.*?/\s*(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens\s+per\s+second", line)
    if m_prompt:
        with engine._timing_lock:
            slot = engine._active_timing_slot
            if slot is not None:
                entry = engine._slot_timings.setdefault(slot, {})
                entry["prompt_tokens"] = int(m_prompt.group(1))
                entry["prompt_tps"] = float(m_prompt.group(2))
                entry["updated_at"] = time.time()
        return

    m_eval = re.search(r"\beval\s+time\s*=.*?/\s*(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens\s+per\s+second", line)
    if m_eval:
        with engine._timing_lock:
            slot = engine._active_timing_slot
            if slot is not None:
                entry = engine._slot_timings.setdefault(slot, {})
                entry["gen_tokens"] = int(m_eval.group(1))
                entry["gen_tps"] = float(m_eval.group(2))
                entry["updated_at"] = time.time()
        return

    m_total = re.search(r"total\s+time\s*=.*?/\s*(\d+)\s+tokens", line)
    if m_total:
        with engine._timing_lock:
            slot = engine._active_timing_slot
            if slot is not None:
                entry = engine._slot_timings.setdefault(slot, {})
                entry["total_tokens"] = int(m_total.group(1))
                entry["updated_at"] = time.time()
        return


def _normalize_image_urls(image_values: Any) -> List[str]:
    if not isinstance(image_values, list):
        return []

    normalized: List[str] = []
    for item in image_values:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return normalized


def _normalize_content_parts(parts: Any) -> List[Dict[str, Any]]:
    if not isinstance(parts, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue

        ptype = str(part.get("type", "")).strip().lower()
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                normalized.append({"type": "text", "text": text})
        elif ptype == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url

            if isinstance(url, str) and url.strip():
                normalized.append({
                    "type": "image_url",
                    "image_url": {"url": url.strip()},
                })

    return normalized


def _merge_text_and_images_to_content(
    text: str,
    image_urls: Optional[List[str]],
    existing_parts: Optional[List[Dict[str, Any]]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    images = _normalize_image_urls(image_urls)

    if existing_parts:
        parts = list(existing_parts)
        for url in images:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    if not images:
        return text

    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for url in images:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in str(model_name or "").lower()

def _apply_thinking_tag(
    messages: List[Dict[str, Any]],
    enable_thinking: Optional[bool],
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """對 Qwen 模型最後一則 user 訊息附加 /think 或 /no_think。"""
    if enable_thinking is None or not _is_qwen_model(model_name) or not messages:
        return messages

    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break

    if last_user_idx < 0:
        return messages

    tag = " /think" if enable_thinking else " /no_think"
    msg = messages[last_user_idx]
    content = msg.get("content")

    if isinstance(content, str):
        stripped_content = content.rstrip()
        if not (stripped_content.endswith("/think") or stripped_content.endswith("/no_think")):
            msg["content"] = content + tag
            logger.info(f"[LlamaServer] Applied thinking tag '{tag}' to last user message")
        return messages

    if isinstance(content, list):
        last_text_idx = -1
        for idx in range(len(content) - 1, -1, -1):
            part = content[idx]
            if isinstance(part, dict) and str(part.get("type", "")).strip().lower() == "text":
                last_text_idx = idx
                break

        if last_text_idx >= 0:
            part = content[last_text_idx]
            text_value = str(part.get("text") or "")
            stripped_text = text_value.rstrip()
            if not (stripped_text.endswith("/think") or stripped_text.endswith("/no_think")):
                part["text"] = text_value + tag
                logger.info(f"[LlamaServer] Applied thinking tag '{tag}' to last user text part")
        else:
            content.append({"type": "text", "text": tag.strip()})
            logger.info(f"[LlamaServer] Added thinking tag '{tag}' as new user text part")

    return messages


def _build_messages(request: Dict[str, Any], model_name: str = "") -> List[Dict[str, Any]]:
    prompt = request.get("prompt", None)
    messages = request.get("messages", None)

    if messages is None:
        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, str) and prompt:
            messages = [{"role": "user", "content": prompt}]

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages/prompt is required")

    normalized: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip() or "user"
        raw_content = m.get("content", "")

        tool_calls = m.get("tool_calls")
        tool_call_id = m.get("tool_call_id")
        
        text_content = ""
        content_parts: Optional[List[Dict[str, Any]]] = None

        if isinstance(raw_content, list):
            content_parts = _normalize_content_parts(raw_content)
        else:
            text_content = str(raw_content or "")

        msg_obj = {"role": role}

        if tool_calls is not None:
            msg_obj["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg_obj["tool_call_id"] = tool_call_id

        if content_parts:
            msg_obj["content"] = content_parts
        else:
            # Only put text_content if it's not empty OR if there are no tool_calls
            if text_content or (not tool_calls and not tool_call_id):
                msg_obj["content"] = text_content

        if not msg_obj.get("content") and not tool_calls and not tool_call_id:
            continue
            
        normalized.append(msg_obj)

    if not normalized:
        raise ValueError("messages content is empty")

    params = request.get("params") or {}
    request_images = _normalize_image_urls(params.get("images"))
    if request_images:
        last_user_idx = -1
        for idx in range(len(normalized) - 1, -1, -1):
            if normalized[idx].get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx >= 0:
            msg = normalized[last_user_idx]
            msg["content"] = _merge_text_and_images_to_content(
                text=str(msg.get("content") or ""),
                image_urls=request_images,
                existing_parts=msg.get("content") if isinstance(msg.get("content"), list) else None,
            )
        else:
            logger.warning("[LlamaServer] images provided but no user message found; images are ignored")

    normalized = _apply_thinking_tag(
        normalized,
        (params.get("enable_thinking") if isinstance(params.get("enable_thinking"), bool) else None),
        model_name=model_name,
    )

    return normalized


def _build_payload(engine: Any, request: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    if engine.config is None:
        raise RuntimeError("Engine config not initialized")

    params = request.get("params", {}) or {}
    model_name = engine._resolve_model_name()
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": _build_messages(request, model_name=model_name),
        "max_tokens": params.get("max_new_tokens", 512),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
    }

    extra_body: Dict[str, Any] = {
        "repeat_penalty": params.get("repetition_penalty", 1.1),
    }

    enable_thinking = params.get("enable_thinking") if isinstance(params.get("enable_thinking"), bool) else None
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": enable_thinking,
        }

    top_k = params.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = top_k

    request_extra_body = request.get("extra_body")
    if isinstance(request_extra_body, dict):
        extra_body.update(request_extra_body)

    if stream:
        payload["stream_options"] = {"include_usage": True}
        
    tools = params.get("tools")
    if tools:
        payload["tools"] = tools
        tool_choice = params.get("tool_choice")
        if tool_choice:
            payload["tool_choice"] = tool_choice

    if extra_body:
        payload["extra_body"] = extra_body

    return payload


def _dump_openai_object(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=False)
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _dump_openai_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_openai_object(item) for item in value]
    return value


def _chat_completion(engine: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    if engine.client is None:
        raise RuntimeError("OpenAI client not initialized")

    register_pending_request(engine, request.get("request_id"))
    payload = _build_payload(engine, request, stream=False)
    timeout_sec = float(request.get("params", {}).get("total_timeout", engine.timeout_sec))
    completion = engine.client.chat.completions.create(
        timeout=timeout_sec,
        stream=False,
        **payload,
    )
    return _dump_openai_object(completion) or {}


def _extract_slot_from_event(event: Dict[str, Any]) -> Optional[int]:
    if not isinstance(event, dict):
        return None

    candidate_keys = ["slot_id", "slot", "slotId", "id_slot"]
    for key in candidate_keys:
        value = event.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for key in candidate_keys:
                value = first.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)

    def _walk(obj: Any) -> Optional[int]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if key_l in {"slot", "slot_id", "slotid", "id_slot"}:
                    if isinstance(value, int):
                        return value
                    if isinstance(value, str) and value.isdigit():
                        return int(value)
                nested = _walk(value)
                if nested is not None:
                    return nested
        elif isinstance(obj, list):
            for item in obj:
                nested = _walk(item)
                if nested is not None:
                    return nested
        return None

    return _walk(event)


def _get_timing_for_slot(engine: Any, slot: Optional[int]) -> Optional[Dict[str, Any]]:
    if slot is None:
        return None
    with engine._timing_lock:
        entry = engine._slot_timings.get(slot)
        return dict(entry) if entry else None


def _infer_slot_by_timing(
    engine: Any,
    gen_tokens: Optional[int],
    prompt_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    min_updated_at: Optional[float] = None,
    recent_seconds: float = 10.0,
) -> Optional[int]:
    now = time.time()
    with engine._timing_lock:
        candidates = []
        for slot, entry in engine._slot_timings.items():
            updated_at = float(entry.get("updated_at", 0))
            if (now - updated_at) > recent_seconds:
                continue
            if isinstance(min_updated_at, (int, float)) and updated_at < float(min_updated_at):
                continue

            score = 0
            if isinstance(total_tokens, int) and entry.get("total_tokens") == total_tokens:
                score += 4
            if isinstance(gen_tokens, int) and entry.get("gen_tokens") == gen_tokens:
                score += 2
            if isinstance(prompt_tokens, int) and entry.get("prompt_tokens") == prompt_tokens:
                score += 1

            if score > 0:
                candidates.append((slot, score, updated_at))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]


def _resolve_slot_and_timing(
    engine: Any,
    slot: Optional[int],
    gen_tokens: Optional[int],
    prompt_tokens: Optional[int],
    total_tokens: Optional[int],
    min_updated_at: Optional[float] = None,
    wait_timeout_sec: float = 0.8,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    deadline = time.time() + max(0.0, wait_timeout_sec)
    resolved_slot = slot

    def _fresh_timing(s: Optional[int]) -> Optional[Dict[str, Any]]:
        t = _get_timing_for_slot(engine, s)
        if not t:
            return None
        if isinstance(min_updated_at, (int, float)):
            updated_at = float(t.get("updated_at", 0.0))
            if updated_at < float(min_updated_at):
                return None
        return t

    while time.time() < deadline:
        if resolved_slot is None:
            resolved_slot = _infer_slot_by_timing(
                engine,
                gen_tokens=gen_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                min_updated_at=min_updated_at,
                recent_seconds=10.0,
            )

        timing = _fresh_timing(resolved_slot)
        if timing and (
            isinstance(timing.get("prompt_tps"), (int, float))
            or isinstance(timing.get("gen_tps"), (int, float))
        ):
            return resolved_slot, timing
        time.sleep(0.03)

    return resolved_slot, _fresh_timing(resolved_slot)


def _extract_text_from_choice(choice: Dict[str, Any], include_reasoning: bool = False) -> str:
    """兼容不同 OpenAI-compatible 回應格式抽取文本。"""
    if not isinstance(choice, dict):
        return ""

    def _normalize_reasoning(text: str) -> str:
        text = text or ""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _wrap_reasoning(text: str) -> str:
        normalized = _normalize_reasoning(text)
        if not normalized:
            return ""
        if "<think>" in normalized or "</think>" in normalized:
            return normalized
        return f"<think>{normalized}</think>"

    delta = choice.get("delta")
    if isinstance(delta, dict):
        reasoning_content = delta.get("reasoning_content")
        if include_reasoning and isinstance(reasoning_content, str) and reasoning_content:
            return _wrap_reasoning(reasoning_content)
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content

    message = choice.get("message")
    if isinstance(message, dict):
        reasoning_content = message.get("reasoning_content")
        if include_reasoning and isinstance(reasoning_content, str) and reasoning_content:
            content = message.get("content")
            if isinstance(content, str) and content:
                return f"{_wrap_reasoning(reasoning_content)}{content}"
            return _wrap_reasoning(reasoning_content)
        content = message.get("content")
        if isinstance(content, str) and content:
            return content

    text = choice.get("text")
    if isinstance(text, str) and text:
        return text

    return ""


def _clone_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in tool_calls:
        if isinstance(item, dict):
            normalized.append(copy.deepcopy(item))
    return normalized


def _merge_tool_call_delta(
    accumulator: List[Dict[str, Any]],
    delta_tool_calls: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(delta_tool_calls, list):
        return accumulator

    for raw_item in delta_tool_calls:
        if not isinstance(raw_item, dict):
            continue

        idx = raw_item.get("index")
        target: Optional[Dict[str, Any]] = None

        if isinstance(idx, int) and idx >= 0:
            while len(accumulator) <= idx:
                accumulator.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            target = accumulator[idx]
        else:
            raw_id = raw_item.get("id")
            if isinstance(raw_id, str) and raw_id:
                for existing in accumulator:
                    if existing.get("id") == raw_id:
                        target = existing
                        break
            if target is None:
                target = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                accumulator.append(target)

        raw_id = raw_item.get("id")
        if isinstance(raw_id, str) and raw_id:
            target["id"] = raw_id

        raw_type = raw_item.get("type")
        if isinstance(raw_type, str) and raw_type:
            target["type"] = raw_type

        raw_function = raw_item.get("function")
        if isinstance(raw_function, dict):
            function_target = target.setdefault("function", {})
            if not isinstance(function_target, dict):
                function_target = {}
                target["function"] = function_target

            raw_name = raw_function.get("name")
            if isinstance(raw_name, str) and raw_name:
                function_target["name"] = raw_name

            raw_arguments = raw_function.get("arguments")
            if isinstance(raw_arguments, str):
                function_target["arguments"] = f"{function_target.get('arguments', '')}{raw_arguments}"
            elif raw_arguments is not None:
                function_target["arguments"] = raw_arguments

    return accumulator


def _extract_tool_calls_from_choice(choice: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(choice, dict):
        return []

    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
        return _clone_tool_calls(delta.get("tool_calls"))

    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        return _clone_tool_calls(message.get("tool_calls"))

    return []


def _thinking_enabled(request: Dict[str, Any]) -> bool:
    params = request.get("params") or {}
    return bool("enable_thinking" in params and params.get("enable_thinking") is not None)


def _should_fallback_reasoning_output(model_name: str) -> bool:
    """Qwen 若仍回 reasoning_content，至少不要讓上層變成空串流。"""
    return _is_qwen_model(model_name)


def _strip_think_blocks(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _chat_stream(engine: Any, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if engine.config is None:
        raise RuntimeError("Engine config not initialized")

    payload = _build_payload(engine, request, stream=True)
    timeout_sec = float(request.get("params", {}).get("total_timeout", engine.timeout_sec))
    request_stop_flag = request.get("request_stop_flag")
    register_pending_request(engine, request.get("request_id"))

    if engine.client is None:
        raise RuntimeError("OpenAI client not initialized")

    stream = engine.client.chat.completions.create(
        timeout=timeout_sec,
        stream=True,
        **payload,
    )

    try:
        for event in stream:
            if isinstance(request_stop_flag, ThreadEvent) and request_stop_flag.is_set():
                logger.info("[LlamaServer] request stop flag received, ending stream")
                break
            if engine.stop_generation_flag.is_set():
                logger.info("[LlamaServer] stop flag received, ending stream")
                break

            data = _dump_openai_object(event)
            if not isinstance(data, dict):
                logger.debug("[LlamaServer] Skip non-dict stream event: %s", type(data).__name__)
                continue

            yield data
    finally:
        close_fn = getattr(stream, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def handle_llama_server_generate(request: Dict[str, Any], engine: Any):
    request_id = request.get("request_id")
    if engine.config is None:
        engine.data_queue.put({
            "type": "error",
            "request_id": request_id,
            "error": "Model not loaded"
        })
        return

    started_at = time.time()
    full_text = ""
    usage: Dict[str, Any] = {}
    slot: Optional[int] = None
    first_token_time: Optional[float] = None
    request_stop_flag = request.get("request_stop_flag")
    include_reasoning = _thinking_enabled(request)
    reasoning_fallback = _should_fallback_reasoning_output(engine._resolve_model_name())
    in_thinking_block = False
    tool_calls_acc: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None

    try:
        completion = _chat_completion(engine, request)

        event_slot = _extract_slot_from_event(completion)
        if event_slot is not None:
            slot = event_slot
        elif slot is None:
            slot = get_request_slot_from_trace(engine, request_id)

        choices = completion.get("choices") or []
        if choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            choice_finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
            if isinstance(choice_finish_reason, str) and choice_finish_reason:
                finish_reason = choice_finish_reason

            tool_calls_acc = _extract_tool_calls_from_choice(first_choice)
            full_text = _extract_text_from_choice(
                first_choice,
                include_reasoning=(include_reasoning or reasoning_fallback),
            )
            if full_text:
                first_token_time = time.time()

        if isinstance(completion.get("usage"), dict):
            usage = completion.get("usage")

        if include_reasoning and in_thinking_block:
            full_text += "</think>"
            in_thinking_block = False

        elapsed = max(time.time() - started_at, 1e-6)
        gen_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")

        slot, timing = _resolve_slot_and_timing(
            engine,
            slot=slot,
            gen_tokens=gen_tokens if isinstance(gen_tokens, int) else None,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            min_updated_at=started_at,
        )

        prompt_seconds = None
        gen_seconds = None
        if first_token_time is not None and first_token_time > started_at:
            prompt_seconds = max(first_token_time - started_at, 1e-6)
            gen_seconds = max((time.time() - first_token_time), 1e-6)

        gen_tps = None
        prompt_tps = None
        if isinstance(gen_tokens, (int, float)):
            if gen_seconds is not None:
                gen_tps = gen_tokens / gen_seconds
            else:
                gen_tps = gen_tokens / elapsed
        if isinstance(prompt_tokens, (int, float)) and prompt_seconds is not None:
            prompt_tps = prompt_tokens / prompt_seconds

        if timing:
            if isinstance(timing.get("prompt_tokens"), int):
                prompt_tokens = timing.get("prompt_tokens")
            if isinstance(timing.get("gen_tokens"), int):
                gen_tokens = timing.get("gen_tokens")
            if isinstance(timing.get("total_tokens"), int):
                total_tokens = timing.get("total_tokens")
            if isinstance(timing.get("prompt_tps"), (int, float)):
                prompt_tps = float(timing.get("prompt_tps"))
            if isinstance(timing.get("gen_tps"), (int, float)):
                gen_tps = float(timing.get("gen_tps"))
            if isinstance(prompt_tokens, int) and isinstance(gen_tokens, int):
                total_tokens = prompt_tokens + gen_tokens

        if not finish_reason:
            finish_reason = "tool_calls" if tool_calls_acc else "stop"

        engine.data_queue.put({
            "type": "result",
            "request_id": request_id,
            "result": full_text,
            "tool_calls": tool_calls_acc or None,
            "finish_reason": finish_reason,
            "slot": slot,
            "stopped": (
                (isinstance(request_stop_flag, ThreadEvent) and request_stop_flag.is_set())
                or engine.stop_generation_flag.is_set()
            ),
            "total_tokens": total_tokens,
            "gen_tokens": gen_tokens,
            "gen_tps": gen_tps,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tps,
        })
        finalize_request_trace(engine, request_id, slot=slot)
    except (APIError, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[LlamaServer] Generation error: {e}")
        finalize_request_trace(engine, request_id, slot=slot)
        error_payload = (
            engine.build_runtime_error_payload(e)
            if hasattr(engine, "build_runtime_error_payload")
            else {"error": str(e)}
        )
        if hasattr(engine, "report_runtime_error"):
            engine.report_runtime_error(error_payload)
        engine.data_queue.put({
            "type": "error",
            "request_id": request_id,
            **error_payload,
        })


def handle_llama_server_stream(request: Dict[str, Any], engine: Any):
    request_id = request.get("request_id")
    if engine.config is None:
        engine.data_queue.put({
            "type": "error",
            "request_id": request_id,
            "error": "Model not loaded"
        })
        return

    usage: Dict[str, Any] = {}
    slot: Optional[int] = None
    started_at = time.time()
    first_token_time: Optional[float] = None
    request_stop_flag = request.get("request_stop_flag")
    slot_announced = False
    in_thinking_block = False
    include_reasoning = _thinking_enabled(request)
    reasoning_fallback = _should_fallback_reasoning_output(engine._resolve_model_name())
    tool_calls_acc: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None

    try:
        for event in _chat_stream(engine, request):
            event_slot = _extract_slot_from_event(event)
            if event_slot is not None:
                slot = event_slot
            elif slot is None:
                slot = get_request_slot_from_trace(engine, request_id)

            if slot is not None and not slot_announced:
                engine.data_queue.put({
                    "type": "stream_chunk",
                    "request_id": request_id,
                    "slot": slot,
                    "chunk": "",
                    "done": False,
                    "meta": "slot",
                })
                slot_announced = True

            choices = event.get("choices") or []
            out_chunks: List[str] = []
            if choices:
                first_choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
                choice_finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
                if isinstance(choice_finish_reason, str) and choice_finish_reason:
                    finish_reason = choice_finish_reason

                delta_tool_calls = _extract_tool_calls_from_choice(first_choice)
                if delta_tool_calls:
                    if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
                        tool_calls_acc = _merge_tool_call_delta(tool_calls_acc, delta_tool_calls)
                    else:
                        tool_calls_acc = delta_tool_calls
                    engine.data_queue.put({
                        "type": "stream_chunk",
                        "request_id": request_id,
                        "slot": slot,
                        "chunk": "",
                        "done": False,
                        "tool_calls": delta_tool_calls,
                    })

                if isinstance(delta, dict):
                    reasoning_chunk = delta.get("reasoning_content")
                    content_chunk = delta.get("content")

                    if include_reasoning and isinstance(reasoning_chunk, str) and reasoning_chunk:
                        if reasoning_chunk:
                            if not in_thinking_block:
                                out_chunks.append("<think>\n")
                                in_thinking_block = True
                            out_chunks.append(reasoning_chunk)
                    elif reasoning_fallback and isinstance(reasoning_chunk, str) and reasoning_chunk and not content_chunk:
                        if not in_thinking_block:
                            out_chunks.append("<think>\n")
                            in_thinking_block = True
                        out_chunks.append(reasoning_chunk)

                    if isinstance(content_chunk, str) and content_chunk:
                        if in_thinking_block:
                            out_chunks.append("\n</think>\n")
                            in_thinking_block = False
                        out_chunks.append(content_chunk)
                else:
                    fallback_chunk = _extract_text_from_choice(
                        first_choice,
                        include_reasoning=(include_reasoning or reasoning_fallback),
                    )
                    if fallback_chunk:
                        out_chunks.append(fallback_chunk)

            for chunk in out_chunks:
                if not chunk:
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                engine.data_queue.put({
                    "type": "stream_chunk",
                    "request_id": request_id,
                    "slot": slot,
                    "chunk": chunk,
                    "done": False,
                })

            if isinstance(event.get("usage"), dict):
                usage = event.get("usage")

        if include_reasoning and in_thinking_block:
            engine.data_queue.put({
                "type": "stream_chunk",
                "request_id": request_id,
                "slot": slot,
                "chunk": "</think>",
                "done": False,
            })
            in_thinking_block = False

        elapsed = max(time.time() - started_at, 1e-6)
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")

        slot, timing = _resolve_slot_and_timing(
            engine,
            slot=slot,
            gen_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            min_updated_at=started_at,
        )

        prompt_seconds = None
        gen_seconds = None
        if first_token_time is not None and first_token_time > started_at:
            prompt_seconds = max(first_token_time - started_at, 1e-6)
            gen_seconds = max((time.time() - first_token_time), 1e-6)

        gen_tps = None
        prompt_tps = None
        if isinstance(completion_tokens, (int, float)):
            if gen_seconds is not None:
                gen_tps = completion_tokens / gen_seconds
            else:
                gen_tps = completion_tokens / elapsed
        if isinstance(prompt_tokens, (int, float)) and prompt_seconds is not None:
            prompt_tps = prompt_tokens / prompt_seconds

        if timing:
            if isinstance(timing.get("prompt_tokens"), int):
                prompt_tokens = timing.get("prompt_tokens")
            if isinstance(timing.get("gen_tokens"), int):
                completion_tokens = timing.get("gen_tokens")
            if isinstance(timing.get("total_tokens"), int):
                total_tokens = timing.get("total_tokens")
            if isinstance(timing.get("prompt_tps"), (int, float)):
                prompt_tps = float(timing.get("prompt_tps"))
            if isinstance(timing.get("gen_tps"), (int, float)):
                gen_tps = float(timing.get("gen_tps"))

        if not finish_reason:
            finish_reason = "tool_calls" if tool_calls_acc else "stop"

        engine.data_queue.put({
            "type": "stream_chunk",
            "request_id": request_id,
            "slot": slot,
            "chunk": "",
            "done": True,
            "tool_calls": tool_calls_acc or None,
            "finish_reason": finish_reason,
            "stopped": (
                (isinstance(request_stop_flag, ThreadEvent) and request_stop_flag.is_set())
                or engine.stop_generation_flag.is_set()
            ),
            "total_tokens": total_tokens,
            "gen_tokens": completion_tokens,
            "gen_tps": gen_tps,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tps,
        })
        finalize_request_trace(engine, request_id, slot=slot)
    except Exception as e:
        logger.error(f"[LlamaServer] Streaming error: {e}")
        finalize_request_trace(engine, request_id, slot=slot)
        error_payload = (
            engine.build_runtime_error_payload(e)
            if hasattr(engine, "build_runtime_error_payload")
            else {"error": str(e)}
        )
        if hasattr(engine, "report_runtime_error"):
            engine.report_runtime_error(error_payload)
        engine.data_queue.put({
            "type": "error",
            "request_id": request_id,
            **error_payload,
        })
