"""
Llama Server Runner - Handles inference using llama-server (OpenAI-compatible API).
"""

import re
import time
from typing import Any

from ...settings import configure_logging

logger = configure_logging(__name__)


def bind_task_slot_from_log(engine: Any, task_id: int, slot: int | None) -> None:  # noqa: ANN401 - duck-typed engine
    """Bind a llama-server task id to the matching request and slot in the engine trace."""
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


def handle_server_log_line(engine: Any, line: str) -> None:  # noqa: ANN401 - duck-typed engine
    """Parse llama-server stderr, maintaining task/slot mapping and timing."""
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

    m_prompt = re.search(
        r"prompt\s+eval\s+time\s*=.*?/\s*(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens\s+per\s+second",
        line,
    )
    if m_prompt:
        with engine._timing_lock:
            slot = engine._active_timing_slot
            if slot is not None:
                entry = engine._slot_timings.setdefault(slot, {})
                entry["prompt_tokens"] = int(m_prompt.group(1))
                entry["prompt_tps"] = float(m_prompt.group(2))
                entry["updated_at"] = time.time()
        return

    m_eval = re.search(
        r"\beval\s+time\s*=.*?/\s*(\d+)\s+tokens.*?([0-9]+(?:\.[0-9]+)?)\s+tokens\s+per\s+second",
        line,
    )
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
