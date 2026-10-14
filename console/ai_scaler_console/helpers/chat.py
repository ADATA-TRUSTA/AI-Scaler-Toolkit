"""
Shared chat-settings loading and prompt execution for the examples.

The chat examples all read the same shape of JSON settings file (see
``configs/chat/*.json``) and all run prompts the same way. This module holds
that logic once so the example scripts stay short.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

DEFAULT_MODEL_NAME = "trusta-ast/trusta-ast-default"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class ChatSettings:
    """
    Normalised chat settings loaded from a ``configs/chat/*.json`` file.

    ``backend_url`` and ``api_key`` stay optional: ``None`` means "fall back to
    the .env value", which is what
    :func:`~ai_scaler_console.openai_client.create_client` already does.
    """

    model_name: str = DEFAULT_MODEL_NAME
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    stream: bool = False
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 800
    extra_body: dict[str, Any] | None = None
    prompts: tuple[str, ...] = ()
    continue_on_error: bool = True
    backend_url: str | None = None
    api_key: str | None = None


def load_settings_file(settings_path: str | Path) -> dict[str, Any]:
    """Load a settings JSON file as a raw dict."""
    with open(settings_path, encoding="utf-8") as f:
        return json.load(f)


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_chat_settings(raw: dict[str, Any]) -> ChatSettings:
    """
    Normalise a raw settings dict, applying defaults and validating prompts.

    Raises:
        TypeError: if ``prompts`` is present but is not a list.
        ValueError: if any prompt is not a non-empty string.
    """
    prompts = raw.get("prompts", [])
    if not isinstance(prompts, list):
        raise TypeError("settings.prompts must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in prompts):
        raise ValueError("every item in settings.prompts must be a non-empty string")

    return ChatSettings(
        model_name=str(raw.get("model", DEFAULT_MODEL_NAME)),
        system_prompt=str(raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        stream=bool(raw.get("stream", False)),
        temperature=float(raw.get("temperature", 0.5)),
        top_p=float(raw.get("top_p", 0.9)),
        max_tokens=int(raw.get("max_tokens", 800)),
        extra_body=dict(raw.get("extra_body", {})),
        prompts=tuple(prompts),
        continue_on_error=bool(raw.get("continue_on_error", True)),
        backend_url=raw.get("backend_url") or None,
        api_key=raw.get("api_key") or None,
    )


def load_chat_settings(settings_path: str | Path) -> ChatSettings:
    """Load and normalise a chat settings JSON file."""
    return parse_chat_settings(load_settings_file(settings_path))


def require_prompts(settings: ChatSettings) -> tuple[str, ...]:
    """
    Return the prompt list, raising if it is empty.

    Raises:
        ValueError: if the settings file declared no prompts.
    """
    if not settings.prompts:
        raise ValueError("settings.prompts must be a non-empty list of strings")
    return settings.prompts


def select_prompt(settings: ChatSettings, prompt_index: int) -> str:
    """
    Return one prompt by index.

    Raises:
        ValueError: if the settings file declared no prompts.
        IndexError: if ``prompt_index`` is out of range.
    """
    prompts = require_prompts(settings)
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(f"prompt index {prompt_index} is out of range for {len(prompts)} prompts")
    return prompts[prompt_index]


def run_prompt(client: OpenAI, settings: ChatSettings, user_prompt: str) -> None:
    """
    Run one prompt and print terminal-friendly logs.

    In streaming mode this also reports time-to-first-token, which is the number
    worth watching when several prompts run concurrently.
    """
    print(f"Prompt: {user_prompt}")

    common = {
        "model": settings.model_name,
        "messages": cast(Any, build_messages(settings.system_prompt, user_prompt)),
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "extra_body": settings.extra_body or {},
    }

    if not settings.stream:
        response = client.chat.completions.create(**common, stream=False)
        print(f"Assistant: {response.choices[0].message.content}")
        print(f"[Stats] usage: {response.usage}")
        return

    request_started_at = time.perf_counter()
    stream = client.chat.completions.create(
        **common,
        stream=True,
        stream_options={"include_usage": True},
    )

    final_usage = None
    ttft_seconds: float | None = None
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft_seconds is None:
                ttft_seconds = time.perf_counter() - request_started_at
            print(chunk.choices[0].delta.content, end="", flush=True)
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage
    print()

    if ttft_seconds is not None:
        print(f"[Stats] TTFT: {ttft_seconds:.3f}s")
    else:
        print("[Stats] TTFT: N/A (no streamed token received)")
    if final_usage:
        print(f"[Stats] usage: {final_usage}")
