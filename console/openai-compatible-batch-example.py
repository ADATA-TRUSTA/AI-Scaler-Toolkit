"""
Batch chat example for AST backend OpenAI-compatible endpoint.

Usage:
    uv run openai-compatible-batch-example.py
    uv run openai-compatible-batch-example.py --settings app_settings/app_settings_batch_chat.json

Workflow:
    1) Load a model first (for example with load_model_example.py).
    2) Run this script to send a batch of prompts to /v1/chat/completions.
    3) Unload model when done.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx
from openai import APIError, OpenAI

from helpers.default_backend import load_default_backend_url

DEFAULT_SETTINGS_PATH = "app_settings/app_settings_batch_chat.json"


def load_settings(settings_path: str) -> dict[str, Any]:
    """Load settings from JSON file."""
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_single_prompt(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    stream: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """Run one prompt and return result metadata."""
    print(f"Prompt: {user_prompt}")

    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=build_messages(system_prompt, user_prompt),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )

        final_usage = None
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
            if getattr(chunk, "usage", None):
                final_usage = chunk.usage
        print()

        if final_usage:
            print(f"[Stats] usage: {final_usage}")

        return {"ok": True, "usage": final_usage}

    response = client.chat.completions.create(
        model=model_name,
        messages=build_messages(system_prompt, user_prompt),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        extra_body=extra_body,
    )

    print(f"Assistant: {response.choices[0].message.content}")
    print(f"[Stats] usage: {response.usage}")
    return {"ok": True, "usage": response.usage}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run chat prompts via OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=DEFAULT_SETTINGS_PATH,
        help=f"Settings JSON path (default: {DEFAULT_SETTINGS_PATH})",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    default_backend_url = load_default_backend_url()

    backend_url = settings.get("backend_url", default_backend_url)
    api_key = settings.get("api_key", "your-api-key")
    model_name = settings.get("model", "trusta-ast/trusta-ast-default")
    system_prompt = settings.get("system_prompt", "You are a helpful assistant.")
    stream = bool(settings.get("stream", False))
    temperature = float(settings.get("temperature", 0.5))
    top_p = float(settings.get("top_p", 0.9))
    max_tokens = int(settings.get("max_tokens", 800))
    extra_body = dict(settings.get("extra_body", {}))
    prompts = settings.get("prompts", [])
    continue_on_error = bool(settings.get("continue_on_error", True))

    if not isinstance(prompts, list) or not prompts:
        raise ValueError("settings.prompts must be a non-empty list of strings")
    if not all(isinstance(item, str) and item.strip() for item in prompts):
        raise ValueError("every item in settings.prompts must be a non-empty string")

    base_url = backend_url.rstrip("/") + "/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False),
    )

    print("=" * 70)
    print("Batch Chat")
    print("=" * 70)
    print(f"Backend URL: {backend_url}")
    print(f"Model: {model_name}")
    print(f"Stream: {stream}")
    print(f"Prompt count: {len(prompts)}")

    success_count = 0
    failure_count = 0

    for idx, prompt in enumerate(prompts, start=1):
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(prompts)}]")
        print("-" * 70)
        try:
            result = run_single_prompt(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            if result.get("ok"):
                success_count += 1
        except (APIError, httpx.HTTPError, ValueError) as e:
            failure_count += 1
            print(f"[ERROR] Prompt failed: {e}")
            if not continue_on_error:
                print("Stopping batch because continue_on_error is false.")
                break

    print("\n" + "=" * 70)
    print("Batch Summary")
    print("=" * 70)
    print(f"Success: {success_count}")
    print(f"Failure: {failure_count}")


if __name__ == "__main__":
    main()
