"""
Batch image chat example for AST backend OpenAI-compatible endpoint.

Usage:
    uv run openai-compatible-image-batch-example.py
    uv run openai-compatible-image-batch-example.py --settings app_settings/app_settings_batch_chat_image.json

Workflow:
    1) Load a multimodal model first (for example with inference-config-vllm.json
       and vllm_mm_image_limit > 0).
    2) Run this script to send multiple image chat tasks to /v1/chat/completions.
    3) Unload model when done.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import httpx
from openai import APIError, OpenAI

from helpers.default_backend import load_default_backend_url

DEFAULT_SETTINGS_PATH = "app_settings/app_settings_batch_chat_image.json"


def load_settings(settings_path: str) -> dict[str, Any]:
    """Load settings from JSON file."""
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_to_base64(image_path: str) -> str:
    """Encode local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Map file extension to image MIME type."""
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix, "image/png")


def to_image_url(source: str) -> str:
    """Convert image source into URL or data URI."""
    if source.startswith(("http://", "https://", "data:")):
        return source

    image_path = Path(source)
    if not image_path.exists() or not image_path.is_file():
        raise ValueError(f"Image file not found: {source}")

    base64_image = encode_image_to_base64(source)
    mime_type = get_image_mime_type(source)
    return f"data:{mime_type};base64,{base64_image}"


def build_user_content(prompt: str, images: list[str]) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal user content."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for source in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": to_image_url(source)},
            }
        )
    return content


def run_single_task(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    prompt: str,
    images: list[str],
    stream: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """Run one multimodal prompt and return metadata."""
    print(f"Prompt: {prompt}")
    print(f"Images ({len(images)}): {images}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(prompt, images)},
    ]

    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
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
        messages=messages,
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
        description="Batch run multimodal chat tasks via OpenAI-compatible endpoint.",
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
    system_prompt = settings.get(
        "system_prompt",
        "You are a helpful assistant that can analyze images.",
    )
    stream = bool(settings.get("stream", False))
    temperature = float(settings.get("temperature", 0.5))
    top_p = float(settings.get("top_p", 0.9))
    max_tokens = int(settings.get("max_tokens", 1024))
    extra_body = dict(settings.get("extra_body", {}))
    tasks = settings.get("tasks", [])
    continue_on_error = bool(settings.get("continue_on_error", True))

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("settings.tasks must be a non-empty list")

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"settings.tasks[{index - 1}] must be an object")
        prompt = task.get("prompt")
        images = task.get("images")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"settings.tasks[{index - 1}].prompt must be a non-empty string"
            )
        if not isinstance(images, list) or not images:
            raise ValueError(
                f"settings.tasks[{index - 1}].images must be a non-empty list"
            )
        if not all(isinstance(item, str) and item.strip() for item in images):
            raise ValueError(
                f"settings.tasks[{index - 1}].images must contain non-empty strings"
            )

    base_url = backend_url.rstrip("/") + "/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False),
    )

    print("=" * 70)
    print("Batch Image Chat")
    print("=" * 70)
    print(f"Backend URL: {backend_url}")
    print(f"Model: {model_name}")
    print(f"Stream: {stream}")
    print(f"Task count: {len(tasks)}")

    success_count = 0
    failure_count = 0

    for idx, task in enumerate(tasks, start=1):
        prompt = task["prompt"]
        images = task["images"]

        print("\n" + "-" * 70)
        print(f"[{idx}/{len(tasks)}]")
        print("-" * 70)

        try:
            result = run_single_task(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=prompt,
                images=images,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            if result.get("ok"):
                success_count += 1
        except (APIError, OSError, httpx.HTTPError, ValueError) as e:
            failure_count += 1
            print(f"[ERROR] Task failed: {e}")
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
