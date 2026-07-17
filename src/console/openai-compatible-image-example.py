"""
Example of calling AST backend `/v1/chat/completions` (multimodal)
using the OpenAI Python SDK.

This matches the `runChatCompletions` payload format in Trusta-AST-Frontend
when `image_url` content parts are included, and is the standard no-UI
integration path for multimodal models.

Supported image sources:
    1) Local image: automatically converted to Base64 Data URI
    2) Remote image: pass HTTP/HTTPS URL directly
    3) Multiple images can be mixed in one message

Workflow:
    1) Load a multimodal-capable model first via `load_model_example.py`, e.g.:
         - transformers + Gemma 3 / Qwen-VL family
         - vLLM with `vllm_mm_image_limit` configured (see inference-config-vllm.json)
    2) Run this file to send image chat requests to `<backend_url>/v1/chat/completions`.
    3) Run `unload_model_example.py` when done to release resources.

Note: when using vLLM, `vllm_mm_image_limit` must be set at load time,
otherwise image input will not be enabled by the backend.
"""

import base64
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
import httpx

from helpers.default_backend import load_default_backend_url

# ---- Backend settings ----
BACKEND_URL = load_default_backend_url()
BASE_URL = BACKEND_URL.rstrip("/") + "/v1"
API_KEY = "your-api-key"
MODEL_NAME = "trusta-ast/trusta-ast-default"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False),
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image as Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from file extension."""
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


def build_image_data_uri(image_path: str) -> str:
    """Convert a local image to Data URI."""
    base64_image = encode_image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)
    return f"data:{mime_type};base64,{base64_image}"


def _default_extra_body() -> dict:
    """Common AST extra_body example (all optional, shown for reference)."""
    return {
        "top_k": 50,
        "repetition_penalty": 1.1,
        "total_timeout": 300,
        "enable_thinking": False,
    }


# ----------------------------------------------------------------------------
# Chat: local image (streaming)
# ----------------------------------------------------------------------------
def chat_with_local_image(image_path: str, user_message: str) -> None:
    """Send a local image and stream the response."""
    image_data_uri = build_image_data_uri(image_path)

    content = [
        {"type": "text", "text": user_message},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_default_extra_body(),
    )

    print(f"Image: {image_path}")
    print(f"Prompt: {user_message}")
    print("Answer: ", end="", flush=True)

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ----------------------------------------------------------------------------
# Chat: image URL (streaming)
# ----------------------------------------------------------------------------
def chat_with_url_image(image_url: str, user_message: str) -> None:
    """Chat with an HTTP/HTTPS image URL."""
    content = [
        {"type": "text", "text": user_message},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_default_extra_body(),
    )

    print(f"Image URL: {image_url}")
    print(f"Prompt: {user_message}")
    print("Answer: ", end="", flush=True)

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ----------------------------------------------------------------------------
# Chat: multiple images (mixed local paths and URLs, streaming)
# ----------------------------------------------------------------------------
def chat_with_multiple_images(image_sources: list[str], user_message: str) -> None:
    """Chat with multiple images in one request (local paths and URLs can be mixed)."""
    content: list[dict[str, Any]] = [{"type": "text", "text": user_message}]

    for source in image_sources:
        if source.startswith(("http://", "https://")):
            content.append({"type": "image_url", "image_url": {"url": source}})
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": build_image_data_uri(source)},
                }
            )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=cast(
            Any,
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze images.",
                },
                {"role": "user", "content": content},
            ],
        ),
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_default_extra_body(),
    )

    print(f"Images ({len(image_sources)}): {image_sources}")
    print(f"Prompt: {user_message}")
    print("Answer: ", end="", flush=True)

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ----------------------------------------------------------------------------
# Chat: local image (non-stream)
# ----------------------------------------------------------------------------
def chat_with_image_non_stream(image_path: str, user_message: str) -> None:
    """Get full response and usage stats in non-stream mode."""
    image_data_uri = build_image_data_uri(image_path)

    content = [
        {"type": "text", "text": user_message},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stream=False,
        extra_body=_default_extra_body(),
    )

    print(f"Image: {image_path}")
    print(f"Prompt: {user_message}")
    print(f"Answer: {response.choices[0].message.content}")
    print(f"\n[Stats] usage: {response.usage}")


if __name__ == "__main__":
    # ===== Example 1: local image (streaming) =====
    chat_with_local_image(
        image_path="./your-image.jpg",
        user_message="Please describe the content of this image.",
    )

    # ===== Example 2: remote image URL =====
    # chat_with_url_image(
    #     image_url="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    #     user_message="What is in this image?",
    # )

    # ===== Example 3: multiple images (mixed local and URL) =====
    # chat_with_multiple_images(
    #     image_sources=[
    #         "./your-image.jpg",
    #         "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    #     ],
    #     user_message="Please compare the differences between these images.",
    # )

    # ===== Example 4: non-stream mode =====
    # chat_with_image_non_stream(
    #     image_path="./your-image.jpg",
    #     user_message="Please describe the content of this image.",
    # )
