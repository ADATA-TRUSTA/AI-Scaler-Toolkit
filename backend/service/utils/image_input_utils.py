"""Utilities for normalizing multimodal image inputs."""

from __future__ import annotations

import base64
import mimetypes
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from starlette.datastructures import UploadFile

_IMAGE_DATA_URL_PREFIXES = ("data:image/", "data:application/octet-stream")
_BASE64_ALLOWED_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
)


def guess_mime_type(filename: Optional[str], content_type: Optional[str]) -> str:
    """Infer MIME type for an uploaded or local file."""
    if content_type and "/" in content_type:
        return content_type

    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed

    return "application/octet-stream"


async def upload_file_to_data_url(upload: UploadFile) -> str:
    """Convert an uploaded file to a data URL."""
    content = await upload.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{upload.filename or 'unnamed'}' is empty",
        )

    mime_type = guess_mime_type(upload.filename, upload.content_type)
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _read_file_to_data_url(file_path: str) -> str:
    with open(file_path, "rb") as handle:
        content = handle.read()

    if not content:
        raise HTTPException(status_code=400, detail=f"Image file is empty: {file_path}")

    mime_type = guess_mime_type(file_path, None)
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def looks_like_base64_payload(value: str) -> bool:
    """Heuristically detect if a string is likely raw base64."""
    raw = value.strip()
    if len(raw) < 32 or len(raw) % 4 != 0:
        return False
    return all(ch in _BASE64_ALLOWED_CHARS for ch in raw)


def normalize_image_input(value: Any) -> Optional[str]:
    """Normalize an image input into a supported string representation."""
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    raw = value.strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered.startswith(_IMAGE_DATA_URL_PREFIXES):
        return raw

    if lowered.startswith(("http://", "https://")):
        return raw

    if lowered.startswith("file://"):
        file_path = raw[7:]
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail=f"Image file not found: {file_path}")
        return _read_file_to_data_url(file_path)

    if os.path.isfile(raw):
        return _read_file_to_data_url(raw)

    if looks_like_base64_payload(raw):
        return f"data:image/png;base64,{raw}"

    return raw


def normalize_openai_messages_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize image fields in OpenAI-format messages."""
    normalized = deepcopy(messages)

    for message in normalized:
        message_attachment_urls = extract_image_urls_from_attachment_payload(
            message.get("attachments")
        )
        content = message.get("content")
        if not isinstance(content, list):
            if message_attachment_urls:
                message["content"] = _merge_image_urls_into_content(content, message_attachment_urls)
            message.pop("attachments", None)
            continue

        for part in content:
            if not isinstance(part, dict):
                continue

            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = normalize_image_input(image_url.get("url"))
                    if url:
                        image_url["url"] = url
                elif isinstance(image_url, str):
                    url = normalize_image_input(image_url)
                    if url:
                        part["image_url"] = {"url": url}
            elif part_type == "image":
                image_value = normalize_image_input(part.get("image"))
                if image_value:
                    part["type"] = "image_url"
                    part["image_url"] = {"url": image_value}
                    part.pop("image", None)
            elif part_type == "input_image":
                image_value = normalize_image_input(
                    part.get("image_url")
                    or part.get("image")
                    or part.get("url")
                    or part.get("content")
                )
                if image_value:
                    part["type"] = "image_url"
                    part["image_url"] = {"url": image_value}
                    part.pop("image", None)
                    part.pop("url", None)
                    part.pop("content", None)
            elif part_type == "input_text":
                text_value = part.get("text") or part.get("content")
                if isinstance(text_value, str):
                    part["type"] = "text"
                    part["text"] = text_value
                    part.pop("content", None)

        if message_attachment_urls:
            message["content"] = _merge_image_urls_into_content(content, message_attachment_urls)
        message.pop("attachments", None)

    return normalized


def _attachment_to_image_url(attachment: Any) -> Optional[str]:
    if attachment is None:
        return None

    if isinstance(attachment, str):
        return normalize_image_input(attachment)

    if not isinstance(attachment, dict):
        return None

    nested_image_url = attachment.get("image_url")
    if isinstance(nested_image_url, dict):
        url = normalize_image_input(nested_image_url.get("url"))
        if url:
            return url
    elif isinstance(nested_image_url, str):
        url = normalize_image_input(nested_image_url)
        if url:
            return url

    for key in ["url", "uri", "src", "path", "file_path", "filePath"]:
        url = normalize_image_input(attachment.get(key))
        if url:
            return url

    content = attachment.get("content") or attachment.get("buffer") or attachment.get("data")
    if isinstance(content, str) and content.strip():
        mime_type = (
            attachment.get("mimeType")
            or attachment.get("contentType")
            or guess_mime_type(
                attachment.get("name") or attachment.get("filename"),
                None,
            )
        )
        encoding = str(attachment.get("encoding") or "").strip().lower()
        normalized_content = content.strip()
        if normalized_content.lower().startswith(_IMAGE_DATA_URL_PREFIXES):
            return normalized_content
        if encoding == "base64" or looks_like_base64_payload(normalized_content):
            return f"data:{mime_type};base64,{normalized_content}"
        return normalize_image_input(normalized_content)

    return None


def extract_image_urls_from_attachment_payload(attachments: Any) -> List[str]:
    """Extract normalized image URLs from OpenClaw-style attachment payloads."""
    if attachments is None:
        return []

    if isinstance(attachments, list):
        values = attachments
    else:
        values = [attachments]

    image_urls: List[str] = []
    for attachment in values:
        image_url = _attachment_to_image_url(attachment)
        if image_url:
            image_urls.append(image_url)
    return image_urls


def _merge_image_urls_into_content(content: Any, image_urls: List[str]) -> List[Dict[str, Any]]:
    image_parts = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]

    if isinstance(content, list):
        return content + image_parts

    if isinstance(content, str):
        parts: List[Dict[str, Any]] = []
        if content.strip():
            parts.append({"type": "text", "text": content})
        parts.extend(image_parts)
        return parts

    return image_parts


def append_uploads_to_last_user_message(
    messages: List[Dict[str, Any]], uploaded_image_urls: List[str]
) -> List[Dict[str, Any]]:
    """Append uploaded images to the last user message content."""
    if not uploaded_image_urls:
        return messages

    normalized = deepcopy(messages)
    last_user_idx = -1
    for idx in range(len(normalized) - 1, -1, -1):
        role = str(normalized[idx].get("role", "")).strip().lower()
        if role == "user":
            last_user_idx = idx
            break

    if last_user_idx < 0:
        normalized.append({"role": "user", "content": ""})
        last_user_idx = len(normalized) - 1

    content = normalized[last_user_idx].get("content")
    image_parts = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in uploaded_image_urls
    ]

    if isinstance(content, str):
        parts: List[Dict[str, Any]] = []
        if content.strip():
            parts.append({"type": "text", "text": content})
        parts.extend(image_parts)
        normalized[last_user_idx]["content"] = parts
        return normalized

    if isinstance(content, list):
        content.extend(image_parts)
        return normalized

    normalized[last_user_idx]["content"] = image_parts
    return normalized
