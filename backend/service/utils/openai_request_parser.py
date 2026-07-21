"""Helpers for parsing OpenAI-compatible chat completion requests."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile

from .image_input_utils import (
    append_uploads_to_last_user_message,
    extract_image_urls_from_attachment_payload,
    normalize_openai_messages_images,
    upload_file_to_data_url,
)

_IMAGE_UPLOAD_FIELD_NAMES = {
    "image",
    "images",
    "file",
    "files",
    "attachment",
    "attachments",
}

_INT_FIELDS = {"max_tokens", "top_k", "seed", "n", "rag_top_k"}
_FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
}
_BOOL_FIELDS = {"stream", "use_rag", "reset_history", "rag_include_sources"}
_LOG_TRUNCATION_LENGTH = 160
_DATA_URL_LOG_PREFIX_LENGTH = 96
_JSON_ATTACHMENT_KEYS = ("attachments", "attachment", "images", "image", "files", "file")


def _parse_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    raw = value.strip()
    if not raw:
        return value

    if raw[0] in {"{", "["}:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON form field: {exc}") from exc

    return value


def _coerce_scalar_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    for key in _INT_FIELDS:
        if key in payload and isinstance(payload[key], str) and payload[key].strip():
            payload[key] = int(payload[key])

    for key in _FLOAT_FIELDS:
        if key in payload and isinstance(payload[key], str) and payload[key].strip():
            payload[key] = float(payload[key])

    for key in _BOOL_FIELDS:
        if key in payload and isinstance(payload[key], str):
            payload[key] = payload[key].strip().lower() in {"1", "true", "yes", "on"}

    return payload


async def _extract_uploaded_images(form_items: List[tuple[str, Any]]) -> List[str]:
    uploaded_image_urls: List[str] = []
    for key, value in form_items:
        if not isinstance(value, UploadFile):
            continue

        is_named_image_field = key.lower() in _IMAGE_UPLOAD_FIELD_NAMES
        is_image_content = bool(value.content_type and value.content_type.startswith("image/"))
        if not is_named_image_field and not is_image_content:
            continue

        uploaded_image_urls.append(await upload_file_to_data_url(value))

    return uploaded_image_urls


async def parse_openai_chat_request_payload(http_request: Request) -> Dict[str, Any]:
    """Parse JSON or multipart/form-data for /v1/chat/completions."""
    content_type = http_request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await http_request.form()
        form_items = list(form.multi_items())

        payload: Dict[str, Any] = {}
        payload_field = form.get("payload") or form.get("request")
        if isinstance(payload_field, str) and payload_field.strip():
            try:
                parsed_payload = json.loads(payload_field)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid multipart payload JSON: {exc}") from exc
            if not isinstance(parsed_payload, dict):
                raise HTTPException(status_code=400, detail="Multipart payload must be a JSON object")
            payload.update(parsed_payload)

        for key, value in form_items:
            if key in {"payload", "request"}:
                continue
            if isinstance(value, UploadFile):
                continue
            payload[key] = _parse_json_field(value)

        payload = _coerce_scalar_fields(payload)
        uploaded_image_urls = await _extract_uploaded_images(form_items)

        messages = payload.get("messages")
        if messages is None:
            message_text = str(payload.get("message", "") or "")
            payload["messages"] = [{"role": "user", "content": message_text}]
        elif not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="'messages' must be a list")

        payload["messages"] = normalize_openai_messages_images(payload["messages"])
        payload["messages"] = append_uploads_to_last_user_message(
            payload["messages"], uploaded_image_urls
        )
        return payload

    try:
        payload = await http_request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type. Use application/json or multipart/form-data",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON request body must be an object")

    json_attachment_urls: List[str] = []
    for key in _JSON_ATTACHMENT_KEYS:
        json_attachment_urls.extend(
            extract_image_urls_from_attachment_payload(payload.get(key))
        )

    if isinstance(payload.get("messages"), list):
        payload["messages"] = normalize_openai_messages_images(payload["messages"])
        payload["messages"] = append_uploads_to_last_user_message(
            payload["messages"], json_attachment_urls
        )
    return payload


def sanitize_openai_request_for_logging(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Redact oversized or binary-like values before logging request payloads."""
    try:
        sanitized = json.loads(json.dumps(payload, ensure_ascii=False))
    except Exception:
        return {"raw": str(payload)}

    messages = sanitized.get("messages")
    if isinstance(messages, list):
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and len(content) > _LOG_TRUNCATION_LENGTH:
                message["content"] = f"{content[:_LOG_TRUNCATION_LENGTH]}...(len={len(content)})"
                continue

            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = str(part.get("type", "")).strip().lower()
                if part_type == "text":
                    text = part.get("text")
                    if isinstance(text, str) and len(text) > _LOG_TRUNCATION_LENGTH:
                        part["text"] = f"{text[:_LOG_TRUNCATION_LENGTH]}...(len={len(text)})"
                    continue

                if part_type != "image_url":
                    continue

                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        image_url["url"] = (
                            f"{url[:_DATA_URL_LOG_PREFIX_LENGTH]}...(len={len(url)})"
                        )
                elif isinstance(image_url, str) and image_url.startswith("data:"):
                    part["image_url"] = (
                        f"{image_url[:_DATA_URL_LOG_PREFIX_LENGTH]}...(len={len(image_url)})"
                    )

    for key in _JSON_ATTACHMENT_KEYS:
        if key not in sanitized:
            continue

        attachments_value = sanitized.get(key)
        if isinstance(attachments_value, list):
            sanitized[key] = {"count": len(attachments_value)}
        elif attachments_value is not None:
            sanitized[key] = {"count": 1}

    return sanitized
