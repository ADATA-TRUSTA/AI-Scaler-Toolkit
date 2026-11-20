"""Regression: image inputs must never read files from the host."""

import base64

import pytest
from fastapi import HTTPException

from service.utils.image_input_utils import (
    normalize_image_input,
    normalize_openai_messages_images,
)


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "file://C:/Users/x/Pictures/a.png",
        r"C:\Users\x\a.png",
        r"\\host\share\a.png",
        "//host/share/a.png",
        "/etc/passwd",
        "../secret.png",
        "~/a.png",
    ],
)
def test_local_paths_are_rejected(value, tmp_path):
    with pytest.raises(HTTPException) as exc:
        normalize_image_input(value)
    assert exc.value.status_code == 400


def test_existing_file_is_not_read(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x" * 64)
    with pytest.raises(HTTPException):
        normalize_image_input(str(f))
    with pytest.raises(HTTPException):
        normalize_image_input(f.as_uri())


def test_allowed_forms_pass_through():
    assert normalize_image_input("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    assert normalize_image_input("https://example.com/a.png") == "https://example.com/a.png"
    raw = base64.b64encode(b"\xff\xd8\xff" + b"0" * 60).decode()
    assert normalize_image_input(raw) == f"data:image/png;base64,{raw}"


def test_message_image_url_with_file_is_rejected():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]}]
    with pytest.raises(HTTPException):
        normalize_openai_messages_images(msgs)
