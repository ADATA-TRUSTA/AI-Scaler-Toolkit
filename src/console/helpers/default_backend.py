"""Helpers for loading default backend URL from config file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_BACKEND_SETTINGS_PATH = (
    Path(__file__).resolve().parent.parent
    / "app_settings"
    / "default_backend_settings.json"
)


def load_default_backend_url(
    settings_path: str | Path = DEFAULT_BACKEND_SETTINGS_PATH,
) -> str:
    """Load default backend URL from JSON settings file."""
    path = Path(settings_path)
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    backend_url = data.get("backend_url")
    if not isinstance(backend_url, str) or not backend_url.strip():
        raise ValueError(
            f"backend_url must be a non-empty string in settings file: {path}"
        )
    return backend_url.strip()
