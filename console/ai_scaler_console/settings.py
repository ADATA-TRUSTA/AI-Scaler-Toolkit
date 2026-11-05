"""
Runtime settings resolved from environment variables.

Values are read from the process environment, falling back to a ``.env`` file
found by walking up from the current working directory. See ``.env.example``
for the full list of recognised variables.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_FALSEY = frozenset({"0", "false", "no", "off", ""})

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip()
"""Base URL of the AI Scaler backend, without the ``/v1`` suffix."""

_API_KEY_PLACEHOLDER = "EMPTY"

API_KEY: str = os.getenv("API_KEY", "").strip() or _API_KEY_PLACEHOLDER
"""The backend's API key, or the placeholder ``EMPTY`` when none is set.

A backend on this machine takes no key. One on another machine asks for the key set
in AI Scaler Toolkit's settings, as ``Authorization: Bearer <key>``, and answers only
its inference API (``/v1/``) unless it was set to allow the rest. Without the key every
request from another machine is refused, and wrong keys lock the address out for a
while. ``EMPTY`` stays as the fallback because the OpenAI SDK refuses an empty
``api_key``; it is never sent by :class:`~ai_scaler_console.client.AIClient`.
"""

VERIFY_SSL: bool = os.getenv("VERIFY_SSL", "true").strip().lower() not in _FALSEY
"""Whether to verify the backend's TLS certificate.

Only set ``VERIFY_SSL=false`` when the backend serves https with a self-signed
certificate. It has no effect on plain-http backends.
"""


def auth_headers(api_key: str | None = None) -> dict[str, str]:
    """
    Return the ``Authorization`` header for a real key, or no header at all.

    The placeholder is not a key: sent to a backend on another machine it is just a
    wrong one, which counts toward that address's lockout.

    Args:
        api_key: Override for :data:`API_KEY`. ``None`` falls back to it.
    """
    key = (API_KEY if api_key is None else api_key).strip()
    if not key or key == _API_KEY_PLACEHOLDER:
        return {}
    return {"Authorization": f"Bearer {key}"}


def api_base_url(backend_url: str | None = None) -> str:
    """
    Return the OpenAI-compatible ``/v1`` base URL for a backend.

    Args:
        backend_url: Override for :data:`BACKEND_URL`. Falsey values fall back
            to the configured default.
    """
    return (backend_url or BACKEND_URL).rstrip("/") + "/v1"
