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

API_KEY: str = os.getenv("API_KEY", "").strip() or "EMPTY"
"""Placeholder credential.

The backend does not authenticate incoming requests -- ``/v1/chat/completions``
has no auth dependency or middleware. This value exists only because the OpenAI
SDK refuses to construct a client with an empty ``api_key``. ``EMPTY`` is the
same placeholder the backend uses for its own internal engine clients.
"""

VERIFY_SSL: bool = os.getenv("VERIFY_SSL", "true").strip().lower() not in _FALSEY
"""Whether to verify the backend's TLS certificate.

Only set ``VERIFY_SSL=false`` when the backend serves https with a self-signed
certificate. It has no effect on plain-http backends.
"""


def api_base_url(backend_url: str | None = None) -> str:
    """
    Return the OpenAI-compatible ``/v1`` base URL for a backend.

    Args:
        backend_url: Override for :data:`BACKEND_URL`. Falsey values fall back
            to the configured default.
    """
    return (backend_url or BACKEND_URL).rstrip("/") + "/v1"
