"""
Environment parsing in ai_scaler_console.settings.

The module reads os.environ at import time, so each case reloads it under a
patched environment. VERIFY_SSL is the one worth pinning: it decides whether TLS
verification happens at all, and it is parsed from a string.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType

import pytest

import ai_scaler_console.settings as settings_module


def reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    """Reload settings with exactly the given variables set."""
    for name in ("BACKEND_URL", "API_KEY", "VERIFY_SSL"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Stop python-dotenv reinstating a developer's local .env.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    return importlib.reload(settings_module)


def test_defaults_when_nothing_is_set(clean_env: pytest.MonkeyPatch) -> None:
    s = reload_settings(clean_env)

    assert s.BACKEND_URL == "http://127.0.0.1:8000"
    assert s.API_KEY == "EMPTY"
    assert s.VERIFY_SSL is True


def test_values_are_taken_from_the_environment(clean_env: pytest.MonkeyPatch) -> None:
    s = reload_settings(
        clean_env,
        BACKEND_URL="https://backend.example:9443",
        API_KEY="my-key",
        VERIFY_SSL="false",
    )

    assert s.BACKEND_URL == "https://backend.example:9443"
    assert s.API_KEY == "my-key"
    assert s.VERIFY_SSL is False


@pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off", ""])
def test_verify_ssl_falsey_spellings(clean_env: pytest.MonkeyPatch, value: str) -> None:
    assert reload_settings(clean_env, VERIFY_SSL=value).VERIFY_SSL is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "anything-else"])
def test_verify_ssl_defaults_to_on_for_anything_else(
    clean_env: pytest.MonkeyPatch, value: str
) -> None:
    """Verification must fail safe: only explicit falsey spellings disable it."""
    assert reload_settings(clean_env, VERIFY_SSL=value).VERIFY_SSL is True


def test_blank_api_key_falls_back_to_the_placeholder(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The OpenAI SDK refuses an empty api_key, so blank must not pass through."""
    assert reload_settings(clean_env, API_KEY="   ").API_KEY == "EMPTY"


def test_surrounding_whitespace_is_stripped(clean_env: pytest.MonkeyPatch) -> None:
    s = reload_settings(clean_env, BACKEND_URL="  http://host:8000  ")

    assert s.BACKEND_URL == "http://host:8000"


@pytest.mark.parametrize(
    ("backend_url", "expected"),
    [
        ("http://host:8000", "http://host:8000/v1"),
        ("http://host:8000/", "http://host:8000/v1"),
        ("http://host:8000///", "http://host:8000/v1"),
        ("https://host", "https://host/v1"),
    ],
)
def test_api_base_url_appends_v1_exactly_once(backend_url: str, expected: str) -> None:
    assert settings_module.api_base_url(backend_url) == expected


def test_api_base_url_falls_back_to_the_configured_backend() -> None:
    assert settings_module.api_base_url() == settings_module.api_base_url(
        settings_module.BACKEND_URL
    )


@pytest.mark.parametrize("falsey", [None, ""])
def test_api_base_url_treats_falsey_overrides_as_absent(falsey: str | None) -> None:
    assert settings_module.api_base_url(falsey) == settings_module.api_base_url()


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    """Leave the module as the rest of the suite expects to find it."""
    yield
    importlib.reload(settings_module)
