"""
Transport selection in ai_scaler_console.openai_client.

The subtle rule here is that httpx2 treats ``timeout=None`` as "no timeout at
all", not "use the default". An earlier version built its client from a kwargs
dict and passed the timeout unconditionally, which silently removed every
timeout whenever a caller omitted one.
"""

from __future__ import annotations

import httpx2
import pytest

from ai_scaler_console import openai_client, settings


def test_no_custom_transport_when_verification_is_on_and_no_timeout_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the SDK's own transport alone when nothing needs overriding."""
    monkeypatch.setattr(settings, "VERIFY_SSL", True)

    assert openai_client._sync_transport(None) is None
    assert openai_client._async_transport(None) is None


def test_transport_is_built_when_verification_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "VERIFY_SSL", False)

    sync = openai_client._sync_transport(None)
    assert isinstance(sync, httpx2.Client)
    sync.close()


def test_transport_is_built_when_a_timeout_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "VERIFY_SSL", True)

    sync = openai_client._sync_transport(30.0)
    assert isinstance(sync, httpx2.Client)
    assert sync.timeout.read == 30.0
    sync.close()


def test_omitting_a_timeout_never_means_infinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The regression this module exists for.

    With verification off and no timeout asked for, the client must still carry
    httpx2's default timeout rather than none at all.
    """
    monkeypatch.setattr(settings, "VERIFY_SSL", False)

    sync = openai_client._sync_transport(None)
    assert sync is not None
    assert sync.timeout.read is not None, "omitting a timeout disabled timeouts entirely"
    sync.close()


@pytest.mark.parametrize(
    ("verify_ssl", "timeout", "expected"),
    [
        (True, None, False),
        (True, 10.0, True),
        (False, None, True),
        (False, 10.0, True),
    ],
)
def test_needs_custom_transport_truth_table(
    monkeypatch: pytest.MonkeyPatch, verify_ssl: bool, timeout: float | None, expected: bool
) -> None:
    monkeypatch.setattr(settings, "VERIFY_SSL", verify_ssl)

    assert openai_client._needs_custom_transport(timeout) is expected


def test_create_client_uses_the_configured_backend_and_placeholder_key() -> None:
    client = openai_client.create_client()

    assert str(client.base_url).rstrip("/") == settings.api_base_url().rstrip("/")
    assert client.api_key == settings.API_KEY


def test_create_client_honours_overrides() -> None:
    client = openai_client.create_client(
        api_key="explicit-key", backend_url="http://other-host:9000"
    )

    assert str(client.base_url).rstrip("/") == "http://other-host:9000/v1"
    assert client.api_key == "explicit-key"


@pytest.mark.parametrize("blank", [None, ""])
def test_blank_overrides_fall_back_to_configuration(blank: str | None) -> None:
    client = openai_client.create_client(api_key=blank, backend_url=blank)

    assert client.api_key == settings.API_KEY
    assert str(client.base_url).rstrip("/") == settings.api_base_url().rstrip("/")


def test_async_client_is_bound_the_same_way() -> None:
    client = openai_client.create_async_client(backend_url="http://other-host:9000")

    assert str(client.base_url).rstrip("/") == "http://other-host:9000/v1"
    assert client.api_key == settings.API_KEY
