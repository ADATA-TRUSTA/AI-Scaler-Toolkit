"""
AIClient response and error handling.

Every branch is exercised through httpx2.MockTransport rather than a live
backend, so the mapping from transport failures to AIClientError is pinned
without needing a server. This is the layer the requests -> httpx2 migration
rewrote, so it is the layer most worth locking down.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx2
import pytest

from ai_scaler_console import AIClient, AIClientError

Handler = Callable[[httpx2.Request], httpx2.Response]


def client_with(handler: Handler) -> AIClient:
    """Build an AIClient whose transport is driven by ``handler``."""
    client = AIClient(base_url="http://mock-backend")
    client.session.close()
    client.session = httpx2.Client(transport=httpx2.MockTransport(handler))
    return client


LOADED_STATUS = {
    "loaded": True,
    "model_name": "trusta-ast/trusta-ast-default",
    "engine": "llama_server",
}


def test_success_parses_into_the_response_model() -> None:
    client = client_with(lambda _: httpx2.Response(200, json=LOADED_STATUS))
    status = client.get_status()

    assert status.loaded is True
    assert status.model_name == "trusta-ast/trusta-ast-default"
    assert status.engine.value == "llama_server"


def test_success_without_a_model_returns_the_raw_response() -> None:
    client = client_with(lambda _: httpx2.Response(200, json={"ok": True}))
    response = client.unload_model()

    assert isinstance(response, httpx2.Response)
    assert response.json() == {"ok": True}


def test_4xx_carries_the_parsed_json_body() -> None:
    client = client_with(lambda _: httpx2.Response(422, json={"detail": "bad config"}))

    with pytest.raises(AIClientError) as excinfo:
        client.unload_model()

    assert excinfo.value.status_code == 422
    assert excinfo.value.response_content == {"detail": "bad config"}


def test_5xx_with_a_non_json_body_falls_back_to_text() -> None:
    client = client_with(lambda _: httpx2.Response(500, text="internal boom"))

    with pytest.raises(AIClientError) as excinfo:
        client.unload_model()

    assert excinfo.value.status_code == 500
    assert excinfo.value.response_content == "internal boom"


def test_200_that_fails_validation_is_reported_as_a_parse_failure() -> None:
    # An engine value outside the enum: the request succeeded, the body did not.
    client = client_with(lambda _: httpx2.Response(200, json={"engine": "no-such-engine"}))

    with pytest.raises(AIClientError) as excinfo:
        client.get_status()

    assert "Failed to parse response model" in excinfo.value.message
    assert excinfo.value.status_code == 200


def test_200_that_is_not_json_at_all_is_reported_as_a_parse_failure() -> None:
    client = client_with(lambda _: httpx2.Response(200, text="<html>nope</html>"))

    with pytest.raises(AIClientError) as excinfo:
        client.get_status()

    assert "Failed to parse response model" in excinfo.value.message


def test_connect_error_becomes_a_network_error_without_a_status() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = client_with(refuse)

    with pytest.raises(AIClientError) as excinfo:
        client.unload_model()

    assert "Network error connecting to" in str(excinfo.value)
    assert excinfo.value.status_code is None


def test_read_timeout_becomes_a_network_error() -> None:
    def stall(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    client = client_with(stall)

    with pytest.raises(AIClientError) as excinfo:
        client.get_model_list()

    assert "Network error connecting to" in str(excinfo.value)


def test_context_manager_closes_the_connection_pool() -> None:
    with AIClient(base_url="http://mock-backend") as client:
        session = client.session

    assert session.is_closed


def test_base_url_defaults_to_the_configured_backend() -> None:
    from ai_scaler_console import settings

    client = AIClient()
    try:
        assert client.base_url == settings.BACKEND_URL.rstrip("/")
    finally:
        client.close()


def test_trailing_slash_is_stripped_from_base_url() -> None:
    client = AIClient(base_url="http://mock-backend/")
    try:
        assert client.base_url == "http://mock-backend"
    finally:
        client.close()


def test_request_path_is_composed_from_base_url_and_endpoint() -> None:
    seen: dict[str, str] = {}

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx2.Response(200, json=LOADED_STATUS)

    client = client_with(capture)
    client.get_status()

    assert seen["url"] == "http://mock-backend/inference/status"
    assert seen["method"] == "GET"
