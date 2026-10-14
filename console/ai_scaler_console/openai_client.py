"""
Factories for OpenAI-compatible clients pointed at the AI Scaler backend.

Every example needs the same three things: the ``/v1`` base URL, the placeholder
api key, and a transport that honours
:data:`~ai_scaler_console.settings.VERIFY_SSL`. Building them here keeps that
boilerplate out of the example scripts.
"""

from __future__ import annotations

import httpx2
from openai import AsyncOpenAI, OpenAI

from ai_scaler_console import settings


def _needs_custom_transport(timeout: float | None) -> bool:
    """
    Whether the SDK's own transport has to be replaced.

    Only true when TLS verification is switched off or a timeout is requested;
    otherwise the OpenAI SDK's defaults are left alone.
    """
    return timeout is not None or not settings.VERIFY_SSL


def _sync_transport(timeout: float | None) -> httpx2.Client | None:
    """Build a sync transport, or None to keep the SDK default."""
    if not _needs_custom_transport(timeout):
        return None
    # Passing timeout=None to httpx2 means "no timeout at all", which is not the
    # same as "use the default", so only pass it when one was asked for.
    if timeout is None:
        return httpx2.Client(verify=settings.VERIFY_SSL)
    return httpx2.Client(verify=settings.VERIFY_SSL, timeout=timeout)


def _async_transport(timeout: float | None) -> httpx2.AsyncClient | None:
    """Build an async transport, or None to keep the SDK default."""
    if not _needs_custom_transport(timeout):
        return None
    if timeout is None:
        return httpx2.AsyncClient(verify=settings.VERIFY_SSL)
    return httpx2.AsyncClient(verify=settings.VERIFY_SSL, timeout=timeout)


def create_client(
    *,
    api_key: str | None = None,
    backend_url: str | None = None,
    timeout: float | None = None,
) -> OpenAI:
    """
    Create a synchronous OpenAI client bound to the backend.

    Args:
        api_key: Override for the placeholder key. Rarely needed.
        backend_url: Override for the configured backend URL.
        timeout: Request timeout in seconds. Uses the SDK default when omitted.
    """
    return OpenAI(
        api_key=api_key or settings.API_KEY,
        base_url=settings.api_base_url(backend_url),
        http_client=_sync_transport(timeout),
    )


def create_async_client(
    *,
    api_key: str | None = None,
    backend_url: str | None = None,
    timeout: float | None = None,
) -> AsyncOpenAI:
    """
    Create an asynchronous OpenAI client bound to the backend.

    Args:
        api_key: Override for the placeholder key. Rarely needed.
        backend_url: Override for the configured backend URL.
        timeout: Request timeout in seconds. Uses the SDK default when omitted.
    """
    return AsyncOpenAI(
        api_key=api_key or settings.API_KEY,
        base_url=settings.api_base_url(backend_url),
        http_client=_async_transport(timeout),
    )
