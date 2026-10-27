"""
The guard in front of vLLM's admin endpoints, loaded into vLLM itself (``--middleware``).

vLLM mounts its admin endpoints all at once, under VLLM_SERVER_DEV_MODE: sleep and wake
(what GPU sharing tools use to hand the card to something else), cache resets (what
cleanup_generation_memory uses), and with them ``/collective_rpc``, the weight-update and
RLHF endpoints and ``/server_info``. ``--api-key`` does not cover any of them, and
``/collective_rpc`` reads its body whatever the content type, so a web page open on this
machine could post to it although vLLM listens on 127.0.0.1 only.

So with them on, this lets through the few that are needed and nothing else:

* ``/sleep``, ``/wake_up``, ``/is_sleeping`` and the cache resets answer, to programs on
  this machine. A request a browser made -- it carries Origin or Sec-Fetch-* -- or one
  addressed to a host name other than localhost (DNS rebinding) is refused.
* Every other admin endpoint answers 404, as it does with the admin endpoints off.

Which paths are admin endpoints is read from the vLLM that is running (the routers under
``vllm.entrypoints.serve.dev``), so one that a later vLLM adds is refused rather than
missed; the list below is the fallback, vLLM 0.28's.

This file is imported by vLLM's interpreter from a directory of its own on PYTHONPATH,
not as part of the service: it must not import the service, and nothing beside it is
put on vLLM's path.
"""

from __future__ import annotations

import importlib
import ipaddress
import json
import logging
import pkgutil
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

logger = logging.getLogger("ast_vllm_guard")

ALLOWED = frozenset(
    {
        "/sleep",
        "/wake_up",
        "/is_sleeping",
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    }
)

# vLLM 0.28's admin endpoints, for when its routers cannot be read.
FALLBACK_ADMIN = frozenset(
    {
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
        "/pause",
        "/resume",
        "/abort_requests",
        "/is_paused",
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/start_draft_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/update_weight_version",
        "/weight_info",
        "/get_world_size",
        "/collective_rpc",
        "/server_info",
        "/sleep",
        "/wake_up",
        "/is_sleeping",
    }
)

_DEV_PACKAGE = "vllm.entrypoints.serve.dev"
_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain"})


def admin_paths() -> frozenset[str]:
    """The admin endpoint paths of the vLLM installed here, or the fallback list."""
    try:
        package = importlib.import_module(_DEV_PACKAGE)
        found: set[str] = set()
        for info in pkgutil.walk_packages(package.__path__, f"{_DEV_PACKAGE}."):
            if not info.name.endswith(".api_router"):
                continue
            router = getattr(importlib.import_module(info.name), "router", None)
            found.update(getattr(route, "path", "") for route in getattr(router, "routes", []))
        found.discard("")
    except Exception:  # noqa: BLE001 - any failure here means "use the known list"
        logger.warning("could not read vLLM's admin routes; using the vLLM 0.28 list")
        return FALLBACK_ADMIN
    return frozenset(found) or FALLBACK_ADMIN


def _is_loopback_host(host_header: str) -> bool:
    name = host_header.strip().lower()
    if name.startswith("["):
        name = name[1:].partition("]")[0]
    elif name.count(":") == 1:
        name = name.partition(":")[0]
    if name in _LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def refusal(path: str, headers: dict[str, str], blocked: frozenset[str]) -> int | None:
    """The status to refuse a request with, or None to pass it on."""
    if path in blocked:
        return 404
    if path not in ALLOWED:
        return None
    if "origin" in headers or any(name.startswith("sec-fetch-") for name in headers):
        return 403
    if not _is_loopback_host(headers.get("host", "")):
        return 403
    return None


class AdminGuard:
    """ASGI middleware: vLLM's ``--middleware ast_vllm_guard.AdminGuard``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.blocked = admin_paths() - ALLOWED
        logger.warning(
            "admin endpoints limited to %s for programs on this machine; %d others refused",
            ", ".join(sorted(ALLOWED)),
            len(self.blocked),
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Refuse what is not allowed, pass the rest to vLLM."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        root = scope.get("root_path", "")
        path = scope.get("path", "")
        if root and path.startswith(root):
            path = path[len(root) :] or "/"
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        status = refusal(path.rstrip("/") or "/", headers, self.blocked)
        if status is None:
            await self.app(scope, receive, send)
            return
        detail = "Not Found" if status == 404 else "Not available to web pages."
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
