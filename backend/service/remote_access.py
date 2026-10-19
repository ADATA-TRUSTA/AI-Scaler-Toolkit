"""
Who may reach this service, from where, and with what.

The service listens on 127.0.0.1 unless it is opened to the network (SERVICE_HOST=0.0.0.0,
which the desktop application does when "Let another computer use this machine's model"
is on). It had no authentication at all, so once open, anyone on the network could chat
with the loaded model, load or unload others, start downloads and conversions, and open
the web interface -- the exposure behind the reported attacks on unauthenticated Ollama
servers, which went in through its model-management API. That switch exists so another
computer's Hermes can use the model, which needs the inference API and nothing else.

So, by where a request comes from:

* Another machine: `Authorization: Bearer <SERVICE_API_KEY>` -- the header every
  OpenAI-compatible client already sends -- and only what SERVICE_REMOTE_ACCESS allows:
  the inference API under /v1/ by default, the management API as well with "all". Without
  a configured key it is refused, and an address that keeps sending wrong keys is
  locked out for a while (KeyLockout). `/health` is open, so another machine can tell the
  service is reachable before it has a key. The web interface and the API docs are
  refused even with a key: they were never meant for another computer.
* This machine: everything, no key -- the application's own window, a local Hermes and
  local coding agents keep working as before. But "this machine" includes every web page
  open in a browser here, and a page can send requests to 127.0.0.1 too. So a request
  that a browser marks as coming from another site is refused, and so is one addressed
  to a host name other than localhost (DNS rebinding: a name the attacker controls,
  pointed at 127.0.0.1, so the browser treats the service as that site's own).

"Where it comes from" is the TCP peer address the server saw, never a header a client
can set (X-Forwarded-For and the like). There is no proxy in front of this service.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
import time
from collections import deque
from collections.abc import Callable
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# What another machine may reach with the key, by SERVICE_REMOTE_ACCESS. Allowlists, so a
# route added later is closed to the network until someone decides otherwise. Everything
# else -- the web interface, its assets, /docs -- is this machine's only.
REMOTE_PREFIXES: dict[str, tuple[str, ...]] = {
    # The OpenAI-compatible API: chat, completions, embeddings, the model list.
    "inference": ("/v1/",),
    # Plus the management API. These take file paths and launch arguments.
    "all": (
        "/v1/",
        "/inference/",
        "/config/",
        "/training/",
        "/rag/",
        "/system/",
        "/examples/",
    ),
}

# Reachable from anywhere, without a key: says the service is up and nothing else.
OPEN_PATHS: frozenset[str] = frozenset({"/health"})

logger = logging.getLogger(__name__)

# Host names a request from this machine may be addressed to, beside IP literals.
_LOCAL_HOST_NAMES: frozenset[str] = frozenset({"localhost", "localhost.localdomain"})


def is_local(host: str | None) -> bool:
    """Whether a peer address is this machine. No address (a Unix socket) is local."""
    if host is None:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _host_name(netloc: str) -> str:
    """The host part of a Host header or URL authority: no port, no brackets, lower case."""
    return (urlsplit(f"//{netloc}").hostname or "").lower()


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _is_loopback_name(name: str) -> bool:
    return (
        name in _LOCAL_HOST_NAMES
        or name.endswith(".localhost")
        or (_is_ip(name) and is_local(name))
    )


def is_local_origin(origin: str) -> bool:
    """Whether a browser Origin is a page served from this machine (any port)."""
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and _is_loopback_name((parts.hostname or "").lower())


def local_refusal(headers: dict[str, str]) -> tuple[int, str] | None:
    """
    Why a request from this machine is refused, as (status, detail); None to allow it.

    Only browsers send Origin and Sec-Fetch-*, so clients that are not a web page -- the
    application's own window aside, which is a page on this origin -- are unaffected.
    """
    host = headers.get("host", "")
    name = _host_name(host) if host else ""
    if name and not (_is_ip(name) or _is_loopback_name(name)):
        return 403, "Requests to this service must be addressed to localhost or an IP address."

    origin = headers.get("origin")
    if origin is not None:
        # "null" is a sandboxed frame or a file:// page, and is refused with the rest.
        if not is_local_origin(origin):
            return 403, "Web pages from other sites cannot use this service."
        return None
    # No Origin: a navigation, or a cross-site load such as an <img> -- which can still
    # carry a side effect on a GET. Browsers mark those.
    if (
        headers.get("sec-fetch-site") == "cross-site"
        and headers.get("sec-fetch-mode") != "navigate"
    ):
        return 403, "Web pages from other sites cannot use this service."
    return None


def _bearer(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").strip().partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def remote_refusal(
    path: str, authorization: str | None, api_key: str, access: str = "inference"
) -> tuple[int, str] | None:
    """Why a request from another machine is refused, as (status, detail); None to allow it."""
    if path in OPEN_PATHS:
        return None
    prefixes = REMOTE_PREFIXES.get(access, REMOTE_PREFIXES["inference"])
    if not path.startswith(prefixes):
        if path.startswith(REMOTE_PREFIXES["all"]):
            return 403, (
                "Other computers can use the inference API (/v1/) only. The rest of the API "
                "is available on the computer running AI Scaler Toolkit."
            )
        return 403, "This is available on the computer running AI Scaler Toolkit only."
    if not api_key:
        return 403, (
            "Access from other computers is not enabled on this service. "
            "Turn it on in AI Scaler Toolkit's settings, which also sets its API key."
        )
    supplied = _bearer(authorization)
    if not supplied or not secrets.compare_digest(supplied.encode(), api_key.encode()):
        return 401, "A valid API key is required: send it as 'Authorization: Bearer <key>'."
    return None


class KeyLockout:
    """
    Addresses that sent a wrong key too often, refused for a while.

    A key the user typed can be short; a random one cannot be guessed, but nothing stops
    a machine on the network from trying. Ten wrong keys in a minute is no client that
    was given the key -- a misconfigured one gets there too, and a pause is what it needs
    as well. Held in memory: a restart forgets it, which costs an attacker a restart of a
    service they cannot reach.
    """

    def __init__(
        self,
        limit: int = 10,
        window: float = 60.0,
        duration: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self.duration = duration
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._until: dict[str, float] = {}

    def remaining(self, address: str) -> float:
        """Seconds this address is still refused for; 0 when it is not."""
        until = self._until.get(address)
        if until is None:
            return 0.0
        left = until - self._clock()
        if left <= 0:
            del self._until[address]
            return 0.0
        return left

    def record_failure(self, address: str) -> None:
        """Count a wrong key; lock the address out once there are too many."""
        now = self._clock()
        recent = self._failures.setdefault(address, deque())
        recent.append(now)
        while recent and now - recent[0] > self.window:
            recent.popleft()
        if len(recent) >= self.limit:
            self._until[address] = now + self.duration
            del self._failures[address]
            # The address, never what was sent: a wrong key can be a real one mistyped.
            logger.warning(
                "Refusing %s for %.0f s after %d wrong API keys in %.0f s",
                address,
                self.duration,
                self.limit,
                self.window,
            )


class RemoteAccessMiddleware:
    """
    Apply remote_refusal to every request from another machine, and local_refusal to
    every request from this one.

    Plain ASGI rather than BaseHTTPMiddleware, so streamed responses (chat, the training
    log) pass through untouched.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key: Callable[[], str],
        access: Callable[[], str] = lambda: "inference",
        lockout: KeyLockout | None = None,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.access = access
        self.lockout = lockout or KeyLockout()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check a request against where it came from, then pass it on or refuse it."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        client = scope.get("client")
        address = client[0] if client else None
        path = scope.get("path", "")
        extra: dict[str, str] = {}
        if is_local(address):
            refusal = local_refusal(headers)
        elif path not in OPEN_PATHS and (left := self.lockout.remaining(str(address))):
            refusal = (429, "Too many wrong API keys from this address. Try again later.")
            extra = {"Retry-After": str(int(left) + 1)}
        else:
            refusal = remote_refusal(
                path,
                headers.get("authorization"),
                self.api_key() or "",
                self.access() or "inference",
            )
            if refusal is not None and refusal[0] == 401:
                self.lockout.record_failure(str(address))
                extra = {"WWW-Authenticate": "Bearer"}
        if refusal is None:
            await self.app(scope, receive, send)
            return
        status, detail = refusal
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": detail[:120]})
            return
        await JSONResponse({"detail": detail}, status_code=status, headers=extra)(
            scope, receive, send
        )
