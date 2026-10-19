"""
What a request may and may not change about the inference servers this service starts.

llama-server and vLLM are started with arguments from the load request, and the request
can add its own (llama_server_extra_args, vllm_server_extra_args), appended last so they
win. That is how a model is tuned -- context, batch, offload, chat template -- and it is
also how a request could move a server off loopback, give it a key of its own, open its
CORS, make it serve or write files, or (vLLM's --middleware) import code. Those flags
belong to this service and are refused here, before anything starts.
"""

from __future__ import annotations

import ipaddress

from . import settings

# Flags that decide who can reach a server and what it does beyond inference.
LLAMA_SERVER_RESERVED_FLAGS: frozenset[str] = frozenset(
    {
        "--host",
        "--port",
        "--reuse-port",
        "--path",
        "--api-prefix",
        "--api-key",
        "--api-key-file",
        "--cors-origins",
        "--cors-methods",
        "--cors-headers",
        "--cors-credentials",
        "--no-cors-credentials",
        "--ssl-key-file",
        "--ssl-cert-file",
        "--slot-save-path",
        "--log-file",
    }
)

VLLM_SERVER_RESERVED_FLAGS: frozenset[str] = frozenset(
    {
        "--host",
        "--port",
        "--uds",
        "--api-key",
        "--allowed-origins",
        "--allowed-methods",
        "--allowed-headers",
        "--allow-credentials",
        "--ssl-keyfile",
        "--ssl-certfile",
        "--ssl-ca-certs",
        "--root-path",
        "--middleware",
    }
)


def local_service_origins() -> list[str]:
    """The web origins this service is served from, for the servers' CORS lists."""
    port = settings.SERVICE_PORT
    return [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]


def reserved_flags_in(args: list[str] | None, reserved: frozenset[str]) -> list[str]:
    """The reserved flags among args, in either spelling: "--flag value" or "--flag=value"."""
    found: list[str] = []
    for arg in args or []:
        flag = str(arg).split("=", 1)[0].strip()
        if flag in reserved and flag not in found:
            found.append(flag)
    return found


def is_loopback_host(host: str | None) -> bool:
    """Whether a bind address keeps a server on this machine."""
    name = (host or "").strip().strip("[]").lower()
    if name in ("localhost", ""):
        return name == "localhost"
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False
