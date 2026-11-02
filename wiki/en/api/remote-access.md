---
title: Remote access and the trust boundary
summary: Who may call the service — this machine, other machines, and web pages in a browser each have their own rules; only the inference API is open by default, and wrong keys get locked out.
updatedAt: 2026-10-29
---

# Remote access and the trust boundary

## The design

The service listens on `127.0.0.1` by default. Letting another computer use the model (Hermes on
another machine, for example) means opening the service to the network — and **that needs the
inference API and nothing else**.

The management API takes file paths and launch arguments, and that is exactly how unauthenticated
model servers get attacked. So the design is not "add authentication" but **narrow what is opened
to the network down to what it is for**.

## Three sources, three sets of rules

The decision is based on **the TCP peer address**, never on a header the client can set.

| Source | Needs | Can use |
|---|---|---|
| **This machine** (loopback) | No key | Everything |
| **Another machine** | `Authorization: Bearer <SERVICE_API_KEY>` | Per `SERVICE_REMOTE_ACCESS`; **only `/v1/`** by default |
| **A web page in a browser** | — | Only pages this machine serves; pages from other sites are refused |

### Other machines

| Setting | Notes |
|---|---|
| `SERVICE_API_KEY` | When empty, other machines are always refused (remote access is off) |
| `SERVICE_REMOTE_ACCESS=inference` (default) | `/v1/` only |
| `SERVICE_REMOTE_ACCESS=all` | Adds `/inference/`, `/config/`, `/training/`, `/rag/`, `/system/`, `/examples/` |

- It is an **allow-list**: routes added later stay closed to the network until someone decides to open them.
- **The web interface and `/docs` are always local only**, key or not.
- **The desktop app always runs with `inference`.**

> ⚠️ Use `all` only behind your own protection: a VPN (Tailscale, for example), or a reverse proxy
> with TLS and authentication. **Never expose it directly to the internet** — whoever holds the key
> can reach far more than the model.

### `/health`

The one path open to the network, so another machine can check the service is reachable before it
has a key. **Without a key it only says the service is alive**; from this machine or with the right
key, it returns everything (including `supported_engines`).

### Wrong keys get locked out

**10 wrong keys within a minute → that address is locked for 5 minutes** (`429`).

- The lockout lives in memory and is forgotten on restart.
- The log records the address, not what was sent — a wrong key may be a mistyped real one.
- **A refused path is not a wrong key**, so a mistyped URL does not lock you out.

## "This machine" includes every web page in its browser

The easiest point to miss: any site open in a browser on this machine can call `127.0.0.1`.
So local requests pass one more check:

| Situation | Result |
|---|---|
| `Origin` is a page this machine serves (any port) | ✅ |
| `Origin` is another site, or `null` (sandboxed frame, `file://`) | ❌ `403` |
| No `Origin`, but `Sec-Fetch-Site: cross-site` and not a navigation | ❌ `403` (a cross-site `<img>` and the like; even a GET can have effects) |
| `Host` is neither localhost nor an IP | ❌ `403` (DNS rebinding protection) |
| Not a browser (no `Origin`, no `Sec-Fetch-*`) | ✅ Scripts, agents and coding assistants are unaffected |

CORS allows only local origins, without credentials.

## Protection further down

| Target | Measure |
|---|---|
| llama-server | Started with an internal key (random when unset); CORS allows only this service; loopback only |
| vLLM's admin endpoints | Sleep/wake and similar for local programs only; `/collective_rpc`, `/update_weights` and the rest return 404 |
| Extra args in a load request | Flags the service owns (`--host`, `--port`, `--api-key`, `--cors-*`…) are refused |
| `llama_server_url` | Loopback only |

## Behind a reverse proxy

**This layer depends on the peer address, and a proxy changes it.** Neither outcome is what you want:

| Proxy | Peer address the backend sees | Result |
|---|---|---|
| Does not send `X-Forwarded-For` | The proxy (127.0.0.1) | **Every outside request counts as local: everything is open** |
| Sends `X-Forwarded-For` | The real client | **The web interface and management API all return `403`** |

So behind a proxy, **authentication is the proxy's job**. See [Behind a reverse proxy](../operations/reverse-proxy.md).

## Related

- [Error model](errors.md)
- [Troubleshooting by symptom](../operations/troubleshooting.md)
- [`docs/usage.en.md`: connecting other applications](../../../docs/usage.en.md#connecting-other-applications)
