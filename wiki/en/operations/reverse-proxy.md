---
title: Behind a reverse proxy
summary: What to watch when serving AST through nginx or another reverse proxy — the path prefix, SSE buffering, timeouts, and how a proxy defeats the backend's remote-access checks.
updatedAt: 2026-10-29
---

# Behind a reverse proxy

The desktop app needs no reverse proxy. This page is for deployments that put the service behind
nginx or similar for several users.

## The one thing that matters most: the proxy does authentication

The backend decides where a request comes from by **the TCP peer address** (see
[Remote access](../api/remote-access.md)), and a proxy changes it:

| Proxy setting | Peer the backend sees | Result |
|---|---|---|
| No `X-Forwarded-For` | The proxy itself, 127.0.0.1 | **Every outside request counts as local: everything is open.** `SERVICE_API_KEY` and `SERVICE_REMOTE_ACCESS` have no effect |
| Sends `X-Forwarded-For` | The real client | **The web interface and management API all return `403`** (*Access from other computers is not enabled*). uvicorn trusts this header from 127.0.0.1 by default |

**Neither is what you want.** In practice:

- do **not** send `X-Forwarded-For` / `X-Real-IP`, so the backend treats requests as local;
- **authenticate at the proxy**: `auth_basic`, mTLS or an upstream SSO, with TLS;
- or expose only `/v1/` through the proxy and keep the management API off it.

> The backend cannot tell a proxy is in front of it, so **it will not warn you**. After deploying,
> call `/config/models` from another machine without credentials and confirm it is refused.

## Path prefix

The backend has **no `/api` prefix**. To serve it under `/api/`, let a trailing slash strip it:

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000/;   # the trailing / strips /api
}
```

## SSE needs buffering off

Proxies buffer responses by default; with SSE, events pile up and arrive all at once.

```nginx
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_buffering off;
proxy_cache off;
chunked_transfer_encoding on;
proxy_read_timeout 3600s;
```

| Setting | Why |
|---|---|
| `proxy_buffering off` | **The key one**; without it nothing is sent live |
| `proxy_http_version 1.1` + `Connection ""` | Keeps upstream connections alive |
| `proxy_read_timeout` | Training and long prefills take a while; the 60 s default cuts the stream |

The backend's SSE responses also carry `X-Accel-Buffering: no`, but only nginx honours it; another
proxy will ignore it.

Affected endpoints: `POST /v1/chat/completions` (streaming) and `GET /training/{job_id}/log/stream`.

## Timeouts

The backend's generation timeout defaults to 1800 s (`DEFAULT_GENERATION_TIMEOUT`), and when
streaming, **the wait for the first byte is the whole prefill** — minutes for a long context. The
proxy's `proxy_read_timeout` has to keep up, or long contexts drop before the first response.

## Related

- [Remote access and the trust boundary](../api/remote-access.md)
- [Troubleshooting by symptom](troubleshooting.md)
