"""Unit tests for /v1/chat/completions backend routing and concurrency.

The endpoint routes through _backend_generate[_stream]: server engines
(uses_async_http() True) take the direct async HTTP path (agenerate /
agenerate_stream); everything else falls back to the sync worker path
bridged onto the event loop. These tests pin the branch selection, error
mapping (500 / 507 OOM / SSE error events), stream passthrough, client
cancellation, and true concurrent overlap — all with stubbed backends.

Techniques (proven in this repo):
- TestClient + monkeypatch for request/response cases
- direct body_iterator drive for cancellation (TestClient and ASGITransport
  both buffer streams)
- httpx2.ASGITransport + asyncio.gather for the concurrency case
"""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx2
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import service.app as app_module
from service.app import app

pytestmark = pytest.mark.inference

# As this machine: see test_remote_access.py.
client = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


def _mark_loaded(monkeypatch, engine="llama_server"):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(engine=engine, model_path=None, model_name="demo"),
        raising=False,
    )


def _poison(name):
    def _fail(**kwargs):
        raise AssertionError(f"{name} must not be called on this path")

    return _fail


def _body(text="hi", stream=False, **extra):
    return {
        "model": "demo",
        "stream": stream,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def _sse_payloads(sse_lines):
    return [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.startswith("data:") and line.strip() != "data: [DONE]"
    ]


# ------------------------------ non-stream routing ------------------------------
def test_non_stream_routes_async_path(monkeypatch):
    _mark_loaded(monkeypatch, engine="llama_server")
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    async def _fake_agenerate(**kwargs):
        return {"result": "from-async", "prompt_tokens": 2, "gen_tokens": 3, "total_tokens": 5}

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)
    monkeypatch.setattr("service.app.model_manager.generate", _poison("generate"))

    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "from-async"
    assert data["usage"]["total_tokens"] == 5


def test_non_stream_routes_worker_path(monkeypatch):
    _mark_loaded(monkeypatch, engine="transformers")
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: False)

    def _fake_generate(**kwargs):
        return {"result": "from-worker", "total_tokens": 4}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)
    monkeypatch.setattr("service.app.model_manager.agenerate", _poison("agenerate"))

    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "from-worker"


def test_non_stream_async_error_maps_500_and_oom_507(monkeypatch):
    _mark_loaded(monkeypatch)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    async def _fail(**kwargs):
        # NOTE: the OOM heuristic substring-matches "oom", so the message
        # must avoid it (e.g. "boom" would map to 507).
        raise RuntimeError("backend failure")

    monkeypatch.setattr("service.app.model_manager.agenerate", _fail)
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 500
    assert resp.json()["detail"] == "backend failure"

    async def _oom(**kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("service.app.model_manager.agenerate", _oom)
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 507
    detail = resp.json()["detail"]
    assert detail["is_oom"] is True


# ------------------------------ stream passthrough ------------------------------
def test_stream_async_chunk_passthrough_and_usage(monkeypatch):
    _mark_loaded(monkeypatch)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    async def _fake_stream(**kwargs):
        yield {"chunk": "a", "done": False}
        yield {"chunk": "b", "done": False}
        yield {"chunk": "", "done": True, "prompt_tokens": 2, "gen_tokens": 2, "total_tokens": 4}

    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_stream)
    monkeypatch.setattr(
        "service.app.model_manager.generate_stream", _poison("generate_stream")
    )

    body = _body(stream=True, stream_options={"include_usage": True})
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = _sse_payloads(sse_lines)
    deltas = [
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("content")
        for p in payloads
    ]
    assert "a" in deltas and "b" in deltas
    assert deltas.index("a") < deltas.index("b")  # order preserved
    assert any(
        ((p.get("choices") or [{}])[0].get("finish_reason")) == "stop" for p in payloads
    )
    assert any(p.get("usage", {}).get("total_tokens") == 4 for p in payloads)
    assert sse_lines[-1].strip() == "data: [DONE]"


def test_stream_backend_error_yields_sse_error_event(monkeypatch):
    """Backend errors mid-stream must surface as a structured SSE error
    payload — not be swallowed as a fake client disconnect (regression guard
    for RuntimeError previously being in the disconnect tuple)."""
    _mark_loaded(monkeypatch)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    async def _fake_stream(**kwargs):
        yield {"chunk": "partial", "done": False}
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_stream)

    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = _sse_payloads(sse_lines)
    errors = [p["error"] for p in payloads if p.get("error")]
    assert errors, f"no SSE error event in: {payloads}"
    assert "engine exploded" in errors[0]["message"]
    assert errors[0]["type"] == "server_error"


def test_stream_worker_bridge_order_and_error(monkeypatch):
    _mark_loaded(monkeypatch, engine="transformers")
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: False)
    monkeypatch.setattr(
        "service.app.model_manager.agenerate_stream", _poison("agenerate_stream")
    )

    # ordered chunks across the producer-thread bridge
    def _fake_stream_ok(**kwargs):
        for i in range(5):
            time.sleep(0.01)
            yield {"chunk": f"c{i}", "done": False}
        yield {"chunk": "", "done": True}

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_stream_ok)
    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as resp:
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]
    payloads = _sse_payloads(sse_lines)
    deltas = [
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("content")
        for p in payloads
    ]
    assert [d for d in deltas if d and d.startswith("c")] == [f"c{i}" for i in range(5)]
    assert sse_lines[-1].strip() == "data: [DONE]"

    # a worker exception crossing the bridge becomes an SSE error event
    def _fake_stream_err(**kwargs):
        yield {"chunk": "x", "done": False}
        raise RuntimeError("worker blew up")

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_stream_err)
    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as resp:
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]
    payloads = _sse_payloads(sse_lines)
    errors = [p["error"] for p in payloads if p.get("error")]
    assert errors and "worker blew up" in errors[0]["message"]


# ------------------------------ cancellation ------------------------------
def test_stream_cancellation_stops_backend(monkeypatch):
    """Client disconnect (task cancellation) must trigger a backend stop for
    the in-flight request. TestClient/ASGITransport buffer streams, so the
    endpoint's StreamingResponse body_iterator is driven directly."""
    _mark_loaded(monkeypatch)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    stopped = []
    monkeypatch.setattr(
        "service.app.model_manager.stop_generation",
        lambda request_id=None: stopped.append(request_id) or {"status": "success"},
    )

    async def _endless_stream(**kwargs):
        yield {"chunk": "first", "done": False}
        while True:
            await asyncio.sleep(0.02)
            yield {"chunk": ".", "done": False}

    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _endless_stream)

    payload = json.dumps(_body(stream=True)).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
    }

    async def _receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def _run():
        resp = await app_module.openai_chat_completions(Request(scope, _receive))
        received = []

        async def _consume():
            async for chunk in resp.body_iterator:
                received.append(chunk)

        task = asyncio.create_task(_consume())
        deadline = time.time() + 5
        while len(received) < 3 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert len(received) >= 3, "stream never started"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    assert stopped, "stop_generation was not called on client disconnect"


# ------------------------------ concurrency ------------------------------
def test_concurrent_mixed_stream_and_non_stream(monkeypatch):
    """3 stream + 3 non-stream requests in flight together: every response
    carries its own marker (no cross-talk) and executions genuinely overlap
    (max_active >= 2 given MAX_CONCURRENT_GENERATIONS defaults to 8)."""
    _mark_loaded(monkeypatch)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: True)

    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def _marker_from(kwargs):
        return kwargs["prompt"][-1]["content"]

    async def _track(coro_body):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            return await coro_body
        finally:
            with lock:
                state["active"] -= 1

    async def _fake_agenerate(**kwargs):
        return await _track(_agen_body(_marker_from(kwargs)))

    async def _agen_body(marker):
        await asyncio.sleep(0.05)
        return {"result": f"echo:{marker}", "total_tokens": 1}

    async def _fake_stream(**kwargs):
        marker = _marker_from(kwargs)
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            for i in range(3):
                await asyncio.sleep(0.05)
                yield {"chunk": f"{marker}-{i}", "done": False}
            yield {"chunk": "", "done": True}
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)
    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_stream)

    async def _run():
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as ac:

            async def _non_stream(marker):
                r = await ac.post("/v1/chat/completions", json=_body(marker))
                assert r.status_code == 200
                return marker, r.json()["choices"][0]["message"]["content"]

            async def _stream(marker):
                r = await ac.post(
                    "/v1/chat/completions", json=_body(marker, stream=True)
                )
                assert r.status_code == 200
                deltas = [
                    ((p.get("choices") or [{}])[0].get("delta") or {}).get("content")
                    or ""
                    for p in _sse_payloads(r.text.splitlines())
                ]
                return marker, "".join(deltas)

            tasks = []
            for i in range(3):
                tasks.append(_non_stream(f"ns-{i}"))
                tasks.append(_stream(f"st-{i}"))
            return await asyncio.gather(*tasks)

    outputs = asyncio.run(_run())
    assert len(outputs) == 6
    all_markers = [m for m, _ in outputs]
    for marker, text in outputs:
        assert marker in text, f"{marker} missing from its own response: {text!r}"
        for other in all_markers:
            if other != marker:
                assert other not in text, f"cross-talk: {other} leaked into {marker}"
    assert state["max_active"] >= 2, "requests never overlapped"
