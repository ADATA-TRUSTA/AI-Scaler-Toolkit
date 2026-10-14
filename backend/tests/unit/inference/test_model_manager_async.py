"""Unit tests for ModelManager's async server-engine path.

Covers the uses_async_http gate, async-client caching/invalidation, the
per-request async stop-event registry, the dual-path stop_generation, and
agenerate / agenerate_stream delegation — all against stubbed clients.

ModelManager is a process-wide singleton; tests construct a private instance
via object.__new__ so the module-level ``model_manager`` is never mutated.
"""
import asyncio
from types import SimpleNamespace

import pytest

import service.model_manager as mm_module
from service.config_models import InferenceConfig, InferenceEngine
from service.model_manager import ModelManager

pytestmark = pytest.mark.inference


@pytest.fixture()
def mm():
    """A private (non-singleton) ModelManager instance."""
    instance = object.__new__(ModelManager)
    ModelManager.__init__(instance)
    return instance


def _cfg(engine):
    return InferenceConfig(model_name="demo", engine=engine)


# ------------------------------ gate & client cache ------------------------------
def test_uses_async_http_gate(mm, monkeypatch):
    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    for engine, expected in (
        (InferenceEngine.LLAMA_SERVER, True),
        (InferenceEngine.VLLM, True),
        (InferenceEngine.TRANSFORMERS, False),
    ):
        mm.config = _cfg(engine)
        assert mm.uses_async_http() is expected, engine

    monkeypatch.setattr(mm, "is_loaded", lambda: False)
    mm.config = _cfg(InferenceEngine.LLAMA_SERVER)
    assert mm.uses_async_http() is False  # not loaded -> False regardless


def test_get_async_client_cache_and_invalidation(mm, monkeypatch):
    built = []

    def _fake_client_for_config(config, **kw):
        client = SimpleNamespace(base_url=(config.llama_server_url or "http://a") + "/v1")
        built.append(client)
        return client

    monkeypatch.setattr(mm_module, "client_for_config", _fake_client_for_config)

    mm.config = _cfg(InferenceEngine.LLAMA_SERVER)
    mm.config.llama_server_url = "http://a"

    first = mm._get_async_client()
    second = mm._get_async_client()
    assert first is second  # same endpoint -> cached instance
    # endpoint compare goes through resolve_server_endpoint: no probe clients
    assert len(built) == 1

    mm.config.llama_server_url = "http://b"
    third = mm._get_async_client()
    assert third is not first
    assert third.base_url == "http://b/v1"
    # the replaced client is queued for aclose on the next acquire
    assert first in mm._stale_async_clients


def test_get_async_client_requires_config(mm):
    mm.config = None
    with pytest.raises(RuntimeError, match="No model configuration"):
        mm._get_async_client()


def test_unload_model_drops_cached_client(mm, monkeypatch):
    mm.config = _cfg(InferenceEngine.LLAMA_SERVER)
    mm._async_client = SimpleNamespace(base_url="http://a/v1")
    mm._async_client_endpoint = {"base_url": "http://a/v1", "api_key": "EMPTY"}

    monkeypatch.setattr(
        mm.inference_process, "get_status", lambda: {"loaded": True, "is_loading": False}
    )
    monkeypatch.setattr(mm.inference_process, "unload_model", lambda: None)

    out = mm.unload_model()
    assert out["status"] == "success"
    assert mm._async_client is None
    assert mm._async_client_endpoint is None
    assert mm.config is None


# ------------------------------ stop registry ------------------------------
def test_async_stop_registry(mm):
    event = mm._register_async_stop("r1")
    assert isinstance(event, asyncio.Event) and not event.is_set()

    assert mm._signal_async_stop("r1") is True
    assert event.is_set()

    mm._unregister_async_stop("r1")
    assert mm._signal_async_stop("r1") is False

    # None request_id: register/unregister are no-ops; signal with nothing
    # registered reports False
    assert mm._register_async_stop(None) is None
    assert mm._signal_async_stop(None) is False
    mm._unregister_async_stop(None)  # must not raise

    # signal without request_id broadcasts to every registered request
    e1 = mm._register_async_stop("r2")
    e2 = mm._register_async_stop("r3")
    assert mm._signal_async_stop(None) is True
    assert e1.is_set() and e2.is_set()
    mm._unregister_async_stop("r2")
    mm._unregister_async_stop("r3")


def test_stop_generation_dual_path(mm, monkeypatch):
    worker_calls = []
    monkeypatch.setattr(
        mm.inference_process,
        "stop_generation",
        lambda request_id=None: worker_calls.append(request_id)
        or {"status": "success", "message": "worker stop"},
    )

    # (a) registered async request: resolved without touching the worker
    mm._register_async_stop("async-1")
    out = mm.stop_generation(request_id="async-1")
    assert out["status"] == "success"
    assert "async" in out["message"]
    assert worker_calls == []

    # (b) unknown id falls through to the worker path
    out = mm.stop_generation(request_id="worker-1")
    assert out["message"] == "worker stop"
    assert worker_calls == ["worker-1"]


def test_stop_generation_stop_all_signals_async_and_worker(mm, monkeypatch):
    worker_calls = []
    monkeypatch.setattr(
        mm.inference_process,
        "stop_generation",
        lambda request_id=None: worker_calls.append(request_id)
        or {"status": "success", "message": "worker stop"},
    )

    event = mm._register_async_stop("s1")
    out = mm.stop_generation(request_id=None)
    # stop-all reaches both paths: async events signalled AND worker stopped
    assert event.is_set()
    assert worker_calls == [None]
    assert out["status"] == "success"
    mm._unregister_async_stop("s1")


def test_stop_generation_stop_all_succeeds_without_worker(mm, monkeypatch):
    monkeypatch.setattr(
        mm.inference_process,
        "stop_generation",
        lambda request_id=None: {"status": "error", "message": "No active worker process"},
    )

    event = mm._register_async_stop("s2")
    out = mm.stop_generation(request_id=None)
    # the async signal succeeded, so the worker-side error must not surface
    assert event.is_set()
    assert out["status"] == "success"
    mm._unregister_async_stop("s2")


# ------------------------------ delegation ------------------------------
def test_as_messages_normalisation(mm):
    assert ModelManager._as_messages("hi", None) == [{"role": "user", "content": "hi"}]
    assert ModelManager._as_messages("hi", "sys") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    msgs = [{"role": "user", "content": "q"}]
    assert ModelManager._as_messages(msgs, None) == msgs
    assert ModelManager._as_messages(msgs, "sys")[0] == {
        "role": "system",
        "content": "sys",
    }


def test_agenerate_delegates(mm, monkeypatch):
    captured = {}

    class _StubClient:
        async def generate(self, messages, params, request_id=None):
            captured["messages"] = messages
            captured["params"] = params
            captured["request_id"] = request_id
            return {"result": "ok", "gen_tokens": 1}

    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: _StubClient())

    out = asyncio.run(
        mm.agenerate(
            prompt="hi",
            system_prompt="sys",
            max_new_tokens=7,
            temperature=0.1,
            request_id="rq",
        )
    )
    assert out == {"result": "ok", "gen_tokens": 1}
    assert captured["request_id"] == "rq"
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}
    assert captured["params"]["max_new_tokens"] == 7
    assert captured["params"]["temperature"] == 0.1


def test_agenerate_requires_loaded_model(mm, monkeypatch):
    monkeypatch.setattr(mm, "is_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="not loaded"):
        asyncio.run(mm.agenerate(prompt="hi"))


def test_agenerate_stop_aborts_non_stream(mm, monkeypatch):
    started = asyncio.Event()

    class _StubClient:
        async def generate(self, messages, params, request_id=None):
            started.set()
            await asyncio.sleep(30)
            return {"result": "never"}

    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: _StubClient())

    async def _run():
        task = asyncio.create_task(mm.agenerate(prompt="hi", request_id="stop-me"))
        await started.wait()
        out = mm.stop_generation(request_id="stop-me")
        assert out["status"] == "success"
        return await task

    result = asyncio.run(_run())
    assert result.get("stopped") is True
    assert result["result"] == ""
    assert "stop-me" not in mm._async_stop_events  # unregistered on exit


def test_agenerate_stream_stop_event_lifecycle(mm, monkeypatch):
    seen = {}

    class _StubClient:
        async def generate_stream(self, messages, params, request_id=None, stop_event=None):
            seen["stop_event"] = stop_event
            yield {"chunk": "a", "done": False}
            yield {"chunk": "", "done": True, "gen_tokens": 1}

    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: _StubClient())

    async def _run():
        items = []
        async for item in mm.agenerate_stream(prompt="hi", request_id="rs"):
            # mid-iteration: the registry must hold this request's event,
            # and it must be the very object handed to the client
            assert "rs" in mm._async_stop_events
            assert mm._async_stop_events["rs"] is seen["stop_event"]
            items.append(item)
        return items

    items = asyncio.run(_run())
    assert [i.get("chunk") for i in items] == ["a", ""]
    assert "rs" not in mm._async_stop_events  # unregistered on completion


def test_agenerate_stream_unregisters_on_error(mm, monkeypatch):
    class _StubClient:
        async def generate_stream(self, messages, params, request_id=None, stop_event=None):
            yield {"chunk": "a", "done": False}
            raise RuntimeError("mid-stream boom")

    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: _StubClient())

    async def _run():
        async for _ in mm.agenerate_stream(prompt="hi", request_id="err-1"):
            pass

    with pytest.raises(RuntimeError, match="mid-stream boom"):
        asyncio.run(_run())
    assert "err-1" not in mm._async_stop_events  # cleaned up by finally
