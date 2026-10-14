"""Acceptance tests for the dev -> refactor/inference-concurrency migration gaps.

Each test encodes the behaviour the queue path had and the direct async HTTP path
(service/inference/async_server_client.py) has to keep. Every gap is closed, so all of
these are permanent regression tests — nothing is xfail, and a gap that comes back
fails the suite instead of being tolerated.

Two of them landed narrower than the queue path, because this client cannot see what
the engine could; each says so where it is implemented, in build_chat_payload and
classify_server_error.

Reference for the expected behaviour:
  git show dev:service/inference/generate/llama_server_runner.py
  git show dev:service/inference/engines/llama_server_engine.py  (prepare_request_for_generation)
"""
import asyncio
from types import SimpleNamespace

import pytest

import service.model_manager as mm_module
from service.config_models import InferenceConfig, InferenceEngine
from service.inference.async_server_client import client_for_config
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


# ----------------------------- stubbed OpenAI client -----------------------------
class _Stream:
    def __init__(self, events, delays=None):
        self._events = list(events)
        self._delays = list(delays or [0] * len(self._events))
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        delay = self._delays.pop(0)
        if delay:
            await asyncio.sleep(delay)
        return self._events.pop(0)

    async def close(self):
        self.closed = True


class _FakeOpenAI:
    """Stub AsyncOpenAI: records create() kwargs, returns handler result."""

    def __init__(self, handler):
        self.captured = []
        self._handler = handler
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)

    async def _create(self, **kwargs):
        self.captured.append(kwargs)
        result = self._handler(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    async def _list(self, timeout=None):
        return SimpleNamespace(data=[SimpleNamespace(id="stub-model")])

    async def close(self):
        pass


def _delta(content=None, reasoning=None):
    return SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)


def _event(delta=None, usage=None):
    choices = [SimpleNamespace(delta=delta, finish_reason=None)] if delta else []
    return SimpleNamespace(choices=choices, usage=usage)


def _usage(prompt, completion, total):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


def _completion(content="ok", reasoning=None, usage=None):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=usage or _usage(1, 1, 2),
    )


def _server_client(engine, handler, model="m"):
    """Real AsyncServerClient built for the engine, with a stubbed OpenAI client."""
    client = client_for_config(_cfg(engine), model=model)
    client.client = _FakeOpenAI(handler)
    return client


async def _collect_stream(client, params):
    items = []
    async for item in client.generate_stream([{"role": "user", "content": "hi"}], params):
        items.append(item)
    return items


# --------------------------- reasoning_content forwarding ---------------------------
# reasoning deltas are forwarded on their own key, never merged into content.
def test_stream_forwards_reasoning_on_its_own_key():
    events = [
        _event(_delta(reasoning="step one of thinking")),
        _event(_delta(reasoning=" step two")),
        _event(_delta(content="Final answer")),
        _event(usage=_usage(100, 50, 150)),
    ]
    client = _server_client(
        InferenceEngine.LLAMA_SERVER, lambda kw: _Stream(list(events))
    )
    items = asyncio.run(_collect_stream(client, {"enable_thinking": True}))
    content = "".join(i.get("chunk", "") for i in items)
    reasoning = "".join(i.get("reasoning", "") for i in items)

    assert reasoning == "step one of thinking step two"
    assert content == "Final answer"
    assert "<think>" not in content and "</think>" not in content


def test_stream_reasoning_only_response_still_forwards_it():
    """A turn that produces nothing but thinking must not arrive empty.

    It used to be forwarded as <think> content so the stream was non-empty;
    now it arrives on the reasoning key instead, which callers read separately.
    """
    events = [
        _event(_delta(reasoning="only thinking, no final content")),
        _event(usage=_usage(10, 5, 15)),
    ]
    client = _server_client(
        InferenceEngine.LLAMA_SERVER,
        lambda kw: _Stream(list(events)),
        model="Qwen3-4B-Instruct",
    )
    # enable_thinking intentionally absent: reasoning is forwarded regardless.
    items = asyncio.run(_collect_stream(client, {}))
    reasoning = "".join(i.get("reasoning", "") for i in items)
    assert reasoning == "only thinking, no final content"


def test_generate_returns_reasoning_content_separately():
    client = _server_client(
        InferenceEngine.LLAMA_SERVER,
        lambda kw: _completion(content="Final answer", reasoning="my chain of thought"),
    )
    res = asyncio.run(
        client.generate([{"role": "user", "content": "hi"}], {"enable_thinking": True})
    )
    assert res["reasoning_content"] == "my chain of thought"
    assert res["result"] == "Final answer"
    assert "<think>" not in res["result"]


# --------------------------- penalty param name per engine ---------------------------
# build_chat_payload names the penalty per engine; this guards the end-to-end llama path.
def test_llama_payload_uses_native_repeat_penalty_name():
    client = _server_client(InferenceEngine.LLAMA_SERVER, lambda kw: _completion())
    asyncio.run(
        client.generate([{"role": "user", "content": "hi"}], {"repetition_penalty": 1.1})
    )
    extra_body = client.client.captured[0].get("extra_body") or {}
    assert extra_body.get("repeat_penalty") == 1.1


def test_vllm_payload_keeps_repetition_penalty_name():
    """Control: vLLM's native name IS repetition_penalty — must not be renamed."""
    client = _server_client(InferenceEngine.VLLM, lambda kw: _completion())
    asyncio.run(
        client.generate([{"role": "user", "content": "hi"}], {"repetition_penalty": 1.1})
    )
    extra_body = client.client.captured[0].get("extra_body") or {}
    assert extra_body.get("repetition_penalty") == 1.1
    assert "repeat_penalty" not in extra_body


# --------------------------- Qwen thinking tag injection ---------------------------
# _apply_thinking_tag appends ' /no_think' (or ' /think') to the last user message
# for Qwen models.
def test_qwen_no_think_tag_appended_to_last_user_message():
    client = _server_client(
        InferenceEngine.LLAMA_SERVER, lambda kw: _completion(), model="Qwen3-4B-Instruct"
    )
    asyncio.run(
        client.generate([{"role": "user", "content": "hello"}], {"enable_thinking": False})
    )
    messages = client.client.captured[0]["messages"]
    assert str(messages[-1]["content"]).rstrip().endswith("/no_think")


# --------------------------- client caching / close lifecycle ---------------------------
# _get_async_client compares the resolved endpoint
# (no throwaway probe clients) and queues replaced clients so the next
# _acquire_async_client awaits their aclose. Permanent regression tests.
def _track_aclose(client):
    client._test_closed = False
    orig = client.aclose

    async def tracked():
        client._test_closed = True
        await orig()

    client.aclose = tracked
    return client


def test_get_async_client_does_not_leak_discarded_clients(mm, monkeypatch):
    built = []
    orig = mm_module.client_for_config

    def counting(config, **kw):
        client = _track_aclose(orig(config, **kw))
        built.append(client)
        return client

    monkeypatch.setattr(mm_module, "client_for_config", counting)
    mm.config = _cfg(InferenceEngine.LLAMA_SERVER)

    async def _run():
        c1 = mm._get_async_client()
        c2 = mm._get_async_client()
        assert c1 is c2
        assert len(built) == 1, "unchanged endpoint must not build new clients"
        discarded = [c for c in built if c is not mm._async_client]
        try:
            assert not discarded
        finally:
            for c in built:
                await c.client.close()

    asyncio.run(_run())


def test_engine_swap_closes_previous_client(mm):
    mm.config = _cfg(InferenceEngine.LLAMA_SERVER)

    async def _run():
        old = _track_aclose(await mm._acquire_async_client())
        mm.config = _cfg(InferenceEngine.VLLM)
        new = await mm._acquire_async_client()
        assert new is not old
        try:
            assert old._test_closed, "previous engine's client was never closed"
        finally:
            await old.client.close()
            await new.client.close()

    asyncio.run(_run())


# --------------------------- prompt-cache reuse across turns ---------------------------
def test_llama_payload_requests_prompt_cache():
    client = _server_client(InferenceEngine.LLAMA_SERVER, lambda kw: _completion())
    asyncio.run(client.generate([{"role": "user", "content": "hi"}], {}))
    extra_body = client.client.captured[0].get("extra_body") or {}
    # id_slot pinning needs the engine's slot bookkeeping, which this client has no
    # handle on; llama-server still prefix-matches across slots without it.
    assert extra_body.get("cache_prompt") is True


# --------------------------- tps must not include the other phase ---------------------------
def test_gen_tps_excludes_prompt_processing_time():
    prefill_s = 0.25
    events = [
        _event(_delta(content="a")),
        _event(_delta(content="b")),
        _event(usage=_usage(300, 30, 330)),
    ]
    client = _server_client(
        InferenceEngine.LLAMA_SERVER,
        lambda kw: _Stream(list(events), delays=[prefill_s, 0, 0]),
    )
    done = asyncio.run(_collect_stream(client, {}))[-1]
    # decode itself is near-instant here: decode-based tps is huge, while
    # total-elapsed-based tps is capped around 30/0.25 = 120
    assert done["gen_tps"] > 300, f"gen_tps={done['gen_tps']:.0f} includes prefill time"


def test_prompt_tps_excludes_decode_time():
    events = [
        _event(_delta(content="a")),
        _event(_delta(content="b")),
        _event(_delta(content="c")),
        _event(usage=_usage(300, 30, 330)),
    ]
    # first token almost immediately, then a slow decode phase
    client = _server_client(
        InferenceEngine.LLAMA_SERVER,
        lambda kw: _Stream(list(events), delays=[0.02, 0.15, 0.15, 0]),
    )
    done = asyncio.run(_collect_stream(client, {}))[-1]
    # dev-style: 300 / ~0.02s >> 3000; total-elapsed-based caps around 300/0.32 = 940
    assert done["prompt_tps"] > 3000, (
        f"prompt_tps={done['prompt_tps']:.0f} includes decode time"
    )


# --------------------------- structured runtime-error classification ---------------------------
def test_server_crash_error_carries_classification():
    client = _server_client(
        InferenceEngine.LLAMA_SERVER,
        lambda kw: RuntimeError("Server disconnected without sending a response."),
    )
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client.generate([{"role": "user", "content": "hi"}], {}))
    payload = getattr(excinfo.value, "error_payload", None)
    assert isinstance(payload, dict)
    assert payload.get("error_type")
    assert "is_oom" in payload


# --------------------------- images must reach the payload ---------------------------
def test_agenerate_images_reach_the_payload(mm, monkeypatch):
    image_url = "data:image/png;base64,AAAA"
    client = _server_client(InferenceEngine.LLAMA_SERVER, lambda kw: _completion())
    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: client)

    asyncio.run(mm.agenerate(prompt="what is in the picture", images=[image_url]))

    messages = client.client.captured[0]["messages"]
    last_user = next(m for m in reversed(messages) if m.get("role") == "user")
    content = last_user.get("content")
    assert isinstance(content, list), "images were not merged into multi-part content"
    urls = [
        (p.get("image_url") or {}).get("url")
        for p in content
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    assert image_url in urls
