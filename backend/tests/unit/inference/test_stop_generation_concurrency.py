"""stop_generation behaviour for the async server-engine path, single and
multi concurrency.

Exercises the REAL AsyncServerClient.generate_stream abort logic (stop_event
checked per streamed event) through ModelManager, against stubbed OpenAI
streams — no network.

Passing tests document behaviour that works today and must keep working:
  - targeted stop (exact request_id) aborts a single in-flight stream
  - targeted stop during prompt processing aborts before any content
  - with concurrent streams, stopping one leaves the other untouched
  - the stop-event registry is empty once streams finish

Global stop: stop_generation(request_id=None) broadcasts
to every registered async stop event before falling through to the worker,
matching dev's worker-wide stop flag. The global-stop tests below are
permanent regression tests.
"""
import asyncio
from types import SimpleNamespace

import pytest

from service.config_models import InferenceConfig, InferenceEngine
from service.inference.async_server_client import client_for_config
from service.model_manager import ModelManager

pytestmark = pytest.mark.inference

TOTAL_CHUNKS = 20
CHUNK_DELAY_S = 0.01
WAIT_TIMEOUT_S = 5.0


@pytest.fixture()
def mm():
    """A private (non-singleton) ModelManager instance."""
    instance = object.__new__(ModelManager)
    ModelManager.__init__(instance)
    return instance


def _cfg(engine):
    return InferenceConfig(model_name="demo", engine=engine)


# ----------------------------- stubbed OpenAI stream -----------------------------
class _Stream:
    def __init__(self, events, delays):
        self._events = list(events)
        self._delays = list(delays)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        await asyncio.sleep(self._delays.pop(0))
        return self._events.pop(0)

    async def close(self):
        self.closed = True


def _delta(content):
    return SimpleNamespace(content=content, tool_calls=None)


def _events(n=TOTAL_CHUNKS):
    events = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=_delta(f"t{i}"), finish_reason=None)],
            usage=None,
        )
        for i in range(n)
    ]
    events.append(
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=n, total_tokens=10 + n),
        )
    )
    return events


class _FakeOpenAI:
    """Each create() call returns a fresh, independent stream."""

    def __init__(self, first_event_delay=CHUNK_DELAY_S):
        self._first_event_delay = first_event_delay
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)

    async def _create(self, **kwargs):
        events = _events()
        # first delay simulates prompt processing; the usage tail event is instant
        delays = [self._first_event_delay] + [CHUNK_DELAY_S] * (len(events) - 2) + [0]
        return _Stream(events, delays)

    async def _list(self, timeout=None):
        return SimpleNamespace(data=[SimpleNamespace(id="stub-model")])

    async def close(self):
        pass


def _make_manager(mm, monkeypatch, first_event_delay=CHUNK_DELAY_S):
    client = client_for_config(_cfg(InferenceEngine.LLAMA_SERVER), model="m")
    client.client = _FakeOpenAI(first_event_delay=first_event_delay)
    monkeypatch.setattr(mm, "is_loaded", lambda: True)
    monkeypatch.setattr(mm, "_get_async_client", lambda: client)
    monkeypatch.setattr(
        mm.inference_process,
        "stop_generation",
        lambda request_id=None: {"status": "success", "message": "worker stop (no-op)"},
    )
    return mm


async def _consume(mm, request_id, sink):
    async for item in mm.agenerate_stream(prompt="hi", request_id=request_id):
        sink.append(item)


def _content_count(sink):
    return sum(1 for i in sink if i.get("chunk"))


async def _wait_until(predicate, what):
    deadline = asyncio.get_running_loop().time() + WAIT_TIMEOUT_S
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, f"timeout waiting for {what}"
        await asyncio.sleep(0.005)


# ------------------------------ single concurrency ------------------------------
def test_targeted_stop_aborts_single_inflight_stream(mm, monkeypatch):
    _make_manager(mm, monkeypatch)

    async def scenario():
        sink = []
        task = asyncio.create_task(_consume(mm, "req-a", sink))
        await _wait_until(lambda: _content_count(sink) >= 3, "first chunks")

        result = mm.stop_generation(request_id="req-a")
        await asyncio.wait_for(task, WAIT_TIMEOUT_S)
        return sink, result

    sink, result = asyncio.run(scenario())

    assert result["status"] == "success" and "async" in result["message"]
    done = sink[-1]
    assert done.get("done") is True and done.get("stopped") is True
    assert _content_count(sink) < TOTAL_CHUNKS, "stream ran to completion despite stop"
    assert mm._async_stop_events == {}, "stop-event registry not cleaned up"


def test_targeted_stop_during_prompt_processing(mm, monkeypatch):
    # first event (= end of prefill) arrives late; stop is issued before it
    _make_manager(mm, monkeypatch, first_event_delay=0.3)

    async def scenario():
        sink = []
        task = asyncio.create_task(_consume(mm, "req-a", sink))
        await _wait_until(lambda: "req-a" in mm._async_stop_events, "request registration")
        await asyncio.sleep(0.02)  # still inside the simulated prefill

        result = mm.stop_generation(request_id="req-a")
        await asyncio.wait_for(task, WAIT_TIMEOUT_S)
        return sink, result

    sink, result = asyncio.run(scenario())

    assert result["status"] == "success"
    done = sink[-1]
    assert done.get("done") is True and done.get("stopped") is True
    assert _content_count(sink) == 0, "content leaked out after a prefill-time stop"


# ------------------------------ multi concurrency ------------------------------
def test_concurrent_streams_targeted_stop_is_isolated(mm, monkeypatch):
    _make_manager(mm, monkeypatch)

    async def scenario():
        sink_a, sink_b = [], []
        task_a = asyncio.create_task(_consume(mm, "req-a", sink_a))
        task_b = asyncio.create_task(_consume(mm, "req-b", sink_b))
        await _wait_until(
            lambda: _content_count(sink_a) >= 3 and _content_count(sink_b) >= 3,
            "both streams to produce chunks",
        )

        result = mm.stop_generation(request_id="req-a")
        await asyncio.wait_for(asyncio.gather(task_a, task_b), WAIT_TIMEOUT_S)
        return sink_a, sink_b, result

    sink_a, sink_b, result = asyncio.run(scenario())

    assert result["status"] == "success"
    # stopped stream: aborted early, flagged
    done_a = sink_a[-1]
    assert done_a.get("stopped") is True
    assert _content_count(sink_a) < TOTAL_CHUNKS
    # sibling stream: untouched, ran to completion
    done_b = sink_b[-1]
    assert done_b.get("done") is True and done_b.get("stopped") is not True
    assert _content_count(sink_b) == TOTAL_CHUNKS
    assert mm._async_stop_events == {}


# ------------------------------- global stop -------------------------------
def test_global_stop_aborts_single_inflight_stream(mm, monkeypatch):
    _make_manager(mm, monkeypatch)

    async def scenario():
        sink = []
        task = asyncio.create_task(_consume(mm, "req-a", sink))
        await _wait_until(lambda: _content_count(sink) >= 2, "first chunks")

        mm.stop_generation(request_id=None)
        await asyncio.wait_for(task, WAIT_TIMEOUT_S)
        return sink

    sink = asyncio.run(scenario())

    done = sink[-1]
    assert done.get("stopped") is True, "global stop did not abort the async stream"
    assert _content_count(sink) < TOTAL_CHUNKS


def test_global_stop_aborts_all_concurrent_streams(mm, monkeypatch):
    _make_manager(mm, monkeypatch)

    async def scenario():
        sink_a, sink_b = [], []
        task_a = asyncio.create_task(_consume(mm, "req-a", sink_a))
        task_b = asyncio.create_task(_consume(mm, "req-b", sink_b))
        await _wait_until(
            lambda: _content_count(sink_a) >= 2 and _content_count(sink_b) >= 2,
            "both streams to produce chunks",
        )

        mm.stop_generation(request_id=None)
        await asyncio.wait_for(asyncio.gather(task_a, task_b), WAIT_TIMEOUT_S)
        return sink_a, sink_b

    sink_a, sink_b = asyncio.run(scenario())

    for name, sink in (("req-a", sink_a), ("req-b", sink_b)):
        done = sink[-1]
        assert done.get("stopped") is True, f"{name} was not aborted by global stop"
        assert _content_count(sink) < TOTAL_CHUNKS, f"{name} ran to completion"
