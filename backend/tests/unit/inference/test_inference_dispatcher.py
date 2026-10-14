"""Unit tests for the ModelInferenceProcess dispatcher demux.

A single dispatcher thread is the only reader of the shared data_queue and
fans each response out to the owning request's private queue. These tests pin
routing, FIFO order, cross-talk isolation, lifecycle and the generate /
generate_stream consumer paths — all without a real multiprocessing worker
(a thread queue satisfies the ``get(timeout=...)`` interface the loop uses).
"""
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from service.inference.model_inference_process import ModelInferenceProcess

pytestmark = pytest.mark.inference


@pytest.fixture()
def mip():
    """A ModelInferenceProcess wired to fake queues; dispatcher always stopped."""
    m = ModelInferenceProcess()
    m.current_status = "ready"
    m.data_queue = queue.Queue()
    sent = []
    m.request_queue = SimpleNamespace(put=lambda item, **kw: sent.append(item))
    m._sent_commands = sent  # test-only backdoor to inspect worker commands
    m.process = SimpleNamespace(is_alive=lambda: True, pid=0)
    yield m
    m._stop_dispatcher()


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ------------------------------ routing ------------------------------
def test_dispatcher_routes_by_request_id_no_crosstalk(mip):
    mip._start_dispatcher()
    qa = mip._register_request("a")
    qb = mip._register_request("b")

    interleaved = [
        {"request_id": "a", "type": "stream_chunk", "chunk": "a1"},
        {"request_id": "b", "type": "stream_chunk", "chunk": "b1"},
        {"request_id": "a", "type": "stream_chunk", "chunk": "a2"},
        {"request_id": "b", "type": "stream_chunk", "chunk": "b2"},
    ]
    for item in interleaved:
        mip.data_queue.put(item)

    assert _wait_until(lambda: qa.qsize() == 2 and qb.qsize() == 2)
    assert [qa.get_nowait()["chunk"] for _ in range(2)] == ["a1", "a2"]
    assert [qb.get_nowait()["chunk"] for _ in range(2)] == ["b1", "b2"]


def test_dispatcher_drops_unknown_and_missing_request_id(mip):
    mip._start_dispatcher()
    qa = mip._register_request("a")

    mip.data_queue.put({"request_id": "ghost", "type": "stream_chunk", "chunk": "x"})
    mip.data_queue.put({"type": "stream_chunk", "chunk": "y"})  # no request_id
    mip.data_queue.put({"request_id": "a", "type": "stream_chunk", "chunk": "ok"})

    assert _wait_until(lambda: qa.qsize() == 1)
    assert qa.get_nowait()["chunk"] == "ok"
    # the drops must not have killed the loop
    mip.data_queue.put({"request_id": "a", "type": "stream_chunk", "chunk": "ok2"})
    assert _wait_until(lambda: qa.qsize() == 1)


def test_unregister_stops_delivery(mip):
    mip._start_dispatcher()
    qa = mip._register_request("a")
    qb = mip._register_request("b")
    assert mip._is_request_active("a") and mip._is_request_active("b")

    mip._unregister_request("a")
    assert not mip._is_request_active("a")

    mip.data_queue.put({"request_id": "a", "type": "stream_chunk", "chunk": "late"})
    mip.data_queue.put({"request_id": "b", "type": "stream_chunk", "chunk": "b1"})

    assert _wait_until(lambda: qb.qsize() == 1)
    assert qa.qsize() == 0


def test_notify_active_requests_broadcasts_shutdown_error(mip):
    qa = mip._register_request("a")
    qb = mip._register_request("b")

    mip._notify_active_requests("worker stopping", error_type="ProcessInterrupted")

    for rid, q in (("a", qa), ("b", qb)):
        item = q.get_nowait()
        assert item["type"] == "error"
        assert item["request_id"] == rid
        assert item["error"] == "worker stopping"
        assert item["error_type"] == "ProcessInterrupted"
        assert item["recoverable"] is True and item["fatal"] is False


# ------------------------------ lifecycle ------------------------------
def test_dispatcher_lifecycle_restart_and_stop(mip):
    mip._start_dispatcher()
    first = mip._dispatcher_thread
    assert first is not None and first.is_alive()

    mip._start_dispatcher()  # restart replaces the thread
    second = mip._dispatcher_thread
    assert second is not None and second is not first and second.is_alive()
    assert _wait_until(lambda: not first.is_alive())

    mip._stop_dispatcher()
    assert mip._dispatcher_thread is None
    assert _wait_until(lambda: not second.is_alive())


def test_dispatcher_exits_when_data_queue_is_none(mip):
    mip.data_queue = None
    mip._start_dispatcher()
    thread = mip._dispatcher_thread
    assert _wait_until(lambda: not thread.is_alive())


def test_start_dispatcher_clears_stale_registrations(mip):
    mip._register_request("stale")
    mip._start_dispatcher()
    assert not mip._is_request_active("stale")


# ------------------------------ generate ------------------------------
def test_generate_full_path_result_and_cleanup(mip):
    mip._start_dispatcher()
    result_box = {}

    def _consume():
        result_box["out"] = mip.generate(
            "hi", {"total_timeout": 5}, request_id="r1"
        )

    t = threading.Thread(target=_consume)
    t.start()
    assert _wait_until(lambda: mip._is_request_active("r1"))

    # registration must precede the worker command
    assert mip._sent_commands and mip._sent_commands[0]["command"] == "generate"
    assert mip._sent_commands[0]["request_id"] == "r1"

    mip.data_queue.put(
        {
            "type": "result",
            "request_id": "r1",
            "result": "ok",
            "gen_tokens": 3,
            "prompt_tokens": 2,
            "total_tokens": 5,
            "finish_reason": "stop",
        }
    )
    t.join(timeout=5)
    out = result_box["out"]
    assert out["result"] == "ok"
    assert out["gen_tokens"] == 3 and out["total_tokens"] == 5
    assert out["finish_reason"] == "stop"
    assert not mip._is_request_active("r1")  # unregistered after completion


def test_generate_error_fatal_vs_recoverable(mip):
    mip._start_dispatcher()

    def _run(request_id, payload):
        box = {}

        def _consume():
            try:
                mip.generate("hi", {"total_timeout": 5}, request_id=request_id)
            except Exception as e:  # noqa: BLE001
                box["exc"] = e

        t = threading.Thread(target=_consume)
        t.start()
        assert _wait_until(lambda: mip._is_request_active(request_id))
        mip.data_queue.put({**payload, "request_id": request_id})
        t.join(timeout=5)
        return box.get("exc")

    exc = _run(
        "fatal-1",
        {"type": "error", "error": "boom", "fatal": True, "error_type": "X"},
    )
    assert isinstance(exc, RuntimeError) and "boom" in str(exc)
    assert mip.current_status == "error"

    mip.current_status = "ready"
    exc = _run(
        "soft-1",
        {
            "type": "error",
            "error": "soft oom",
            "is_oom": True,
            "recoverable": True,
            "fatal": False,
        },
    )
    assert isinstance(exc, RuntimeError) and "soft oom" in str(exc)
    assert mip.current_status == "ready"  # recoverable must not flip status


def test_generate_raises_when_worker_dies(mip):
    mip._start_dispatcher()
    mip.process = SimpleNamespace(is_alive=lambda: False, pid=0)

    with pytest.raises(RuntimeError, match="died during generation"):
        mip.generate("hi", {"total_timeout": 5}, request_id="dead-1")
    assert mip.current_status == "error"


def test_generate_total_timeout(mip):
    mip._start_dispatcher()
    start = time.time()
    with pytest.raises(TimeoutError):
        mip.generate("hi", {"total_timeout": 0.3}, request_id="slow-1")
    assert time.time() - start < 5


# ------------------------------ generate_stream ------------------------------
def test_generate_stream_two_concurrent_consumers_interleaved(mip):
    """Two threads consume streams while chunks arrive interleaved on the
    shared data_queue: each consumer must see only its own chunks, in order."""
    mip._start_dispatcher()
    results = {"s1": [], "s2": []}
    done_payloads = {}

    def _consume(rid):
        for item in mip.generate_stream("hi", {"total_timeout": 10}, request_id=rid):
            if item.get("done"):
                done_payloads[rid] = item
            elif item.get("chunk"):
                results[rid].append(item["chunk"])

    t1 = threading.Thread(target=_consume, args=("s1",))
    t2 = threading.Thread(target=_consume, args=("s2",))
    t1.start()
    t2.start()
    assert _wait_until(
        lambda: mip._is_request_active("s1") and mip._is_request_active("s2")
    )

    feed = [
        {"request_id": "s1", "type": "stream_chunk", "chunk": "c1", "done": False},
        {"request_id": "s2", "type": "stream_chunk", "chunk": "d1", "done": False},
        {"request_id": "s1", "type": "stream_chunk", "chunk": "c2", "done": False},
        {"request_id": "s2", "type": "stream_chunk", "chunk": "d2", "done": False},
        {
            "request_id": "s1",
            "type": "stream_chunk",
            "chunk": "",
            "done": True,
            "gen_tokens": 2,
            "total_tokens": 4,
        },
        {
            "request_id": "s2",
            "type": "stream_chunk",
            "chunk": "",
            "done": True,
            "gen_tokens": 2,
            "total_tokens": 4,
        },
    ]
    for item in feed:
        mip.data_queue.put(item)

    t1.join(timeout=5)
    t2.join(timeout=5)
    assert results["s1"] == ["c1", "c2"]
    assert results["s2"] == ["d1", "d2"]
    assert done_payloads["s1"]["gen_tokens"] == 2
    assert done_payloads["s2"]["total_tokens"] == 4
    assert not mip._is_request_active("s1") and not mip._is_request_active("s2")


def test_generate_stream_command_sent_after_registration(mip):
    mip._start_dispatcher()

    def _consume():
        for _ in mip.generate_stream("hi", {"total_timeout": 5}, request_id="sc-1"):
            pass

    t = threading.Thread(target=_consume)
    t.start()
    assert _wait_until(lambda: len(mip._sent_commands) == 1)
    assert mip._sent_commands[0]["command"] == "generate_stream"
    assert mip._is_request_active("sc-1")  # registered before/at command time

    mip.data_queue.put(
        {"request_id": "sc-1", "type": "stream_chunk", "chunk": "", "done": True}
    )
    t.join(timeout=5)
    assert not t.is_alive()
