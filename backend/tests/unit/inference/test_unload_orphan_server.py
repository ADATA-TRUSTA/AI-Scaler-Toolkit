"""Unload waits for the engine, and a server the worker leaves behind is stopped (F-0086).

The worker used to get 7.5 s to unload before it was terminated. A Strata engine saving a
large conversation and stopping its server took longer, so the worker died mid-unload and
the server, in a session of its own, kept its RAM and VRAM after the API said the model
was unloaded. These tests pin the two halves of the fix without a real model: the manager
waits for the engine's "unloaded" report, and whatever server is still running once the
worker is gone is stopped from the API process -- real child processes stand in for it.
"""

import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest
from fastapi.testclient import TestClient

import service.app as app_module
import service.inference.model_inference_process as mip_module
from service.inference.model_inference_process import (
    ModelInferenceProcess,
    stop_orphaned_server,
)
from service.utils.process_identity import process_identity

pytestmark = pytest.mark.inference

posix_only = pytest.mark.skipif(os.name == "nt", reason="sessions and process groups are POSIX")


@pytest.fixture()
def mip():
    """A manager wired to thread queues and a worker that looks alive."""
    m = ModelInferenceProcess()
    m.current_status = "ready"
    m.status_queue = queue.Queue()
    sent: list = []
    m.request_queue = SimpleNamespace(put=lambda item, **kw: sent.append(item))
    m._sent_commands = sent  # test-only backdoor to inspect worker commands
    m.process = SimpleNamespace(is_alive=lambda: True, pid=0)
    return m


# ------------------------------ waiting for the engine ------------------------------
def test_unload_waits_for_the_engines_report_and_no_longer(mip, monkeypatch):
    monkeypatch.setattr(mip_module, "UNLOAD_WAIT_SECONDS", 10.0)
    threading.Timer(0.6, lambda: mip.status_queue.put({"status": "unloaded"})).start()

    started = time.monotonic()
    mip._send_unload_and_wait()
    elapsed = time.monotonic() - started

    assert mip._sent_commands == [{"command": "unload"}]
    # past the report, and well short of the wait's bound
    assert 0.5 <= elapsed < 5
    assert mip.current_status == "unloaded"


def test_an_engine_slower_than_the_old_budget_is_still_waited_for(mip, monkeypatch):
    """The old 2.5 s sleep plus 5 s join is what cut a large Strata unload short."""
    monkeypatch.setattr(mip_module, "UNLOAD_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(mip_module.time, "sleep", lambda s: None)
    clock = {"now": 0.0}

    def monotonic():
        clock["now"] += 0.5  # every poll moves the clock half a second
        if clock["now"] >= 12.0 and mip.status_queue.empty():
            mip.status_queue.put({"status": "unloaded"})
        return clock["now"]

    monkeypatch.setattr(mip_module.time, "monotonic", monotonic)
    mip._send_unload_and_wait()
    assert mip._unloaded_reports == 1
    assert clock["now"] >= 12.0


def test_unload_gives_up_waiting_after_the_bound(mip, monkeypatch):
    monkeypatch.setattr(mip_module, "UNLOAD_WAIT_SECONDS", 0.5)
    started = time.monotonic()
    mip._send_unload_and_wait()
    assert 0.5 <= time.monotonic() - started < 3
    assert mip._unloaded_reports == 0


def test_unload_stops_waiting_when_the_worker_dies(mip, monkeypatch):
    monkeypatch.setattr(mip_module, "UNLOAD_WAIT_SECONDS", 10.0)
    mip.current_status = "idle"  # nothing to mark failed: only the wait is under test
    alive = {"value": True}
    mip.process = SimpleNamespace(is_alive=lambda: alive["value"], pid=0, join=lambda timeout: None)
    threading.Timer(0.3, lambda: alive.update(value=False)).start()

    started = time.monotonic()
    mip._send_unload_and_wait()
    assert time.monotonic() - started < 3


def test_unload_is_sent_once(mip, monkeypatch):
    monkeypatch.setattr(mip_module, "UNLOAD_WAIT_SECONDS", 0.2)
    mip._send_unload_and_wait()
    mip._send_unload_and_wait()
    assert mip._sent_commands == [{"command": "unload"}]


# ------------------------------ stopping what is left ------------------------------
# A stand-in for Strata's serve/server.py: it starts a child (the engine), prints the
# child's PID, and exits when a line arrives on stdin, leaving the child running. On POSIX
# the child gets a process group of its own in the server's session, as the real engine
# does (measured: engine PGID 1971055, server PGID and both SIDs 1971050) -- so stopping
# the server's group alone would miss it.
_SERVER = r"""
import os, subprocess, sys
kwargs = {"process_group": 0} if os.name != "nt" else {}
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], **kwargs)
print(child.pid, flush=True)
sys.stdin.readline()
"""


def _start_server(own_session: bool) -> tuple[subprocess.Popen, int, float]:
    kwargs = {"start_new_session": True} if own_session else {}
    server = subprocess.Popen(
        [sys.executable, "-c", _SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        **kwargs,
    )
    assert server.stdout is not None
    child = int(server.stdout.readline())
    return server, child, process_identity(server.pid)


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _kill(*pids: int) -> None:
    for pid in pids:
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass


@posix_only
def test_an_engine_whose_server_already_exited_is_reached_through_the_session():
    """What F-0086 left: server.py gone or going, the engine still holding the memory."""
    server, engine, create_time = _start_server(own_session=True)
    try:
        # the layout that makes the session, not the group, the thing to stop
        assert os.getpgid(engine) != os.getpgid(server.pid)
        assert os.getsid(engine) == server.pid
        assert server.stdin is not None
        server.stdin.write("\n")
        server.stdin.flush()
        server.wait(timeout=10)  # the server is gone; its engine is not
        assert not _gone(engine)

        stopped = stop_orphaned_server(server.pid, create_time, own_session=True)

        assert engine in stopped
        _wait_gone(engine)
        assert _gone(engine)
    finally:
        _kill(engine)


@posix_only
def test_a_running_server_and_its_engine_are_both_stopped():
    server, engine, create_time = _start_server(own_session=True)
    try:
        stopped = stop_orphaned_server(server.pid, create_time, own_session=True)
        server.wait(timeout=10)
        assert {server.pid, engine} <= set(stopped)
        assert _gone(engine)
    finally:
        _kill(server.pid, engine)


def test_a_server_without_its_own_session_is_stopped_with_its_children():
    server, engine, create_time = _start_server(own_session=False)
    try:
        stopped = stop_orphaned_server(server.pid, create_time, own_session=False)
        server.wait(timeout=10)
        assert {server.pid, engine} <= set(stopped)
        assert _gone(engine)
    finally:
        _kill(server.pid, engine)


def test_a_reused_pid_is_left_alone():
    """Another process holding the PID now has another start time, and is not ours."""
    server, engine, create_time = _start_server(own_session=False)
    try:
        assert stop_orphaned_server(server.pid, create_time - 100, own_session=False) == []
        assert not _gone(server.pid)
        assert not _gone(engine)
    finally:
        _kill(server.pid, engine)
        server.wait(timeout=10)


@posix_only
def test_a_session_number_now_led_by_another_process_is_left_alone():
    """The old session ended and its number went to a new leader: none of it is ours."""
    server, engine, create_time = _start_server(own_session=True)
    try:
        assert stop_orphaned_server(server.pid, create_time - 100, own_session=True) == []
        assert not _gone(server.pid)
    finally:
        _kill(server.pid, engine)
        server.wait(timeout=10)


def _wait_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_nothing_left_is_nothing_to_do():
    server, engine, create_time = _start_server(own_session=False)
    _kill(server.pid, engine)
    server.wait(timeout=10)
    _wait_gone(engine)
    assert stop_orphaned_server(server.pid, create_time, own_session=False) == []


# ------------------------------ wired into the manager ------------------------------
def _report(server: subprocess.Popen, create_time: float, own_session: bool) -> dict:
    return {
        "status": "server_started",
        "pid": server.pid,
        "create_time": create_time,
        "own_session": own_session,
    }


def test_stop_process_stops_a_server_the_worker_left(mip):
    own_session = os.name != "nt"
    server, engine, create_time = _start_server(own_session=own_session)
    try:
        # reported by the engine, not yet read: stop_process reads it before it reaps
        mip.status_queue.put(_report(server, create_time, own_session))
        mip.process = SimpleNamespace(is_alive=lambda: False, pid=0)  # the worker is gone

        mip.stop_process()

        server.wait(timeout=10)
        assert _gone(engine)
        assert mip._engine_servers == []
    finally:
        _kill(server.pid, engine)


def test_a_crashed_worker_does_not_leave_its_server_running(mip):
    own_session = os.name != "nt"
    server, engine, create_time = _start_server(own_session=own_session)
    try:
        mip.request_queue = queue.Queue()  # drained by the cleanup
        mip.status_queue.put({"status": "loading"})
        mip.status_queue.put(_report(server, create_time, own_session))
        mip.process = SimpleNamespace(is_alive=lambda: False, pid=0, join=lambda timeout: None)

        mip._cleanup_dead_process()

        server.wait(timeout=10)
        assert _gone(engine)
    finally:
        _kill(server.pid, engine)


def test_a_server_the_engine_stopped_itself_is_not_touched_again(mip):
    """The normal unload: by the time the worker is stopped there is nothing left."""
    server, engine, create_time = _start_server(own_session=False)
    _kill(server.pid, engine)
    server.wait(timeout=10)
    _wait_gone(engine)
    mip.status_queue.put(_report(server, create_time, False))
    mip.process = SimpleNamespace(is_alive=lambda: False, pid=0)

    mip.stop_process()

    assert mip._engine_servers == []


# ------------------------------ the unload route ------------------------------
def test_the_unload_route_ends_running_generations_before_it_unloads(monkeypatch):
    """A generation still running made Strata's unload wait for it (37 s for 2,500 tokens)."""
    order: list = []
    manager = app_module.model_manager
    monkeypatch.setattr(manager, "abort_generations", lambda: order.append("abort") or True)
    monkeypatch.setattr(
        manager, "unload_model", lambda: order.append("unload") or {"status": "success"}
    )
    # a client on this machine: other machines only get inference (test_remote_access.py)
    client = TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")

    response = client.post("/inference/unload_model")

    assert response.status_code == 200
    assert order == ["abort", "unload"]
