"""ServiceManager.stop: only a service that answers with our fingerprint is killed."""

from launcher import service_manager
from launcher.service_manager import ServiceManager


def _manager(monkeypatch, status):
    killed = []
    monkeypatch.setattr(ServiceManager, "port", property(lambda self: 8571))
    monkeypatch.setattr(service_manager, "is_port_free", lambda port: False)
    monkeypatch.setattr(ServiceManager, "_wait_for_port", lambda self, seconds: False)
    monkeypatch.setattr(service_manager, "listening_pid", lambda port: 4242)
    monkeypatch.setattr(service_manager, "probe", lambda port: status)
    monkeypatch.setattr(service_manager, "terminate_pid", lambda pid: killed.append(pid))
    # No real request to a port a test does not own.
    monkeypatch.setattr(service_manager, "unload_model", lambda port, timeout: True)
    return ServiceManager(), killed


class _Process:
    """A service this launcher started: it exits when terminated, after `wait` is called."""

    pid = 999

    def __init__(self):
        self.alive = True
        self.waited: list[float] = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout):
        self.waited.append(timeout)

    def kill(self):
        self.killed = True
        self.alive = False


def test_unload_is_asked_for_before_the_service_is_terminated(monkeypatch):
    manager, _ = _manager(monkeypatch, "ours")
    order = []
    monkeypatch.setattr(service_manager, "unload_model",
                        lambda port, timeout: order.append(("unload", port, timeout)) or True)
    process = _Process()
    process.terminate = lambda: order.append(("terminate",)) or setattr(process, "alive", False)
    manager._process = process
    manager.stop()
    assert [step[0] for step in order] == ["unload", "terminate"]
    assert order[0][1] == 8571
    # Within the backend's own 120 s wait for an engine, so that it can finish.
    assert order[0][2] == service_manager.UNLOAD_REQUEST_SECONDS


def test_a_program_that_is_not_ours_is_not_asked_to_unload(monkeypatch):
    manager, _ = _manager(monkeypatch, "foreign")
    asked = []
    monkeypatch.setattr(service_manager, "unload_model", lambda port, timeout: asked.append(port))
    manager.stop()
    assert asked == []


def test_a_failed_unload_still_stops_the_service(monkeypatch):
    manager, _ = _manager(monkeypatch, "ours")
    monkeypatch.setattr(service_manager, "unload_model", lambda port, timeout: False)
    process = _Process()
    manager._process = process
    manager.stop()
    assert process.terminated


def test_the_wait_after_terminate_outlasts_the_backends_engine_wait(monkeypatch):
    """The 20 s that used to cut a Strata unload off: the backend waits up to 120 s."""
    manager, _ = _manager(monkeypatch, "ours")
    process = _Process()
    manager._process = process
    manager.stop()
    assert process.waited and process.waited[0] > 120.0
    assert service_manager.STOP_SECONDS > 120.0
    assert service_manager.GRACEFUL_STOP_SECONDS > 120.0


def test_time_spent_unloading_comes_out_of_the_wait(monkeypatch):
    manager, _ = _manager(monkeypatch, "ours")
    clock = iter([1000.0, 1100.0])  # the deadline is set at 1000; 100 s have gone when read again
    monkeypatch.setattr(service_manager.time, "monotonic", lambda: next(clock))
    process = _Process()
    manager._process = process
    manager.stop(timeout=130.0)
    assert 29.0 < process.waited[0] <= 30.0


def test_unload_model_reports_a_refused_connection_as_false():
    assert service_manager.unload_model(1, timeout=0.5) is False


def test_foreign_program_on_the_port_is_left_alone(monkeypatch):
    manager, killed = _manager(monkeypatch, "foreign")
    manager.stop()
    assert killed == []


def test_our_service_still_holding_the_port_is_stopped(monkeypatch):
    manager, killed = _manager(monkeypatch, "ours")
    manager.stop()
    assert killed == [4242]


def test_graceful_exit_reaches_a_hidden_console_process(tmp_path):
    """Ctrl+Break lets a console process shut itself down (Windows launcher stop path)."""
    import subprocess
    import sys
    import time

    import pytest

    from launcher import config

    if not config.IS_WINDOWS:
        pytest.skip("Windows console events")
    marker = tmp_path / "graceful.txt"
    child = (
        "import signal, sys, time\n"
        "def h(*a):\n"
        "    open(sys.argv[1], 'w').write('ok'); sys.exit(0)\n"
        "signal.signal(signal.SIGBREAK, h)\n"
        "for _ in range(300): time.sleep(0.2)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", child, str(marker)],
                            creationflags=config.subprocess_flags())
    try:
        time.sleep(1.5)
        assert service_manager.request_graceful_exit(proc.pid)
        proc.wait(timeout=10)
    finally:
        proc.kill()
    assert marker.read_text() == "ok"
