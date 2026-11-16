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
    return ServiceManager(), killed


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
