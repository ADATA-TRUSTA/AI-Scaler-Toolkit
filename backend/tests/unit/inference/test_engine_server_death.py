"""Regression: a crashed engine server must not leave the model reported as loaded."""

import subprocess
import sys

import pytest

from service.inference.model_inference_process import ModelInferenceProcess
from service.utils.process_identity import process_identity


def _ready(monkeypatch):
    m = ModelInferenceProcess()
    monkeypatch.setattr(m, "update_status", lambda: None)
    m.current_status = "ready"
    return m


def test_loaded_while_the_engine_server_runs(monkeypatch):
    m = _ready(monkeypatch)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        m._engine_servers = [(proc.pid, process_identity(proc.pid), False)]
        assert m.is_loaded()
    finally:
        proc.kill()
        proc.wait()


def test_not_loaded_once_the_engine_server_exits(monkeypatch):
    m = _ready(monkeypatch)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    ct = process_identity(proc.pid)
    m._engine_servers = [(proc.pid, ct, False)]
    proc.kill()
    proc.wait()
    assert not m.is_loaded()
    assert m.current_status == "error"
    assert m.error_type == "EngineServerExited"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_wall_clock_step_does_not_make_a_live_server_look_dead(monkeypatch):
    """psutil's create_time moves with the boot time; a 3 s step must not read as a crash."""
    m = _ready(monkeypatch)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        m._engine_servers = [(proc.pid, process_identity(proc.pid), False)]
        import psutil._pslinux as pslinux

        real = pslinux.boot_time
        monkeypatch.setattr(pslinux, "boot_time", lambda: real() + 3.0)
        assert m.is_loaded()
    finally:
        proc.kill()
        proc.wait()
