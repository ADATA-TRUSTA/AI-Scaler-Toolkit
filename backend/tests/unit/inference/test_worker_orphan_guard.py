"""Regression: the worker must notice a vanished parent and exit on its own."""

import subprocess
import sys
from unittest import mock

import psutil
import pytest

from service.inference.model_inference_process import _parent_is_alive
from service.utils.process_identity import process_identity

pytestmark = pytest.mark.inference


def test_parent_is_alive_for_current_process():
    me = psutil.Process()
    assert _parent_is_alive(me.pid, process_identity(me.pid))


def test_parent_is_alive_false_for_dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not _parent_is_alive(proc.pid, None)


def test_parent_is_alive_false_on_pid_reuse():
    me = psutil.Process()
    identity = process_identity(me.pid)
    assert identity is not None
    assert not _parent_is_alive(me.pid, identity - 1000)


def test_alive_then_dead_transition():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    identity = process_identity(proc.pid)
    assert _parent_is_alive(proc.pid, identity)
    proc.kill()
    proc.wait()
    assert not _parent_is_alive(proc.pid, identity)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_identity_ignores_wall_clock_steps() -> None:
    """A clock step moves psutil's create_time (boot time + ticks) but not the identity."""
    import psutil._pslinux as pslinux

    me = psutil.Process()
    identity = process_identity(me.pid)
    before = psutil.Process(me.pid).create_time()
    real = pslinux.boot_time
    with mock.patch.object(pslinux, "boot_time", lambda: real() + 3.0):
        assert psutil.Process(me.pid).create_time() == pytest.approx(before + 3.0)
        assert process_identity(me.pid) == identity
        assert _parent_is_alive(me.pid, identity)
