"""A failed training run must stay visible through /training/status and error_details."""

import queue

import pytest

from service.training.training_process import TrainingProcessManager

pytestmark = pytest.mark.finetune


class _FakeProcess:
    def __init__(self, alive: bool):
        self.alive = alive
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        pass

    def kill(self):
        self.alive = False


def _manager(alive: bool, reports: list[dict]) -> TrainingProcessManager:
    mgr = TrainingProcessManager()
    mgr.process = _FakeProcess(alive)
    mgr.status_q = queue.Queue()
    for r in reports:
        mgr.status_q.put(r)
    mgr.current_status = "running"
    mgr.current_session_id = "job-1"
    return mgr


_QUANTIZE_FAILED = {
    "status": "error",
    "error": "llama-quantize binary not found.",
    "traceback": "Traceback ... FileNotFoundError",
}


def test_error_reported_by_a_worker_that_already_exited_is_kept():
    """
    Found live: training succeeded, post-training Q4_K_M conversion failed, and the
    worker reported that and exited. The status poll saw a dead worker first, took
    it for a silent crash and ran cleanup(), which reset to idle and threw the queued
    report away -- /training/status said idle/error=null and error_details said no
    error occurred, although the job log held the traceback.
    """
    mgr = _manager(alive=False, reports=[_QUANTIZE_FAILED])

    status = mgr.get_status()
    assert status.status == "error"
    assert status.is_training is False
    assert status.error == "llama-quantize binary not found."

    details = mgr.get_error_details()
    assert details is not None
    assert details["error"] == "llama-quantize binary not found."
    assert details["error_traceback"] == "Traceback ... FileNotFoundError"

    # Polling again must not lose it either.
    assert mgr.get_status().status == "error"


def test_error_reported_by_a_live_worker_is_not_masked_as_stopped():
    # stop_training() marks the run "stopped"; the error has to win.
    mgr = _manager(alive=True, reports=[_QUANTIZE_FAILED])
    process = mgr.process

    status = mgr.get_status()
    assert process.terminated is True
    assert status.status == "error"
    assert status.error == "llama-quantize binary not found."


def test_worker_that_dies_without_a_report_is_an_error():
    mgr = _manager(alive=False, reports=[])

    status = mgr.get_status()
    assert status.status == "error"
    assert status.error == "Worker process terminated unexpectedly"
    assert mgr.get_error_details() is not None


def test_completed_run_is_not_turned_into_an_error():
    mgr = _manager(alive=False, reports=[{"status": "completed"}])

    status = mgr.get_status()
    assert status.status == "completed"
    assert status.error is None
    assert mgr.get_error_details() is None
