import os
import sys
import threading
import time
import unittest
from collections import deque
from queue import Queue
from unittest.mock import MagicMock

import pytest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from service.inference.engines.llama_server_engine import LlamaServerEngine
from service.inference.model_inference_process import ModelInferenceProcess

pytestmark = pytest.mark.inference


def _drive_stream_expecting_error(process_manager, request_id, feed, params=None):
    """Consume generate_stream in a thread and inject a response mid-flight.

    The dispatcher refactor routes responses through per-request queues that
    generate_stream registers on first iteration, so test payloads must be
    injected only after the request is registered. ``feed(process_manager)``
    runs at that point; the raised exception (if any) is returned.
    """
    captured = {}

    def _consume():
        try:
            next(
                process_manager.generate_stream(
                    prompt="hello",
                    params=params or {"total_timeout": 5},
                    request_id=request_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the exception is the result
            captured["exc"] = exc

    thread = threading.Thread(target=_consume)
    thread.start()
    deadline = time.time() + 2
    while not process_manager._is_request_active(request_id) and time.time() < deadline:
        time.sleep(0.01)
    feed(process_manager)
    thread.join(timeout=5)
    return captured.get("exc")


class TestLlamaServerRuntimeErrorHandling(unittest.TestCase):
    def test_normalize_base_url_strips_v1_suffix(self):
        engine = LlamaServerEngine(
            status_queue=Queue(),
            data_queue=Queue(),
            stop_event=MagicMock(),
            stop_generation_flag=MagicMock(),
        )

        self.assertEqual(
            engine._normalize_base_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080",
        )

    def test_build_runtime_error_payload_marks_crash_as_fatal(self):
        engine = LlamaServerEngine(
            status_queue=Queue(),
            data_queue=Queue(),
            stop_event=MagicMock(),
            stop_generation_flag=MagicMock(),
        )
        engine.base_url = "http://127.0.0.1:65530"
        engine.managed_process = True
        engine.server_process = MagicMock()
        engine.server_process.poll.return_value = 134
        engine._stderr_recent_lines = deque(
            [
                "srv  process_chun: processing image...",
                "encoding image slice...",
                "CUDA error: out of memory",
                "cuMemCreate(&handle, reserve_size, &prop, 0)",
            ],
            maxlen=120,
        )

        payload = engine.build_runtime_error_payload(
            RuntimeError("Server disconnected without sending a response.")
        )

        self.assertTrue(payload["fatal"])
        self.assertTrue(payload["is_oom"])
        self.assertEqual(payload["error_type"], "LlamaServerOOM")
        self.assertIn("exited with code 134", payload["error"])
        self.assertIn("CUDA error: out of memory", payload["error"])

    def test_generate_stream_fatal_error_updates_manager_state(self):
        process_manager = ModelInferenceProcess()
        process_manager.current_status = "ready"
        process_manager.request_queue = Queue()
        process_manager.data_queue = Queue()
        process_manager.process = MagicMock()
        process_manager.process.is_alive.return_value = True
        process_manager._start_dispatcher()

        def _feed(manager):
            manager.data_queue.put(
                {
                    "type": "error",
                    "request_id": "req-runtime-fatal",
                    "error": "llama-server process exited with code 134",
                    "error_type": "LlamaServerOOM",
                    "error_traceback": "CUDA error: out of memory",
                    "is_oom": True,
                    "recoverable": False,
                    "fatal": True,
                }
            )

        try:
            exc = _drive_stream_expecting_error(
                process_manager, "req-runtime-fatal", _feed
            )
        finally:
            process_manager._stop_dispatcher()

        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(process_manager.current_status, "error")
        self.assertEqual(process_manager.loading_error, "llama-server process exited with code 134")
        self.assertEqual(process_manager.error_type, "LlamaServerOOM")
        self.assertEqual(process_manager.error_traceback, "CUDA error: out of memory")
        self.assertTrue(process_manager.is_oom_error)

    def test_get_error_details_consumes_status_queue_error(self):
        process_manager = ModelInferenceProcess()
        process_manager.current_status = "ready"
        process_manager.status_queue = Queue()
        process_manager.status_queue.put(
            {
                "status": "error",
                "error": "llama-server crashed",
                "error_type": "LlamaServerProcessExited",
                "error_traceback": "recent stderr excerpt",
                "is_oom": True,
            }
        )
        process_manager.process = MagicMock()
        process_manager.process.is_alive.return_value = True

        details = process_manager.get_error_details()

        self.assertIsNotNone(details)
        self.assertEqual(details["error"], "llama-server crashed")
        self.assertEqual(details["error_type"], "LlamaServerProcessExited")
        self.assertEqual(details["error_traceback"], "recent stderr excerpt")
        self.assertTrue(details["is_oom"])

    def test_generate_stream_receives_unload_interrupt_error(self):
        process_manager = ModelInferenceProcess()
        process_manager.current_status = "ready"
        process_manager.request_queue = Queue()
        process_manager.data_queue = Queue()
        process_manager.process = MagicMock()
        process_manager.process.is_alive.return_value = True

        def _feed(manager):
            # Delivers straight into the per-request queue generate_stream
            # registered; no dispatcher involvement (mirrors unload flow).
            manager._notify_active_requests(
                "Generation interrupted because model was unloaded",
                error_type="ModelUnloaded",
            )

        exc = _drive_stream_expecting_error(process_manager, "req-unload", _feed)

        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("model was unloaded", str(exc))

    def test_stop_managed_server_drains_threads_and_waits_for_shutdown(self):
        engine = LlamaServerEngine(
            status_queue=Queue(),
            data_queue=Queue(),
            stop_event=MagicMock(),
            stop_generation_flag=MagicMock(),
        )

        proc = MagicMock()
        proc.poll.return_value = None
        stdout_thread = MagicMock()
        stdout_thread.is_alive.side_effect = [True, False]
        stderr_thread = MagicMock()
        stderr_thread.is_alive.side_effect = [True, False]

        engine.managed_process = True
        engine.server_process = proc
        engine._stdout_pump_thread = stdout_thread
        engine._stderr_pump_thread = stderr_thread
        engine._wait_for_server_shutdown = MagicMock(return_value=True)

        engine._stop_managed_server()

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        stdout_thread.join.assert_called_once_with(timeout=2.0)
        stderr_thread.join.assert_called_once_with(timeout=2.0)
        engine._wait_for_server_shutdown.assert_called_once_with(timeout_sec=5.0)
        self.assertIsNone(engine.server_process)
        self.assertFalse(engine.managed_process)
        self.assertIsNone(engine._stdout_pump_thread)
        self.assertIsNone(engine._stderr_pump_thread)

    def test_stop_managed_server_drains_threads_when_process_already_exited(self):
        engine = LlamaServerEngine(
            status_queue=Queue(),
            data_queue=Queue(),
            stop_event=MagicMock(),
            stop_generation_flag=MagicMock(),
        )

        proc = MagicMock()
        proc.poll.return_value = 0
        stdout_thread = MagicMock()
        stdout_thread.is_alive.return_value = False
        stderr_thread = MagicMock()
        stderr_thread.is_alive.return_value = False

        engine.managed_process = True
        engine.server_process = proc
        engine._stdout_pump_thread = stdout_thread
        engine._stderr_pump_thread = stderr_thread
        engine._wait_for_server_shutdown = MagicMock()

        engine._stop_managed_server()

        proc.terminate.assert_not_called()
        proc.wait.assert_not_called()
        engine._wait_for_server_shutdown.assert_not_called()
        self.assertIsNone(engine._stdout_pump_thread)
        self.assertIsNone(engine._stderr_pump_thread)
        self.assertIsNone(engine.server_process)
        self.assertFalse(engine.managed_process)

    def test_stop_process_skips_join_thread_for_ipc_queues(self):
        process_manager = ModelInferenceProcess()
        process_manager.process = MagicMock()
        process_manager.process.pid = 12345
        process_manager.process.is_alive.return_value = False

        request_queue = MagicMock()
        status_queue = MagicMock()
        data_queue = MagicMock()
        request_queue.join_thread.side_effect = AssertionError("join_thread should not be called")
        status_queue.join_thread.side_effect = AssertionError("join_thread should not be called")
        data_queue.join_thread.side_effect = AssertionError("join_thread should not be called")

        process_manager.request_queue = request_queue
        process_manager.status_queue = status_queue
        process_manager.data_queue = data_queue

        process_manager.stop_process("Generation interrupted because model was unloaded")

        request_queue.cancel_join_thread.assert_called_once()
        status_queue.cancel_join_thread.assert_called_once()
        data_queue.cancel_join_thread.assert_called_once()
        request_queue.close.assert_called_once()
        status_queue.close.assert_called_once()
        data_queue.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()