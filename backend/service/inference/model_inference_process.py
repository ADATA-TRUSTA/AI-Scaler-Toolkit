"""
Model Inference Process - Separate process for model loading and inference to avoid OOM issues
Uses multiprocessing to isolate model operations and ensure proper VRAM cleanup on process termination.
"""

import os
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty
from queue import Queue as ThreadQueue
from typing import Any, cast

import torch

from ..config_models import InferenceConfig, InferenceEngine

# Import settings BEFORE torch/transformers (sets HF_HOME environment variable)
from ..settings import configure_logging
from .engines.llama_server_engine import LlamaServerEngine

# Import engines
from .engines.transformers_engine import TransformersEngine
from .engines.vllm_engine import VllmEngine
from .model_utils import _cleanup_memory

logger = configure_logging(__name__)


def _model_worker_process(
    request_queue: Queue,
    status_queue: Queue,
    data_queue: Queue,
    stop_event: EventClass,
    stop_generation_flag: EventClass,
) -> None:
    """
    Model worker process - handles model loading and inference requests.

    Request format:
        {"command": "load", "config": {...}}
        {"command": "generate", "request_id": "...", "prompt": "...", "params": {...}}
        {"command": "generate_stream", "request_id": "...", "prompt": "...", "params": {...}}
        {"command": "unload"}

    Response format (separate queues):
        status_queue: status updates {"status": "loading|ready|error", ...}
        data_queue: inference results, stream chunks, errors {"type": "result|stream_chunk|error", ...}
    """
    engine = None

    try:
        logger.info("[Worker] Model worker process started")

        while not stop_event.is_set():
            try:
                # Wait for a request with a timeout so the stop signal can be checked
                try:
                    request = request_queue.get(timeout=1.0)
                except Empty:
                    continue

                command = request.get("command")

                if command == "load":
                    try:
                        status_queue.put({"status": "loading", "stage": "config"})
                        config_dict = request.get("config", {})
                        config = InferenceConfig(**config_dict)

                        # Clean up previous engine if exists
                        if engine:
                            try:
                                stop_generation_flag.set()
                                engine.unload()
                            except Exception:
                                pass
                            engine = None

                        if config.engine == InferenceEngine.LLAMA_SERVER:
                            logger.info("[Worker] Selecting LlamaServerEngine")
                            engine = LlamaServerEngine(
                                status_queue,
                                data_queue,
                                stop_event,
                                stop_generation_flag,
                            )
                        elif config.engine == InferenceEngine.VLLM:
                            logger.info("[Worker] Selecting VllmEngine")
                            engine = VllmEngine(
                                status_queue,
                                data_queue,
                                stop_event,
                                stop_generation_flag,
                            )
                        else:
                            logger.info("[Worker] Selecting TransformersEngine")
                            engine = TransformersEngine(
                                status_queue,
                                data_queue,
                                stop_event,
                                stop_generation_flag,
                            )

                        engine.load_model(config)

                    except Exception as e:
                        logger.exception(f"[Worker] Failed to load model: {e}")
                        error_traceback = traceback.format_exc()
                        logger.exception(error_traceback)

                        # Cleanup execution
                        if engine:
                            try:
                                stop_generation_flag.set()
                                engine.unload()
                            except Exception:
                                pass
                            engine = None

                        _cleanup_memory()

                        error_str = str(e)
                        error_type = type(e).__name__

                        status_queue.put(
                            {
                                "status": "error",
                                "error": error_str,
                                "error_type": error_type,
                                "error_traceback": error_traceback,
                            }
                        )

                        error_str_lower = error_str.lower()
                        if (
                            "out of memory" in error_str_lower
                            or "oom" in error_str_lower
                            or "cuda" in error_str_lower
                        ):
                            logger.exception(
                                "[Worker] OOM/CUDA error detected, forcing process exit to release VRAM..."
                            )
                            status_queue.put(
                                {
                                    "status": "error",
                                    "error": f"OOM Error: {error_str}",
                                    "error_type": error_type,
                                    "is_oom": True,
                                    "message": "Process will exit to release GPU memory",
                                }
                            )
                            _cleanup_memory()
                            os._exit(1)

                elif command == "generate":
                    stop_generation_flag.clear()
                    if engine is None:
                        data_queue.put(
                            {
                                "type": "error",
                                "request_id": request.get("request_id"),
                                "error": "Engine not initialized/Model not loaded",
                            }
                        )
                        continue
                    # Only in-process engines (transformers) generate in the
                    # worker; server engines are served via direct async HTTP and
                    # would hit the BaseEngine error default here.
                    engine.generate(request)

                elif command == "generate_stream":
                    stop_generation_flag.clear()
                    if engine is None:
                        data_queue.put(
                            {
                                "type": "error",
                                "request_id": request.get("request_id"),
                                "error": "Engine not initialized/Model not loaded",
                            }
                        )
                        continue
                    engine.generate_stream(request)

                elif command == "stop_generation":
                    # The worker only serves in-process engines (transformers),
                    # which honour the global stop flag. Server engines are
                    # stopped in the main process via the async client's stop
                    # event and never reach here.
                    req_id = request.get("request_id")
                    stop_generation_flag.set()
                    logger.info(f"[Worker] Global stop_generation_flag set (request_id={req_id})")

                elif command == "unload":
                    logger.info("[Worker] Unload command received")
                    # Stop all in-flight generation first
                    stop_generation_flag.set()
                    if engine:
                        engine.unload()
                        engine = None
                    else:
                        status_queue.put(
                            {
                                "status": "unloaded",
                                "message": "Model unloaded (no active engine)",
                            }
                        )

                elif command == "apply_chat_template":
                    if engine:
                        engine.apply_chat_template(request)
                    else:
                        data_queue.put(
                            {
                                "type": "error",
                                "request_id": request.get("request_id"),
                                "error": "Model not loaded",
                            }
                        )

                elif command == "cleanup_generation_memory":
                    logger.info("[Worker] cleanup_generation_memory command received")
                    if engine:
                        if isinstance(engine, LlamaServerEngine):
                            engine.cleanup_generation_memory(request)
                        else:
                            # Other engines have no slot support; keep the global cleanup behavior
                            engine.cleanup_generation_memory()
                    else:
                        _cleanup_memory()
                        data_queue.put({"type": "cleanup", "result": "memory cleaned (no engine)"})

                else:
                    logger.warning(f"[Worker] Unknown command: {command}")

            except Exception as e:
                logger.exception(f"[Worker] Request handling error: {e}")
                logger.exception(traceback.format_exc())

        logger.info("[Worker] Stop event received, cleaning up...")

    except Exception as e:
        logger.exception(f"[Worker] Worker process error: {e}")
        logger.exception(traceback.format_exc())

    finally:
        if engine:
            try:
                stop_generation_flag.set()
                engine.unload()
            except Exception:
                pass
        _cleanup_memory()
        logger.info("[Worker] Model worker process terminated")
        os._exit(0)


class ModelInferenceProcess:
    """
    Model inference process manager.

    Manages a dedicated inference process so VRAM can be fully reclaimed on OOM.
    Uses a split-queue architecture:
    - status_queue: status updates (loading, ready, error, ...)
    - data_queue: inference results, stream chunks, error messages
    """

    def __init__(self) -> None:
        self.process: Process | None = None
        self.request_queue: Queue | None = None
        self.status_queue: Queue | None = None  # Dedicated to status updates
        self.data_queue: Queue | None = None  # Dedicated to inference results
        self.stop_event: EventClass | None = None  # Used to abort the whole process
        self.stop_generation_flag: EventClass | None = None  # Used to stop the current generation
        self.current_status = "idle"
        self.current_config: InferenceConfig | None = None
        self.device: str | None = None
        self.loading_error: str | None = None
        self.error_type: str | None = None
        self.error_traceback: str | None = None
        self.is_oom_error: bool = False
        self.last_stop_time: float | None = None  # Timestamp of the last stop_generation
        # Tracks whether an unload command was already sent to the worker, to avoid log noise
        self._unload_sent: bool = False
        # Device map statistics
        self.device_map_summary: str | None = None  # e.g. "cuda:0:30, cpu:10"
        self.total_modules: int | None = None  # Total module count
        self.layer_lines: list | None = None  # Sample layer placements
        # GPU memory usage (reported by the worker process)
        self.memory_usage: dict[str, float] | None = None
        # Per-request delivery queues. A single dispatcher thread is the only
        # reader of the shared multiprocessing ``data_queue``; it fans each
        # response out to the owning request's queue. Every consumer then
        # blocks solely on its own queue, which preserves per-request FIFO
        # order and removes the cross-consumer reordering / busy-spin that a
        # shared queue with multiple concurrent readers suffered from.
        self._request_queues: dict[str, ThreadQueue] = {}
        self._request_queues_lock = threading.Lock()
        self._dispatcher_thread: threading.Thread | None = None
        self._dispatcher_stop = threading.Event()

    def _register_request(self, request_id: str) -> ThreadQueue:
        """
        Create the per-request delivery queue *before* the request is sent.

        Registering first guarantees the dispatcher can always route the very
        first response (no window where a reply arrives before we are listening).
        """
        response_queue: ThreadQueue = ThreadQueue()
        with self._request_queues_lock:
            self._request_queues[request_id] = response_queue
        return response_queue

    def _unregister_request(self, request_id: str) -> None:
        with self._request_queues_lock:
            self._request_queues.pop(request_id, None)

    def _clear_request_queues(self) -> None:
        with self._request_queues_lock:
            self._request_queues.clear()

    def _is_request_active(self, request_id: str) -> bool:
        with self._request_queues_lock:
            return request_id in self._request_queues

    def _start_dispatcher(self) -> None:
        """Start the single reader thread that demuxes ``data_queue``."""
        self._stop_dispatcher()  # ensure any stale thread is gone
        self._clear_request_queues()
        self._dispatcher_stop.clear()
        thread = threading.Thread(
            target=self._dispatch_loop,
            name="inference-data-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread = thread
        thread.start()

    def _stop_dispatcher(self) -> None:
        self._dispatcher_stop.set()
        thread = self._dispatcher_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._dispatcher_thread = None

    def _dispatch_loop(self) -> None:
        """
        Single reader of ``data_queue``: route each response to its owner.

        Because there is exactly one reader, responses for a given request stay
        in the order the worker produced them. Responses whose owner is no
        longer registered (finished/aborted) are dropped, matching the previous
        best-effort semantics.
        """
        while not self._dispatcher_stop.is_set():
            data_queue = self.data_queue
            if data_queue is None:
                break
            try:
                response = data_queue.get(timeout=0.2)
            except Empty:
                continue
            except (ValueError, OSError, EOFError):
                # Queue was closed/disposed underneath us -> stop cleanly.
                break
            except Exception as e:
                logger.debug(f"[Dispatcher] error reading data_queue: {e}")
                break

            request_id = response.get("request_id") if isinstance(response, dict) else None
            if not request_id:
                logger.debug("[Dispatcher] dropping response without request_id")
                continue

            with self._request_queues_lock:
                response_queue = self._request_queues.get(request_id)
            if response_queue is None:
                # Owner already finished/aborted -> drop.
                continue
            response_queue.put(response)

    def _notify_active_requests(
        self,
        message: str,
        *,
        error_type: str = "ProcessInterrupted",
    ) -> None:
        # Deliver directly into the per-request queues rather than the shared
        # data_queue: this stays reliable even while the dispatcher is being
        # torn down, and cannot race with data_queue disposal.
        with self._request_queues_lock:
            targets = list(self._request_queues.items())
        if not targets:
            return

        for request_id, response_queue in targets:
            try:
                response_queue.put(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": message,
                        "error_type": error_type,
                        "fatal": False,
                        "recoverable": True,
                        "is_oom": False,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to notify active request {request_id} during shutdown: {e}")

    def _build_queue_unavailable_error(self, action: str) -> RuntimeError:
        if self.current_status == "idle":
            return RuntimeError(f"{action} interrupted because model was unloaded")
        return RuntimeError(f"{action} failed because worker IPC queue is unavailable")

    def _dispose_ipc_queue(self, attr_name: str) -> None:
        queue_obj = getattr(self, attr_name, None)
        if queue_obj is None:
            return

        try:
            queue_obj.cancel_join_thread()
        except Exception:
            pass

        try:
            queue_obj.close()
        except Exception as e:
            logger.debug(f"Error closing {attr_name}: {e}")

        setattr(self, attr_name, None)

    def start_process(self) -> None:
        """Start the worker process."""
        if self.process and self.process.is_alive():
            logger.warning("Worker process already running")
            return

        self._unload_sent = False

        # Create the IPC objects (split queues)
        self.request_queue = Queue()
        self.status_queue = Queue()  # Dedicated to status updates
        self.data_queue = Queue()  # Dedicated to inference results
        self.stop_event = Event()  # Init the event used to abort the whole process
        self.stop_generation_flag = Event()  # Init the stop-generation flag

        # Start the single dispatcher thread: it is the only reader of data_queue
        # and fans each response out to the owning request's queue.
        self._start_dispatcher()

        # Create and start the process (daemon=True ensures it exits with the main process)
        self.process = Process(
            target=_model_worker_process,
            args=(
                self.request_queue,
                self.status_queue,
                self.data_queue,
                self.stop_event,
                self.stop_generation_flag,
            ),
            daemon=True,
        )

        self.process.start()
        logger.info(f"Worker process started (PID: {self.process.pid})")

    def stop_process(self, interrupt_message: str | None = None) -> None:
        """Stop the worker process."""
        if not self.process:
            logger.info("No worker process to stop")
            return

        pid = self.process.pid if self.process else "N/A"
        logger.info(f"Stopping worker process (PID: {pid})...")

        self._notify_active_requests(
            interrupt_message or "Generation interrupted because worker process is stopping",
            error_type="ModelUnloaded" if interrupt_message else "ProcessInterrupted",
        )

        # If the process is still alive, try a graceful shutdown
        if self.process.is_alive():
            # 1. Send the unload command
            if not self._unload_sent:  # Send only once
                try:
                    if self.request_queue:
                        self.request_queue.put({"command": "unload"}, timeout=1)
                        self._unload_sent = True
                except Exception as e:
                    logger.warning(f"Failed to send unload command: {e}")

            # 2. Set the stop event
            try:
                if self.stop_event:
                    self.stop_event.set()
            except Exception as e:
                logger.warning(f"Failed to set stop event: {e}")

            # 3. Wait for the process to exit gracefully
            self.process.join(timeout=5)

            # 4. Still alive? try terminate
            if self.process.is_alive():
                logger.warning(f"Process {pid} still alive, terminating...")
                try:
                    self.process.terminate()
                    self.process.join(timeout=3)
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")

            # 5. Still alive? force kill
            if self.process.is_alive():
                logger.error(f"Process {pid} still alive, killing...")
                try:
                    self.process.kill()
                    self.process.join(timeout=2)
                except Exception as e:
                    logger.exception(f"Error killing process: {e}")

            # 6. Final check
            if self.process.is_alive():
                logger.error(f"Failed to stop process {pid}!")
            else:
                logger.info(f"Worker process {pid} stopped successfully")

        # Reset state
        self.current_status = "idle"
        self.current_config = None
        self.device = None
        self.process = None
        self._unload_sent = False

        # Stop the dispatcher before closing the shared queues, so nothing is disposed mid-read.
        self._stop_dispatcher()
        self._clear_request_queues()

        # Close the queues and set them to None
        self._dispose_ipc_queue("request_queue")
        self._dispose_ipc_queue("status_queue")
        self._dispose_ipc_queue("data_queue")

    def load_model(self, config: InferenceConfig) -> None:
        """Load the model (asynchronous)."""
        if not self.process or not self.process.is_alive():
            self.start_process()

        # start_process already created the IPC queue; defensive check to narrow the type
        if self.request_queue is None:
            raise self._build_queue_unavailable_error("Load")
        # Send the load command
        self.request_queue.put({"command": "load", "config": config.model_dump()})

        self.current_status = "loading"
        self.current_config = config
        self.loading_error = None
        self.error_type = None
        self.error_traceback = None
        self.is_oom_error = False
        logger.info(f"Load command sent for model: {config.model_name}")

    def unload_model(self) -> None:
        """Unload the model - terminate the worker process to fully reclaim VRAM."""
        if not self.process or not self.process.is_alive():
            logger.info("No worker process to unload")
            return

        logger.info("Unloading model by terminating worker process...")

        # Log GPU state before unload (worker-reported memory, not the main process's CUDA stats)
        if self.memory_usage:
            try:
                allocated_before = float(self.memory_usage.get("allocated_gb", 0.0))
                reserved_before = float(self.memory_usage.get("reserved_gb", 0.0))
                logger.info(
                    "GPU memory before unload (from worker): "
                    f"{allocated_before:.2f} GB allocated, {reserved_before:.2f} GB reserved"
                )
            except Exception as e:
                logger.debug(f"Could not read worker memory_usage: {e}")
        else:
            logger.info("GPU memory before unload: worker memory_usage not available")

        # ⚠️ Important: if stop_generation was called recently, wait for the generation thread to fully stop
        if self.last_stop_time is not None:
            time_since_stop = time.time() - self.last_stop_time
            if time_since_stop < 3.0:  # Unloading less than 3s after a stop
                wait_time = 3.0 - time_since_stop
                logger.warning(f"⚠️ Unload called {time_since_stop:.1f}s after stop_generation")
                logger.info(
                    f"Waiting additional {wait_time:.1f}s for generation thread to fully stop..."
                )
                time.sleep(wait_time)
            # Reset the stop timestamp
            self.last_stop_time = None

        # Clear the stop flag
        if self.stop_generation_flag and self.stop_generation_flag.is_set():
            self.stop_generation_flag.clear()

        # Step 1: send the unload command so the worker cleans up first.
        # A live worker process implies the IPC queue exists (see start_process).
        if self.request_queue is None:
            raise self._build_queue_unavailable_error("Unload")
        if not self._unload_sent:
            try:
                self.request_queue.put({"command": "unload"}, timeout=1)
                self._unload_sent = True
                # Wait for the worker to finish cleanup (up to 2.5s)
                time.sleep(2.5)
            except Exception as e:
                logger.warning(f"Failed to send unload command: {e}")
        else:
            logger.debug("Unload command already sent earlier; skipping duplicate send")

        # Step 2: terminate the whole process to guarantee VRAM is fully released
        self.update_status()
        self.stop_process("Generation interrupted because model was unloaded")

        # Clear the main process's GPU cache again and refresh memory_usage for later queries
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated_after = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(
                    "GPU memory after unload (main process view): "
                    f"{allocated_after:.2f} GB allocated"
                )
        except Exception as e:
            logger.debug(f"Could not clean GPU cache: {e}")

        # Update state
        self.current_status = "idle"
        self.current_config = None
        self.device = None
        self.loading_error = None
        self.error_type = None
        self.error_traceback = None
        self.is_oom_error = False

        logger.info("✅ Model unloaded and worker process terminated")

    def stop_generation(self, request_id: str | None = None) -> dict[str, Any]:
        """
        Stop an in-flight generation on the worker process.

        Server engines (llama-server/vLLM) are normally stopped in the main
        process via the async client's per-request stop event; requests that
        reach this method target the worker path (transformers) or arrive
        without a registered per-request queue.
        """
        if not self.process or not self.process.is_alive():
            logger.warning("No active worker process to stop generation")
            return {"status": "error", "message": "No active worker process"}

        if request_id and not self._is_request_active(request_id):
            logger.warning(f"No active generation for request_id={request_id}")
            # An unknown request_id may still belong to a transformers
            # generation (which does not register per-request queues here),
            # so fall through and let the worker decide instead of returning.

        if not self.request_queue:
            logger.warning("Request queue is not initialized")
            return {"status": "error", "message": "Request queue not initialized"}

        try:
            # Determine whether the active engine is LlamaServerEngine: its
            # generations never block the worker loop, so a request-scoped
            # stop must not set the global flag (that would kill unrelated
            # generations on the shared server).
            is_llama_server = False
            if self.current_config and hasattr(self.current_config, "engine"):
                if self.current_config.engine == InferenceEngine.LLAMA_SERVER:
                    is_llama_server = True

            self.request_queue.put(
                {"command": "stop_generation", "request_id": request_id}, timeout=1
            )

            if self.stop_generation_flag:
                # Without a specific request_id, or with a blocking in-process
                # engine (transformers), the generate call occupies the worker
                # loop, so only the global flag can interrupt it immediately.
                if not request_id or not is_llama_server:
                    self.stop_generation_flag.set()

            self.last_stop_time = time.time()
            logger.info(f"Stop generation command sent (request_id={request_id})")
            return {
                "status": "success",
                "message": "Stop signal sent to worker process",
                "request_id": request_id,
            }
        except Exception as e:
            logger.warning(f"Failed to send stop_generation command: {e}")
            if self.stop_generation_flag:
                # Fallback: legacy behaviour, broadcast stop to the worker.
                self.stop_generation_flag.set()
                self.last_stop_time = time.time()
                logger.info("Fallback stop_generation flag set")
                return {
                    "status": "success",
                    "message": "Stop signal sent via fallback flag",
                    "request_id": request_id,
                }
            return {"status": "error", "message": "Stop generation not available"}

    def generate(
        self, prompt: str, params: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        """Non-streaming inference (synchronous) - polls without blocking."""
        if self.current_status != "ready":
            raise RuntimeError("Model not ready for inference")

        request_id = request_id or str(uuid.uuid4())

        if self.request_queue is None or self.data_queue is None:
            raise self._build_queue_unavailable_error("Generation")

        # Register the dedicated queue before sending, so the dispatcher catches the first response.
        response_queue = self._register_request(request_id)

        try:
            try:
                self.request_queue.put(
                    {
                        "command": "generate",
                        "request_id": request_id,
                        "prompt": prompt,
                        "params": params,
                    }
                )
            except (ValueError, OSError, EOFError):
                raise self._build_queue_unavailable_error("Generation") from None

            # Wait on this request's dedicated queue (dispatcher guarantees it only holds our responses)
            # Read total_timeout from params, defaulting to 300s
            total_timeout = params.get("total_timeout", 300)
            timeout_start = time.time()

            while True:
                try:
                    # Use a short timeout (0.1s) to avoid blocking the event loop for long
                    response = response_queue.get(timeout=0.1)
                except Empty:
                    # Queue is empty, keep polling
                    # Check the total timeout
                    if time.time() - timeout_start > total_timeout:
                        logger.exception(f"[Manager] Generation total timeout ({total_timeout}s)")
                        raise TimeoutError(
                            f"Generation timeout after {total_timeout} seconds"
                        ) from None

                    # Check whether the process is still alive
                    if self.process and not self.process.is_alive():
                        self.current_status = "error"
                        self.loading_error = "Worker process died during generation"
                        raise RuntimeError("Worker process died during generation") from None

                    # Move on to the next poll
                    continue

                if response.get("type") == "result":
                    result_payload = {"result": response.get("result", "")}
                    if response.get("slot") is not None:
                        result_payload["slot"] = response.get("slot")
                    if response.get("tool_calls") is not None:
                        result_payload["tool_calls"] = response.get("tool_calls")
                    if response.get("finish_reason") is not None:
                        result_payload["finish_reason"] = response.get("finish_reason")
                    for key in [
                        "total_tokens",
                        "gen_tokens",
                        "gen_tps",
                        "prompt_tokens",
                        "prompt_tps",
                    ]:
                        if response.get(key) is not None:
                            result_payload[key] = response.get(key)
                    return result_payload
                elif response.get("type") == "error":
                    error_msg = response.get("error", "Unknown error")
                    is_oom = response.get("is_oom", False)
                    recoverable = response.get("recoverable", False)
                    fatal = response.get("fatal", False)
                    error_type = response.get("error_type")
                    error_traceback = response.get("error_traceback")

                    if fatal or (is_oom and not recoverable):
                        # Fatal OOM (keeps the original logic)
                        self.current_status = "error"
                        self.loading_error = error_msg
                        self.error_type = error_type or ("OOMError" if is_oom else "RuntimeError")
                        self.error_traceback = error_traceback
                        self.is_oom_error = True
                        logger.error(
                            "[Manager] Fatal generation error detected; marking model as error"
                        )
                    elif is_oom and recoverable:
                        # Recoverable OOM: leave the status unchanged, just advise
                        logger.warning(
                            "[Manager] Recoverable OOM during generate – model kept loaded"
                        )
                    raise RuntimeError(error_msg)
                # Other types (should not happen): ignore and keep polling
        finally:
            self._unregister_request(request_id)

    def generate_stream(
        self, prompt: str, params: dict[str, Any], request_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Streaming inference (generator) - polls without blocking."""
        if self.current_status != "ready":
            raise RuntimeError("Model not ready for inference")

        request_id = request_id or str(uuid.uuid4())

        if self.request_queue is None or self.data_queue is None:
            raise self._build_queue_unavailable_error("Stream generation")

        # Register the dedicated queue before sending, so the dispatcher catches the first response.
        response_queue = self._register_request(request_id)

        try:
            try:
                self.request_queue.put(
                    {
                        "command": "generate_stream",
                        "request_id": request_id,
                        "prompt": prompt,
                        "params": params,
                    }
                )
            except (ValueError, OSError, EOFError):
                raise self._build_queue_unavailable_error("Stream generation") from None

            # Stream results from this request's dedicated queue (the dispatcher guarantees order and ownership)
            # Read total_timeout from params, defaulting to 300s
            total_timeout = params.get("total_timeout", 300)
            timeout_start = time.time()

            while True:
                try:
                    # Use a short timeout (0.1s) to avoid blocking the event loop for long
                    response = response_queue.get(timeout=0.1)
                except Empty:
                    # Queue is empty, keep polling
                    # Check the total timeout
                    if time.time() - timeout_start > total_timeout:
                        logger.exception(
                            f"[Manager] Stream generation total timeout ({total_timeout}s)"
                        )
                        raise TimeoutError(
                            f"Generation timeout after {total_timeout} seconds"
                        ) from None

                    # Check whether the process is still alive
                    if self.process and not self.process.is_alive():
                        self.current_status = "error"
                        self.loading_error = "Worker process died during generation"
                        raise RuntimeError("Worker process died during generation") from None

                    # Move on to the next poll
                    continue

                if response.get("type") == "stream_chunk":
                    chunk = response.get("chunk", "")
                    done = response.get("done", False)
                    is_slot_meta = response.get("meta") == "slot"

                    if chunk or is_slot_meta:
                        chunk_payload = {"chunk": chunk, "done": False}
                        if response.get("slot") is not None:
                            chunk_payload["slot"] = response.get("slot")
                        if response.get("chunk_tokens") is not None:
                            chunk_payload["chunk_tokens"] = response.get("chunk_tokens")
                        if is_slot_meta:
                            chunk_payload["meta"] = "slot"
                        if response.get("tool_calls") is not None:
                            chunk_payload["tool_calls"] = response.get("tool_calls")
                        if response.get("finish_reason") is not None:
                            chunk_payload["finish_reason"] = response.get("finish_reason")
                        yield chunk_payload

                    if done:
                        done_payload = {"chunk": "", "done": True}
                        if response.get("slot") is not None:
                            done_payload["slot"] = response.get("slot")
                        if response.get("stopped") is not None:
                            done_payload["stopped"] = response.get("stopped")
                        if response.get("tool_calls") is not None:
                            done_payload["tool_calls"] = response.get("tool_calls")
                        if response.get("finish_reason") is not None:
                            done_payload["finish_reason"] = response.get("finish_reason")
                        for key in [
                            "total_tokens",
                            "gen_tokens",
                            "gen_tps",
                            "prompt_tokens",
                            "prompt_tps",
                        ]:
                            if response.get(key) is not None:
                                done_payload[key] = response.get(key)
                        yield done_payload
                        break
                elif response.get("type") == "error":
                    error_msg = response.get("error", "Unknown error")
                    is_oom = response.get("is_oom", False)
                    recoverable = response.get("recoverable", False)
                    fatal = response.get("fatal", False)
                    error_type = response.get("error_type")
                    error_traceback = response.get("error_traceback")

                    if fatal or (is_oom and not recoverable):
                        self.current_status = "error"
                        self.loading_error = error_msg
                        self.error_type = error_type or ("OOMError" if is_oom else "RuntimeError")
                        self.error_traceback = error_traceback
                        self.is_oom_error = True
                        logger.error(
                            "[Manager] Fatal stream generation error detected; marking model as error"
                        )
                    elif is_oom and recoverable:
                        logger.warning("[Manager] Recoverable OOM in stream – model kept loaded")
                    raise RuntimeError(error_msg)
                # Other types (should not happen): ignore and keep polling
        finally:
            self._unregister_request(request_id)

    def update_status(self) -> None:
        """Update state (reading from status_queue)."""
        if self.status_queue is None:
            return

        # Drain all status messages from the queue first
        while not self.status_queue.empty():
            try:
                response = self.status_queue.get_nowait()

                # Handle the status message (no type check needed, status_queue only carries statuses)
                status = response.get("status")

                if status in ["loading", "ready", "idle", "unloaded", "error"]:
                    self.current_status = status

                if status == "ready":
                    self.device = response.get("device")
                    self.loading_error = None
                    self.error_type = None
                    self.error_traceback = None
                    self.is_oom_error = False
                    # Update device map statistics
                    self.device_map_summary = response.get("device_map_summary")
                    self.total_modules = response.get("total_modules")
                    self.layer_lines = response.get("layer_lines")
                    # Update GPU memory usage (reported by the worker process)
                    self.memory_usage = response.get("memory_usage")
                elif status == "error":
                    # Capture the detailed error message
                    self.loading_error = response.get("error")
                    self.error_type = response.get("error_type")
                    self.error_traceback = response.get("error_traceback")
                    self.is_oom_error = response.get("is_oom", False)
                    self.current_status = "error"
                elif status == "unloaded":
                    self.current_config = None
                    self.device = None
                    self.loading_error = None
                    self.error_type = None
                    self.error_traceback = None
                    self.is_oom_error = False
                    # Clear device map statistics
                    self.device_map_summary = None
                    self.total_modules = None
                    self.layer_lines = None
                    # Clear GPU memory statistics
                    self.memory_usage = None

            except Empty:
                break

        # Check whether the process died unexpectedly (e.g. an OOM crash)
        if self.process and not self.process.is_alive():
            if self.current_status in ["loading", "ready"]:
                # No error message yet (the process crashed before it could send one)
                if not self.loading_error:
                    logger.error(
                        "[Process] Worker process terminated unexpectedly without error message"
                    )
                    self.loading_error = (
                        "Worker process crashed unexpectedly (possible OOM or system kill)"
                    )
                    self.error_type = "ProcessTerminated"
                else:
                    # An error message already exists; just log that the process died
                    logger.error(
                        f"[Process] Worker process terminated after error: {self.error_type or 'Unknown'}"
                    )

                self.current_status = "error"
                # Release resources
                self._cleanup_dead_process()

    def _cleanup_dead_process(self) -> None:
        """Clean up a dead process."""
        logger.info("Cleaning up dead worker process...")
        if self.process:
            try:
                self.process.join(timeout=1)
            except Exception:
                pass
            self.process = None
        self._unload_sent = False

        # Stop the dispatcher so it does not keep reading the data_queue about to be cleaned up.
        self._stop_dispatcher()

        # Drain the queues and set them to None
        if self.request_queue:
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except Exception:
                    break
            self.request_queue = None

        if self.status_queue:
            while not self.status_queue.empty():
                try:
                    self.status_queue.get_nowait()
                except Exception:
                    break
            self.status_queue = None

        if self.data_queue:
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except Exception:
                    break
            self.data_queue = None

        self.current_config = None
        self.device = None
        self._clear_request_queues()
        logger.info("Dead worker process cleaned up")

    def get_status(self) -> dict[str, Any]:
        """Get the current status."""
        self.update_status()

        # Check whether the process is still running
        if (
            self.process
            and not self.process.is_alive()
            and self.current_status not in ["idle", "error"]
        ):
            self.current_status = "error"
            if not self.loading_error:
                self.loading_error = "Worker process terminated unexpectedly"
                self.error_type = "ProcessTerminated"

        status = {
            "status": self.current_status,
            "loaded": self.current_status == "ready",
            "is_loading": self.current_status == "loading",
            "loading_error": self.loading_error,
            "error_type": self.error_type,
            "is_oom": self.is_oom_error,
            "model_name": (self.current_config.model_name if self.current_config else None),
            "model_path": (self.current_config.model_path if self.current_config else None),
            "quantization": (self.current_config.quantization if self.current_config else None),
            "device": self.device,
            "process_alive": self.process.is_alive() if self.process else False,
            # Added: device map allocation statistics
            "device_allocation": {
                "summary": self.device_map_summary,  # e.g. "cuda:0:30, cpu:10"
                "total_modules": self.total_modules,  # Total module count
                "layer_lines": self.layer_lines,  # Sample layer placements (first 10)
            },
            # GPU memory usage (reported by the worker process)
            "memory_usage": self.memory_usage,
            # llama.cpp-specific info
            "n_gpu_layers": (self.current_config.n_gpu_layers if self.current_config else None),
            "n_ctx": self.current_config.n_ctx if self.current_config else None,
            "n_batch": self.current_config.n_batch if self.current_config else None,
            "llama_server_extra_args": (
                self.current_config.llama_server_extra_args if self.current_config else None
            ),
        }

        # Enable for a detailed traceback (optional, keeps the response small)
        # status["error_traceback"] = self.error_traceback

        return status

    def cleanup_generation_memory(self, slot: int | None = None) -> dict[str, Any]:
        """Send a soft cleanup command to free transient generation memory (model stays loaded)."""
        if self.current_status != "ready":
            return {"status": "error", "message": "Model not ready"}
        if not self.request_queue or not (self.process and self.process.is_alive()):
            return {"status": "error", "message": "Worker process not alive"}

        req_id = str(uuid.uuid4())  # Not actually used, kept for format consistency
        # The guard above ruled out request_queue being None; data_queue shares its lifetime
        if self.data_queue is None:
            return {"status": "error", "message": "Worker process not alive"}
        self.request_queue.put(
            {"command": "cleanup_generation_memory", "request_id": req_id, "slot": slot}
        )
        # Read the response (up to 5s)
        start = time.time()
        while time.time() - start < 5:
            try:
                resp = self.data_queue.get(timeout=0.5)
                if resp.get("type") == "cleanup":
                    return {"status": "success", "message": resp.get("result")}
                # Put non-cleanup responses back
                self.data_queue.put(resp)
            except Empty:
                continue
        return {"status": "timeout", "message": "Cleanup command timed out"}

    def is_loaded(self) -> bool:
        """Check whether the model is loaded."""
        self.update_status()
        return self.current_status == "ready"

    def get_tokenizer_proxy(self) -> "TokenizerProxy | None":
        """
        Get the tokenizer proxy object.

        Returns a lightweight proxy that supports apply_chat_template and similar methods.
        """
        if not self.is_loaded():
            return None

        return TokenizerProxy(self)

    def apply_chat_template(self, messages: list, **kwargs: Any) -> str | None:  # noqa: ANN401 - forwarded to tokenizer.apply_chat_template
        """
        Apply the chat template (internal method, called by TokenizerProxy).

        Args:
            messages: List of conversation messages
            **kwargs: Extra arguments forwarded to apply_chat_template, e.g.:
                     - add_generation_prompt: bool
                     - enable_thinking: bool (for models that support it)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        request_id = str(uuid.uuid4())

        # A ready model implies a live worker, so the IPC queues are never None
        if self.request_queue is None or self.data_queue is None:
            raise self._build_queue_unavailable_error("apply_chat_template")
        # Send the request
        self.request_queue.put(
            {
                "command": "apply_chat_template",
                "request_id": request_id,
                "messages": messages,
                "template_kwargs": kwargs,
            }
        )

        # Wait for the response (10s timeout) - read from data_queue
        timeout = 10
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.data_queue.get(timeout=0.5)

                if response.get("request_id") == request_id:
                    if response.get("type") == "result":
                        return response.get("result")
                    elif response.get("type") == "error":
                        error_msg = response.get("error", "Unknown error")
                        raise RuntimeError(f"apply_chat_template failed: {error_msg}")
                else:
                    # Not our response, put it back on the queue
                    self.data_queue.put(response)

            except Empty:
                continue

        raise TimeoutError("apply_chat_template request timed out")

    def get_error_details(self) -> dict[str, Any] | None:
        """Get detailed error info (including the traceback)."""
        self.update_status()

        if self.current_status != "error":
            return None

        return {
            "error": self.loading_error,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
            "is_oom": self.is_oom_error,
            "process_alive": self.process.is_alive() if self.process else False,
        }

    def cleanup(self) -> None:
        """Release resources."""
        self.stop_process()


class TokenizerProxy:
    """
    Tokenizer proxy class.

    Exposes the common tokenizer methods, delegating to the tokenizer in the separate process.
    """

    def __init__(self, inference_process: ModelInferenceProcess) -> None:
        self.inference_process = inference_process

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,  # noqa: ANN401 - forwarded to tokenizer.apply_chat_template
    ) -> str:
        """
        Apply the chat template.

        Args:
            messages: List of conversation messages, e.g. [{"role": "user", "content": "..."}, ...]
            tokenize: Whether to return token ids (only False is supported for now)
            add_generation_prompt: Whether to append the generation prompt
            **kwargs: Extra arguments, e.g. enable_thinking=True/False (for models that support it)

        Returns:
            The formatted prompt string

        Examples:
            # Standard usage
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            # Enable thinking mode (DeepSeek, QwQ, etc.)
            tokenizer.apply_chat_template(messages, enable_thinking=True)

            # Disable thinking mode
            tokenizer.apply_chat_template(messages, enable_thinking=False)
        """
        if tokenize:
            raise NotImplementedError("TokenizerProxy only supports tokenize=False")

        # Merge all arguments
        template_kwargs = {"add_generation_prompt": add_generation_prompt}
        template_kwargs.update(kwargs)

        # On success the worker always returns a formatted string (the None path raises),
        # so narrow it to str
        return cast(
            str,
            self.inference_process.apply_chat_template(messages=messages, **template_kwargs),
        )
