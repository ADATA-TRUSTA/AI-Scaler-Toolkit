"""
Model Inference Process - Separate process for model loading and inference to avoid OOM issues
Uses multiprocessing to isolate model operations and ensure proper VRAM cleanup on process termination
"""

import os
import time
import uuid
import traceback
import threading
from typing import Optional, Dict, Any
from multiprocessing import Process, Queue, Event
from multiprocessing.synchronize import Event as EventClass
from queue import Empty

import torch

# Import settings BEFORE torch/transformers (sets HF_HOME environment variable)
from ..settings import configure_logging
from ..config_models import InferenceConfig, InferenceEngine

# Import engines
from .engines.transformers_engine import TransformersEngine
from .engines.llama_server_engine import LlamaServerEngine
from .engines.vllm_engine import VllmEngine
from .model_utils import _cleanup_memory

logger = configure_logging(__name__)


def _model_worker_process(
    request_queue: Queue,
    status_queue: Queue,
    data_queue: Queue,
    stop_event: EventClass,
    stop_generation_flag: EventClass,
):
    """
    模型工作進程 - 處理模型加載和推理請求

    請求格式:
        {"command": "load", "config": {...}}
        {"command": "generate", "request_id": "...", "prompt": "...", "params": {...}}
        {"command": "generate_stream", "request_id": "...", "prompt": "...", "params": {...}}
        {"command": "unload"}

    響應格式 (分離隊列):
        status_queue: 狀態更新 {"status": "loading|ready|error", ...}
        data_queue: 推理結果、Stream chunks、錯誤 {"type": "result|stream_chunk|error", ...}
    """
    engine = None
    active_generation_flags: Dict[str, threading.Event] = {}

    def _run_generation_task(task_request: Dict[str, Any], is_stream: bool):
        """在 worker 內以執行緒執行生成，供 llama-server 並發使用。"""
        req_id = task_request.get("request_id")
        local_stop_flag = threading.Event()

        if req_id:
            active_generation_flags[req_id] = local_stop_flag

        # 只在 worker 內部透傳，讓引擎改用 request 專屬 stop flag
        task_request["request_stop_flag"] = local_stop_flag

        try:
            if engine is None:
                data_queue.put(
                    {
                        "type": "error",
                        "request_id": req_id,
                        "error": "Engine not initialized/Model not loaded",
                    }
                )
                return

            if is_stream:
                engine.generate_stream(task_request)
            else:
                engine.generate(task_request)
        except Exception as e:
            logger.error(f"[Worker] Generation task error (request_id={req_id}): {e}")
            data_queue.put(
                {
                    "type": "error",
                    "request_id": req_id,
                    "error": str(e),
                }
            )
        finally:
            if req_id:
                active_generation_flags.pop(req_id, None)

    try:
        logger.info("[Worker] Model worker process started")

        while not stop_event.is_set():
            try:
                # 等待請求，設置超時以便檢查停止信號
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
                                engine.unload()
                            except:
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
                        logger.error(f"[Worker] Failed to load model: {e}")
                        error_traceback = traceback.format_exc()
                        logger.error(error_traceback)

                        # Cleanup execution
                        if engine:
                            try:
                                engine.unload()
                            except:
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
                            logger.error(
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
                    if isinstance(engine, (LlamaServerEngine, VllmEngine)):
                        worker = threading.Thread(
                            target=_run_generation_task,
                            args=(dict(request), False),
                            daemon=True,
                        )
                        worker.start()
                    else:
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
                    if isinstance(engine, (LlamaServerEngine, VllmEngine)):
                        worker = threading.Thread(
                            target=_run_generation_task,
                            args=(dict(request), True),
                            daemon=True,
                        )
                        worker.start()
                    else:
                        engine.generate_stream(request)

                elif command == "stop_generation":
                    req_id = request.get("request_id")
                    if req_id:
                        # 對於帶有 request_id 的情況（例如來自 API 但引擎是 TransformersEngine），
                        # 如果我們找到對應的 flag，就停止該請求。
                        stop_flag = active_generation_flags.get(req_id)
                        if stop_flag:
                            stop_flag.set()
                            logger.info(
                                f"[Worker] stop_generation sent to request_id={req_id}"
                            )
                        else:
                            logger.warning(
                                f"[Worker] stop_generation target not found: request_id={req_id}"
                            )
                            # 針對未使用 active_generation_flags 的引擎（如 TransformersEngine），設置全局停止標誌
                            # LlamaServerEngine 與 VllmEngine 均透過 threading 管理 per-request flag，
                            # 不需要回落至全局旗標（避免誤停其他並發請求）
                            if not isinstance(engine, (LlamaServerEngine, VllmEngine)):
                                stop_generation_flag.set()
                                logger.info(
                                    f"[Worker] Global stop_generation_flag set for request_id={req_id} (not found in active flags)"
                                )
                    else:
                        # 舊行為：停止當前所有生成
                        stop_generation_flag.set()
                        for rid, flag in list(active_generation_flags.items()):
                            flag.set()
                            logger.info(
                                f"[Worker] stop_generation broadcast to request_id={rid}"
                            )

                elif command == "unload":
                    logger.info("[Worker] Unload command received")
                    # 先停止所有進行中生成
                    stop_generation_flag.set()
                    for rid, flag in list(active_generation_flags.items()):
                        flag.set()
                        logger.info(
                            f"[Worker] stopping active generation before unload: request_id={rid}"
                        )
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
                            # 其他引擎不支援 slot，維持既有全域清理行為
                            engine.cleanup_generation_memory()
                    else:
                        _cleanup_memory()
                        data_queue.put(
                            {"type": "cleanup", "result": "memory cleaned (no engine)"}
                        )

                else:
                    logger.warning(f"[Worker] Unknown command: {command}")

            except Exception as e:
                logger.error(f"[Worker] Request handling error: {e}")
                logger.error(traceback.format_exc())

        logger.info("[Worker] Stop event received, cleaning up...")

    except Exception as e:
        logger.error(f"[Worker] Worker process error: {e}")
        logger.error(traceback.format_exc())

    finally:
        if engine:
            try:
                engine.unload()
            except:
                pass
        _cleanup_memory()
        logger.info("[Worker] Model worker process terminated")
        os._exit(0)


class ModelInferenceProcess:
    """
    模型推理進程管理器

    管理獨立的模型推理進程，確保 OOM 時能完全清理 VRAM
    使用分離隊列架構：
    - status_queue: 狀態更新（loading, ready, error等）
    - data_queue: 推理結果、Stream chunks、錯誤消息
    """

    def __init__(self):
        self.process: Optional[Process] = None
        self.request_queue: Optional[Queue] = None
        self.status_queue: Optional[Queue] = None  # 專門用於狀態更新
        self.data_queue: Optional[Queue] = None  # 專門用於推理結果
        self.stop_event: Optional[EventClass] = None  # 用來中止整個進程
        self.stop_generation_flag: Optional[EventClass] = None  # 用于停止当前生成
        self.current_status = "idle"
        self.current_config: Optional[InferenceConfig] = None
        self.device: Optional[str] = None
        self.loading_error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.error_traceback: Optional[str] = None
        self.is_oom_error: bool = False
        self.last_stop_time: Optional[float] = None  # 记录最后一次停止生成的时间
        # 標記是否已向 worker 發送過 unload 指令，避免重複發送造成日誌噪音
        self._unload_sent: bool = False
        # Device map 統計資訊
        self.device_map_summary: Optional[str] = None  # 例如: "cuda:0:30, cpu:10"
        self.total_modules: Optional[int] = None  # 總模組數
        self.layer_lines: Optional[list] = None  # 層級分配範例
        # GPU 記憶體使用情況（從 worker 進程報告）
        self.memory_usage: Optional[Dict[str, float]] = None
        self.active_request_ids = set()
        self._active_request_ids_lock = threading.Lock()

    def _mark_request_active(self, request_id: str):
        with self._active_request_ids_lock:
            self.active_request_ids.add(request_id)

    def _mark_request_inactive(self, request_id: str):
        with self._active_request_ids_lock:
            self.active_request_ids.discard(request_id)

    def _is_request_active(self, request_id: str) -> bool:
        with self._active_request_ids_lock:
            return request_id in self.active_request_ids

    def _snapshot_active_request_ids(self) -> list[str]:
        with self._active_request_ids_lock:
            return list(self.active_request_ids)

    def _notify_active_requests(
        self,
        message: str,
        *,
        error_type: str = "ProcessInterrupted",
    ) -> None:
        if not self.data_queue:
            return

        request_ids = self._snapshot_active_request_ids()
        if not request_ids:
            return

        for request_id in request_ids:
            try:
                self.data_queue.put(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": message,
                        "error_type": error_type,
                        "fatal": False,
                        "recoverable": True,
                        "is_oom": False,
                    },
                    block=False,
                )
            except Exception as e:
                logger.debug(
                    f"Failed to notify active request {request_id} during shutdown: {e}"
                )

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

    def start_process(self):
        """啟動工作進程"""
        if self.process and self.process.is_alive():
            logger.warning("Worker process already running")
            return

        self._unload_sent = False

        # 創建進程間通信對象（分離隊列）
        self.request_queue = Queue()
        self.status_queue = Queue()  # 專門用於狀態更新
        self.data_queue = Queue()  # 專門用於推理結果
        self.stop_event = Event()  # 初始化用來中止整個進程
        self.stop_generation_flag = Event()  # 初始化停止生成标志

        # 創建並啟動進程（daemon=True 確保主進程退出時自動終止）
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

    def stop_process(self, interrupt_message: Optional[str] = None):
        """停止工作進程"""
        if not self.process:
            logger.info("No worker process to stop")
            return

        pid = self.process.pid if self.process else "N/A"
        logger.info(f"Stopping worker process (PID: {pid})...")

        self._notify_active_requests(
            interrupt_message
            or "Generation interrupted because worker process is stopping",
            error_type="ModelUnloaded" if interrupt_message else "ProcessInterrupted",
        )

        # 如果進程還活著，嘗試優雅關閉
        if self.process.is_alive():
            # 1. 發送卸載命令
            if not self._unload_sent:  # 只在尚未發送過時送一次
                try:
                    if self.request_queue:
                        self.request_queue.put({"command": "unload"}, timeout=1)
                        self._unload_sent = True
                except Exception as e:
                    logger.warning(f"Failed to send unload command: {e}")

            # 2. 設置停止事件
            try:
                if self.stop_event:
                    self.stop_event.set()
            except Exception as e:
                logger.warning(f"Failed to set stop event: {e}")

            # 3. 等待進程優雅退出
            self.process.join(timeout=5)

            # 4. 如果還活著，嘗試 terminate
            if self.process.is_alive():
                logger.warning(f"Process {pid} still alive, terminating...")
                try:
                    self.process.terminate()
                    self.process.join(timeout=3)
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")

            # 5. 如果還活著，強制 kill
            if self.process.is_alive():
                logger.error(f"Process {pid} still alive, killing...")
                try:
                    self.process.kill()
                    self.process.join(timeout=2)
                except Exception as e:
                    logger.error(f"Error killing process: {e}")

            # 6. 最後檢查
            if self.process.is_alive():
                logger.error(f"Failed to stop process {pid}!")
            else:
                logger.info(f"Worker process {pid} stopped successfully")

        # 清理狀態
        self.current_status = "idle"
        self.current_config = None
        self.device = None
        with self._active_request_ids_lock:
            self.active_request_ids.clear()
        self.process = None
        self._unload_sent = False

        # 關閉隊列並設置為 None
        self._dispose_ipc_queue("request_queue")
        self._dispose_ipc_queue("status_queue")
        self._dispose_ipc_queue("data_queue")

    def load_model(self, config: InferenceConfig):
        """加載模型（異步）"""
        if not self.process or not self.process.is_alive():
            self.start_process()

        # 發送加載命令
        self.request_queue.put({"command": "load", "config": config.model_dump()})

        self.current_status = "loading"
        self.current_config = config
        self.loading_error = None
        self.error_type = None
        self.error_traceback = None
        self.is_oom_error = False
        logger.info(f"Load command sent for model: {config.model_name}")

    def unload_model(self):
        """卸載模型 - 終止 worker process 以徹底釋放 VRAM"""
        if not self.process or not self.process.is_alive():
            logger.info("No worker process to unload")
            return

        logger.info("Unloading model by terminating worker process...")

        # 記錄卸載前的 GPU 狀態（使用 worker 回報的記憶體資訊，而不是主進程自己的 CUDA 統計）
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

        # ⚠️ 重要：如果最近调用了 stop_generation，需要等待生成线程完全停止
        if self.last_stop_time is not None:
            time_since_stop = time.time() - self.last_stop_time
            if time_since_stop < 3.0:  # 如果停止后不到3秒就卸载
                wait_time = 3.0 - time_since_stop
                logger.warning(
                    f"⚠️ Unload called {time_since_stop:.1f}s after stop_generation"
                )
                logger.info(
                    f"Waiting additional {wait_time:.1f}s for generation thread to fully stop..."
                )
                time.sleep(wait_time)
            # 重置停止时间
            self.last_stop_time = None

        # 清除停止标志
        if self.stop_generation_flag and self.stop_generation_flag.is_set():
            self.stop_generation_flag.clear()

        # 方法 1：發送 unload 命令讓 worker 先清理
        if not self._unload_sent:
            try:
                self.request_queue.put({"command": "unload"}, timeout=1)
                self._unload_sent = True
                # 等待 worker 完成清理（最多 2.5 秒）
                time.sleep(2.5)
            except Exception as e:
                logger.warning(f"Failed to send unload command: {e}")
        else:
            logger.debug("Unload command already sent earlier; skipping duplicate send")

        # 方法 2：終止整個進程以確保 VRAM 完全釋放
        self.update_status()
        self.stop_process("Generation interrupted because model was unloaded")

        # 再次清理主進程的 GPU 緩存，並更新 memory_usage 供後續查詢
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

        # 更新狀態
        self.current_status = "idle"
        self.current_config = None
        self.device = None
        self.loading_error = None
        self.error_type = None
        self.error_traceback = None
        self.is_oom_error = False

        logger.info("✅ Model unloaded and worker process terminated")

    def stop_generation(self, request_id: Optional[str] = None):
        """停止当前正在进行的生成"""
        if not self.process or not self.process.is_alive():
            logger.warning("No active worker process to stop generation")
            return {"status": "error", "message": "No active worker process"}

        if request_id and not self._is_request_active(request_id):
            logger.warning(f"No active generation for request_id={request_id}")
            # 即使 request_id 找不到, 也有可能是 transformer_engine, 因此我們不 return, 而是往下執行, 由 worker 判斷
            # return {
            #     "status": "error",
            #     "message": "No active generation",
            #     "request_id": request_id,
            # }

        if not self.request_queue:
            logger.warning("Request queue is not initialized")
            return {"status": "error", "message": "Request queue not initialized"}

        try:
            # 針對可能使用 transformers 但 manager 層帶有 request_id 的情況
            # worker 層在收到的時候，若發現在 active_generation_flags 找不到且為 transformers_engine
            # 就會自動 fallback 到 stop_generation_flag
            # 判断当前使用的引擎是否为 LlamaServerEngine
            is_llama_server = False
            if self.current_config and hasattr(self.current_config, "engine"):
                if self.current_config.engine == InferenceEngine.LLAMA_SERVER:
                    is_llama_server = True

            self.request_queue.put(
                {"command": "stop_generation", "request_id": request_id}, timeout=1
            )

            if self.stop_generation_flag:
                # 如果沒有特定的 request_id，或者當前使用的是阻塞式引擎(TransformersEngine/LlamaCppEngine)
                # 這些引擎會阻塞 worker 中讀取隊列的迴圈，因此需要全局 flag 立即中斷
                if not request_id or not is_llama_server:
                    self.stop_generation_flag.set()

            self.last_stop_time = time.time()  # 记录停止时间
            logger.info(f"Stop generation command sent (request_id={request_id})")
            return {
                "status": "success",
                "message": "Stop signal sent to worker process",
                "request_id": request_id,
            }
        except Exception as e:
            logger.warning(f"Failed to send stop_generation command: {e}")
            if self.stop_generation_flag:
                # fallback：舊行為，廣播停止
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
        self, prompt: str, params: Dict[str, Any], request_id: Optional[str] = None
    ) -> str:
        """非串流推理（同步）- 使用非阻塞方式轮询"""
        if self.current_status != "ready":
            raise RuntimeError("Model not ready for inference")

        request_id = request_id or str(uuid.uuid4())
        self._mark_request_active(request_id)

        if self.request_queue is None or self.data_queue is None:
            raise self._build_queue_unavailable_error("Generation")

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
            raise self._build_queue_unavailable_error("Generation")

        # 等待結果 - 從 data_queue 讀取
        # 從 params 獲取 total_timeout，預設 300 秒
        total_timeout = params.get("total_timeout", 300)
        timeout_start = time.time()
        data_queue = self.data_queue

        try:
            while True:
                try:
                    # 使用短超时（0.1秒）以避免长时间阻塞事件循环
                    if data_queue is None:
                        raise self._build_queue_unavailable_error("Generation")

                    response = data_queue.get(timeout=0.1)

                    if response.get("request_id") == request_id:
                        if response.get("type") == "result":
                            result_payload = {"result": response.get("result", "")}
                            if response.get("slot") is not None:
                                result_payload["slot"] = response.get("slot")
                            if response.get("tool_calls") is not None:
                                result_payload["tool_calls"] = response.get(
                                    "tool_calls"
                                )
                            if response.get("finish_reason") is not None:
                                result_payload["finish_reason"] = response.get(
                                    "finish_reason"
                                )
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
                                # 致命型 OOM（仍採用原邏輯）
                                self.current_status = "error"
                                self.loading_error = error_msg
                                self.error_type = error_type or (
                                    "OOMError" if is_oom else "RuntimeError"
                                )
                                self.error_traceback = error_traceback
                                self.is_oom_error = True
                                logger.error(
                                    "[Manager] Fatal generation error detected; marking model as error"
                                )
                            elif is_oom and recoverable:
                                # 可恢復 OOM：不改變狀態，附加建議
                                logger.warning(
                                    "[Manager] Recoverable OOM during generate – model kept loaded"
                                )
                            raise RuntimeError(error_msg)
                    else:
                        # ⚠️ 不是当前请求的响应，放回队列末尾
                        # 可能是其他并发请求的响应
                        try:
                            data_queue.put(response, block=False)
                        except Exception:
                            logger.warning(
                                f"[Manager] Failed to requeue non-matching response: {response.get('type')}"
                            )

                except RuntimeError:
                    raise
                except (ValueError, OSError, EOFError):
                    raise self._build_queue_unavailable_error("Generation")

                except Empty:
                    # 队列为空，继续轮询
                    # 检查总超时
                    if time.time() - timeout_start > total_timeout:
                        logger.error(
                            f"[Manager] Generation total timeout ({total_timeout}s)"
                        )
                        raise TimeoutError(
                            f"Generation timeout after {total_timeout} seconds"
                        )

                    # 检查进程是否还活着
                    if self.process and not self.process.is_alive():
                        self.current_status = "error"
                        self.loading_error = "Worker process died during generation"
                        raise RuntimeError("Worker process died during generation")

                    # 继续下一次轮询
                    continue
        finally:
            self._mark_request_inactive(request_id)

    def generate_stream(
        self, prompt: str, params: Dict[str, Any], request_id: Optional[str] = None
    ):
        """串流推理（生成器）- 使用非阻塞方式轮詢"""
        if self.current_status != "ready":
            raise RuntimeError("Model not ready for inference")

        request_id = request_id or str(uuid.uuid4())
        self._mark_request_active(request_id)

        if self.request_queue is None or self.data_queue is None:
            raise self._build_queue_unavailable_error("Stream generation")

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
            raise self._build_queue_unavailable_error("Stream generation")

        # 流式返回結果 - 從 data_queue 讀取
        # 從 params 獲取 total_timeout，預設 300 秒
        total_timeout = params.get("total_timeout", 300)
        timeout_start = time.time()
        data_queue = self.data_queue

        try:
            while True:
                try:
                    # 使用短超时（0.1秒）以避免长时间阻塞事件循环
                    if data_queue is None:
                        raise self._build_queue_unavailable_error("Stream generation")

                    response = data_queue.get(timeout=0.1)

                    if response.get("request_id") == request_id:
                        if response.get("type") == "stream_chunk":
                            chunk = response.get("chunk", "")
                            done = response.get("done", False)
                            is_slot_meta = response.get("meta") == "slot"

                            if chunk or is_slot_meta:
                                chunk_payload = {"chunk": chunk, "done": False}
                                if response.get("slot") is not None:
                                    chunk_payload["slot"] = response.get("slot")
                                if response.get("chunk_tokens") is not None:
                                    chunk_payload["chunk_tokens"] = response.get(
                                        "chunk_tokens"
                                    )
                                if is_slot_meta:
                                    chunk_payload["meta"] = "slot"
                                if response.get("tool_calls") is not None:
                                    chunk_payload["tool_calls"] = response.get(
                                        "tool_calls"
                                    )
                                if response.get("finish_reason") is not None:
                                    chunk_payload["finish_reason"] = response.get(
                                        "finish_reason"
                                    )
                                yield chunk_payload

                            if done:
                                done_payload = {"chunk": "", "done": True}
                                if response.get("slot") is not None:
                                    done_payload["slot"] = response.get("slot")
                                if response.get("stopped") is not None:
                                    done_payload["stopped"] = response.get("stopped")
                                if response.get("tool_calls") is not None:
                                    done_payload["tool_calls"] = response.get(
                                        "tool_calls"
                                    )
                                if response.get("finish_reason") is not None:
                                    done_payload["finish_reason"] = response.get(
                                        "finish_reason"
                                    )
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
                                self.error_type = error_type or (
                                    "OOMError" if is_oom else "RuntimeError"
                                )
                                self.error_traceback = error_traceback
                                self.is_oom_error = True
                                logger.error(
                                    "[Manager] Fatal stream generation error detected; marking model as error"
                                )
                            elif is_oom and recoverable:
                                logger.warning(
                                    "[Manager] Recoverable OOM in stream – model kept loaded"
                                )
                            raise RuntimeError(error_msg)
                    else:
                        # ⚠️ 不是当前请求的响应，放回队列末尾
                        # 可能是其他并发请求的响应
                        try:
                            data_queue.put(response, block=False)
                        except Exception:
                            logger.warning(
                                f"[Manager] Failed to requeue non-matching response: {response.get('type')}"
                            )

                except RuntimeError:
                    raise
                except (ValueError, OSError, EOFError):
                    raise self._build_queue_unavailable_error("Stream generation")

                except Empty:
                    # 队列为空，继续轮询
                    # 检查总超时
                    if time.time() - timeout_start > total_timeout:
                        logger.error(
                            f"[Manager] Stream generation total timeout ({total_timeout}s)"
                        )
                        raise TimeoutError(
                            f"Generation timeout after {total_timeout} seconds"
                        )

                    # 检查进程是否还活着
                    if self.process and not self.process.is_alive():
                        self.current_status = "error"
                        self.loading_error = "Worker process died during generation"
                        raise RuntimeError("Worker process died during generation")

                    # 继续下一次轮询
                    continue
        finally:
            self._mark_request_inactive(request_id)

    def update_status(self):
        """更新狀態（從 status_queue 讀取）"""
        if self.status_queue is None:
            return

        # 先處理所有隊列中的狀態消息
        while not self.status_queue.empty():
            try:
                response = self.status_queue.get_nowait()

                # 處理狀態消息（無需檢查 type，status_queue 只包含狀態）
                status = response.get("status")

                if status in ["loading", "ready", "idle", "unloaded", "error"]:
                    self.current_status = status

                if status == "ready":
                    self.device = response.get("device")
                    self.loading_error = None
                    self.error_type = None
                    self.error_traceback = None
                    self.is_oom_error = False
                    # 更新 device map 統計資訊
                    self.device_map_summary = response.get("device_map_summary")
                    self.total_modules = response.get("total_modules")
                    self.layer_lines = response.get("layer_lines")
                    # 更新 GPU 記憶體使用（從 worker 進程報告）
                    self.memory_usage = response.get("memory_usage")
                elif status == "error":
                    # 捕獲詳細的錯誤訊息
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
                    # 清除 device map 統計資訊
                    self.device_map_summary = None
                    self.total_modules = None
                    self.layer_lines = None
                    # 清除 GPU 記憶體統計
                    self.memory_usage = None

            except Empty:
                break

        # 檢查進程是否意外終止（例如 OOM 導致的崩潰）
        if self.process and not self.process.is_alive():
            if self.current_status in ["loading", "ready"]:
                # 如果還沒有錯誤訊息（進程崩潰前沒來得及發送）
                if not self.loading_error:
                    logger.error(
                        "[Process] Worker process terminated unexpectedly without error message"
                    )
                    self.loading_error = "Worker process crashed unexpectedly (possible OOM or system kill)"
                    self.error_type = "ProcessTerminated"
                else:
                    # 如果已經有錯誤訊息，只記錄進程已終止
                    logger.error(
                        f"[Process] Worker process terminated after error: {self.error_type or 'Unknown'}"
                    )

                self.current_status = "error"
                # 清理資源
                self._cleanup_dead_process()

    def _cleanup_dead_process(self):
        """清理已終止的進程"""
        logger.info("Cleaning up dead worker process...")
        if self.process:
            try:
                self.process.join(timeout=1)
            except:
                pass
            self.process = None
        self._unload_sent = False

        # 清理隊列並設置為 None
        if self.request_queue:
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except:
                    break
            self.request_queue = None

        if self.status_queue:
            while not self.status_queue.empty():
                try:
                    self.status_queue.get_nowait()
                except:
                    break
            self.status_queue = None

        if self.data_queue:
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    break
            self.data_queue = None

        self.current_config = None
        self.device = None
        with self._active_request_ids_lock:
            self.active_request_ids.clear()
        logger.info("Dead worker process cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """獲取當前狀態"""
        self.update_status()

        # 檢查進程是否仍在運行
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
            "model_name": (
                self.current_config.model_name if self.current_config else None
            ),
            "model_path": (
                self.current_config.model_path if self.current_config else None
            ),
            "quantization": (
                self.current_config.quantization if self.current_config else None
            ),
            "device": self.device,
            "process_alive": self.process.is_alive() if self.process else False,
            # 新增：device map 分配統計資訊
            "device_allocation": {
                "summary": self.device_map_summary,  # 例如: "cuda:0:30, cpu:10"
                "total_modules": self.total_modules,  # 總模組數
                "layer_lines": self.layer_lines,  # 層級分配範例（前 10 個）
            },
            # GPU 記憶體使用情況（從 worker 進程報告）
            "memory_usage": self.memory_usage,
            # llama.cpp 特有資訊
            "n_gpu_layers": (
                self.current_config.n_gpu_layers if self.current_config else None
            ),
            "n_ctx": self.current_config.n_ctx if self.current_config else None,
            "n_batch": self.current_config.n_batch if self.current_config else None,
            "llama_server_extra_args": (
                self.current_config.llama_server_extra_args
                if self.current_config
                else None
            ),
        }

        # 如果需要詳細的 traceback（可選，避免響應過大）
        # status["error_traceback"] = self.error_traceback

        return status

    def cleanup_generation_memory(self, slot: Optional[int] = None) -> Dict[str, Any]:
        """向工作進程發送 soft cleanup 指令以釋放暫存生成記憶體（不卸載模型）。"""
        if self.current_status != "ready":
            return {"status": "error", "message": "Model not ready"}
        if not self.request_queue or not (self.process and self.process.is_alive()):
            return {"status": "error", "message": "Worker process not alive"}

        req_id = str(uuid.uuid4())  # 不真正使用，但保持格式一致
        self.request_queue.put(
            {"command": "cleanup_generation_memory", "request_id": req_id, "slot": slot}
        )
        # 讀取回應（最多 5 秒）
        start = time.time()
        while time.time() - start < 5:
            try:
                resp = self.data_queue.get(timeout=0.5)
                if resp.get("type") == "cleanup":
                    return {"status": "success", "message": resp.get("result")}
                # 非 cleanup 回應放回
                self.data_queue.put(resp)
            except Empty:
                continue
        return {"status": "timeout", "message": "Cleanup command timed out"}

    def is_loaded(self) -> bool:
        """檢查模型是否已加載"""
        self.update_status()
        return self.current_status == "ready"

    def get_tokenizer_proxy(self):
        """獲取 tokenizer 代理對象

        返回一個輕量級代理，支持 apply_chat_template 等方法
        """
        if not self.is_loaded():
            return None

        return TokenizerProxy(self)

    def apply_chat_template(self, messages: list, **kwargs) -> Optional[str]:
        """應用 chat template（內部方法，由 TokenizerProxy 調用）

        Args:
            messages: 對話消息列表
            **kwargs: 傳遞給 apply_chat_template 的額外參數，如：
                     - add_generation_prompt: bool
                     - enable_thinking: bool (適用於支援的模型)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        request_id = str(uuid.uuid4())

        # 發送請求
        self.request_queue.put(
            {
                "command": "apply_chat_template",
                "request_id": request_id,
                "messages": messages,
                "template_kwargs": kwargs,
            }
        )

        # 等待響應（超時 10 秒）- 從 data_queue 讀取
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
                    # 不是我們的響應，重新放回隊列
                    self.data_queue.put(response)

            except Empty:
                continue

        raise TimeoutError("apply_chat_template request timed out")

    def get_error_details(self) -> Optional[Dict[str, Any]]:
        """獲取詳細的錯誤信息（包括 traceback）"""
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

    def cleanup(self):
        """清理資源"""
        self.stop_process()


class TokenizerProxy:
    """Tokenizer 代理類

    提供 tokenizer 的常用方法，實際調用在獨立進程中的 tokenizer
    """

    def __init__(self, inference_process: ModelInferenceProcess):
        self.inference_process = inference_process

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """應用 chat template

        Args:
            messages: 對話消息列表，格式 [{"role": "user", "content": "..."}, ...]
            tokenize: 是否返回 token ids（目前僅支持 False）
            add_generation_prompt: 是否添加生成提示符
            **kwargs: 額外參數，如 enable_thinking=True/False（適用於支援的模型）

        Returns:
            格式化後的提示詞字符串

        Examples:
            # 標準使用
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            # 啟用思考模式（DeepSeek、QwQ 等）
            tokenizer.apply_chat_template(messages, enable_thinking=True)

            # 禁用思考模式
            tokenizer.apply_chat_template(messages, enable_thinking=False)
        """
        if tokenize:
            raise NotImplementedError("TokenizerProxy only supports tokenize=False")

        # 合併所有參數
        template_kwargs = {"add_generation_prompt": add_generation_prompt}
        template_kwargs.update(kwargs)

        return self.inference_process.apply_chat_template(
            messages=messages, **template_kwargs
        )
