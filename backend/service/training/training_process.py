"""Training process worker and manager.

This module mirrors the design of `inference/model_inference_process.py` but for
finetuning (LoRA / QLoRA / full-parameter) using Hugging Face Trainer + DeepSpeed.

High-level design:
- TrainingWorkerProcess: spawned via multiprocessing.Process, owns GPU resources.
- TrainingProcessManager: lives in main FastAPI process, sends commands and
  receives status updates via multiprocessing queues.

NOTE: This is an initial skeleton; details are wired from `training_manager.py`.
"""

from __future__ import annotations

import os

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

from dataclasses import asdict
from multiprocessing import Process, Queue, Event
from multiprocessing.synchronize import Event as EventClass
from queue import Empty
from typing import Optional, Dict, Any, Callable

# Import settings BEFORE torch/transformers (sets HF_HOME environment variable)
from ..settings import configure_logging, HF_HOME, REDIS_HOST, REDIS_PORT, REDIS_DB
from ..config_models import TrainingConfig, TrainingMethod, TrainingStatus, TrainingLog, ResourceLog
from ..model_registry import model_registry, FinetunedModelInfo
from ..utils.token_utils import load_hf_token
from ..utils.system_monitor import system_monitor
from ..utils.conversion_manager import conversion_manager

import time
import uuid
import json
from typing import List
try:
    import redis
except ImportError:
    redis = None


# New core modules
from .core import (
    ModelLoader,
    load_training_dataset,
    StrategyFactory,
    save_training_results,
)

import torch
from transformers import Trainer

logger = configure_logging(__name__)


def _create_redis_client():
    """Create Redis client from REDIS_URL first, then host/port/db fallback."""
    if not redis:
        return None

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        return redis.Redis.from_url(redis_url, decode_responses=True)

    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        username=redis_username,
        password=redis_password,
        decode_responses=True,
    )


def _resolve_deepspeed_config(training_config: TrainingConfig) -> Optional[str]:
    """Resolve a deepspeed config path from TrainingConfig.

    支援兩種方式：
    1. 若 `training_config.deepspeed_config` 已是明確路徑，直接使用。
    2. 若 `training_config.deepspeed_profile` 存在，會從
       `service/configs/deepspeed/<profile>.json` 解析。
    
    若配置中包含 nvme offload 路徑，會清空該路徑內的舊檔案。
    """
    import json
    import shutil
    from pathlib import Path
    import tempfile

    # 1) explicit path takes precedence
    explicit = getattr(training_config, "deepspeed_config", None)
    if explicit:
        config_path = str(explicit)
    else:
        # 2) profile-based lookup — validate name before constructing path
        profile = getattr(training_config, "deepspeed_profile", None)
        if not profile:
            return None

        # Reject any profile name that contains path separators or traversal sequences
        # to prevent directory traversal attacks (e.g. "../../etc/passwd")
        if "/" in profile or "\\" in profile or ".." in profile:
            raise ValueError(
                f"[TrainingWorker] Invalid deepspeed_profile '{profile}': "
                "profile names must not contain path separators or '..'"
            )

        base = Path("service/configs/deepspeed")
        cfg_path = base / f"{profile}.json"
        # Resolve and verify the path stays within the expected directory
        resolved = cfg_path.resolve()
        expected_base = base.resolve()
        if not str(resolved).startswith(str(expected_base) + os.sep) and resolved != expected_base:
            raise ValueError(
                f"[TrainingWorker] DeepSpeed profile path '{resolved}' escapes the allowed directory"
            )
        if not resolved.is_file():
            logger.warning(f"[TrainingWorker] DeepSpeed profile '{profile}' not found at {cfg_path}")
            return None

        logger.info(f"[TrainingWorker] Using DeepSpeed profile '{profile}': {cfg_path}")
        config_path = str(cfg_path)

    # 3) Override nvme path if offload_folder is provided
    offload_folder = getattr(training_config, "offload_folder", None)
    if offload_folder:
        try:
            with open(config_path, 'r') as f:
                ds_config = json.load(f)
            
            modified = False
            abs_offload_folder = str(Path(offload_folder).resolve())
            
            if "zero_optimization" in ds_config:
                zero_opt = ds_config["zero_optimization"]
                
                # Update optimizer offload
                if "offload_optimizer" in zero_opt and zero_opt["offload_optimizer"].get("device") == "nvme":
                    zero_opt["offload_optimizer"]["nvme_path"] = abs_offload_folder
                    modified = True
                
                # Update parameter offload
                if "offload_param" in zero_opt and zero_opt["offload_param"].get("device") == "nvme":
                    zero_opt["offload_param"]["nvme_path"] = abs_offload_folder
                    modified = True
            
            if modified:
                # Create temp config file
                fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="ds_config_override_", text=True)
                with os.fdopen(fd, 'w') as f:
                    json.dump(ds_config, f, indent=2)
                
                logger.info(f"[TrainingWorker] Overridden DeepSpeed config saved to {temp_path} with nvme_path={abs_offload_folder}")
                config_path = temp_path
                
        except Exception as e:
            logger.warning(f"[TrainingWorker] Failed to override DeepSpeed config: {e}")

    # 檢查並清空 nvme offload 路徑
    try:
        with open(config_path, 'r') as f:
            ds_config = json.load(f)
        
        nvme_paths = set()
        
        # 檢查 optimizer offload nvme path
        zero_opt = ds_config.get("zero_optimization", {})
        offload_opt = zero_opt.get("offload_optimizer", {})
        if offload_opt.get("device") == "nvme":
            nvme_path = offload_opt.get("nvme_path")
            if nvme_path:
                nvme_paths.add(nvme_path)
        
        # 檢查 parameter offload nvme path
        offload_param = zero_opt.get("offload_param", {})
        if offload_param.get("device") == "nvme":
            nvme_path = offload_param.get("nvme_path")
            if nvme_path:
                nvme_paths.add(nvme_path)
        
        # Block paths that are clearly system directories — never valid offload targets.
        # Any absolute path the user deliberately configured is allowed (other disk, NVMe, etc.).
        _BLOCKED_SYSTEM_PREFIXES = (
            "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
            "/lib", "/lib64", "/boot", "/sys", "/proc", "/dev", "/run",
        )

        for nvme_path in nvme_paths:
            if not nvme_path:
                continue
            nvme_dir = Path(nvme_path).resolve()
            # Reject non-absolute sources — relative paths could unexpectedly resolve
            # against the current working directory in ways that are hard to audit.
            if not nvme_dir.is_absolute():
                logger.warning(
                    f"[TrainingWorker] Skipping nvme path '{nvme_path}': "
                    "only absolute paths are allowed for nvme offload directories"
                )
                continue
            if any(str(nvme_dir) == p or str(nvme_dir).startswith(p + "/") for p in _BLOCKED_SYSTEM_PREFIXES):
                logger.warning(
                    f"[TrainingWorker] Skipping nvme path '{nvme_path}': "
                    "path resolves to a protected system directory"
                )
                continue
            if nvme_dir.exists() and nvme_dir.is_dir():
                try:
                    # 刪除資料夾內所有內容
                    for item in nvme_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info(f"[TrainingWorker] Cleared nvme offload directory: {nvme_path}")
                except Exception as clean_err:
                    logger.warning(f"[TrainingWorker] Failed to clear nvme directory {nvme_path}: {clean_err}")
            elif not nvme_dir.exists():
                # 如果目錄不存在，創建它
                try:
                    nvme_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"[TrainingWorker] Created nvme offload directory: {nvme_path}")
                except Exception as create_err:
                    logger.warning(f"[TrainingWorker] Failed to create nvme directory {nvme_path}: {create_err}")
    
    except Exception as e:
        logger.warning(f"[TrainingWorker] Failed to process nvme paths from config: {e}")
    
    return config_path




def _cleanup_training_resources(
    trainer=None, 
    model=None, 
    base_model=None, 
    tokenizer=None, 
    dataset=None,
    deepspeed_initialized=False
):
    """徹底清理訓練相關資源，釋放 GPU 和系統記憶體
    
    Args:
        trainer: Trainer 實例
        model: 模型實例（可能是 PEFT 模型）
        base_model: 基礎模型實例
        tokenizer: Tokenizer 實例
        dataset: Dataset 實例
        deepspeed_initialized: DeepSpeed 是否已初始化
    """
    import gc
    
    logger.info("[TrainingWorker] Starting resource cleanup...")
    
    # 1. 清理 Trainer
    if trainer is not None:
        try:
            # 嘗試清理 trainer 內部狀態
            if hasattr(trainer, 'model'):
                trainer.model = None
            if hasattr(trainer, 'optimizer'):
                trainer.optimizer = None
            if hasattr(trainer, 'lr_scheduler'):
                trainer.lr_scheduler = None
            del trainer
            logger.debug("[TrainingWorker] Trainer cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Trainer cleanup warning: {e}")
    
    # 2. 清理模型
    if model is not None:
        try:
            # 如果是 PEFT 模型，先清理 adapter
            if hasattr(model, 'base_model'):
                model.base_model = None
            del model
            logger.debug("[TrainingWorker] Model cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Model cleanup warning: {e}")
    
    # 3. 清理基礎模型
    if base_model is not None:
        try:
            del base_model
            logger.debug("[TrainingWorker] Base model cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Base model cleanup warning: {e}")
    
    # 4. 清理 Tokenizer
    if tokenizer is not None:
        try:
            del tokenizer
            logger.debug("[TrainingWorker] Tokenizer cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Tokenizer cleanup warning: {e}")
    
    # 5. 清理 Dataset
    if dataset is not None:
        try:
            del dataset
            logger.debug("[TrainingWorker] Dataset cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Dataset cleanup warning: {e}")
    
    # 6. 清理 DeepSpeed distributed process group
    if deepspeed_initialized:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("[TrainingWorker] Destroyed distributed process group")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Failed to destroy process group: {e}")
    
    # 7. 強制多次 Python 垃圾回收
    for i in range(3):
        gc.collect()
    logger.debug("[TrainingWorker] Python garbage collection completed (3 passes)")
    
    # 8. 清理 CUDA 記憶體
    if torch.cuda.is_available():
        try:
            # 清空所有 CUDA 設備的 cache
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # 重置記憶體統計
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            logger.info("[TrainingWorker] CUDA memory cleared on all devices")
        except Exception as e:
            logger.warning(f"[TrainingWorker] CUDA cleanup warning: {e}")
    
    logger.info("[TrainingWorker] Resource cleanup completed")


def _convert_training_output_to_q4_k_m(
    training_config: TrainingConfig,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Convert the saved training output to GGUF and quantize it to Q4_K_M."""
    output_dir = str(training_config.output_dir)
    logger.info(
        "[TrainingWorker] Starting post-training GGUF conversion for %s",
        output_dir,
    )
    result = conversion_manager.convert_and_quantize(
        model_path=output_dir,
        output_dir=output_dir,
        intermediate_outtype="f16",
        quantization_type="Q4_K_M",
        work_dir=training_config.offload_folder,
        status_callback=status_callback,
    )
    logger.info(
        "[TrainingWorker] Post-training GGUF conversion completed: %s",
        result["quantized_output_path"],
    )
    return result


def _training_worker_process(
    request_q: Queue,
    status_q: Queue,
    stop_event: EventClass,
):
    """Worker process entry point for training.

    Commands:
        {"command": "start", "config": {...}}
        {"command": "stop"}

    Status messages from worker:
        - {"status": "starting"}      # 開始階段 (刪除先前資料與暫用檔)
        - {"status": "initializing"}  # 準備階段（載入 tokenizer/dataset/model 等）
        - {"status": "running"}       # 進入訓練主迴圈
        - {"status": "completed"}     # 訓練完成且模型已儲存
        - {"status": "stopped"}       # 收到 stop 指令後停止
        - {"status": "error", "error": str, "traceback": str, "is_oom": bool}  # 發生錯誤或 OOM
        - {"status": "progress", "step": int, "total": int, "progress": float, "loss": float|None}  # 訓練進度

    備註:
        - OOM/CUDA 錯誤時 is_oom=true，worker 可能直接退出以釋放 VRAM。
        - progress 訊息不改變 current_status，僅提供最新的訓練指標。
    """
    # Suppress "The current process just got forked..." warning from tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Workaround for PEFT multi-GPU issues: Force training to use only the first GPU.
    # This prevents "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED" and device mismatches
    # when using generic TrainingArguments without deepspeed distributed setup or when PEFT
    # fails to handle multiple visible devices correctly.
    # We set this inside the worker process so it doesn't affect the main process or inference workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Skip DeepSpeed's CUDA version mismatch check.
    # torch 2.11.0+cu130 bundles CUDA 13.0 internally, but the system-level nvcc may report
    # a different version (e.g. 12.0). DeepSpeed's JIT builder compares the two and refuses
    # to compile ops like CPUAdam when they differ. CPUAdam is a CPU-only op and does not
    # actually need nvcc, so skipping this check is safe.
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"

    # Reduce CUDA memory fragmentation by using expandable segments allocator.
    # This helps avoid OOM when there is reserved-but-unallocated memory.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    trainer: Optional[Trainer] = None
    training_config: Optional[TrainingConfig] = None
    deepspeed_initialized = False
    worker_session_id: Optional[str] = None
    redis_client = None
    _key_metrics = lambda sid: f"training:history:{sid}:metrics"
    _key_resources = lambda sid: f"training:history:{sid}:resources"

    try:
        logger.info("[TrainingWorker] process started")
        hf_token = load_hf_token()
        if hf_token:
            os.environ.setdefault("HF_HUB_TOKEN", hf_token)

        while not stop_event.is_set():
            try:
                cmd = request_q.get(timeout=1.0)
            except Empty:
                continue

            command = cmd.get("command")

            if command == "start":
                # 局部變數用於後續清理
                model = None
                base_model = None
                tokenizer = None
                try:
                    worker_session_id = cmd.get("session_id")
                    cfg_dict = cmd.get("config", {})
                    training_config = TrainingConfig(**cfg_dict)

                    current_nodes = ["start fine-tune"]
                    status_q.put({"status": "node", "node": " -> ".join(current_nodes)})

                    # Init Redis in worker for passive logging (no API required)
                    if redis and worker_session_id:
                        try:
                            redis_client = _create_redis_client()
                            redis_client.ping()
                        except Exception as e:
                            logger.warning(f"[TrainingWorker] Failed to connect to Redis: {e}")
                            redis_client = None
                    
                    ds_config = None
                    if getattr(training_config, "use_deepspeed", False):
                        ds_config = _resolve_deepspeed_config(training_config)
                        current_nodes.append("inititial deepspeed")
                        status_q.put({"status": "node", "node": " -> ".join(current_nodes)})

                    # 檢查輸出目錄是否已有檔案存在
                    from pathlib import Path
                    output_dir = Path(training_config.output_dir)
                    if output_dir.exists():
                        # 檢查目錄內是否有任何檔案或子目錄
                        dir_contents = list(output_dir.iterdir())
                        if dir_contents:
                            error_msg = (
                                f"Output directory '{output_dir}' already exists and contains files. "
                                f"Please clear the directory before starting training to avoid conflicts."
                            )
                            logger.error(f"[TrainingWorker] {error_msg}")
                            status_q.put({
                                "status": "error",
                                "error": error_msg,
                                "traceback": "",
                                "is_oom": False,
                            })
                            continue

                    status_q.put({"status": "initializing"})

                    # 1. Prepare Strategy (needs config only)
                    strategy = StrategyFactory.get_strategy(training_config, ds_config)

                    # 1.1 Initialize TrainingArguments early (Crucial for DeepSpeed/DeviceMap)
                    training_args = strategy.get_training_args()

                    # 2. Load Tokenizer
                    model_loader = ModelLoader(training_config, hf_token)
                    tokenizer = model_loader.load_tokenizer()

                    # 3. Load Dataset
                    dataset = load_training_dataset(training_config.dataset_path)

                    # 4. Preprocess Dataset (Tokenization if needed)
                    # This happens BEFORE model loading to save memory
                    dataset = strategy.preprocess_dataset(dataset, tokenizer)

                    # 5. Load Model
                    model = model_loader.load_model()

                    # 6. Prepare Trainer
                    trainer = strategy.prepare_trainer(model, tokenizer, dataset, training_args)

                    # --- training with progress callbacks ---------------------------------
                    
                    # Prime system_monitor for disk IO to establish baseline before training loop
                    # This ensures the first log captures activity since start of training.
                    # Utilizes the worker process's own SystemMonitor instance.
                    monitor_path = training_config.offload_folder if training_config.offload_folder else "/"
                    try:
                        system_monitor.get_disk_resource("usage", path=monitor_path, calc_size=False)
                    except Exception as e:
                        logger.warning(f"[TrainingWorker] Failed to prime system monitor: {e}")

                    status_q.put({"status": "running"})

                    total_steps = None
                    if trainer.state and trainer.state.max_steps:
                        total_steps = trainer.state.max_steps

                    def _log_progress():
                        """Send a lightweight progress snapshot to status_q.

                        This is called from inside the worker process only.
                        """
                        try:
                            state = trainer.state
                            if not state:
                                return
                            current_step = int(state.global_step or 0)
                            max_steps = int(state.max_steps or (total_steps or 0) or 1)
                            progress = float(current_step) / float(max_steps) if max_steps > 0 else 0.0
                            last_log = state.log_history[-1] if state.log_history else {}
                            # NOTE:
                            # `on_log` can be triggered by different events (train log / eval / save).
                            # Some log entries may not contain `loss` at all (e.g. only learning_rate/epoch),
                            # and forcing such entries to `loss=0` pollutes history.
                            # Prefer `loss`, fall back to `eval_loss`, otherwise keep it as None.
                            loss_raw = None
                            if isinstance(last_log, dict):
                                if last_log.get("loss") is not None:
                                    loss_raw = last_log.get("loss")
                                elif last_log.get("eval_loss") is not None:
                                    loss_raw = last_log.get("eval_loss")
                            loss_val: Optional[float] = None
                            if loss_raw is not None:
                                try:
                                    loss_val = float(loss_raw)
                                except Exception:
                                    loss_val = None
                            # Detect accuracy if available in log_history
                            # Evaluation logs usually have "eval_accuracy" or "accuracy"
                            acc = last_log.get("eval_accuracy") or last_log.get("mean_token_accuracy")

                            # Capture Disk IO stats for the interval since last call (logging step interval)
                            # This provides the average MB/s during the step(s), rather than instantaneous snapshot.
                            disk_read_mbps = 0.0
                            disk_write_mbps = 0.0
                            try:
                                m_path = training_config.offload_folder if training_config.offload_folder else "/"
                                d_res = system_monitor.get_disk_resource("usage", path=m_path, calc_size=False)
                                if d_res.main:
                                    disk_read_mbps = d_res.main.read_speed_mbps or 0.0
                                    disk_write_mbps = d_res.main.write_speed_mbps or 0.0
                            except Exception:
                                pass

                            # Build training log
                            ts = time.time()
                            t_log = None
                            if loss_val is not None:
                                t_log = {
                                    "timestamp": ts,
                                    "step": current_step,
                                    "loss": loss_val,
                                    "learning_rate": last_log.get("learning_rate"),
                                    "epoch": last_log.get("epoch"),
                                    "accuracy": acc,
                                }

                            # Build resource log (minimal fields, /system/resources naming)
                            cpu_payload = None
                            gpu_payload = None
                            disk_payload = None
                            try:
                                cpu_info = system_monitor.get_cpu_resource("usage", force_by_process=True)
                                gpu_info = system_monitor.get_gpu_resource("usage", force_by_process=True)
                                offload_path = training_config.offload_folder if training_config and training_config.offload_folder else None
                                should_calc = bool(offload_path) and bool(getattr(training_config, "use_deepspeed", False))
                                disk_info = system_monitor.get_disk_resource(
                                    "usage",
                                    path=offload_path if (offload_path and getattr(training_config, "use_deepspeed", False)) else "/",
                                    calc_size=should_calc,
                                )

                                if cpu_info:
                                    dram_payload = None
                                    if cpu_info.dram and cpu_info.dram.system_used_gb is not None:
                                        dram_payload = {
                                            "total_gb": cpu_info.dram.total_gb,
                                            "used_gb": cpu_info.dram.used_gb
                                            }
                                    cpu_payload = {
                                        "cpu_util_percent": cpu_info.cpu_util_percent if cpu_info.cpu_util_percent is not None else 0.0,
                                        "dram": dram_payload,
                                    }

                                if gpu_info:
                                    gpus_payload = []
                                    if gpu_info.gpus:
                                        for g in gpu_info.gpus:
                                            gpus_payload.append(
                                                {
                                                    "index": g.index,
                                                    "name": g.name,
                                                    "gpu_util": g.gpu_util if g.gpu_util is not None else 0.0,
                                                    "used_gb": g.used_gb if g.used_gb is not None else 0.0,
                                                    "total_gb": g.total_gb,
                                                    "temperature": g.temperature,
                                                }
                                            )
                                    gpu_payload = {
                                        "available": bool(getattr(gpu_info, "available", False)),
                                        "gpus": gpus_payload,
                                    }

                                if disk_info and disk_info.main:
                                    disk_payload = {
                                        "mounts": [],
                                        "main": {
                                            "total_gb": disk_info.main.total_gb,
                                            "path": disk_info.main.path,
                                            "percent": disk_info.main.percent,
                                            "read_speed_mbps": disk_read_mbps,
                                            "write_speed_mbps": disk_write_mbps,
                                            "folder_size_gb": getattr(disk_info.main, "folder_size_gb", None),
                                        },
                                    }
                            except Exception:
                                pass

                            r_log = {
                                "timestamp": ts,
                                "cpu": cpu_payload,
                                "gpu": gpu_payload,
                                "disk": disk_payload,
                            }

                            # Store to Redis directly from worker (passive by logging_steps)
                            if redis_client and worker_session_id:
                                try:
                                    # Only persist metrics logs when a valid loss is present.
                                    # This keeps `TrainingLog.loss` consistent and avoids extra `loss=0` records.
                                    if t_log is not None:
                                        redis_client.rpush(_key_metrics(worker_session_id), json.dumps(t_log))
                                    redis_client.rpush(_key_resources(worker_session_id), json.dumps(r_log))
                                except Exception as e:
                                    logger.warning(f"[TrainingWorker] Failed to push logs to Redis: {e}")

                            status_q.put(
                                {
                                    "status": "progress",
                                    "step": current_step,
                                    "total": max_steps,
                                    "progress": progress,
                                    "loss": loss_val,
                                    "learning_rate": last_log.get("learning_rate"),
                                    "epoch": last_log.get("epoch"),
                                    "accuracy": acc,
                                }
                            )
                        except Exception:
                            # progress 回報失敗不應該中斷訓練
                            pass

                    # 透過 custom callback 在 log/save 時回報進度
                    from transformers import TrainerCallback, TrainerControl, TrainerState

                    class _ProgressCallback(TrainerCallback):
                        def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
                            _log_progress()

                    trainer.add_callback(_ProgressCallback())

                    trainer.train()

                    # 7. Save Results
                    current_nodes.append("save fine-tune")
                    status_q.put({"status": "node", "node": " -> ".join(current_nodes)})
                    save_training_results(trainer, tokenizer, training_config)
                    current_nodes.append("complete")
                    status_q.put({"status": "node", "node": " -> ".join(current_nodes)})
                    
                    # 訓練成功完成後，先清理 GPU / DeepSpeed 資源，再進行 GGUF 轉換
                    _cleanup_training_resources(
                        trainer=trainer,
                        model=model,
                        base_model=base_model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        deepspeed_initialized=deepspeed_initialized
                    )
                    # 清理後重置變數
                    trainer = None
                    model = None
                    base_model = None
                    tokenizer = None
                    dataset = None

                    logger.info("[TrainingWorker] Saved training artifacts. Starting GGUF Q4_K_M conversion...")
                    def conversion_callback(step: str):
                        current_nodes.append(step)
                        status_q.put({"status": "node", "node": " -> ".join(current_nodes)})

                    conversion_result = _convert_training_output_to_q4_k_m(training_config, status_callback=conversion_callback)

                    status_q.put(
                        {
                            "status": "completed",
                            "gguf_path": conversion_result["quantized_output_path"],
                            "gguf_quantization": conversion_result["quantization_type"],
                        }
                    )
                    
                    # 訓練完成後退出 worker process，確保完全釋放資源
                    logger.info("[TrainingWorker] Training completed successfully, exiting worker process...")
                    return  # 退出 worker process

                except Exception as e:  # start command failure
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(f"[TrainingWorker] Training failed: {e}\n{tb}")

                    # 檢查是否為 CUDA OOM / CUDA 錯誤
                    error_str = str(e).strip()
                    if not error_str:
                        try:
                            error_str = str(getattr(e, "args", [""])[0]).strip()
                        except Exception:
                            error_str = ""
                    if not error_str:
                        try:
                            error_str = tb.strip().splitlines()[-1]
                        except Exception:
                            error_str = "Unknown error"
                    lower = error_str.lower()
                    is_oom = False
                    try:
                        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                            isinstance(e, RuntimeError)
                            and ("out of memory" in lower or "cuda" in lower or "oom" in lower)
                        )
                    except Exception:
                        is_oom = False

                    # 先回報一則一般錯誤，方便 manager 記錄 traceback
                    status_q.put(
                        {
                            "status": "error",
                            "error": error_str,
                            "traceback": tb,
                            "is_oom": is_oom,
                        }
                    )

                    # 無論是否 OOM，都先進行資源清理
                    logger.info("[TrainingWorker] Cleaning up resources after error...")
                    _cleanup_training_resources(
                        trainer=trainer,
                        model=model,
                        base_model=base_model,
                        tokenizer=tokenizer,
                        dataset=dataset if 'dataset' in locals() else None,
                        deepspeed_initialized=deepspeed_initialized
                    )

                    if is_oom:
                        # 若為 OOM / CUDA 錯誤，直接退出整個進程以確保 VRAM 完全釋放
                        logger.error(
                            "[TrainingWorker] OOM/CUDA error detected, forcing process exit to release VRAM..."
                        )
                        try:
                            status_q.put(
                                {
                                    "status": "error",
                                    "error": f"OOM Error: {error_str}",
                                    "traceback": tb,
                                    "is_oom": True
                                }
                            )
                        except Exception:
                            pass

                        # 直接結束進程，確保 CUDA context 釋放
                        import sys
                        sys.exit(1)
                    else:
                        # 非 OOM 錯誤，清理後也退出 worker process
                        logger.info("[TrainingWorker] Error occurred, exiting worker process after cleanup...")
                        return  # 退出 worker process

            elif command == "stop":
                logger.info("[TrainingWorker] stop command received")
                if trainer is not None:
                    try:
                        trainer._max_steps = trainer.state.global_step  # type: ignore[attr-defined]
                    except Exception:
                        pass
                
                # 清理訓練資源
                _cleanup_training_resources(
                    trainer=trainer,
                    model=model if 'model' in locals() else None,
                    base_model=base_model if 'base_model' in locals() else None,
                    tokenizer=tokenizer if 'tokenizer' in locals() else None,
                    dataset=dataset if 'dataset' in locals() else None,
                    deepspeed_initialized=deepspeed_initialized
                )
                trainer = None
                
                status_q.put({"status": "stopped"})
                
                # Stop 命令後退出 worker process
                logger.info("[TrainingWorker] Stop command processed, exiting worker process...")
                return  # 退出 worker process

        # 在正常結束循環時也進行最終清理
        logger.info("[TrainingWorker] Worker loop ended, performing final cleanup...")
        _cleanup_training_resources(
            trainer=trainer,
            model=model if 'model' in locals() else None,
            base_model=base_model if 'base_model' in locals() else None,
            tokenizer=tokenizer if 'tokenizer' in locals() else None,
            dataset=dataset if 'dataset' in locals() else None,
            deepspeed_initialized=deepspeed_initialized
        )

        logger.info("[TrainingWorker] stop_event set; worker exiting")

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(f"[TrainingWorker] Fatal error: {e}\n{tb}")
        try:
            status_q.put({"status": "error", "error": str(e), "traceback": tb})
        except Exception:
            pass
        
        # 在 fatal error 發生時也要清理資源
        try:
            logger.info("[TrainingWorker] Cleaning up resources after fatal error...")
            _cleanup_training_resources(
                trainer=trainer if 'trainer' in locals() else None,
                model=model if 'model' in locals() else None,
                base_model=base_model if 'base_model' in locals() else None,
                tokenizer=tokenizer if 'tokenizer' in locals() else None,
                dataset=dataset if 'dataset' in locals() else None,
                deepspeed_initialized=deepspeed_initialized
            )
        except Exception as cleanup_err:
            logger.error(f"[TrainingWorker] Cleanup after fatal error failed: {cleanup_err}")
    
    finally:
        # Final cleanup: 最後一道防線，確保 distributed process group 被清理
        try:
            if deepspeed_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("[TrainingWorker] Destroyed distributed process group in finally block")
        except Exception as final_err:
            logger.debug(f"[TrainingWorker] Final cleanup: {final_err}")


class TrainingProcessManager:
    """Manage a dedicated training worker process.

    This mirrors `ModelInferenceProcess` but focused on a single training job
    at a time. It is used by `TrainingManager` to decouple heavy GPU work from
    the FastAPI process.
    """

    def __init__(self) -> None:
        self.process: Optional[Process] = None
        self.request_q: Optional[Queue] = None
        self.status_q: Optional[Queue] = None
        self.stop_event: Optional[EventClass] = None

        # current_status 狀態說明:
        # - "idle": 閒置中 (初始狀態或已重置)
        # - "initializing": 正在初始化 (載入模型/數據集)
        # - "running": 正在訓練中
        # - "completed": 訓練正常完成
        # - "stopped": 訓練被手動停止
        # - "error": 訓練發生錯誤 (OOM 或其他例外)
        self.current_status: str = "idle"
        self.last_error: Optional[str] = None
        self.last_traceback: Optional[str] = None
        self.current_config: Optional[TrainingConfig] = None
        # progress fields
        self.current_step: int = 0
        self.total_steps: int = 0
        self.progress: float = 0.0
        self.loss: Optional[float] = None
        self.current_epoch: float = 0.0
        # track OOM flag for last error (optional, for future /training/error_details)
        self.last_is_oom = False

        # In-memory history fallback
        self.history: Dict[str, Dict[str, Any]] = {}

        # Redis connection
        self.redis_client = None
        if redis:
            try:
                self.redis_client = _create_redis_client()
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None

        # Format helpers
        self._key_metrics = lambda sid: f"training:history:{sid}:metrics"
        self._key_resources = lambda sid: f"training:history:{sid}:resources"

        self.current_session_id: Optional[str] = None


    def _ensure_process(self) -> None:
        if self.process and self.process.is_alive():
            return
        self.request_q = Queue()
        self.status_q = Queue()
        self.stop_event = Event()
        self.process = Process(
            target=_training_worker_process,
            args=(self.request_q, self.status_q, self.stop_event),
            daemon=True,
        )
        self.process.start()
        self.current_status = "idle"
        self.last_error = None
        self.last_traceback = None

    def start_training(self, config: TrainingConfig) -> Dict[str, Any]:
        if self.current_status in {"running", "initializing"}:
            raise RuntimeError("Training already in progress")
        
        # 清空之前的訓練狀態，避免抓到先前遺留的結果
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        self.loss = None
        self.current_epoch = 0.0
        self.total_epochs = 0
        self.last_error = None
        self.last_traceback = None
        self.last_is_oom = False
        logger.info("Cleared previous training status before starting new training")
        
        self._ensure_process()
        self.last_error = "start fine-tune"
        assert self.request_q is not None
        self.current_config = config
        
        # Setup session and history
        self.current_session_id = str(uuid.uuid4())
        self.history[self.current_session_id] = {
            "training_logs": [],
            "resource_logs": [],
            "config": config.model_dump(),
            "start_time": time.time()
        }
        
        # Clear Redis history if exists (unlikely for new UUID)
        if self.redis_client:
            try:
                self.redis_client.delete(self._key_metrics(self.current_session_id))
                self.redis_client.delete(self._key_resources(self.current_session_id))
            except Exception:
                pass

        self.request_q.put({"command": "start", "config": config.model_dump(), "session_id": self.current_session_id})
        self.current_status = "starting"
        return {
            "status": "starting",
            "session_id": self.current_session_id
        }


    def stop_training(self) -> Dict[str, Any]:
        if not self.process:
            return {"status": "not_running"}
            
        logger.info("[TrainingManager] Force stopping training process...")
        
        # 直接終止進程，不等待 queue
        if self.process.is_alive():
            try:
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    logger.warning("[TrainingManager] Process did not terminate, killing it...")
                    self.process.kill()
                    self.process.join(timeout=1)
            except Exception as e:
                logger.error(f"[TrainingManager] Error stopping process: {e}")
        
        # 清理資源引用
        self.process = None
        self.request_q = None
        self.status_q = None
        self.stop_event = None
        
        self.current_status = "stopped"
        return {"status": "stopped"}

    def _drain_status(self) -> None:
        if not self.status_q:
            return
        while True:
            try:
                msg = self.status_q.get_nowait()
            except Empty:
                break
            except (AttributeError, ValueError):
                # Queue might be closed or None during cleanup race condition
                break
            status = msg.get("status")
            # Only update current_status for state-changing statuses, not "progress" or "node"
            if status and status not in {"progress", "node"}:
                self.current_status = status

            if status == "node":
                self.last_error = msg.get("node")

            if status == "error":
                self.last_error = msg.get("error")
                self.last_traceback = msg.get("traceback")
                self.last_is_oom = bool(msg.get("is_oom"))

                # 若 worker 回報 OOM 或其他致命錯誤，停止訓練並清理進程
                # 讓下一次訓練可以在乾淨的 worker 上重新啟动
                if self.last_is_oom:
                    logger.error("[TrainingManager] CUDA OOM detected in training worker; stopping training.")
                else:
                    logger.error("[TrainingManager] Training worker reported error; stopping training.")
                # 呼叫 stop_training() 來觸发清理流程
                self.stop_training()
            elif status == "progress":
                # incremental progress report from worker
                self.current_step = int(msg.get("step", self.current_step))
                self.total_steps = int(msg.get("total", self.total_steps or 0) or 0)
                self.progress = float(msg.get("progress", self.progress))
                loss_val = msg.get("loss")
                lr = msg.get("learning_rate")
                epoch_val = msg.get("epoch")

                if loss_val is not None:
                    try:
                        self.loss = float(loss_val)
                    except Exception:
                        pass
                
                if epoch_val is not None:
                    try:
                        self.current_epoch = float(epoch_val)
                    except Exception:
                        pass

    def get_status(self) -> TrainingStatus:
        # Check if worker process died unexpectedly (e.g., from sys.exit(1) on OOM)
        if self.process and not self.process.is_alive():
            if self.current_status in {"running", "initializing"}:
                # Worker was active but died - likely OOM or fatal error
                if not self.last_error:
                    self.last_error = "Worker process terminated unexpectedly"
                self.current_status = "error"
                # Clean up the dead process
                self.cleanup()
        
        self._drain_status()
        if self.current_status not in {"starting", "running", "initializing"}:
            # Training not active, but preserve progress info if completed
            return TrainingStatus(
                is_training=False,
                progress=self.progress,
                current_step=self.current_step,
                total_steps=self.total_steps,
                loss=self.loss,
                current_epoch=self.current_epoch,
                total_epochs=self.current_config.num_train_epochs if self.current_config else None,
                status=self.current_status,
                session_id=self.current_session_id,
                error=self.last_error,  # 添加錯誤信息
                config=None, # 僅在訓練國過程中返回 config
            )
        return TrainingStatus(
            is_training=True,
            progress=self.progress,
            current_step=self.current_step,
            total_steps=self.total_steps,
            loss=self.loss,
            current_epoch=self.current_epoch,
            total_epochs=self.current_config.num_train_epochs if self.current_config else None,
            status=self.current_status,
            session_id=self.current_session_id,
            error=self.last_error,  # 訓練中將進度狀態傳給 error 供前端顯示
            config=self.current_config,
        )
    
    def get_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        # 1. Try Redis
        if self.redis_client:
            try:
                metrics_key = self._key_metrics(session_id)
                resources_key = self._key_resources(session_id)
                
                # Fetch all list items
                m_list = self.redis_client.lrange(metrics_key, 0, -1)
                r_list = self.redis_client.lrange(resources_key, 0, -1)
                
                # Redis returns list of strings, parse them
                training_logs = [json.loads(x) for x in m_list] if m_list else []
                resource_logs = [json.loads(x) for x in r_list] if r_list else []
                
                if training_logs or resource_logs:
                    return {
                        "training_logs": training_logs,
                        "resource_logs": resource_logs,
                        "session_id": session_id
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch history from Redis: {e}")
        
        # 2. Fallback to in-memory
        return self.history.get(session_id)
    
    def get_error_details(self) -> Optional[Dict[str, Any]]:
        """
        獲取詳細的錯誤信息（包括完整的 traceback）
        格式與 inference/error_details 一致
        """
        if not self.last_error:
            return None
        
        return {
            "error": self.last_error,
            "error_type": "OOM" if self.last_is_oom else "TrainingError",
            "is_oom": self.last_is_oom,
            "error_traceback": self.last_traceback,
            "process_alive": self.process.is_alive() if self.process else False,
        }

    def cleanup(self) -> None:
        if self.process and self.process.is_alive():
            try:
                if self.stop_event is not None:
                    self.stop_event.set()
                self.process.terminate()
                self.process.join(timeout=5)
            except Exception:
                pass
        self.process = None
        self.request_q = None
        self.status_q = None
        self.stop_event = None
        self.current_status = "idle"
        self.last_error = None
        self.last_traceback = None
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        self.loss = None
        self.current_epoch = 0.0


training_process_manager = TrainingProcessManager()
