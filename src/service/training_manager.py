"""High-level Training Manager facade.

This module exposes the same interface used by `service.app` but delegates the
actual heavy training work to a dedicated worker process implemented in
`service/training/training_process.py`.

Design goals:
- Support LoRA / QLoRA / full-parameter finetuning using HuggingFace Trainer.
- Allow DeepSpeed ZeRO / ZeRO-Infinity configs to be plugged via TrainingConfig.
- Run finetune inside a separate process (like inference) so that blocking or
  OOM in training will not freeze the main FastAPI process.
"""

from pathlib import Path
from threading import Lock
from typing import Optional

from .config_models import TrainingConfig, TrainingStatus
from .settings import configure_logging
from .training.training_process import training_process_manager

logger = configure_logging(__name__)


class TrainingManager:
    """Singleton facade used by FastAPI routes.

    Public API is intentionally small and stable:
    - start_training(config: TrainingConfig) -> dict
    - get_status() -> TrainingStatus
    - stop_training() -> dict

    Internally this forwards calls to `training_process_manager`, which owns
    the worker process and does all GPU-heavy work.
    """

    _instance: Optional["TrainingManager"] = None
    _lock: Lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._config_dir = Path("service/configs")
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info("TrainingManager initialized (process-based)")

    # --- public facade methods -------------------------------------------------

    def start_training(self, config: TrainingConfig):
        """開始訓練（非阻塞）

        This will:
        - Save the current training configuration for debugging/inspection.
        - Ask `TrainingProcessManager` to start a worker process if needed.
        - Send a `start` command with the TrainingConfig to the worker.

        Returns a small dict for the API handler.
        """
        # 保留與舊實作相同的並發保護語意：一次只允許一個訓練
        with self._lock:
            status = training_process_manager.get_status()
            if status.is_training:
                raise RuntimeError("Training is already in progress")

            self._save_config(config)
            logger.info(f"Starting training with method={config.method}, model={config.model_name}")

            result = training_process_manager.start_training(config)
            logger.info("Training start command sent to worker process")
            return result

    def get_status(self) -> TrainingStatus:
        """獲取訓練狀態（由 worker process 回報的快照）。"""
        return training_process_manager.get_status()
    
    def get_history(self, session_id: str):
        """Get full training history (system metrics & progress)."""
        return training_process_manager.get_history(session_id)
    
    def get_error_details(self):
        """
        獲取詳細的錯誤信息（包括完整的 traceback）
        格式與 inference/error_details 一致
        """
        return training_process_manager.get_error_details()

    def stop_training(self):
        """請求停止訓練（非同步，嘗試優雅結束 worker）。"""
        with self._lock:
            result = training_process_manager.stop_training()
            logger.info("Training stop requested via worker process")
            return result

    # --- internal helpers ------------------------------------------------------

    def _save_config(self, config: TrainingConfig) -> None:
        """Persist the last requested training configuration for debugging."""
        import json

        cfg_path = self._config_dir / "current_training_config.json"
        try:
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Training configuration saved to {cfg_path}")
        except Exception as e:
            logger.warning(f"Failed to save training configuration: {e}")


# Singleton instance used by app.py
training_manager = TrainingManager()
