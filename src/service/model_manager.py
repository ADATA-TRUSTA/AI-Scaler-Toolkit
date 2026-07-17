"""
Model Manager with Singleton Pattern for Inference
Uses separate process for model loading/inference to avoid OOM VRAM issues
"""

import torch
from typing import Any, Dict, Optional, Iterator, List
from pathlib import Path
from threading import Lock

from .config_models import InferenceConfig
from .inference.model_inference_process import ModelInferenceProcess
from .settings import configure_logging

logger = configure_logging(__name__)


class ModelManager:
    """
    Singleton Model Manager for inference
    使用獨立進程進行模型加載和推理，確保 OOM 時能完全清理 VRAM
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize singleton internal state once."""
        if getattr(self, "_initialized", False):
            return

        # 使用獨立進程管理模型
        self.inference_process = ModelInferenceProcess()
        self.config: Optional[InferenceConfig] = None
        self.pending_config: Optional[InferenceConfig] = None
        self._initialized = True

    # ---------------------- Utility Helpers ----------------------
    def _save_config(self, config: InferenceConfig) -> None:
        """Persist current inference config to configs/current_inference_config.json.
        Safe best-effort; logs warning on error instead of raising.
        """
        try:
            base_dir = Path(__file__).parent / "configs"
            base_dir.mkdir(parents=True, exist_ok=True)
            path = base_dir / "current_inference_config.json"
            path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist inference config: {e}")

    def prepare_config(self, config: InferenceConfig):
        """第一階段：僅檢查與存檔配置，並將 config 設為當前/待載入。
        不執行 tokenizer / model 權重下載。讓 /inference/status 立即可見設定。
        """
        with self._lock:
            if self.is_loaded():
                raise RuntimeError("Model already loaded. Please unload first.")
            if self.inference_process.current_status == "loading":
                raise RuntimeError("Model is already loading.")

            # 保存配置檔案並更新狀態
            self._save_config(config)
            self.config = config  # 立即對外回報使用者設定
            self.pending_config = config
        return True

    def start_loading(self, config: InferenceConfig):
        """外部呼叫：分階段載入，立即設置 config，在獨立進程中執行權重載入。"""
        self.prepare_config(config)

        # 在獨立進程中加載模型
        self.inference_process.load_model(config)

        return True

    def unload_model(self):
        """卸載模型"""
        with self._lock:
            # 檢查是否有模型已加載或正在加載
            status = self.inference_process.get_status()

            if not status.get("loaded") and not status.get("is_loading"):
                logger.info("No model loaded to unload (idempotent success).")
                return {"status": "success", "message": "No model loaded"}

            logger.info("Unloading model...")

            # 卸載進程中的模型
            self.inference_process.unload_model()

            self.config = None
            self.pending_config = None

            logger.info("✅ Model unloaded successfully")
            return {"status": "success", "message": "Model unloaded successfully"}

    def stop_generation(self, request_id: Optional[str] = None):
        """停止当前正在进行的生成"""
        return self.inference_process.stop_generation(request_id=request_id)

    def is_loaded(self) -> bool:
        """檢查模型是否已加載"""
        return self.inference_process.is_loaded()

    def get_tokenizer(self):
        """獲取 tokenizer 實例

        注意：由於 tokenizer 在獨立進程中，這裡返回一個代理對象，
        僅支持 apply_chat_template 等常用方法
        """
        if not self.is_loaded():
            return None

        # 返回一個代理對象，提供 tokenizer 的常用方法
        return self.inference_process.get_tokenizer_proxy()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: Optional[str] = None,
        total_timeout: int = 300,
        enable_thinking: Optional[bool] = None,
        images: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        """
        生成文本（非串流）

        Args:
            total_timeout: 生成總超時時間（秒），預設 300 秒
            enable_thinking: 是否啟用思考模式
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Please load a model first.")

        # Format prompt with system prompt if provided
        if isinstance(prompt, str):
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
        else:
            # If prompt is a list (messages), we assume it's self-contained
            full_prompt = prompt

        # 發送到推理進程
        params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "total_timeout": total_timeout,
            "enable_thinking": enable_thinking,
            "images": images,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        return self.inference_process.generate(
            full_prompt, params, request_id=request_id
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: Optional[str] = None,
        total_timeout: int = 300,
        enable_thinking: Optional[bool] = None,
        images: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """
        生成文本（串流模式）

        Args:
            total_timeout: 生成總超時時間（秒），預設 300 秒
            enable_thinking: 是否啟用思考模式
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Please load a model first.")

        # Format prompt with system prompt if provided
        if isinstance(prompt, str):
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
        else:
            # If prompt is a list (messages), we assume it's self-contained
            full_prompt = prompt

        logger.debug(f"Starting generate_stream with prompt: {full_prompt[:100]}...")  # Log first 100 chars of prompt

        # 發送到推理進程（生成器）
        params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "total_timeout": total_timeout,
            "enable_thinking": enable_thinking,
            "images": images,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        for chunk in self.inference_process.generate_stream(
            full_prompt, params, request_id=request_id
        ):
            yield chunk

    def get_status(self) -> dict:
        """獲取當前狀態"""
        # 從推理進程獲取狀態
        process_status = self.inference_process.get_status()

        # 如果在載入中，優先回報 pending_config；否則回報已載入的 config
        cfg = self.pending_config if process_status.get("is_loading") else self.config

        # 當模型載入完成，清除 pending_config
        if process_status.get("loaded") and self.pending_config is not None:
            self.pending_config = None

        status = {
            "loaded": process_status.get("loaded", False),
            "is_loading": process_status.get("is_loading", False),
            "loading_error": process_status.get("loading_error"),
            "error_type": process_status.get("error_type"),
            "is_oom": process_status.get("is_oom", False),
            "model_name": cfg.model_name if cfg else None,
            "model_path": cfg.model_path if cfg else None,
            "engine": cfg.engine if cfg else "transformers",
            "quantization": cfg.quantization if cfg else None,
            "model_total_memory": cfg.model_total_memory if cfg else None,
            "device_map": cfg.device_map if cfg else None,
            "max_memory": cfg.max_memory if cfg else None,
            "offload_folder": cfg.offload_folder if cfg else None,
            "device": process_status.get("device"),
            "process_alive": process_status.get("process_alive", False),
            "device_allocation": process_status.get(
                "device_allocation"
            ),  # 新增：設備分配統計資訊
            "n_gpu_layers": cfg.n_gpu_layers if cfg else None,
            "n_ctx": cfg.n_ctx if cfg else None,
            "n_batch": cfg.n_batch if cfg else None,
            "llama_server_extra_args": cfg.llama_server_extra_args if cfg else None,
            "vllm_gpu_memory_utilization": (
                cfg.vllm_gpu_memory_utilization if cfg else None
            ),
            "vllm_max_model_len": cfg.vllm_max_model_len if cfg else None,
            "vllm_dtype": cfg.vllm_dtype if cfg else None,
            "vllm_quantization": cfg.vllm_quantization if cfg else None,
            "vllm_enforce_eager": cfg.vllm_enforce_eager if cfg else None,
            "vllm_kv_cache_dtype": cfg.vllm_kv_cache_dtype if cfg else None,
            "vllm_cpu_offload_gb": cfg.vllm_cpu_offload_gb if cfg else None,
            "vllm_kv_offloading_size": (
                cfg.vllm_kv_offloading_size if cfg else None
            ),
            "vllm_tensor_parallel_size": cfg.vllm_tensor_parallel_size if cfg else None,
            "vllm_max_num_seqs": cfg.vllm_max_num_seqs if cfg else None,
            "vllm_max_num_batched_tokens": (
                cfg.vllm_max_num_batched_tokens if cfg else None
            ),
            "vllm_mm_image_limit": cfg.vllm_mm_image_limit if cfg else None,
            "vllm_mm_audio_limit": cfg.vllm_mm_audio_limit if cfg else None,
            "vllm_mm_video_limit": cfg.vllm_mm_video_limit if cfg else None,
            "vllm_hf_overrides": cfg.vllm_hf_overrides if cfg else None,
            "vllm_chat_template": cfg.vllm_chat_template if cfg else None,
        }

        # GPU 記憶體使用情況：優先使用 worker 進程報告的數據
        memory_usage = process_status.get("memory_usage")
        if memory_usage:
            # Worker 進程已報告 GPU 用量（正確的跨進程數據）
            status["memory_usage"] = memory_usage
        elif torch.cuda.is_available() and process_status.get("loaded"):
            # Fallback：嘗試在主進程獲取（注意：這只能看到主進程的用量，通常為 0）
            try:
                status["memory_usage"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
            except:
                pass

        return status

    def get_error_details(self) -> Optional[dict]:
        """獲取詳細的錯誤信息（包括完整的 traceback）"""
        return self.inference_process.get_error_details()

    def cleanup(self):
        """清理資源（應用關閉時調用）"""
        logger.info("Cleaning up ModelManager...")

        try:
            # 停止推理進程
            if self.inference_process:
                self.inference_process.stop_process()
                logger.info("Inference process stopped")
        except Exception as e:
            logger.error(f"Error stopping inference process: {e}")

        # 清理配置
        self.config = None
        self.pending_config = None

        logger.info("ModelManager cleanup completed")

    def __del__(self):
        """析構函數 - 確保資源被清理"""
        try:
            if hasattr(self, "inference_process"):
                self.cleanup()
        except:
            pass

    def force_cleanup_gpu(self):
        """
        強制清理 GPU 記憶體
        當進程因 OOM 崩潰後，強制終止並重啟一個乾淨的進程
        """
        logger.warning("Force cleanup GPU - terminating worker process...")

        with self._lock:
            # 強制停止當前進程
            if (
                self.inference_process.process
                and self.inference_process.process.is_alive()
            ):
                try:
                    self.inference_process.process.terminate()
                    self.inference_process.process.join(timeout=3)
                    if self.inference_process.process.is_alive():
                        self.inference_process.process.kill()
                        self.inference_process.process.join()
                except Exception as e:
                    logger.error(f"Error force terminating process: {e}")

            # 清理狀態
            self.inference_process._cleanup_dead_process()
            self.inference_process.current_status = "idle"
            self.config = None
            self.pending_config = None

            logger.info("✅ GPU force cleanup completed - worker process terminated")
            return {"status": "success", "message": "GPU memory force cleaned"}

    def cleanup_generation_memory(self, slot: Optional[int] = None):
        """軟性清理生成階段暫存記憶體，不卸載模型。

        Returns: dict status payload
        """
        try:
            if not self.is_loaded():
                return {"status": "error", "message": "No model loaded"}
            result = self.inference_process.cleanup_generation_memory(slot=slot)
            return result
        except Exception as e:
            logger.error(f"cleanup_generation_memory failed: {e}")
            return {"status": "error", "message": str(e)}


# Singleton instance
model_manager = ModelManager()
