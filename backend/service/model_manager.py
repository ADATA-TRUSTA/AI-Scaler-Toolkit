"""
Model Manager with Singleton Pattern for Inference
Uses separate process for model loading/inference to avoid OOM VRAM issues.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from threading import Lock
from typing import Any, cast

import torch

from .config_models import InferenceConfig
from .inference.async_server_client import (
    AsyncServerClient,
    client_for_config,
    is_server_engine,
    resolve_server_endpoint,
)
from .inference.model_inference_process import ModelInferenceProcess
from .settings import configure_logging

logger = configure_logging(__name__)


class ModelManager:
    """
    Singleton Model Manager for inference
    Loads and runs the model in a separate process so VRAM can be fully reclaimed on OOM.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "ModelManager":
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize singleton internal state once."""
        if getattr(self, "_initialized", False):
            return

        # Manage the model in a separate process
        self.inference_process = ModelInferenceProcess()
        self.config: InferenceConfig | None = None
        self.pending_config: InferenceConfig | None = None

        # Async HTTP path for server engines (llama-server / vLLM): the main
        # process talks to the managed server directly, bypassing the worker
        # queue. Transformers stays on the worker path.
        self._async_client: AsyncServerClient | None = None
        self._async_client_endpoint: dict[str, Any] | None = None
        self._stale_async_clients: list[AsyncServerClient] = []
        self._async_stop_events: dict[str, asyncio.Event] = {}
        self._async_stop_lock = Lock()

        self._initialized = True

    # ---------------------- Async server-engine path ----------------------
    def uses_async_http(self) -> bool:
        """True when the loaded engine is reachable via direct async HTTP."""
        return self.is_loaded() and is_server_engine(self.config)

    def _get_async_client(self) -> AsyncServerClient:
        """
        Return the cached AsyncServerClient for the active server engine.

        Rebuilds only when the resolved endpoint (base_url or api_key) changed
        — e.g. a hot model swap between engines or a port change. Crucially it
        compares the resolved endpoint (via resolve_server_endpoint) instead of
        building a throwaway client every call, and queues any replaced client
        so its keep-alive connections get closed instead of leaking.
        """
        if self.config is None:
            raise RuntimeError("No model configuration loaded")
        endpoint = resolve_server_endpoint(self.config)
        if self._async_client is None or self._async_client_endpoint != endpoint:
            if self._async_client is not None:
                self._stale_async_clients.append(self._async_client)
            self._async_client = client_for_config(self.config)
            self._async_client_endpoint = endpoint
        return self._async_client

    async def _acquire_async_client(self) -> AsyncServerClient:
        """
        Async wrapper around _get_async_client that also closes any client
        replaced by an endpoint change (aclose can only be awaited here)."""
        client = self._get_async_client()
        if self._stale_async_clients:
            stale, self._stale_async_clients = self._stale_async_clients, []
            for old in stale:
                try:
                    await old.aclose()
                except Exception:  # pragma: no cover - defensive
                    pass
        return client

    def _register_async_stop(self, request_id: str | None) -> asyncio.Event | None:
        if not request_id:
            return None
        event = asyncio.Event()
        with self._async_stop_lock:
            self._async_stop_events[request_id] = event
        return event

    def _unregister_async_stop(self, request_id: str | None) -> None:
        if not request_id:
            return
        with self._async_stop_lock:
            self._async_stop_events.pop(request_id, None)

    def _signal_async_stop(self, request_id: str | None) -> bool:
        """
        Set the stop event for an in-flight async request.

        With no request_id, broadcast to every registered async request
        (stop-all, matching the old worker-path behaviour). Returns True if at
        least one event was signalled.
        """
        with self._async_stop_lock:
            if request_id:
                event = self._async_stop_events.get(request_id)
                events = [event] if event is not None else []
            else:
                events = list(self._async_stop_events.values())
        for event in events:
            event.set()
        return bool(events)

    async def aclose_async_client(self) -> None:
        """Close the active and any stale async clients, releasing connections."""
        clients = [c for c in ([self._async_client] + self._stale_async_clients) if c is not None]
        self._async_client = None
        self._async_client_endpoint = None
        self._stale_async_clients = []
        for client in clients:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    def _close_async_clients_best_effort(self) -> None:
        """
        Close the active + stale async clients from a sync context.

        Schedules the async ``aclose`` on the running event loop if there is one;
        otherwise drops the references (httpx AsyncClient closes itself on GC).
        """
        clients = [c for c in ([self._async_client] + self._stale_async_clients) if c is not None]
        self._async_client = None
        self._async_client_endpoint = None
        self._stale_async_clients = []
        if not clients:
            return

        async def _close_all() -> None:
            for client in clients:
                try:
                    await client.aclose()
                except Exception:  # pragma: no cover - defensive
                    pass

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_close_all())
        except RuntimeError:
            pass  # no running loop; GC will close the underlying httpx clients

    # ---------------------- Utility Helpers ----------------------
    def _save_config(self, config: InferenceConfig) -> None:
        """
        Persist current inference config to configs/current_inference_config.json.
        Safe best-effort; logs warning on error instead of raising.
        """
        try:
            base_dir = Path(__file__).parent / "configs"
            base_dir.mkdir(parents=True, exist_ok=True)
            path = base_dir / "current_inference_config.json"
            path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist inference config: {e}")

    def prepare_config(self, config: InferenceConfig) -> bool:
        """
        Stage one: validate and persist the config, then mark it current/pending.

        No tokenizer or model weight download happens here, so /inference/status
        reflects the settings immediately.
        """
        with self._lock:
            if self.is_loaded():
                raise RuntimeError("Model already loaded. Please unload first.")
            if self.inference_process.current_status == "loading":
                raise RuntimeError("Model is already loading.")

            # Persist the config file and update state
            self._save_config(config)
            self.config = config  # report the user's settings right away
            self.pending_config = config
        return True

    def start_loading(self, config: InferenceConfig) -> bool:
        """Public entry: staged load — set config immediately, load weights in the worker process."""
        self.prepare_config(config)

        # Load the model in the separate process
        self.inference_process.load_model(config)

        return True

    def unload_model(self) -> dict:
        """Unload the model."""
        with self._lock:
            # Check whether a model is loaded or still loading
            status = self.inference_process.get_status()

            if not status.get("loaded") and not status.get("is_loading"):
                logger.info("No model loaded to unload (idempotent success).")
                return {"status": "success", "message": "No model loaded"}

            logger.info("Unloading model...")

            # Unload the model inside the worker process
            self.inference_process.unload_model()

            self.config = None
            self.pending_config = None
            # Close the cached async client (and any pending stale ones) so the
            # httpx keep-alive connections are released rather than leaked. This
            # is sync, so schedule the async close on the running loop when there
            # is one; otherwise fall back to GC.
            self._close_async_clients_best_effort()

            logger.info("✅ Model unloaded successfully")
            return {"status": "success", "message": "Model unloaded successfully"}

    def stop_generation(self, request_id: str | None = None) -> dict:
        """
        Stop in-flight generation(s).

        Async HTTP path: signal the request's stop event (every registered
        event when no request_id is given). A request-scoped hit resolves here;
        otherwise fall through to the worker path so stop-all reaches both.
        """
        async_stopped = self._signal_async_stop(request_id)
        if async_stopped and request_id:
            return {
                "status": "success",
                "message": "Stop signal sent to async generation",
                "request_id": request_id,
            }
        result = self.inference_process.stop_generation(request_id=request_id)
        if async_stopped and result.get("status") != "success":
            # Stop-all with nothing stoppable on the worker: the async signal
            # already succeeded, so don't surface the worker-side error.
            return {
                "status": "success",
                "message": "Stop signal sent to async generation(s)",
            }
        return result

    def is_loaded(self) -> bool:
        """Check whether a model is loaded."""
        return self.inference_process.is_loaded()

    def get_tokenizer(self) -> Any:  # noqa: ANN401 - duck-typed tokenizer proxy from worker process
        """
        Get the tokenizer instance.

        Note: the tokenizer lives in a separate process, so this returns a proxy
        object that only supports common methods such as apply_chat_template.
        """
        if not self.is_loaded():
            return None

        # Return a proxy object exposing the tokenizer's common methods
        return self.inference_process.get_tokenizer_proxy()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
        total_timeout: int = 300,
        enable_thinking: bool | None = None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,  # noqa: ANN401 - OpenAI-style tool selection, passed through
        request_id: str | None = None,
    ) -> dict:
        """
        Generate text (non-streaming).

        Args:
            total_timeout: Total generation timeout in seconds (default 300)
            enable_thinking: Whether to enable thinking mode
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

        # Send to the inference process
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

        # inference_process.generate is annotated -> str but returns a result dict at runtime.
        return cast(
            dict, self.inference_process.generate(full_prompt, params, request_id=request_id)
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
        total_timeout: int = 300,
        enable_thinking: bool | None = None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,  # noqa: ANN401 - OpenAI-style tool selection, passed through
        request_id: str | None = None,
    ) -> Iterator[dict]:
        """
        Generate text (streaming).

        Args:
            total_timeout: Total generation timeout in seconds (default 300)
            enable_thinking: Whether to enable thinking mode
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

        logger.debug(
            f"Starting generate_stream with prompt: {full_prompt[:100]}..."
        )  # Log first 100 chars of prompt

        # Send to the inference process (generator)
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

        yield from self.inference_process.generate_stream(
            full_prompt, params, request_id=request_id
        )

    def _build_params(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        total_timeout: int,
        enable_thinking: bool | None,
        images: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,  # noqa: ANN401 - OpenAI-style tool selection, passed through
    ) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _as_messages(
        prompt: str | list[dict[str, Any]], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        """Normalise a prompt into OpenAI chat messages for the async path."""
        if isinstance(prompt, list):
            messages = list(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        return messages

    async def agenerate(
        self,
        prompt: str | list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
        total_timeout: int = 300,
        enable_thinking: bool | None = None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,  # noqa: ANN401 - OpenAI-style tool selection, passed through
        request_id: str | None = None,
    ) -> dict:
        """Async non-stream generation via direct HTTP (server engines only)."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Please load a model first.")

        params = self._build_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            total_timeout=total_timeout,
            enable_thinking=enable_thinking,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
        )
        client = await self._acquire_async_client()
        messages = self._as_messages(prompt, system_prompt)
        stop_event = self._register_async_stop(request_id)
        try:
            if stop_event is None:
                return await client.generate(messages, params, request_id=request_id)
            # Race the request against its stop event so stop_generation can
            # abort a non-stream generation too. Cancelling the task closes
            # the HTTP request, which makes the server abort generation.
            gen_task = asyncio.ensure_future(
                client.generate(messages, params, request_id=request_id)
            )
            stop_task = asyncio.ensure_future(stop_event.wait())
            try:
                await asyncio.wait({gen_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
                if gen_task.done():
                    return gen_task.result()
                gen_task.cancel()
                try:
                    await gen_task
                except (asyncio.CancelledError, RuntimeError):
                    pass
                return {
                    "result": "",
                    "gen_tokens": 0,
                    "gen_tps": 0.0,
                    "prompt_tokens": 0,
                    "prompt_tps": 0.0,
                    "total_tokens": 0,
                    "stopped": True,
                }
            finally:
                stop_task.cancel()
        finally:
            self._unregister_async_stop(request_id)

    async def agenerate_stream(
        self,
        prompt: str | list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
        total_timeout: int = 300,
        enable_thinking: bool | None = None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,  # noqa: ANN401 - OpenAI-style tool selection, passed through
        request_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Async streaming generation via direct HTTP (server engines only)."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Please load a model first.")

        params = self._build_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            total_timeout=total_timeout,
            enable_thinking=enable_thinking,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
        )
        client = await self._acquire_async_client()
        messages = self._as_messages(prompt, system_prompt)
        stop_event = self._register_async_stop(request_id)
        try:
            async for item in client.generate_stream(
                messages, params, request_id=request_id, stop_event=stop_event
            ):
                yield item
        finally:
            self._unregister_async_stop(request_id)

    def get_status(self) -> dict:
        """Get the current status."""
        # Fetch status from the inference process
        process_status = self.inference_process.get_status()

        # While loading, report pending_config; otherwise report the loaded config
        cfg = self.pending_config if process_status.get("is_loading") else self.config

        # Clear pending_config once the model has finished loading
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
            ),  # added: device allocation stats
            "n_gpu_layers": cfg.n_gpu_layers if cfg else None,
            "n_ctx": cfg.n_ctx if cfg else None,
            "n_batch": cfg.n_batch if cfg else None,
            "llama_server_extra_args": cfg.llama_server_extra_args if cfg else None,
            "vllm_gpu_memory_utilization": (cfg.vllm_gpu_memory_utilization if cfg else None),
            "vllm_max_model_len": cfg.vllm_max_model_len if cfg else None,
            "vllm_dtype": cfg.vllm_dtype if cfg else None,
            "vllm_quantization": cfg.vllm_quantization if cfg else None,
            "vllm_enforce_eager": cfg.vllm_enforce_eager if cfg else None,
            "vllm_kv_cache_dtype": cfg.vllm_kv_cache_dtype if cfg else None,
            "vllm_cpu_offload_gb": cfg.vllm_cpu_offload_gb if cfg else None,
            "vllm_kv_offloading_size": (cfg.vllm_kv_offloading_size if cfg else None),
            "vllm_tensor_parallel_size": cfg.vllm_tensor_parallel_size if cfg else None,
            "vllm_max_num_seqs": cfg.vllm_max_num_seqs if cfg else None,
            "vllm_max_num_batched_tokens": (cfg.vllm_max_num_batched_tokens if cfg else None),
            "vllm_mm_image_limit": cfg.vllm_mm_image_limit if cfg else None,
            "vllm_mm_audio_limit": cfg.vllm_mm_audio_limit if cfg else None,
            "vllm_mm_video_limit": cfg.vllm_mm_video_limit if cfg else None,
            "vllm_hf_overrides": cfg.vllm_hf_overrides if cfg else None,
            "vllm_chat_template": cfg.vllm_chat_template if cfg else None,
            "vllm_tool_call_parser": cfg.vllm_tool_call_parser if cfg else None,
        }

        # GPU memory usage: prefer the numbers reported by the worker process
        memory_usage = process_status.get("memory_usage")
        if memory_usage:
            # The worker process already reported GPU usage (correct cross-process data)
            status["memory_usage"] = memory_usage
        elif torch.cuda.is_available() and process_status.get("loaded"):
            # Fallback: read it in the main process (only sees main-process usage, usually 0)
            try:
                status["memory_usage"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
            except Exception:
                pass

        return status

    def get_error_details(self) -> dict | None:
        """Return detailed error info, including the full traceback."""
        return self.inference_process.get_error_details()

    def cleanup(self) -> None:
        """Release resources (called on application shutdown)."""
        logger.info("Cleaning up ModelManager...")

        try:
            # Stop the inference process
            if self.inference_process:
                self.inference_process.stop_process()
                logger.info("Inference process stopped")
        except Exception as e:
            logger.exception(f"Error stopping inference process: {e}")

        # Clear the config
        self.config = None
        self.pending_config = None
        self._close_async_clients_best_effort()

        logger.info("ModelManager cleanup completed")

    def __del__(self) -> None:
        """Destructor - make sure resources are released."""
        try:
            if hasattr(self, "inference_process"):
                self.cleanup()
        except Exception:
            pass

    def force_cleanup_gpu(self) -> dict:
        """
        Force-clean GPU memory.

        After the process crashes with OOM, kill it and restart a clean one.
        """
        logger.warning("Force cleanup GPU - terminating worker process...")

        with self._lock:
            # Force-stop the current process
            if self.inference_process.process and self.inference_process.process.is_alive():
                try:
                    self.inference_process.process.terminate()
                    self.inference_process.process.join(timeout=3)
                    if self.inference_process.process.is_alive():
                        self.inference_process.process.kill()
                        self.inference_process.process.join()
                except Exception as e:
                    logger.exception(f"Error force terminating process: {e}")

            # Reset state
            self.inference_process._cleanup_dead_process()
            self.inference_process.current_status = "idle"
            self.config = None
            self.pending_config = None
            self._close_async_clients_best_effort()

            logger.info("✅ GPU force cleanup completed - worker process terminated")
            return {"status": "success", "message": "GPU memory force cleaned"}

    def cleanup_generation_memory(self, slot: int | None = None) -> dict:
        """
        Softly release scratch memory from the generation phase without unloading the model.

        Returns: dict status payload
        """
        try:
            if not self.is_loaded():
                return {"status": "error", "message": "No model loaded"}
            result = self.inference_process.cleanup_generation_memory(slot=slot)
            return result
        except Exception as e:
            logger.exception(f"cleanup_generation_memory failed: {e}")
            return {"status": "error", "message": str(e)}


# Singleton instance
model_manager = ModelManager()
