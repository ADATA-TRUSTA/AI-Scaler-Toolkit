"""In-process Transformers inference engine (loads models via HuggingFace)."""

import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import Any

import torch
from accelerate import infer_auto_device_map
from transformers import AutoProcessor, AutoTokenizer

from ...config_models import InferenceConfig, QuantizationType
from ...settings import configure_logging
from ...utils.token_utils import load_hf_token
from ..generate.generator_worker import handle_generate_request, handle_generate_stream_request
from ..generate.gpt_parser import is_gpt_model
from ..load_model import is_peft_model, load_peft_model
from ..load_model.peft_loader import read_base_model_name
from ..model_utils import (
    _cleanup_generation_memory,
    _cleanup_memory,
    _get_model_class_for_loading,
    _get_quantization_config,
    _is_gemma3_model,
    _is_gemma4_model,
    _is_qwen3_model,
    get_smart_device_map,
)
from .base_engine import BaseEngine

logger = configure_logging(__name__)

# XPU Workaround: Intel XPU currently does not support mem_get_info.
# Detect and patch at import time so transformers caching_allocator_warmup
# does not fail.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    try:
        # Probe whether mem_get_info is supported
        _ = torch.xpu.mem_get_info(0)
    except RuntimeError as e:
        if "doesn't support querying the available free memory" in str(e):
            logger.warning(
                "[transformers_engine] XPU doesn't support mem_get_info, applying global workaround"
            )
            # Monkey patch: return fake memory info
            _original_xpu_mem_get_info = torch.xpu.mem_get_info

            def _patched_xpu_mem_get_info(device: Any = None) -> tuple[int, int]:  # noqa: ANN401 - matches torch.xpu.mem_get_info(device) signature; arg unused
                """
                Workaround for XPU devices that don't support mem_get_info.
                Returns fake memory info to allow transformers model loading."""
                # Return (free, total) in bytes - assume ample memory
                return (12 * 1024**3, 16 * 1024**3)

            torch.xpu.mem_get_info = _patched_xpu_mem_get_info
            logger.info("[transformers_engine] XPU mem_get_info workaround applied globally")


class TransformersEngine(BaseEngine):
    """In-process engine backed by HuggingFace Transformers."""

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ) -> None:
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.model = None
        self.tokenizer = None
        self.processor = None

    @staticmethod
    def _normalize_device_label(device: torch.device | int | str) -> str:
        """Normalize a device representation into a UI-friendly label."""
        if isinstance(device, torch.device):
            if device.type == "cuda":
                return f"GPU{0 if device.index is None else device.index}"
            if device.type == "cpu":
                return "DRAM"
            return str(device)

        if isinstance(device, int):
            return f"GPU{device}"

        if isinstance(device, str):
            raw = device.strip()
            low = raw.lower()
            if low == "cpu":
                return "DRAM"
            if low == "disk":
                return "SSD"
            if low == "cuda":
                return "GPU0"
            if low.startswith("cuda:"):
                tail = low.split(":", 1)[1]
                if tail.isdigit():
                    return f"GPU{tail}"
            return raw

        return str(device)

    @staticmethod
    def _get_nested_attr(obj: Any, dotted_path: str) -> Any:  # noqa: ANN401 - walks arbitrary model/config attribute chains
        current = obj
        for part in dotted_path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @classmethod
    def _get_total_layer_count(cls, model: Any) -> int | None:  # noqa: ANN401 - accepts any loaded transformers model
        """Best-effort layer count from the config or common transformer containers."""
        candidate_configs: list[Any] = []
        seen_ids = set()

        def _push_config(candidate: Any) -> None:  # noqa: ANN401 - arbitrary transformers config object

            if candidate is None:
                return
            candidate_id = id(candidate)
            if candidate_id in seen_ids:
                return
            seen_ids.add(candidate_id)
            candidate_configs.append(candidate)

        _push_config(getattr(model, "config", None))
        for cfg_path in (
            "language_model.config",
            "model.config",
            "text_model.config",
            "transformer.config",
        ):
            _push_config(cls._get_nested_attr(model, cfg_path))

        for cfg in candidate_configs:
            for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                value = getattr(cfg, attr_name, None)
                if isinstance(value, int) and value > 0:
                    return value

        for layer_path in (
            "model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "text_model.layers",
        ):
            layers = cls._get_nested_attr(model, layer_path)
            if layers is None:
                continue
            try:
                count = len(layers)
            except TypeError:
                continue
            if count > 0:
                return count

        return None

    @staticmethod
    def _build_single_device_allocation(
        device_label: str, total_layers: int | None
    ) -> tuple[str, int | None, list[str]]:
        """Build the summary for a model that fits entirely on one device."""
        if isinstance(total_layers, int) and total_layers > 0:
            return (
                f"{device_label}: {total_layers}/{total_layers} layers (100%)",
                total_layers,
                [f"Layers 0-{total_layers - 1} -> {device_label}"],
            )

        return (
            f"{device_label}: full model",
            None,
            [f"full model -> {device_label}"],
        )

    def load_model(self, config: InferenceConfig) -> None:
        """Load tokenizer, optional processor, and model weights per config."""
        self.config = config
        model_source = config.model_path or config.model_name

        # Read the HF token (prefer the one in config, otherwise read from file)
        # hf_token is not a declared InferenceConfig field, so access it safely via getattr
        config_hf_token = getattr(config, "hf_token", None)
        hf_token = config_hf_token if config_hf_token else load_hf_token()

        logger.info(f"[Worker] Loading model: {config.model_name}")
        if hf_token:
            logger.info("[Worker] Using HF token for authentication")

        # Load the tokenizer
        self.status_queue.put({"status": "loading", "stage": "tokenizer"})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=config.trust_remote_code,
            token=hf_token,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try loading the optional multimodal processor (ignored if unsupported)
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=config.trust_remote_code,
                token=hf_token,
                local_files_only=True,
            )
            logger.info(
                "[Worker] AutoProcessor loaded (multimodal-capable model may accept images)"
            )
        except Exception as e:
            logger.info(f"[Worker] AutoProcessor not available, fallback to tokenizer-only: {e}")

        logger.info("[Worker] Tokenizer loaded")

        # Prepare model loading arguments
        self.status_queue.put({"status": "loading", "stage": "model_weights"})
        # Qwen3/3.5 MoE expert routing uses softmax gating; float16's dynamic
        # range is too narrow and overflows, picking the wrong expert -> garbled
        # mixed-language output. bfloat16 is required.
        _needs_bfloat16 = (
            _is_gemma3_model(model_source)
            or _is_gemma4_model(model_source)
            or is_gpt_model(model_source)
            or _is_qwen3_model(model_source)
        )
        # Resolve dtype: "auto" picks bf16/fp16 by model; otherwise accept a
        # torch dtype name or common alias, falling back to the auto choice
        # instead of crashing on an unknown string.
        _auto_dtype = torch.bfloat16 if _needs_bfloat16 else torch.float16
        if config.torch_dtype == "auto":
            resolved_dtype = _auto_dtype
        else:
            _dtype_aliases = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
            }
            key = str(config.torch_dtype).lower().strip()
            resolved_dtype = _dtype_aliases.get(key)
            if resolved_dtype is None:
                resolved_dtype = getattr(torch, str(config.torch_dtype), None)
            if not isinstance(resolved_dtype, torch.dtype):
                logger.warning(
                    f"[Worker] Unrecognized torch_dtype '{config.torch_dtype}', "
                    f"falling back to {_auto_dtype}"
                )
                resolved_dtype = _auto_dtype
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "dtype": resolved_dtype,
            "low_cpu_mem_usage": True,
            "token": hf_token,
            "attn_implementation": "eager",
        }
        if _is_qwen3_model(model_source):
            logger.info("[Worker] Qwen3/3.5 detected → forcing bfloat16 for MoE routing stability")

        # Normalize max_memory if provided
        normalized_max_memory = None
        if config.max_memory:

            def _normalize_max_memory(mem: dict) -> dict:
                norm: dict = {}
                for k, v in mem.items():
                    key = k
                    if isinstance(k, str):
                        ks = k.strip()
                        if ks.isdigit():
                            key = int(ks)
                        elif ks.lower().startswith("cuda:") or ks.lower().startswith("xpu:"):
                            tail = ks.split(":", 1)[1]
                            if tail.isdigit():
                                key = int(tail)
                            else:
                                key = ks
                        else:
                            key = ks
                    norm[key] = v
                return norm

            normalized_max_memory = _normalize_max_memory(config.max_memory)

        inferred_device_map = None
        needs_quantization = config.quantization in {
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.NF4,
            QuantizationType.FP4,
        }

        if needs_quantization and config.device_map == "auto" and normalized_max_memory:
            # Use the smart device map
            self.status_queue.put({"status": "loading", "stage": "inferring_device_map"})
            qtype_str = (
                "4bit"
                if config.quantization
                in {QuantizationType.INT4, QuantizationType.NF4, QuantizationType.FP4}
                else ("8bit" if config.quantization == QuantizationType.INT8 else "none")
            )
            logger.info(
                f"[Worker] Attempting smart device map (quantization={qtype_str}) with max_memory={normalized_max_memory}"
            )
            inferred_device_map = get_smart_device_map(
                model_source,
                normalized_max_memory,
                quantization_type=qtype_str,
                token=hf_token,
                trust_remote_code=config.trust_remote_code,
            )
            if inferred_device_map:
                self.status_queue.put({"status": "loading", "stage": "loading_with_inferred_map"})
            else:
                logger.info(
                    "[Worker] Smart device map failed, will fallback to naive meta inference"
                )
                try:
                    # Fallback
                    quantization_config = _get_quantization_config(config.quantization)
                    ModelClass = _get_model_class_for_loading(model_source)
                    logger.info("[Worker] Fallback: Loading meta model for standard inference...")
                    meta_model_kwargs = {
                        "device_map": "meta",
                        "trust_remote_code": config.trust_remote_code,
                        "token": hf_token,
                    }
                    if quantization_config:
                        meta_model_kwargs["quantization_config"] = quantization_config
                    meta_model = ModelClass.from_pretrained(
                        model_source, **meta_model_kwargs, local_files_only=True
                    )
                    logger.info("[Worker] Fallback: Inferring device map without scaling...")

                    inferred_device_map = infer_auto_device_map(
                        meta_model,
                        max_memory=normalized_max_memory,
                    )
                    logger.info(f"[Worker] Fallback inferred device_map: {inferred_device_map}")
                    del meta_model
                    import gc

                    gc.collect()
                    self.status_queue.put(
                        {"status": "loading", "stage": "loading_with_inferred_map"}
                    )
                except Exception as e:
                    logger.warning(f"[Worker] Fallback infer device_map failed, using 'auto': {e}")
                    inferred_device_map = None
        elif not needs_quantization and config.device_map == "auto":
            logger.info("[Worker] No quantization, using device_map='auto' directly")

        # XPU device_map: accelerate does not recognize "xpu:N", so load on CPU
        # first and then move manually via .to(xpu)
        _xpu_target_device = None
        if isinstance(config.device_map, str) and config.device_map.lower().startswith("xpu"):
            _xpu_target_device = config.device_map
            logger.info(
                f"[Worker] XPU device_map detected ({config.device_map}); will load on CPU first then move to XPU"
            )

        if inferred_device_map:
            model_kwargs["device_map"] = inferred_device_map
        elif _xpu_target_device:
            model_kwargs["device_map"] = "cpu"
        elif config.device_map:
            model_kwargs["device_map"] = config.device_map

        if normalized_max_memory:
            model_kwargs["max_memory"] = normalized_max_memory

        if config.offload_folder:
            offload_dir = Path(config.offload_folder)
            if not offload_dir.is_absolute():
                offload_dir = Path(f"{config.offload_folder}/{model_source.replace('/', '_')}")
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_dir)
        else:
            # Default offload folder logic
            # Using current file location relative... careful with paths after move
            # Assuming this file is in service/inference/engines/
            # Base of repo is ../../../
            # We want service/model_offload
            service_dir = Path(__file__).parent.parent.parent  # service/
            default_offload_base = service_dir / "model_offload"
            default_offload_dir = default_offload_base / model_source.replace("/", "_")
            try:
                default_offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs["offload_folder"] = str(default_offload_dir)
                logger.info(f"[Worker] Using default offload folder: {default_offload_dir}")
            except PermissionError as e:
                logger.warning(f"[Worker] Cannot create default offload folder: {e}")
                logger.warning("[Worker] Proceeding without offload folder (may cause OOM)")

        if config.quantization in {
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.NF4,
            QuantizationType.FP4,
        }:
            quantization_config = _get_quantization_config(config.quantization)
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

        is_peft = is_peft_model(model_source)

        if is_peft:
            logger.info(f"[Worker] Detected PEFT/LoRA model at: {model_source}")
            try:
                base_model_name = read_base_model_name(model_source)
            except Exception as e:
                logger.exception(f"[Worker] Failed to read adapter config: {e}")
                raise

            ModelClass = _get_model_class_for_loading(base_model_name)
            logger.info(
                f"[Worker] Loading base model with device_map={model_kwargs.get('device_map', 'None')}"
            )
            base_model = ModelClass.from_pretrained(
                base_model_name, **model_kwargs, local_files_only=True
            )
            self.model = load_peft_model(model_source, base_model, hf_token, **model_kwargs)
            self.model.eval()
            logger.info("[Worker] PEFT model loaded successfully")
        else:
            ModelClass = _get_model_class_for_loading(model_source)
            logger.info(
                f"[Worker] Loading model with device_map={model_kwargs.get('device_map', 'None')}"
            )
            self.model = ModelClass.from_pretrained(
                model_source, **model_kwargs, local_files_only=True
            )
            self.model.eval()
            logger.info("[Worker] Model loaded successfully")

        # XPU: after loading on CPU, move the model to the target device manually
        if _xpu_target_device:
            logger.info(f"[Worker] Moving model to {_xpu_target_device}...")
            self.status_queue.put({"status": "loading", "stage": "moving_to_xpu"})
            self.model = self.model.to(_xpu_target_device)
            self.model.eval()
            logger.info(f"[Worker] Model moved to {_xpu_target_device} successfully")

        # Collect device info
        device_info = None
        device_map_summary = None
        total_modules_count = None
        layer_lines = []

        try:
            device_map = getattr(self.model, "hf_device_map", None)
            if device_map:
                dev_counts: dict[str, int] = {}

                for module_name, dev in device_map.items():
                    dev_str = self._normalize_device_label(dev)
                    dev_counts[dev_str] = dev_counts.get(dev_str, 0) + 1
                    layer_lines.append(f"{module_name} -> {dev_str}")

                total_modules_count = sum(dev_counts.values())
                labels = [f"{d}: {c} layers" for d, c in dev_counts.items()]
                device_map_summary = ", ".join(labels)

                logger.info(
                    f"[Worker] Device map summary: {device_map_summary} (total modules={total_modules_count})"
                )

                # Device info for UI/short display
                devices = set(dev_counts.keys())
                if len(devices) == 1:
                    device_info = list(devices)[0]
                else:
                    device_info = f"multi-device: {', '.join(sorted(devices))}"
            else:
                # For a full single-device load, Transformers/Accelerate may not
                # create hf_device_map at all.
                actual_device = None
                try:
                    actual_device = getattr(self.model, "device", None)
                    if actual_device is None:
                        first_param = next(self.model.parameters())
                        actual_device = first_param.device
                except Exception:
                    actual_device = None

                if actual_device is not None:
                    device_info = self._normalize_device_label(actual_device)
                    total_layers = self._get_total_layer_count(self.model)
                    device_map_summary, total_modules_count, layer_lines = (
                        self._build_single_device_allocation(
                            device_info,
                            total_layers,
                        )
                    )
                    logger.info(
                        f"[Worker] Synthesized single-device allocation: {device_map_summary}"
                    )
                else:
                    device_info = "unknown"

        except Exception as e:
            logger.warning(f"[Worker] Failed to collect device info: {e}")
            device_info = "unknown"

        memory_usage = None
        try:
            # CUDA memory stats
            if torch.cuda.is_available():
                memory_usage = {
                    "device_type": "cuda",
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
            # XPU memory stats (Intel GPU, natively supported by torch)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                try:
                    memory_usage = {
                        "device_type": "xpu",
                        "allocated_gb": torch.xpu.memory_allocated() / (1024**3),
                        "reserved_gb": torch.xpu.memory_reserved() / (1024**3),
                    }
                except Exception as e:
                    logger.debug(f"Failed to get XPU memory usage: {e}")
        except Exception as e:
            logger.debug(f"Failed to get GPU memory usage: {e}")

        # Construct status response
        self.status_queue.put(
            {
                "status": "ready",
                "message": "Model loaded successfully",
                "device": device_info,
                "device_map_summary": device_map_summary,
                "total_modules": total_modules_count,
                "layer_lines": layer_lines,
                "memory_usage": memory_usage,
            }
        )

    def generate(self, request: dict[str, Any]) -> None:
        """Run a non-stream generation request in-process."""
        if self.model is None or self.tokenizer is None:
            self.data_queue.put(
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "error": "Model or Tokenizer not loaded",
                }
            )
            return

        handle_generate_request(
            request,
            self.model,
            self.tokenizer,
            self.processor,
            self.config,
            self.data_queue,
            self.stop_generation_flag,
        )

    def generate_stream(self, request: dict[str, Any]) -> None:
        """Run a streaming generation request in-process."""
        if self.model is None or self.tokenizer is None:
            self.data_queue.put(
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "error": "Model or Tokenizer not loaded",
                }
            )
            return

        handle_generate_stream_request(
            request,
            self.model,
            self.tokenizer,
            self.processor,
            self.config,
            self.data_queue,
            self.stop_generation_flag,
        )

    def unload(self) -> None:
        """Free model, tokenizer, processor, and report memory usage."""
        logger.info("[Worker] Unloading Transformers model...")
        if self.stop_generation_flag.is_set():
            time.sleep(1.0)
            self.stop_generation_flag.clear()

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.config = None
        _cleanup_memory()

        unload_memory_usage = None
        try:
            # CUDA memory stats
            if torch.cuda.is_available():
                unload_memory_usage = {
                    "device_type": "cuda",
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
            # XPU memory stats (Intel GPU, natively supported by torch)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                try:
                    unload_memory_usage = {
                        "device_type": "xpu",
                        "allocated_gb": torch.xpu.memory_allocated() / (1024**3),
                        "reserved_gb": torch.xpu.memory_reserved() / (1024**3),
                    }
                except Exception as e:
                    logger.debug(f"Failed to get XPU memory usage: {e}")
        except Exception:
            pass

        self.status_queue.put(
            {
                "status": "unloaded",
                "message": "Model unloaded successfully",
                "memory_usage": unload_memory_usage,
            }
        )

    def apply_chat_template(self, request: dict[str, Any]) -> None:
        """Render chat messages into a prompt via the tokenizer template."""
        request_id = request.get("request_id")
        if self.tokenizer is None:
            self.data_queue.put(
                {"type": "error", "request_id": request_id, "error": "Tokenizer not loaded"}
            )
            return

        try:
            messages = request.get("messages", [])
            template_kwargs = request.get("template_kwargs", {})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
            self.data_queue.put({"type": "result", "request_id": request_id, "result": prompt})
        except Exception as e:
            logger.exception(f"[Worker] apply_chat_template error: {e}")
            self.data_queue.put({"type": "error", "request_id": request_id, "error": str(e)})

    def cleanup_generation_memory(self) -> None:
        """Release cached generation memory for the loaded model."""
        _cleanup_generation_memory(self.model)
        self.data_queue.put({"type": "cleanup", "result": "generation memory cleaned"})
