"""Model loading utilities: quantization config, model class selection and memory cleanup."""

import gc
from typing import Any, cast

import torch
from accelerate import infer_auto_device_map
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from ..config_models import QuantizationType
from ..settings import configure_logging

logger = configure_logging(__name__)

try:
    from transformers import Gemma3ForConditionalGeneration

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False
    logger.warning("Gemma3ForConditionalGeneration not available in this transformers version")

try:
    from transformers import Gemma4ForConditionalGeneration

    GEMMA4_AVAILABLE = True
except ImportError:
    GEMMA4_AVAILABLE = False
    # Fallback: try AutoModelForImageTextToText (official recommended class for vision-language models)
    try:
        from transformers import AutoModelForImageTextToText as Gemma4ForConditionalGeneration

        GEMMA4_AVAILABLE = True
        logger.info("Using AutoModelForImageTextToText as fallback for Gemma 4")
    except ImportError:
        logger.warning(
            "Neither Gemma4ForConditionalGeneration nor AutoModelForImageTextToText available"
        )

try:
    from transformers import Qwen3_5MoeForConditionalGeneration

    QWEN35_MOE_AVAILABLE = True
except ImportError:
    QWEN35_MOE_AVAILABLE = False
    # Fallback: try AutoModelForImageTextToText (official recommended class)
    try:
        from transformers import AutoModelForImageTextToText

        QWEN35_MOE_AVAILABLE = True
        Qwen3_5MoeForConditionalGeneration = AutoModelForImageTextToText
        logger.info("Using AutoModelForImageTextToText as fallback for Qwen3.5 MoE")
    except ImportError:
        logger.warning(
            "Neither Qwen3_5MoeForConditionalGeneration nor AutoModelForImageTextToText available"
        )


def _get_quantization_config(qtype: QuantizationType) -> BitsAndBytesConfig | None:
    """Return BitsAndBytesConfig for supported quantization types."""
    try:
        if qtype == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        if qtype in {QuantizationType.INT4, QuantizationType.NF4, QuantizationType.FP4}:
            # INT4 defaults to nf4 (bitsandbytes' higher-quality 4-bit type);
            # only an explicit FP4 request uses fp4.
            quant_type = "fp4" if qtype == QuantizationType.FP4 else "nf4"
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quant_type,
                llm_int8_enable_fp32_cpu_offload=True,
            )
    except Exception as e:
        logger.warning(f"Failed to build quantization config for {qtype}: {e}")
    return None


def _mem_str_to_bytes(mem_val: Any) -> int:  # noqa: ANN401 - accepts dynamic int/float/str memory values
    """Convert a memory number/string to bytes."""
    if isinstance(mem_val, (int, float)):
        return int(mem_val)
    if isinstance(mem_val, str):
        s = mem_val.strip()
        units = {
            "GiB": 1024**3,
            "GB": 1000**3,
            "MiB": 1024**2,
            "MB": 1000**2,
            "KiB": 1024,
            "KB": 1000,
        }
        for unit, mult in units.items():
            if s.endswith(unit):
                num_part = s[: -len(unit)].strip()
                try:
                    return int(float(num_part) * mult)
                except ValueError:
                    break
        try:
            return int(float(s))
        except ValueError:
            logger.warning(f"[SmartDispatch] Cannot parse memory value: {mem_val}, defaulting to 0")
            return 0
    logger.warning(f"[SmartDispatch] Unsupported memory value type: {type(mem_val)} -> {mem_val}")
    return 0


def scale_max_memory_for_quantization(
    max_memory: dict[Any, Any], quantization_type: str
) -> dict[Any, int]:
    """
    Scale up GPU max_memory based on quantization type to avoid over-offloading during meta inference.
    """
    if quantization_type == "4bit":
        scale_factor = 3.0
    elif quantization_type == "8bit":
        scale_factor = 1.8
    else:
        scale_factor = 1.0

    scaled_max_memory: dict[Any, int] = {}
    for key, value in max_memory.items():
        raw_bytes = _mem_str_to_bytes(value)
        key_str = str(key).lower()
        if key_str in {"cpu", "disk"}:
            scaled_max_memory[key] = raw_bytes
        else:
            is_gpu = key_str.isdigit() or key_str.startswith("cuda:") or key_str.startswith("xpu:")
            if is_gpu and scale_factor != 1.0:
                scaled_bytes = int(raw_bytes * scale_factor)
            else:
                scaled_bytes = raw_bytes
            scaled_max_memory[key] = scaled_bytes

    logger.info(f"[SmartDispatch] Scaled Max Memory for estimation: {scaled_max_memory}")
    return scaled_max_memory


def _is_gemma3_model(model_name: str) -> bool:
    """Detect whether the model belongs to the Gemma3 family (excluding Gemma 4)."""
    model_name_lower = model_name.lower()
    gemma3_patterns = ["gemma-3", "gemma3", "gemma-2-3"]
    # Exclude Gemma 4 (e.g. gemma-4, gemma4)
    if _is_gemma4_model(model_name):
        return False
    return any(pattern in model_name_lower for pattern in gemma3_patterns)


def _is_gemma4_model(model_name: str) -> bool:
    """Detect whether the model belongs to the Gemma 4 family (multimodal vision-language models)."""
    model_name_lower = model_name.lower()
    gemma4_patterns = ["gemma-4", "gemma4"]
    return any(pattern in model_name_lower for pattern in gemma4_patterns)


def _is_qwen3_model(model_name: str) -> bool:
    """
    Detect whether the model belongs to the Qwen3 / Qwen3.5 family (including MoE variants).

    These models require bfloat16 precision; float16 makes MoE routing numerically
    unstable and produces mixed-language garbage output.
    """
    model_name_lower = model_name.lower()
    qwen3_patterns = ["qwen3", "qwen3.5", "qwen-3", "qwen_3"]
    return any(pattern in model_name_lower for pattern in qwen3_patterns)


def _is_qwen35_moe_model(model_name: str) -> bool:
    """
    Detect whether the model belongs to the Qwen3.5 MoE family (multimodal ConditionalGeneration architecture).

    MoE models such as Qwen3.5-35B-A3B store weights as model.language_model.layers.*,
    so they must be loaded with Qwen3_5MoeForConditionalGeneration; using
    AutoModelForCausalLM (Qwen3_5MoeForCausalLM) breaks weight mapping → garbage output.
    """
    model_name_lower = model_name.lower()
    return (
        "qwen3.5" in model_name_lower
        or "qwen3_5" in model_name_lower
        or "qwen3.5_moe" in model_name_lower
    )


def _get_model_class_for_loading(model_name: str) -> type:
    """Return the appropriate model class based on the model name."""
    if _is_gemma4_model(model_name):
        if GEMMA4_AVAILABLE:
            logger.info("[Worker] Detected Gemma 4 model, using Gemma4ForConditionalGeneration")
            return Gemma4ForConditionalGeneration
        else:
            logger.warning(
                "[Worker] Gemma 4 model detected but Gemma4ForConditionalGeneration not available, falling back to AutoModelForCausalLM"
            )
            return AutoModelForCausalLM
    if _is_gemma3_model(model_name):
        if GEMMA3_AVAILABLE:
            logger.info("[Worker] Detected Gemma3 model, using Gemma3ForConditionalGeneration")
            return Gemma3ForConditionalGeneration
        else:
            logger.warning(
                "[Worker] Gemma3 model detected but Gemma3ForConditionalGeneration not available, falling back to AutoModelForCausalLM"
            )
            return AutoModelForCausalLM
    if _is_qwen35_moe_model(model_name):
        if QWEN35_MOE_AVAILABLE:
            logger.info(
                "[Worker] Detected Qwen3.5 MoE model, using Qwen3_5MoeForConditionalGeneration"
            )
            return Qwen3_5MoeForConditionalGeneration
        else:
            logger.warning(
                "[Worker] Qwen3.5 MoE model detected but Qwen3_5MoeForConditionalGeneration not available, falling back to AutoModelForCausalLM"
            )
            return AutoModelForCausalLM
    return AutoModelForCausalLM


def get_smart_device_map(
    model_id: str,
    max_memory: dict[Any, Any],
    quantization_type: str = "none",
    token: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any] | None:
    """Infer a better device_map."""
    scaled_max_memory = scale_max_memory_for_quantization(max_memory, quantization_type)

    try:
        quantization_config = None
        if quantization_type in {"4bit", "8bit"}:
            if quantization_type == "8bit":
                quantization_config = _get_quantization_config(QuantizationType.INT8)
            else:
                quantization_config = _get_quantization_config(QuantizationType.INT4)

        ModelClass = _get_model_class_for_loading(model_id)
        meta_model_kwargs = {
            "device_map": "meta",
            "trust_remote_code": trust_remote_code,
            "token": token,
        }
        if quantization_config:
            meta_model_kwargs["quantization_config"] = quantization_config

        logger.info("[SmartDispatch] Loading meta model for smart device map inference...")
        meta_model = ModelClass.from_pretrained(
            model_id, **meta_model_kwargs, local_files_only=True
        )

        logger.info("[SmartDispatch] Inferring device map with scaled max_memory...")
        # cast: dict invariance mismatch in transformers stub; dict[Any, int] is valid at runtime
        smart_device_map = infer_auto_device_map(
            meta_model,
            max_memory=cast("dict[int | str, int | str]", scaled_max_memory),
        )
        logger.info(f"[SmartDispatch] Inferred smart device_map: {smart_device_map}")

        del meta_model
        gc.collect()
        return smart_device_map
    except Exception as e:
        logger.warning(f"[SmartDispatch] Failed to infer smart device map: {e}")
        return None


def _cleanup_memory() -> None:
    """Thoroughly free GPU and Python object memory (supports CUDA and XPU)."""
    try:
        for _ in range(3):
            gc.collect()

        # CUDA cleanup
        if torch.cuda.is_available():
            try:
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                logger.debug("[Cleanup] All CUDA devices cache cleared")
            except Exception as e:
                logger.warning(f"[Cleanup] CUDA cleanup error: {e}")

        # XPU cleanup (Intel GPU, natively supported by torch)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                for device_id in range(torch.xpu.device_count()):
                    with torch.xpu.device(device_id):
                        torch.xpu.empty_cache()
                        torch.xpu.synchronize()

                if hasattr(torch.xpu, "reset_peak_memory_stats"):
                    torch.xpu.reset_peak_memory_stats()
                if hasattr(torch.xpu, "reset_accumulated_memory_stats"):
                    torch.xpu.reset_accumulated_memory_stats()

                logger.debug("[Cleanup] All XPU devices cache cleared")
            except Exception as e:
                logger.warning(f"[Cleanup] XPU cleanup error: {e}")

        logger.info("[Cleanup] Memory cleanup completed (CUDA/XPU cache + KV cache + GC)")

    except Exception as e:
        logger.warning(f"[Cleanup] Memory cleanup error: {e}")


def _cleanup_generation_memory(model: torch.nn.Module | None = None) -> None:
    """Free memory allocated specifically during generation."""
    try:
        logger.info("[Cleanup] Starting generation memory cleanup...")

        if model is not None:
            try:
                if hasattr(model, "reset_cache"):
                    model.reset_cache()  # pyright: ignore[reportCallIssue]  # dynamic nn.Module attr; guarded by hasattr
                    logger.debug("[Cleanup] Model cache reset via reset_cache()")

                if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    for layer in model.transformer.h:  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]  # dynamic nn.Module attrs; guarded by hasattr
                        if hasattr(layer, "attn"):
                            for attr in ["past_key_value", "cache", "_cache"]:
                                if hasattr(layer.attn, attr):
                                    setattr(layer.attn, attr, None)
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    for layer in model.model.layers:  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]  # dynamic nn.Module attrs; guarded by hasattr
                        if hasattr(layer, "self_attn"):
                            for attr in ["past_key_value", "cache", "_cache"]:
                                if hasattr(layer.self_attn, attr):
                                    setattr(layer.self_attn, attr, None)

                logger.debug("[Cleanup] Model internal cache cleared")
            except Exception as e:
                logger.debug(f"[Cleanup] Model cache cleanup error (non-critical): {e}")

        for _ in range(3):
            gc.collect()

        # CUDA cleanup
        if torch.cuda.is_available():
            try:
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                logger.debug("[Cleanup] All CUDA devices memory cleared")
            except Exception as e:
                logger.warning(f"[Cleanup] CUDA cleanup error: {e}")

        # XPU cleanup (Intel GPU, natively supported by torch)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                for device_id in range(torch.xpu.device_count()):
                    with torch.xpu.device(device_id):
                        torch.xpu.empty_cache()
                        torch.xpu.synchronize()

                if hasattr(torch.xpu, "reset_peak_memory_stats"):
                    torch.xpu.reset_peak_memory_stats()
                if hasattr(torch.xpu, "reset_accumulated_memory_stats"):
                    torch.xpu.reset_accumulated_memory_stats()

                logger.debug("[Cleanup] All XPU devices memory cleared")
            except Exception as e:
                logger.warning(f"[Cleanup] XPU cleanup error: {e}")

        logger.info("[Cleanup] Generation memory cleanup completed")

    except Exception as e:
        logger.warning(f"[Cleanup] Generation memory cleanup error: {e}")
