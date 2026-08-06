"""
GPU Memory Estimator for Model Inference
Estimates GPU memory requirements for models under different configurations.
Supports reading model configs from Hugging Face and computing parameter counts automatically.
Supports MoE (Mixture of Experts) models.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

# Optional torch import for environment detection
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - estimator should still work without torch
    torch = None

# Optional transformers for config loading
try:
    from transformers import AutoConfig  # type: ignore
except Exception:
    AutoConfig = None


# Parameter counts for common models (unit: Billion parameters)
MODEL_SIZES = {
    # Llama family
    "llama-7b": 7.0,
    "llama-13b": 13.0,
    "llama-30b": 30.0,
    "llama-65b": 65.0,
    "llama-2-7b": 7.0,
    "llama-2-13b": 13.0,
    "llama-2-70b": 70.0,
    "llama-3-8b": 8.0,
    "llama-3-70b": 70.0,
    # Mistral family
    "mistral-7b": 7.0,
    "mixtral-8x7b": 47.0,  # actual active params
    # Qwen family
    "qwen3-4b": 4.0,
    "qwen3-14b": 14.0,
    "qwen3-32b": 32.0,
    "qwen3-Next-80B-A3B-Instruct": 78.59,  # MoE: 512 experts, 10 active per token
    # Gemma family
    "gemma3-4b": 4.0,
    "gemma3-12b": 12.0,
    # TinyLlama
    "tinyllama-1.1b": 1.1,
    # ChatGLM family
    "chatglm-6b": 6.0,
    "chatglm2-6b": 6.0,
    "chatglm3-6b": 6.0,
    # OpenAI GSS-Opt family
    "gss-opt-20b": 20.0,
    "gss-opt-120b": 120.0,
}


class MemoryEstimator:
    """Memory requirement estimator - supports HuggingFace configs and MoE models."""

    def __init__(self) -> None:
        self.model_sizes = MODEL_SIZES
        self._config_cache = {}  # Cache for loaded configs

    def _load_model_config(self, model_name: str) -> Any | None:  # noqa: ANN401 - transformers PretrainedConfig or None
        """
        Load a model config from Hugging Face.

        Args:
            model_name: model name or path

        Returns:
            Model config object, or None
        """
        if model_name in self._config_cache:
            return self._config_cache[model_name]

        if AutoConfig is None:
            logger.warning("transformers is not installed; cannot load model config automatically")
            return None

        try:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True
            )
            self._config_cache[model_name] = config
            logger.info(f"Loaded model config: {model_name}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load model config {model_name}: {e}")
            return None

    def _calculate_params_from_config(self, config: Any) -> tuple[float, bool, dict]:  # noqa: ANN401 - transformers PretrainedConfig
        """
        Compute the model parameter count from a config.

        Args:
            config: model config object

        Returns:
            (params(B), is_moe, details)
        """
        params_info = {
            "total_params": 0,
            "active_params": 0,
            "is_moe": False,
            "num_experts": 0,
            "experts_per_token": 0,
            "architecture": getattr(config, "architectures", ["unknown"])[0]
            if hasattr(config, "architectures")
            else "unknown",
        }

        # Multimodal models (e.g. Gemma 3) - use text_config
        if hasattr(config, "text_config") and config.text_config is not None:
            logger.info("Multimodal model detected; using text_config for the calculation")
            config = config.text_config

        # Detect MoE models
        is_moe = False
        num_experts = 0
        experts_per_token = 1
        moe_intermediate_size = None
        shared_expert_intermediate_size = None

        # Mixtral-style MoE
        if hasattr(config, "num_local_experts"):
            is_moe = True
            num_experts = getattr(config, "num_local_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True

        # DeepSeek-style MoE
        elif hasattr(config, "n_routed_experts"):
            is_moe = True
            num_experts = getattr(config, "n_routed_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True

        # Qwen3-Next-style MoE (uses num_experts)
        elif hasattr(config, "num_experts") and getattr(config, "num_experts", 0) > 1:
            is_moe = True
            num_experts = getattr(config, "num_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
            shared_expert_intermediate_size = getattr(
                config, "shared_expert_intermediate_size", None
            )
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True
            params_info["moe_intermediate_size"] = moe_intermediate_size
            params_info["shared_expert_intermediate_size"] = shared_expert_intermediate_size

        # Basic model parameters
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        vocab_size = getattr(config, "vocab_size", 32000)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # Token embedding params
        embedding_params = vocab_size * hidden_size

        # Attention params (per layer)
        # Q, K, V projections + output projection
        attention_params_per_layer = (
            hidden_size * hidden_size  # Q
            + hidden_size
            * (hidden_size // num_attention_heads)
            * num_key_value_heads  # K (GQA support)
            + hidden_size
            * (hidden_size // num_attention_heads)
            * num_key_value_heads  # V (GQA support)
            + hidden_size * hidden_size  # output projection
        )

        # FFN params (per layer)
        if is_moe:
            # MoE: router + (experts * FFN_size)
            # Each expert holds gate_proj, up_proj, down_proj
            router_params = hidden_size * num_experts

            # Use moe_intermediate_size when present (Qwen3-Next style)
            expert_intermediate_size = (
                moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
            )

            # Params per expert: gate(hidden->intermediate) + up(hidden->intermediate) + down(intermediate->hidden)
            params_per_expert = (
                (hidden_size * expert_intermediate_size)
                + (hidden_size * expert_intermediate_size)
                + (expert_intermediate_size * hidden_size)
            )
            expert_params = num_experts * params_per_expert

            # Shared expert (if present)
            shared_expert_params = 0
            if shared_expert_intermediate_size is not None and shared_expert_intermediate_size > 0:
                shared_expert_params = (
                    (hidden_size * shared_expert_intermediate_size)
                    + (hidden_size * shared_expert_intermediate_size)
                    + (shared_expert_intermediate_size * hidden_size)
                )

            ffn_params_per_layer = router_params + expert_params + shared_expert_params

            # Active params: only the selected experts take part in the computation
            active_expert_params = experts_per_token * params_per_expert
            active_ffn_params_per_layer = (
                router_params + active_expert_params + shared_expert_params
            )
        else:
            # Standard FFN: gate + up + down projections
            ffn_params_per_layer = (
                (hidden_size * intermediate_size)
                + (hidden_size * intermediate_size)
                + (intermediate_size * hidden_size)
            )
            active_ffn_params_per_layer = ffn_params_per_layer

        # Layer norm params (negligible, included for completeness)
        norm_params_per_layer = hidden_size * 2  # pre-attention norm + post-ffn norm

        # Total params
        total_layer_params = num_layers * (
            attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        )

        active_layer_params = num_layers * (
            attention_params_per_layer + active_ffn_params_per_layer + norm_params_per_layer
        )

        # Output layer (lm_head)
        output_params = vocab_size * hidden_size

        # Totals
        total_params = embedding_params + total_layer_params + output_params
        active_params = embedding_params + active_layer_params + output_params

        params_info["total_params"] = total_params / 1e9  # convert to Billion
        params_info["active_params"] = active_params / 1e9
        params_info["hidden_size"] = hidden_size
        params_info["num_layers"] = num_layers
        params_info["intermediate_size"] = intermediate_size
        params_info["vocab_size"] = vocab_size

        # Return total_params (used for memory estimation), is_moe, and the details
        return params_info["total_params"], is_moe, params_info

    def extract_model_size(self, model_name: str) -> float | None:
        """
        Extract the parameter count from a model name, preferring the HuggingFace config.

        Args:
            model_name: model name, e.g. "meta-llama/Llama-2-7b-chat-hf"

        Returns:
            Parameter count (Billion) or None
        """
        # Prefer loading the config from HuggingFace
        config = self._load_model_config(model_name)
        if config is not None:
            try:
                size, is_moe, info = self._calculate_params_from_config(config)
                logger.info(f"Params computed from config: {size:.2f}B (MoE: {is_moe})")
                return size
            except Exception as e:
                logger.warning(f"Failed to compute params from config: {e}")

        # Fall back to name matching
        model_name_lower = model_name.lower()

        # Try the predefined table
        for key, size in self.model_sizes.items():
            if key in model_name_lower:
                return size

        # Try extracting a number from the name
        # e.g. "7b", "13b", "70b"
        import re

        # Match the XXb form
        match = re.search(r"(\d+\.?\d*)b", model_name_lower)
        if match:
            size = float(match.group(1))
            return size

        # Match the XX-billion form
        match = re.search(r"(\d+\.?\d*)-?billion", model_name_lower)
        if match:
            return float(match.group(1))

        logger.warning(f"Cannot extract the parameter count from the model name: {model_name}")
        return None

    def estimate_memory_requirements(
        self,
        model_name: str,
        quantization: str = "none",
        include_activations: bool = True,
        batch_size: int = 1,
        sequence_length: int = 2048,
    ) -> dict:
        """
        Estimate the memory requirements of a model (MoE models supported).

        Args:
            model_name: model name
            quantization: quantization type (none, int8, int4, nf4, fp4)
            include_activations: whether to include activation memory
            batch_size: batch size
            sequence_length: sequence length

        Returns:
            Dict with the memory estimation result
        """
        # Try loading the config for full details
        config = self._load_model_config(model_name)
        is_moe = False
        params_info = {}

        if config is not None:
            try:
                model_size_b, is_moe, params_info = self._calculate_params_from_config(config)
                logger.info(f"Computed from config: {model_size_b:.2f}B params (MoE: {is_moe})")
            except Exception as e:
                logger.warning(f"Config calculation failed; falling back to name extraction: {e}")
                model_size_b = self.extract_model_size(model_name)
        else:
            # Extract the model size (fallback path)
            model_size_b = self.extract_model_size(model_name)

        if model_size_b is None:
            return {
                "error": "Cannot determine the model size",
                "model_name": model_name,
                "suggestion": "Check the model name, or install transformers",
            }

        # Model weight memory
        # For MoE, use total_params (all experts) rather than active_params
        total_params_for_memory = (
            params_info.get("total_params", model_size_b) if is_moe else model_size_b
        )
        model_memory = self._calculate_model_memory(total_params_for_memory, quantization)

        # Activation memory (intermediate results during inference)
        # Use the actual config when available, otherwise estimate.
        # When params_info is truthy, hidden_size/num_layers are already ints
        # (see _calculate_params_from_config).
        hidden_size = cast(
            int,
            params_info.get("hidden_size")
            if params_info
            else self._estimate_hidden_size(model_size_b),
        )
        num_layers = cast(
            int,
            params_info.get("num_layers")
            if params_info
            else self._estimate_num_layers(model_size_b),
        )

        activation_memory = 0
        if include_activations:
            activation_memory = self._calculate_activation_memory_with_config(
                hidden_size, num_layers, batch_size, sequence_length, quantization
            )

        # KV cache memory
        kv_cache_memory = self._calculate_kv_cache_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length
        )

        # Runtime overhead (Python/Torch/CUDA etc.)
        overhead_breakdown = self._estimate_runtime_overhead(quantization)
        overhead_memory = overhead_breakdown["total"]

        # Total memory requirement
        total_memory = model_memory + activation_memory + kv_cache_memory + overhead_memory

        # Recommended minimum GPU memory
        recommended_gpu_memory = total_memory * 1.1  # reserve a 10% safety margin

        # Minimum GPU memory in the hybrid offload mode
        min_gpu_with_offload = (
            activation_memory + kv_cache_memory + overhead_memory + (model_memory * 0.1)
        )

        result = {
            "model_name": model_name,
            "model_size_billions": round(model_size_b, 2),
            "quantization": quantization,
            "memory_breakdown_gb": {
                "model_weights": round(model_memory, 2),
                "activations": round(activation_memory, 2),
                "kv_cache": round(kv_cache_memory, 2),
                "overhead": round(overhead_memory, 2),
                "total": round(total_memory, 2),
            },
            "overhead_details_gb": {
                "python_runtime": round(overhead_breakdown["python_runtime"], 2),
                "pytorch_framework": round(overhead_breakdown["pytorch_framework"], 2),
                "cuda_context": round(overhead_breakdown["cuda_context"], 2),
                "cuda_libraries": round(overhead_breakdown["cuda_libraries"], 2),
                "transformers_lib": round(overhead_breakdown["transformers_lib"], 2),
                "quantization_lib": round(overhead_breakdown["quantization_lib"], 2),
                "cuda_driver": round(overhead_breakdown["cuda_driver"], 2),
                "total": round(overhead_breakdown["total"], 2),
            },
            "recommendations": {
                "full_gpu_memory_gb": round(recommended_gpu_memory, 2),
                "min_gpu_with_cpu_offload_gb": round(min_gpu_with_offload, 2),
                "min_gpu_with_disk_offload_gb": round(min_gpu_with_offload * 0.7, 2),
            },
            "offload_strategies": self._generate_offload_strategies(
                model_memory, activation_memory, kv_cache_memory, overhead_memory
            ),
            "notes": [
                f"Estimate based on a {model_size_b:.2f}B parameter model",
                f"Batch size: {batch_size}, sequence length: {sequence_length}",
            ],
        }

        # MoE-specific info
        if is_moe:
            result["moe_info"] = {
                "is_moe": True,
                "num_experts": params_info.get("num_experts", 0),
                "experts_per_token": params_info.get("experts_per_token", 0),
                "total_params_billions": round(params_info.get("total_params", 0), 2),
                "active_params_billions": round(params_info.get("active_params", 0), 2),
            }
            result["notes"].append(
                f"MoE model: {params_info.get('num_experts')} experts, "
                f"{params_info.get('experts_per_token')} experts per token"
            )
            result["notes"].append(
                f"Total params: {params_info.get('total_params', 0):.2f}B, "
                f"active params: {params_info.get('active_params', 0):.2f}B"
            )

        # Config info
        if params_info:
            result["model_config"] = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "intermediate_size": params_info.get("intermediate_size"),
                "vocab_size": params_info.get("vocab_size"),
                "architecture": params_info.get("architecture", "unknown"),
            }
            result["notes"].append(
                f"Config source: Hugging Face ({params_info.get('architecture')})"
            )
        else:
            result["notes"].append(
                "Config source: estimated (install transformers for the exact config)"
            )

        result["notes"].extend(
            [
                "Actual memory use varies with the framework and runtime (CUDA/driver/libraries)",
                "Reserving a 20% safety margin is recommended",
                (
                    "CUDA environment detected; the extra overhead is included"
                    if (torch and hasattr(torch, "cuda") and torch.cuda.is_available())
                    else "CPU-only environment; overhead is lower"
                ),
            ]
        )

        return result

    def _estimate_runtime_overhead(self, quantization: str) -> dict[str, float]:
        """
        Estimate the detailed runtime memory overhead (GB).

        Includes:
        - Base Python process memory (~0.3 GB)
        - PyTorch framework overhead (~0.5-1.0 GB, version dependent)
        - CUDA context initialization (~0.5-1.0 GB per GPU)
        - CUDA core library workspaces (cuBLAS, cuDNN, cuSPARSE ~0.5-1.5 GB)
        - Transformers library (~0.2-0.5 GB)
        - Quantization library (bitsandbytes ~0.3-0.5 GB)
        - CUDA driver and runtime (~0.2-0.3 GB)

        Returns:
            Dict with the per-item overhead breakdown and the total

        Note: these are empirical values; actual numbers depend on the driver version,
        CUDA version, GPU generation and settings.
        """
        overhead = {
            "python_runtime": 0.3,  # Python process and core libraries
            "pytorch_framework": 0.0,
            "cuda_context": 0.0,
            "cuda_libraries": 0.0,
            "transformers_lib": 0.2,
            "quantization_lib": 0.0,
            "cuda_driver": 0.0,
        }

        # PyTorch framework overhead (depends on CUDA support)
        if torch and torch.cuda.is_available():
            overhead["pytorch_framework"] = 0.8  # the CUDA build is larger

            # Detect GPU info to adjust the estimate
            try:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                compute_capability = (
                    torch.cuda.get_device_capability(0) if device_count > 0 else (0, 0)
                )

                # CUDA context (per GPU)
                # Newer GPUs (Compute Capability >= 8.0, Ampere+) may need more
                if compute_capability[0] >= 8:  # Ampere (A100, RTX 30xx) or newer
                    overhead["cuda_context"] = 0.8 * device_count
                elif compute_capability[0] >= 7:  # Volta/Turing (V100, T4, RTX 20xx)
                    overhead["cuda_context"] = 0.6 * device_count
                else:  # older GPUs
                    overhead["cuda_context"] = 0.5 * device_count

                # CUDA core library workspaces (cuBLAS, cuDNN, cuSPARSE etc.)
                # Adjusted by GPU generation and CUDA version
                if compute_capability[0] >= 8:
                    # Newer GPUs support more CUDA core features (Tensor Cores, etc.)
                    overhead["cuda_libraries"] = 1.1
                elif compute_capability[0] >= 7:
                    overhead["cuda_libraries"] = 0.8
                else:
                    overhead["cuda_libraries"] = 0.5

                # CUDA driver and runtime
                overhead["cuda_driver"] = 0.25

                logger.debug(
                    f"GPU detected: {device_name} (Compute {compute_capability[0]}.{compute_capability[1]})"
                )
            except Exception as e:
                # If GPU detection fails, use conservative estimates
                logger.warning(f"GPU detection failed; using defaults: {e}")
                overhead["cuda_context"] = 0.6
                overhead["cuda_libraries"] = 0.8
                overhead["cuda_driver"] = 0.25
        else:
            # CPU-only environment
            overhead["pytorch_framework"] = 0.5
            overhead["cuda_context"] = 0.0
            overhead["cuda_libraries"] = 0.0
            overhead["cuda_driver"] = 0.0

        # Quantization library (bitsandbytes, only when quantization is used)
        if quantization.lower() in {"int8", "int4", "nf4", "fp4"}:
            # bitsandbytes loads CUDA kernels and quantization constants
            overhead["quantization_lib"] = 0.4 if torch and torch.cuda.is_available() else 0.1

        # Total overhead
        total = sum(overhead.values())
        overhead["total"] = round(total, 2)

        return overhead

    def _calculate_model_memory(self, model_size_b: float, quantization: str) -> float:
        """
        Compute the model weight memory requirement.

        Args:
            model_size_b: model size (Billion parameters)
            quantization: quantization type

        Returns:
            Memory requirement (GB)
        """
        # Bits per parameter
        bits_per_param = {
            "none": 16,  # FP16/BF16
            "fp16": 16,
            "bf16": 16,
            "int8": 8,
            "int4": 4,
            "nf4": 4,
            "fp4": 4,
        }

        bits = bits_per_param.get(quantization.lower(), 16)

        # Compute memory (GB)
        # 1B parameters * bits_per_param / 8 (bytes) / 1024^3 (GB)
        memory_gb = (model_size_b * 1e9 * bits / 8) / (1024**3)

        return memory_gb

    def _calculate_activation_memory(
        self, model_size_b: float, batch_size: int, sequence_length: int, quantization: str
    ) -> float:
        """More precise inference activation memory estimate (fallback path)."""
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        return self._calculate_activation_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length, quantization
        )

    def _calculate_activation_memory_with_config(
        self,
        hidden_size: int,
        num_layers: int,
        batch_size: int,
        sequence_length: int,
        quantization: str,
    ) -> float:
        """Compute activation memory from the actual config."""
        # 16-bit dtypes (none/fp16/bf16) use 2 bytes per element; quantized
        # weights use a rough 1-byte approximation for activations.
        bytes_per_element = 2 if quantization.lower() in ("none", "fp16", "bf16") else 1

        # Inference only needs to keep a small slice of the intermediate layers
        raw_memory = (
            batch_size * sequence_length * hidden_size * num_layers * bytes_per_element
        ) / (1024**3)
        return raw_memory / 30  # roughly 1/30 of training

    def _calculate_kv_cache_memory(
        self, model_size_b: float, batch_size: int, sequence_length: int
    ) -> float:
        """Adjust the KV cache toward realistic values (fallback path)."""
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        return self._calculate_kv_cache_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length
        )

    def _calculate_kv_cache_memory_with_config(
        self, hidden_size: int, num_layers: int, batch_size: int, sequence_length: int
    ) -> float:
        """Compute the KV cache from the actual config."""
        bytes_per_element = 2  # FP16

        kv_cache = (
            2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element
        ) / (1024**3)
        return kv_cache * 0.6  # about 60% is used on average in practice

    def _estimate_hidden_size(self, model_size_b: float) -> int:
        """Estimate the hidden size."""
        if model_size_b <= 1:
            return 2048
        elif model_size_b <= 3:
            return 2560
        elif model_size_b <= 7:
            return 4096
        elif model_size_b <= 13:
            return 5120
        elif model_size_b <= 30:
            return 6656
        elif model_size_b <= 70:
            return 8192
        else:
            return 12288

    def _estimate_num_layers(self, model_size_b: float) -> int:
        """Estimate the number of layers."""
        if model_size_b <= 1:
            return 22
        elif model_size_b <= 3:
            return 32
        elif model_size_b <= 7:
            return 32
        elif model_size_b <= 13:
            return 40
        elif model_size_b <= 30:
            return 60
        elif model_size_b <= 70:
            return 80
        else:
            return 120

    def _generate_offload_strategies(
        self,
        model_memory: float,
        activation_memory: float,
        kv_cache_memory: float,
        overhead_memory: float,
    ) -> list:
        """
        Generate offload strategy suggestions for different GPU memory sizes.
        """
        strategies = []

        # Strategy 1: full GPU (no offload)
        full_gpu = model_memory + activation_memory + kv_cache_memory + overhead_memory
        strategies.append(
            {
                "name": "Full GPU (No Offload)",
                "min_gpu_gb": round(full_gpu * 1.1, 2),
                "description": "All model weights and compute stay on the GPU",
                "performance": "fastest",
                "config": {"offload": "none"},
            }
        )

        # Strategy 2: CPU offload (some weights)
        cpu_offload_50 = (
            (model_memory * 0.5) + activation_memory + kv_cache_memory + overhead_memory
        )
        strategies.append(
            {
                "name": "CPU Offload (50% weights)",
                "min_gpu_gb": round(cpu_offload_50 * 1.1, 2),
                "description": "50% of the model weights offloaded to CPU",
                "performance": "moderate",
                "config": {"offload": "cpu", "device_map": "auto"},
            }
        )

        # Strategy 3: CPU offload (most weights)
        cpu_offload_80 = (
            (model_memory * 0.2) + activation_memory + kv_cache_memory + overhead_memory
        )
        strategies.append(
            {
                "name": "CPU Offload (80% weights)",
                "min_gpu_gb": round(cpu_offload_80 * 1.1, 2),
                "description": "80% of the weights offloaded to CPU; only key layers on GPU",
                "performance": "slower",
                "config": {"offload": "cpu", "device_map": "auto"},
            }
        )

        # Strategy 4: disk offload
        disk_offload = (model_memory * 0.1) + activation_memory + kv_cache_memory + overhead_memory
        strategies.append(
            {
                "name": "Disk Offload (90% weights)",
                "min_gpu_gb": round(disk_offload * 1.1, 2),
                "description": "Most weights offloaded to disk (NVMe recommended)",
                "performance": "slowest, but the lowest GPU requirement",
                "config": {"offload": "disk", "offload_dir": "./offload"},
            }
        )

        return strategies


# Create the global instance
memory_estimator = MemoryEstimator()
