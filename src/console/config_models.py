"""
Configuration models for inference and training
"""

from typing import Optional, List, Dict, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class InferenceEngine(str, Enum):
    """Inference engine."""

    TRANSFORMERS = "transformers"
    LLAMA_SERVER = "llama_server"
    VLLM = "vllm"


class QuantizationType(str, Enum):
    """Quantization type."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit normal float used by QLoRA
    FP4 = "fp4"  # 4-bit float


class TrainingMethod(str, Enum):
    """Training method."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class ListedModel(BaseModel):
    """Single model entry returned by `/config/models`."""

    model_name: str = Field(
        ..., description="Model name, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Model path, output_dir, or GGUF filename returned by backend",
    )
    label: str = Field(..., description="Display label shown in model selector")
    size: str = Field(..., description="Human-readable model size, e.g. ~1.1B")
    max_context_length: Optional[int] = Field(
        default=None,
        description="Maximum context length reported by backend",
    )
    method: Optional[TrainingMethod] = Field(
        default=None,
        description="Training method for fine-tuned models, e.g. lora",
    )


class ModelListResponse(BaseModel):
    """Unified model list response returned by `/config/models`."""

    base_models: List[ListedModel] = Field(default_factory=list)
    finetuned_models: List[ListedModel] = Field(default_factory=list)
    llama_gguf_models: List[ListedModel] = Field(default_factory=list)


# ------ Inference-related configuration models ------
class InferenceConfig(BaseModel):
    """Inference configuration - directly uses Hugging Face Transformers style.

    Example:
    {
        "model_name": "Qwen/Qwen3-4B",
        "quantization": "none",
        "device_map": "auto",
        "model_total_memory": "15GB",
        "max_memory": {"0": "5GB", "cpu": "5GB"},
        "offload_folder": "./offload"
    }
    Or
    {
        "model_name": "Qwen/Qwen3-8B",
        "quantization": "none",
        "device_map": "cpu"
    }
    """

    model_name: str = Field(..., description="Model name or path")
    model_path: Optional[str] = Field(
        default=None,
        description="Local fine-tuned model path (usually output_dir); if provided, it takes precedence over model_name for from_pretrained",
    )
    engine: InferenceEngine = Field(
        default=InferenceEngine.TRANSFORMERS,
        description="Inference engine: transformers (default), llama_server, vllm",
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization type: none, int8, int4, nf4, fp4",
    )
    device_map: Optional[Union[str, Dict]] = Field(
        default="auto",
        description="Device mapping strategy, e.g. 'auto', 'cpu', 'cuda:0' or {'': 0, 'cpu': 'cpu'}, 'balanced_low_0'",
    )
    model_total_memory: Optional[str] = Field(
        default=None, description="Total model memory requirement, e.g. '15GB'"
    )
    max_memory: Optional[Dict[Union[int, str], str]] = Field(
        default=None,
        description="Maximum memory allocation, e.g. {0: '20GB', 'cpu': '50GB'}",
    )
    offload_folder: Optional[str] = Field(
        default=None,
        description="Offload folder path used to offload model weights to disk",
    )
    torch_dtype: str = Field(default="auto", description="Torch dtype")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    use_cache: bool = Field(default=True, description="Use KV cache")

    # llama.cpp-specific configuration
    n_gpu_layers: int = Field(
        default=-1, description="[llama.cpp] Number of GPU layers; -1 means all"
    )
    n_ctx: int = Field(default=4096, description="[llama.cpp] Context length")
    n_batch: int = Field(default=512, description="[llama.cpp] Batch size")

    # llama-server-specific configuration (OpenAI-compatible API)
    llama_server_url: Optional[str] = Field(
        default=None,
        description="[llama_server] Server base URL, e.g. http://127.0.0.1:8080",
    )
    llama_server_api_key: Optional[str] = Field(
        default=None,
        description="[llama_server] API key (if server-side authorization is enabled)",
    )
    llama_server_model: Optional[str] = Field(
        default=None,
        description="[llama_server] Model name used in requests; defaults to model_name when not provided",
    )
    llama_server_timeout: int = Field(
        default=300,
        description="[llama_server] Request timeout in seconds",
        ge=10,
    )
    llama_server_auto_start: bool = Field(
        default=True,
        description="[llama_server] Whether to auto-start a llama-server subprocess during load",
    )
    llama_server_binary: Optional[str] = Field(
        default=None,
        description="[llama_server] Path to llama-server executable (uses environment default if not provided)",
    )
    llama_server_host: str = Field(
        default="127.0.0.1",
        description="[llama_server] Host bound when starting subprocess",
    )
    llama_server_port: int = Field(
        default=5001,
        description="[llama_server] Port used when starting subprocess",
        ge=1,
        le=65535,
    )
    llama_server_np: int = Field(
        default=1,
        description="[llama_server] Parallel generation slots (maps to llama-server -np)",
        ge=1,
    )
    llama_server_health_timeout: int = Field(
        default=300,
        description="[llama_server] Seconds to wait for successful service startup during load",
        ge=5,
    )
    llama_server_extra_args: Optional[List[str]] = Field(
        default=None,
        description="[llama_server] Additional startup args, e.g. ['--mlock', '--no-mmap']",
    )
    llama_server_mmproj: Optional[str] = Field(
        default=None,
        description="[llama_server] Multimodal projector (.gguf) path; automatically appends --mmproj when provided",
    )

    # vLLM OpenAI-compatible server-specific configuration
    vllm_gpu_memory_utilization: float = Field(
        default=0.8,
        description="[vLLM] --gpu-memory-utilization",
        ge=0.05,
        le=0.99,
    )
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        description="[vLLM] --max-model-len; falls back to n_ctx when not provided",
        ge=1,
    )
    vllm_dtype: str = Field(default="auto", description="[vLLM] --dtype")
    vllm_quantization: Optional[str] = Field(
        default=None,
        description="[vLLM] --quantization, e.g. awq/gptq/fp8",
    )
    vllm_enforce_eager: bool = Field(
        default=False,
        description="[vLLM] Whether to enable --enforce-eager",
    )
    vllm_kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="[vLLM] --kv-cache-dtype, e.g. auto/fp8_e5m2/fp8_e4m3",
    )
    vllm_cpu_offload_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="[vLLM] --cpu-offload-gb",
    )
    vllm_kv_offloading_size: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "[vLLM] --kv-offloading-size in GB; "
            "when using tensor parallelism this value is the total across all TP ranks, not per-GPU"
        ),
    )
    vllm_tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="[vLLM] --tensor-parallel-size",
    )
    vllm_max_num_seqs: Optional[int] = Field(
        default=None,
        ge=1,
        description="[vLLM] --max-num-seqs",
    )
    # vLLM multimodal (Vision / Audio / Video) specific configuration
    # Applicable to VLM models such as Gemma 4, Gemma 3n, Qwen-VL, and LLaVA
    vllm_mm_image_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max number of images in --limit-mm-per-prompt; "
            "set this when loading multimodal models (e.g. gemma-4-E2B-it), such as 1"
        ),
    )
    vllm_mm_audio_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max number of audio inputs in --limit-mm-per-prompt; "
            "only needed for audio-capable models such as Gemma 4 E2B/E4B"
        ),
    )
    vllm_mm_video_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="[vLLM] Max number of videos in --limit-mm-per-prompt",
    )
    vllm_hf_overrides: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "[vLLM] --hf-overrides to force override fields in HuggingFace config.json; "
            "for example, if gemma-4-E2B-it is incorrectly detected as text-only, set "
            '{"architectures":["Gemma4ForConditionalGeneration"]} to force multimodal architecture. '
            "Accepts dict or JSON string"
        ),
    )
    vllm_chat_template: Optional[str] = Field(
        default=None,
        description=(
            "[vLLM] --chat-template, path to custom chat template file (.jinja); "
            "when tokenizer_config.json lacks chat_template (e.g. some base or quantized variants), "
            "provide this to support /v1/chat/completions"
        ),
    )

    @field_validator("vllm_chat_template", mode="before")
    @classmethod
    def _normalize_vllm_chat_template(cls, v):
        """Normalize empty chat_template strings to None."""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @field_validator("vllm_hf_overrides", mode="before")
    @classmethod
    def _normalize_vllm_hf_overrides(cls, v):
        """Accept dict/list or JSON string; treat empty strings as unset."""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        if isinstance(v, (dict, list)):
            return v
        raise ValueError("vllm_hf_overrides must be dict, list, JSON string, or None")


class DeviceAllocation(BaseModel):
    """Device allocation statistics."""

    summary: Optional[str] = Field(
        default=None,
        description="Summary of module counts per device, e.g. 'cuda:0:30, cpu:10'",
    )
    total_modules: Optional[int] = Field(
        default=None, description="Total number of model modules"
    )
    layer_lines: Optional[List[str]] = Field(
        default=None,
        description="Layer-level allocation, e.g. ['model.layers.0 -> cuda:0', ...]",
    )


class ModelStatus(BaseModel):
    """Model status."""

    loaded: bool = Field(default=False, description="Whether model is loaded")
    is_loading: bool = Field(
        default=False, description="Whether model is currently loading"
    )
    loading_error: Optional[str] = Field(
        default=None, description="Model loading error message"
    )
    model_name: Optional[str] = Field(default=None, description="Model name")
    model_path: Optional[str] = Field(default=None, description="Model path")
    engine: InferenceEngine = Field(
        default=InferenceEngine.TRANSFORMERS,
        description="Inference engine: transformers (default), llama_server, vllm",
    )
    quantization: Optional[str] = Field(default=None, description="Quantization type")
    model_total_memory: Optional[str] = Field(
        default=None, description="Total model memory requirement"
    )
    device_map: Optional[Union[str, Dict]] = Field(
        default=None, description="Device mapping"
    )
    max_memory: Optional[Dict] = Field(
        default=None, description="Maximum memory limits"
    )
    offload_folder: Optional[str] = Field(default=None, description="Offload folder")
    device: Optional[str] = Field(default=None, description="Device")
    memory_usage: Optional[Dict] = Field(default=None, description="Memory usage")
    device_allocation: Optional[DeviceAllocation] = Field(
        default=None,
        description="Actual device allocation statistics (available after model load)",
    )

    # llama.cpp specific status
    n_gpu_layers: Optional[int] = Field(
        default=None, description="[llama.cpp] Actual number of GPU layers used"
    )
    n_ctx: Optional[int] = Field(default=None, description="[llama.cpp] Context length")
    n_batch: Optional[int] = Field(default=None, description="[llama.cpp] Batch size")
    prefill_strategy: Optional[str] = Field(
        default=None,
        description="[llama_server] Prefill strategy, e.g. slot or cache_prompt",
    )
    llama_capabilities: Optional[List[str]] = Field(
        default=None, description="[llama_server] Capabilities reported by /v1/models"
    )
    slot_restore_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="[llama_server] slot restore result summary"
    )

    # vLLM specific status
    vllm_gpu_memory_utilization: Optional[float] = Field(
        default=None, description="[vLLM] --gpu-memory-utilization"
    )
    vllm_max_model_len: Optional[int] = Field(
        default=None, description="[vLLM] --max-model-len"
    )
    vllm_dtype: Optional[str] = Field(default=None, description="[vLLM] --dtype")
    vllm_quantization: Optional[str] = Field(
        default=None, description="[vLLM] --quantization"
    )
    vllm_enforce_eager: Optional[bool] = Field(
        default=None, description="[vLLM] --enforce-eager"
    )
    vllm_kv_offloading_size: Optional[float] = Field(
        default=None, description="[vLLM] --kv-offloading-size"
    )
    vllm_kv_cache_dtype: Optional[str] = Field(
        default=None, description="[vLLM] --kv-cache-dtype"
    )
    vllm_cpu_offload_gb: Optional[float] = Field(
        default=None, description="[vLLM] --cpu-offload-gb"
    )
    vllm_tensor_parallel_size: Optional[int] = Field(
        default=None, description="[vLLM] --tensor-parallel-size"
    )
    vllm_max_num_seqs: Optional[int] = Field(
        default=None, description="[vLLM] --max-num-seqs"
    )
    vllm_mm_image_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt image limit",
    )
    vllm_mm_audio_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt audio limit",
    )
    vllm_mm_video_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt video limit",
    )
    vllm_hf_overrides: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="[vLLM] --hf-overrides",
    )
    vllm_chat_template: Optional[str] = Field(
        default=None,
        description="[vLLM] --chat-template",
    )


# ------ Inference-related Chat / Stream Pydantic models removed ------
# The previous ChatRequest / ChatResponse / ChatStreamChunk mapped to backend
# `/inference/chat`. The console client now uses the OpenAI-compatible
# endpoint (`<backend_url>/v1/chat/completions`) for all chat. Please use
# the OpenAI Python SDK. See openai-compatible-example.py and
# openai-compatible-image-example.py for examples.


class AppSettings(BaseModel):
    """Application settings (used by fine-tune examples)."""

    backend_url: str = Field(..., description="Backend API URL")
    verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificate"
    )
    finetune_config_path: Optional[str] = Field(
        default=None, description="Fine-tune training config file path"
    )


# ------ Training-related configuration models ------
class TrainingConfig(BaseModel):
    """Training configuration.

    Supports three training methods:
    - full: full-parameter fine-tuning
    - lora: parameter-efficient fine-tuning via LoRA
    - qlora: quantized training with QLoRA + LoRA
    """

    model_name: str = Field(
        ...,
        description="Model name label (from the label field in models registry config)",
    )
    method: TrainingMethod = Field(
        ..., description="Training method: lora / qlora / full"
    )
    dataset_path: str = Field(
        ..., description="Training dataset file path; must be JSON or JSONL format"
    )
    output_dir: str = Field(
        ..., description="Output directory path for fine-tuned model files"
    )
    offload_folder: Optional[str] = Field(
        default="./deepspeed_offload",
        description="Offload folder path; overrides DeepSpeed JSON configuration",
    )

    # LoRA/QLoRA specific
    lora_r: int = Field(
        default=8,
        description="[Only when method = LoRA/QLoRA] LoRA rank; controls trainable parameter count. Higher values may improve results but increase training cost",
    )
    lora_alpha: int = Field(
        default=16,
        description="[Only when method = LoRA/QLoRA] LoRA alpha scaling factor, usually 1-2x lora_r",
    )
    lora_dropout: float = Field(
        default=0.05,
        description="[Only when method = LoRA/QLoRA] LoRA dropout ratio used to reduce overfitting",
    )
    lora_target_modules: Optional[List[str]] = Field(
        default=None,
        description="[Only when method = LoRA/QLoRA] Target module list where LoRA is applied. Use null for defaults (e.g. q_proj, k_proj, v_proj, o_proj)",
    )

    # Training hyperparameters
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        default=1, description="Per-device training batch size"
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        description="Number of gradient accumulation steps before each optimizer update",
    )
    learning_rate: float = Field(
        default=2e-4, description="Learning rate controlling update magnitude"
    )
    warmup_steps: int = Field(
        default=100, description="Warmup steps using lower learning rate"
    )
    logging_steps: int = Field(
        default=10,
        description="Report training progress every N steps (loss, step, etc.)",
    )
    save_steps: int = Field(default=500, description="Save checkpoint every N steps")
    save_total_limit: Optional[int] = Field(
        default=2,
        description="Maximum number of checkpoints to keep; oldest will be deleted automatically",
    )
    max_seq_length: int = Field(
        default=2048,
        description="Maximum token length during training; longer sequences are truncated",
    )

    # Dataset field configuration - choose one of two training data modes
    text_field: Optional[str] = Field(
        default="text",
        description="[Mode 1] Single-field mode: train by predicting continuation in one text field. Field name in dataset, e.g. 'text'. Mutually exclusive with prompt_field/completion_field",
    )
    prompt_field: Optional[str] = Field(
        default=None,
        description="[Mode 2] Two-field mode: separate prompt and response. Prompt field name in dataset, e.g. 'prompt'. Must be used with completion_field",
    )
    completion_field: Optional[str] = Field(
        default=None,
        description="[Mode 2] Two-field mode: separate prompt and response. Completion field name in dataset, e.g. 'completion'. Must be used with prompt_field",
    )
    save_tokenizer: bool = Field(
        default=True,
        description="Whether to save tokenizer to output directory after training",
    )

    # DeepSpeed settings
    use_deepspeed: bool = Field(
        default=False,
        description="Whether to use DeepSpeed offload training (requires deepspeed_config or deepspeed_profile when enabled)",
    )
    deepspeed_config: Optional[str] = Field(
        default=None,
        description="[Option 1] Full path to DeepSpeed config file, e.g. './my_deepspeed.json'. Mutually exclusive with deepspeed_profile",
    )
    deepspeed_profile: Optional[str] = Field(
        default=None,
        description="[Option 2] DeepSpeed profile name; auto-loads service/configs/deepspeed/<profile>.json. Mutually exclusive with deepspeed_config",
    )
    # SFTTrainer specific
    use_sft_trainer: bool = Field(
        default=True,
        description="Whether to use TRL SFTTrainer for training (recommended for instruction tuning)",
    )
    packing: bool = Field(
        default=False,
        description="[Only when use_sft_trainer=True] Whether to enable packing (pack multiple short sequences into one long sequence for better efficiency)",
    )


class TrainingStatus(BaseModel):
    """Training status."""

    is_training: bool = Field(
        default=False, description="Whether training is currently running"
    )
    progress: float = Field(default=0.0, description="Training progress (0-1)")
    current_step: int = Field(default=0, description="Current step")
    total_steps: int = Field(default=0, description="Total steps")
    loss: Optional[float] = Field(default=None, description="Current loss")
    current_epoch: Optional[float] = Field(default=0.0, description="Current epoch")
    total_epochs: Optional[int] = Field(
        default=None, description="Total number of epochs"
    )
    status: Optional[str] = Field(default=None, description="Status message")
    session_id: Optional[str] = Field(
        default=None, description="Current training session ID"
    )
    error: Optional[str] = Field(
        default=None, description="Error message (short description)"
    )
    config: Optional[TrainingConfig] = Field(
        default=None, description="Current or last training configuration"
    )
