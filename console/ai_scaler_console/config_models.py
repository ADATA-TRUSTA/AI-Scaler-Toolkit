"""
Configuration models for inference and training.
"""

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator


class InferenceEngine(StrEnum):
    """Inference engine."""

    TRANSFORMERS = "transformers"
    LLAMA_SERVER = "llama_server"
    VLLM = "vllm"


class QuantizationType(StrEnum):
    """Quantization type."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit normal float used by QLoRA
    FP4 = "fp4"  # 4-bit float


class TrainingMethod(StrEnum):
    """Training method."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class ListedModel(BaseModel):
    """Single model entry returned by `/config/models`."""

    # Nullable because the backend maps these straight off the registry entry, and a
    # finetuned entry (FinetunedModelInfo) may legitimately carry no base_model_name,
    # label, or size. Base and GGUF entries always fill all three.
    model_name: str | None = Field(
        default=None,
        description="Model name, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0; "
        "null for a fine-tuned entry with no recorded base model",
    )
    model_path: str | None = Field(
        default=None,
        description="Model path, output_dir, or GGUF filename returned by backend",
    )
    label: str | None = Field(default=None, description="Display label shown in model selector")
    size: str | None = Field(default=None, description="Human-readable model size, e.g. ~1.1B")
    max_context_length: int | None = Field(
        default=None,
        description="Maximum context length reported by backend",
    )
    # Plain str, not TrainingMethod: the backend stores this as a free-form string
    # (FinetunedModelInfo.method), so a hand-edited registry can hold a value outside
    # the enum and an enum-typed field would fail the whole list. TrainingMethod is a
    # StrEnum, so `entry.method == TrainingMethod.LORA` still works.
    method: str | None = Field(
        default=None,
        description="Training method for fine-tuned models, e.g. lora",
    )


class ModelListResponse(BaseModel):
    """Unified model list response returned by `/config/models`."""

    base_models: list[ListedModel] = Field(default_factory=list)
    finetuned_models: list[ListedModel] = Field(default_factory=list)
    llama_gguf_models: list[ListedModel] = Field(default_factory=list)
    last_finetuned: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Raw contents of the backend's last_finetuned_model.json quick reference; "
            "null when the backend has never finished a fine-tune"
        ),
    )


# ------ Inference-related configuration models ------
class InferenceConfig(BaseModel):
    """
    Inference configuration - directly uses Hugging Face Transformers style.

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
    model_path: str | None = Field(
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
    device_map: str | dict | None = Field(
        default="auto",
        description="Device mapping strategy, e.g. 'auto', 'cpu', 'cuda:0' or {'': 0, 'cpu': 'cpu'}, 'balanced_low_0'",
    )
    model_total_memory: str | None = Field(
        default=None, description="Total model memory requirement, e.g. '15GB'"
    )
    max_memory: dict[int | str, str] | None = Field(
        default=None,
        description="Maximum memory allocation, e.g. {0: '20GB', 'cpu': '50GB'}",
    )
    offload_folder: str | None = Field(
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
    # Matches the backend field. Without it, fitting a MoE model onto a card too
    # small for it meant hand-writing -ncmoe into llama_server_extra_args, and the
    # sizing endpoints name the setting n_cpu_moe -- so their recommendation could
    # not be handed to load_model as-is.
    n_cpu_moe: int = Field(
        default=0,
        ge=0,
        description="[llama.cpp] Keep the MoE expert weights of the first N layers "
        "on the CPU (-ncmoe). 0 disables it.",
    )

    # llama-server-specific configuration (OpenAI-compatible API)
    llama_server_url: str | None = Field(
        default=None,
        description="[llama_server] Server base URL, e.g. http://127.0.0.1:8080",
    )
    llama_server_api_key: str | None = Field(
        default=None,
        description="[llama_server] API key (if server-side authorization is enabled)",
    )
    llama_server_model: str | None = Field(
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
    llama_server_binary: str | None = Field(
        default=None,
        description="[llama_server] Path to llama-server executable (uses environment default if not provided)",
    )
    llama_server_device: str | None = Field(
        default=None,
        description=(
            "[llama_server] Target offload device (maps to --device, e.g. 'Vulkan1' / 'CUDA0'); "
            "llama picks one itself when omitted. Use it to pin the card on multi-GPU hosts "
            "(e.g. Intel iGPU + NVIDIA)"
        ),
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
    llama_server_extra_args: list[str] | None = Field(
        default=None,
        description="[llama_server] Additional startup args, e.g. ['--mlock', '--no-mmap']",
    )
    llama_server_mmproj: str | None = Field(
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
    vllm_max_model_len: int | None = Field(
        default=None,
        description="[vLLM] --max-model-len; falls back to n_ctx when not provided",
        ge=1,
    )
    vllm_dtype: str = Field(default="auto", description="[vLLM] --dtype")
    vllm_quantization: str | None = Field(
        default=None,
        description="[vLLM] --quantization, e.g. awq/gptq/fp8",
    )
    vllm_enforce_eager: bool = Field(
        default=False,
        description="[vLLM] Whether to enable --enforce-eager",
    )
    vllm_kv_cache_dtype: str | None = Field(
        default=None,
        description="[vLLM] --kv-cache-dtype, e.g. auto/fp8_e5m2/fp8_e4m3",
    )
    vllm_cpu_offload_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="[vLLM] --cpu-offload-gb",
    )
    vllm_kv_offloading_size: float | None = Field(
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
    vllm_max_num_seqs: int | None = Field(
        default=None,
        ge=1,
        description="[vLLM] --max-num-seqs",
    )
    vllm_max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="[vLLM] --max-num-batched-tokens",
    )
    # vLLM multimodal (Vision / Audio / Video) specific configuration
    # Applicable to VLM models such as Gemma 4, Gemma 3n, Qwen-VL, and LLaVA
    vllm_mm_image_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max number of images in --limit-mm-per-prompt; "
            "set this when loading multimodal models (e.g. gemma-4-E2B-it), such as 1"
        ),
    )
    vllm_mm_audio_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max number of audio inputs in --limit-mm-per-prompt; "
            "only needed for audio-capable models such as Gemma 4 E2B/E4B"
        ),
    )
    vllm_mm_video_limit: int | None = Field(
        default=None,
        ge=1,
        description="[vLLM] Max number of videos in --limit-mm-per-prompt",
    )
    vllm_hf_overrides: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "[vLLM] --hf-overrides to force override fields in HuggingFace config.json; "
            "for example, if gemma-4-E2B-it is incorrectly detected as text-only, set "
            '{"architectures":["Gemma4ForConditionalGeneration"]} to force multimodal architecture. '
            "Accepts dict or JSON string"
        ),
    )
    vllm_chat_template: str | None = Field(
        default=None,
        description=(
            "[vLLM] --chat-template, path to custom chat template file (.jinja); "
            "when tokenizer_config.json lacks chat_template (e.g. some base or quantized variants), "
            "provide this to support /v1/chat/completions"
        ),
    )
    # The backend defaults this to "auto", so leaving it out of the payload and
    # sending "auto" are the same request -- but only the field makes the third
    # state reachable: naming a parser the model's chat template does not imply,
    # or turning tool calling off for a model whose template does imply one.
    vllm_tool_call_parser: str = Field(
        default="auto",
        description=(
            "[vLLM] Tool-call parser. 'auto' (default) infers it from the model's chat "
            "template and enables tool calling; 'off' serves chat only; any other value "
            "is passed to vLLM as-is (e.g. hermes/qwen3_coder/qwen3_xml/gemma4/"
            "llama3_json/mistral/deepseek_v3)"
        ),
    )

    @field_validator("vllm_chat_template", mode="before")
    @classmethod
    def _normalize_vllm_chat_template(cls, v: object) -> object:
        """Normalize empty chat_template strings to None."""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @field_validator("vllm_hf_overrides", mode="before")
    @classmethod
    def _normalize_vllm_hf_overrides(cls, v: object) -> object:
        """
        Accept a dict or JSON string; treat empty strings as unset.

        The annotation above is the authority on what is allowed. This used to
        also wave `list` through, which the annotation then rejected anyway --
        all that achieved was turning a clear TypeError into a pydantic
        ValidationError.
        """
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        if isinstance(v, dict):
            return v
        raise TypeError("vllm_hf_overrides must be a dict, a JSON string, or None")


class DeviceAllocation(BaseModel):
    """Device allocation statistics."""

    summary: str | None = Field(
        default=None,
        description="Summary of module counts per device, e.g. 'cuda:0:30, cpu:10'",
    )
    total_modules: int | None = Field(default=None, description="Total number of model modules")
    layer_lines: list[str] | None = Field(
        default=None,
        description="Layer-level allocation, e.g. ['model.layers.0 -> cuda:0', ...]",
    )


class ModelStatus(BaseModel):
    """Model status."""

    loaded: bool = Field(default=False, description="Whether model is loaded")
    is_loading: bool = Field(default=False, description="Whether model is currently loading")
    loading_error: str | None = Field(default=None, description="Model loading error message")
    model_name: str | None = Field(default=None, description="Model name")
    model_path: str | None = Field(default=None, description="Model path")
    engine: InferenceEngine = Field(
        default=InferenceEngine.TRANSFORMERS,
        description="Inference engine: transformers (default), llama_server, vllm",
    )
    quantization: str | None = Field(default=None, description="Quantization type")
    model_total_memory: str | None = Field(
        default=None, description="Total model memory requirement"
    )
    device_map: str | dict | None = Field(default=None, description="Device mapping")
    max_memory: dict | None = Field(default=None, description="Maximum memory limits")
    offload_folder: str | None = Field(default=None, description="Offload folder")
    device: str | None = Field(default=None, description="Device")
    memory_usage: dict | None = Field(default=None, description="Memory usage")
    device_allocation: DeviceAllocation | None = Field(
        default=None,
        description="Actual device allocation statistics (available after model load)",
    )

    # llama.cpp specific status
    n_gpu_layers: int | None = Field(
        default=None, description="[llama.cpp] Actual number of GPU layers used"
    )
    n_ctx: int | None = Field(default=None, description="[llama.cpp] Context length")
    n_batch: int | None = Field(default=None, description="[llama.cpp] Batch size")
    prefill_strategy: str | None = Field(
        default=None,
        description="[llama_server] Prefill strategy, e.g. slot or cache_prompt",
    )
    llama_capabilities: list[str] | None = Field(
        default=None, description="[llama_server] Capabilities reported by /v1/models"
    )
    llama_server_binary: str | None = Field(
        default=None,
        description="[llama_server] Absolute path of the binary this load runs",
    )
    slot_restore_summary: dict[str, Any] | None = Field(
        default=None, description="[llama_server] slot restore result summary"
    )
    llama_server_extra_args: list[str] | None = Field(
        default=None,
        description="[llama_server] Extra launch arguments, e.g. ['--mlock', '--no-mmap']",
    )

    # vLLM specific status
    vllm_gpu_memory_utilization: float | None = Field(
        default=None, description="[vLLM] --gpu-memory-utilization"
    )
    vllm_max_model_len: int | None = Field(default=None, description="[vLLM] --max-model-len")
    vllm_dtype: str | None = Field(default=None, description="[vLLM] --dtype")
    vllm_quantization: str | None = Field(default=None, description="[vLLM] --quantization")
    vllm_enforce_eager: bool | None = Field(default=None, description="[vLLM] --enforce-eager")
    vllm_kv_offloading_size: float | None = Field(
        default=None, description="[vLLM] --kv-offloading-size"
    )
    vllm_kv_cache_dtype: str | None = Field(default=None, description="[vLLM] --kv-cache-dtype")
    vllm_cpu_offload_gb: float | None = Field(default=None, description="[vLLM] --cpu-offload-gb")
    vllm_tensor_parallel_size: int | None = Field(
        default=None, description="[vLLM] --tensor-parallel-size"
    )
    vllm_max_num_seqs: int | None = Field(default=None, description="[vLLM] --max-num-seqs")
    vllm_max_num_batched_tokens: int | None = Field(
        default=None, description="[vLLM] --max-num-batched-tokens"
    )
    vllm_mm_image_limit: int | None = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt image limit",
    )
    vllm_mm_audio_limit: int | None = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt audio limit",
    )
    vllm_mm_video_limit: int | None = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt video limit",
    )
    vllm_hf_overrides: str | dict[str, Any] | None = Field(
        default=None,
        description="[vLLM] --hf-overrides",
    )
    vllm_chat_template: str | None = Field(
        default=None,
        description="[vLLM] --chat-template",
    )
    # None rather than "auto": this reports what the running server was given,
    # and a status read before anything is loaded has no answer to report.
    vllm_tool_call_parser: str | None = Field(
        default=None,
        description="[vLLM] --tool-call-parser, or 'off' when tool calling is not served",
    )


# ------ Inference-related Chat / Stream Pydantic models removed ------
# The previous ChatRequest / ChatResponse / ChatStreamChunk mapped to backend
# `/inference/chat`. The console now uses the OpenAI-compatible
# endpoint (`<backend_url>/v1/chat/completions`) for all chat. Please use
# the OpenAI Python SDK. See openai-compatible-example.py and
# openai-compatible-image-example.py for examples.


class AppSettings(BaseModel):
    """Application settings (used by fine-tune examples)."""

    backend_url: str = Field(..., description="Backend API URL")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificate")
    finetune_config_path: str | None = Field(
        default=None, description="Fine-tune training config file path"
    )


# ------ Training-related configuration models ------
class TrainingConfig(BaseModel):
    """
    Training configuration.

    Supports three training methods:
    - full: full-parameter fine-tuning
    - lora: parameter-efficient fine-tuning via LoRA
    - qlora: quantized training with QLoRA + LoRA
    """

    model_name: str = Field(
        ...,
        description="Model name label (from the label field in models registry config)",
    )
    method: TrainingMethod = Field(..., description="Training method: lora / qlora / full")
    dataset_path: str = Field(
        ...,
        description=(
            "Training dataset: a path to a local .json / .jsonl file, or any other value, "
            "which is treated as a Hugging Face Hub dataset id. A Hub id cannot carry a "
            "config/subset or split selector, and skips the local-file format sniffing — set "
            "prompt_field/completion_field/text_field to match its columns"
        ),
    )
    output_dir: str = Field(..., description="Output directory path for fine-tuned model files")
    offload_folder: str | None = Field(
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
    lora_target_modules: list[str] | None = Field(
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
    warmup_steps: int = Field(default=100, description="Warmup steps using lower learning rate")
    logging_steps: int = Field(
        default=10,
        description="Report training progress every N steps (loss, step, etc.)",
    )
    save_steps: int = Field(default=500, description="Save checkpoint every N steps")
    save_total_limit: int | None = Field(
        default=2,
        description="Maximum number of checkpoints to keep; oldest will be deleted automatically",
    )
    max_seq_length: int = Field(
        default=2048,
        description="Maximum token length during training; longer sequences are truncated",
    )

    # Dataset field configuration - choose one of three training data modes
    text_field: str | None = Field(
        default="text",
        description="[Mode 1] Single-field mode: train by predicting continuation in one text field. Field name in dataset, e.g. 'text'. Mutually exclusive with the other modes",
    )
    prompt_field: str | None = Field(
        default=None,
        description="[Mode 2] Two-field mode: separate prompt and response. Prompt field name in dataset, e.g. 'prompt'. Must be used with completion_field",
    )
    completion_field: str | None = Field(
        default=None,
        description="[Mode 2] Two-field mode: separate prompt and response. Completion field name in dataset, e.g. 'completion'. Must be used with prompt_field",
    )
    messages_field: str | None = Field(
        default=None,
        description=(
            "[Mode 3] OpenAI chat format mode: the dataset has a messages field "
            "(list of {role, content}). TRL applies the model's chat template automatically "
            "and computes loss on assistant tokens only. Left empty, a messages field in the "
            "dataset is detected automatically"
        ),
    )

    # Evaluation / held-out test set
    eval_split_ratio: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of the training set held out as a test set (0~1, e.g. 0.1 means 10%). "
            "Empty or 0 skips evaluation. The held-out data is evaluated periodically during "
            "training and once more after training ends"
        ),
    )
    eval_steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between test-set evaluations. Defaults to logging_steps when empty. "
            "Only effective when eval_split_ratio is set"
        ),
    )
    eval_split_seed: int = Field(
        default=42,
        description=(
            "Random seed for the test-set split, so the same data always yields the same split"
        ),
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
    deepspeed_config: str | None = Field(
        default=None,
        description="[Option 1] Full path to DeepSpeed config file, e.g. './my_deepspeed.json'. Mutually exclusive with deepspeed_profile",
    )
    deepspeed_profile: str | None = Field(
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
    gradient_checkpointing: bool = Field(
        default=True,
        description=(
            "Whether to enable gradient checkpointing (trades compute time for memory; "
            "essential for large-model training)"
        ),
    )
    use_flash_attention: bool = Field(
        default=False,
        description=(
            "Whether to enable Triton-accelerated attention: tries flash_attention_2 first "
            "(needs the flash-attn package), falling back to PyTorch's built-in sdpa (also a "
            "Triton kernel) when unavailable. Speeds up attention and lowers VRAM usage"
        ),
    )
    num_gpus: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("num_gpus", "num_gpu"),
        description=(
            "GPU count used for fine-tuning. 1 = single GPU (default); >1 launches multi-GPU "
            "distributed training via deepspeed --num_gpus and lifts the "
            "CUDA_VISIBLE_DEVICES restriction automatically"
        ),
    )


class TrainingStatus(BaseModel):
    """Training status."""

    is_training: bool = Field(default=False, description="Whether training is currently running")
    progress: float = Field(default=0.0, description="Training progress (0-1)")
    current_step: int = Field(default=0, description="Current step")
    total_steps: int = Field(default=0, description="Total steps")
    loss: float | None = Field(default=None, description="Current loss")
    current_epoch: float | None = Field(default=0.0, description="Current epoch")
    total_epochs: int | None = Field(default=None, description="Total number of epochs")
    status: str | None = Field(default=None, description="Coarse status message")
    phase: str | None = Field(
        default=None, description="Fine-grained lifecycle phase (see the backend's core.Phase)"
    )
    phase_detail: str | None = Field(
        default=None, description="Human-readable detail for the current phase"
    )
    session_id: str | None = Field(default=None, description="Current training session ID")
    job_id: str | None = Field(
        default=None, description="Job ID (alias of session_id; universal key for logs)"
    )
    error: str | None = Field(
        default=None, description="Real error message only (progress lives in `phase`)"
    )
    config: TrainingConfig | None = Field(
        default=None, description="Current or last training configuration"
    )


# ------ Structured training-log models (`/training/{job_id}/log[/stream]`) ------
# These mirror service/training/core/training_events.py. The backend writes every
# event to a durable events.jsonl, so the SSE stream and the polling endpoint carry
# exactly the same records.
class TrainingPhase(StrEnum):
    """Fine-grained training lifecycle phase, in the order the backend emits them."""

    PENDING = "pending"  # written into meta.json, never emitted as an event
    QUEUED = "queued"  # first stage event of every run, not queue-only
    RESOLVING_DEEPSPEED = "resolving_deepspeed"
    LAUNCHING_WORKERS = "launching_workers"
    LOADING_TOKENIZER = "loading_tokenizer"
    LOADING_DATASET = "loading_dataset"
    LOADING_MODEL = "loading_model"
    PREPARING_TRAINER = "preparing_trainer"
    TRAINING = "training"
    SAVING = "saving"
    MERGING_LORA = "merging_lora"
    CONVERTING_GGUF = "converting_gguf"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    CANCELLED = "cancelled"  # reserved for the future job queue


TERMINAL_PHASES = frozenset(
    {
        TrainingPhase.COMPLETED,
        TrainingPhase.STOPPED,
        TrainingPhase.ERROR,
        TrainingPhase.CANCELLED,
    }
)


class TrainingEventType(StrEnum):
    """Kind of record in the backend's events.jsonl."""

    STAGE = "stage"  # lifecycle transition
    RESOURCE = "resource"  # GPU / DRAM / SSD snapshot
    METRIC = "metric"  # trainer step metrics
    ERROR = "error"  # a real error / traceback
    INFO = "info"  # env snapshot, eval summaries, forwarded worker messages


class TrainingEvent(BaseModel):
    """One record from a job's training log."""

    seq: int = Field(default=0, description="Monotonic sequence number; also the SSE event id")
    ts: float | None = Field(default=None, description="Unix timestamp the event was written")
    job_id: str | None = Field(default=None, description="Job the event belongs to")
    # Deliberately str, not TrainingEventType: an unknown future event kind must not
    # fail the whole stream. Compare against TrainingEventType members, which are StrEnum.
    type: str = Field(default="", description="Event kind: stage/resource/metric/error/info")
    owner: str | None = Field(default=None, description="Event owner, when the backend sets one")
    phase: str | None = Field(default=None, description="Lifecycle phase carried by the event")
    msg: str | None = Field(default=None, description="Short human-readable message")
    data: dict[str, Any] | None = Field(default=None, description="Event-kind-specific payload")

    @property
    def is_terminal(self) -> bool:
        """Whether this event ends the job, matching the backend's is_terminal_event()."""
        if self.type == TrainingEventType.ERROR:
            return True
        return self.type == TrainingEventType.STAGE and self.phase in TERMINAL_PHASES

    @property
    def is_eval_metric(self) -> bool:
        """
        Whether this is a held-out evaluation metric rather than a training one.

        Eval points are interleaved into the same stream, so charting every metric
        event mixes them into the training loss curve.
        """
        if self.type != TrainingEventType.METRIC:
            return False
        return (self.data or {}).get("split") == "eval"


class TrainingLogEventsResponse(BaseModel):
    """Backlog of training-log events (`/training/{job_id}/log`)."""

    job_id: str = Field(..., description="Job the events belong to")
    cursor: int = Field(
        default=0, description="Highest event seq returned; pass as the next `since`"
    )
    events: list[TrainingEvent] = Field(default_factory=list)


class TrainingErrorDetails(BaseModel):
    """Detailed training error (`/training/error_details`)."""

    has_error: bool = Field(default=False, description="False during a normal run")
    error: str | None = Field(default=None, description="Short error string")
    error_type: str | None = Field(default=None, description="Exception class name")
    is_oom: bool = Field(default=False, description="Whether the failure was an OOM")
    error_traceback: str | None = Field(default=None, description="Full traceback")
    process_alive: bool = Field(default=False, description="Whether the worker is still alive")
    message: str | None = Field(default=None, description="Set to a notice when has_error is False")
