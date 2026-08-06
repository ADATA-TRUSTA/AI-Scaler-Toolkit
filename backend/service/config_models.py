"""Configuration models for inference and training."""

from enum import StrEnum
from typing import Any, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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


class InferenceSharedFields(BaseModel):
    """Fields shared by inference config and status."""

    model_name: str | None = Field(default=None, description="Model name or path")
    model_path: str | None = Field(
        default=None,
        description=(
            "Path to a locally fine-tuned model = output_dir; when set, it takes "
            "precedence over model_name for from_pretrained"
        ),
    )
    engine: InferenceEngine = Field(
        default=InferenceEngine.TRANSFORMERS,
        description="Inference engine: transformers (default), llama_server, vllm",
    )
    quantization: QuantizationType | str | None = Field(
        default=None,
        description="Quantization type: none, int8, int4, nf4, fp4",
    )
    device_map: str | dict | None = Field(
        default="auto",
        description=(
            "Device map strategy, e.g. 'auto', 'cpu', 'cuda:0', "
            "{'': 0, 'cpu': 'cpu'}, 'balanced_low_0'"
        ),
    )
    model_total_memory: str | None = Field(
        default=None, description="Total memory the model needs, e.g. '15GB'"
    )
    max_memory: dict[int | str, str] | None = Field(
        default=None, description="Max memory per device, e.g. {0: '20GB', 'cpu': '50GB'}"
    )
    offload_folder: str | None = Field(
        default=None, description="Offload folder path, used to offload model weights to disk"
    )

    # Snapshot fields shared by GGUF / llama-server
    n_gpu_layers: int | None = Field(
        default=None, description="[llama_server] GPU layer count; -1 means all"
    )
    n_ctx: int | None = Field(default=None, description="[llama_server] Context length")
    n_batch: int | None = Field(default=None, description="[llama_server] Batch size")
    llama_server_extra_args: list[str] | None = Field(
        default=None,
        description="[llama_server] Extra launch arguments, e.g. ['--mlock', '--no-mmap']",
    )

    # Shared vLLM fields
    vllm_gpu_memory_utilization: float | None = Field(
        default=None,
        description="[vLLM] --gpu-memory-utilization",
    )
    vllm_max_model_len: int | None = Field(
        default=None,
        description="[vLLM] --max-model-len; falls back to n_ctx when omitted",
    )
    vllm_dtype: str | None = Field(default=None, description="[vLLM] --dtype")
    vllm_quantization: str | None = Field(
        default=None,
        description="[vLLM] --quantization, e.g. awq/gptq/fp8",
    )
    vllm_enforce_eager: bool | None = Field(
        default=None,
        description="[vLLM] Whether to pass --enforce-eager",
    )
    vllm_kv_cache_dtype: str | None = Field(
        default=None,
        description="[vLLM] --kv-cache-dtype, e.g. auto/fp8_e5m2/fp8_e4m3",
    )
    vllm_cpu_offload_gb: float | None = Field(
        default=None,
        description="[vLLM] --cpu-offload-gb",
    )
    vllm_tensor_parallel_size: int | None = Field(
        default=None,
        description="[vLLM] --tensor-parallel-size",
    )
    vllm_max_num_seqs: int | None = Field(
        default=None,
        description="[vLLM] --max-num-seqs",
    )
    vllm_max_num_batched_tokens: int | None = Field(
        default=None,
        description="[vLLM] --max-num-batched-tokens",
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
    vllm_kv_offloading_size: float | None = Field(
        default=None,
        description=(
            "[vLLM] --kv-offloading-size (in GB; under multi-GPU TP this is the total "
            "across all TP ranks, not per GPU)"
        ),
    )
    vllm_hf_overrides: str | dict[str, Any] | None = Field(
        default=None,
        description="[vLLM] --hf-overrides",
    )
    vllm_chat_template: str | None = Field(
        default=None,
        description="[vLLM] --chat-template",
    )


class InferenceConfig(InferenceSharedFields):
    """
    Inference config - uses the Hugging Face Transformers format directly.

    Examples:
    {
        "model_name": "Qwen/Qwen3-4B",
        "quantization": "none",
        "device_map": "auto",
        "model_total_memory": "15GB",
        "max_memory": {"0": "5GB", "cpu": "5GB"},
        "offload_folder": "./offload"
    }
    or
    {
        "model_name": "Qwen/Qwen3-8B",
        "quantization": "none",
        "model_total_memory": "20GB",
        "device_map": "cpu"
    }
    """

    # pydantic idiom: narrow the optional parent field (str | None) to required str.
    model_name: str = Field(..., description="Model name or path")  # pyright: ignore[reportGeneralTypeIssues]
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization type: none, int8, int4, nf4, fp4",
    )
    torch_dtype: str = Field(default="auto", description="Torch dtype")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    use_cache: bool = Field(default=True, description="Use KV cache")

    # Config shared by GGUF / llama-server
    n_gpu_layers: int = Field(
        default=-1, description="[llama_server] GPU layer count; -1 means all"
    )
    n_ctx: int = Field(default=4096, description="[llama_server] Context length")
    n_batch: int = Field(default=512, description="[llama_server] Batch size")

    # llama-server specific config (OpenAI-compatible API)
    llama_server_url: str | None = Field(
        default=None,
        description="[llama_server] Server base URL, e.g. http://127.0.0.1:8080",
    )
    llama_server_api_key: str | None = Field(
        default=None, description="[llama_server] API key (when the server requires auth)"
    )
    llama_server_model: str | None = Field(
        default=None,
        description="[llama_server] Model name used in requests; defaults to model_name",
    )
    llama_server_timeout: int = Field(
        default=300,
        description="[llama_server] Request timeout in seconds",
        ge=10,
    )
    llama_server_auto_start: bool = Field(
        default=True,
        description="[llama_server] Whether the engine starts a llama-server subprocess on load",
    )
    llama_server_binary: str | None = Field(
        default=None,
        description="[llama_server] llama-server binary path (uses the env default when omitted)",
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
        default="127.0.0.1", description="[llama_server] Host the subprocess binds to"
    )
    llama_server_port: int = Field(
        default=5001,
        description="[llama_server] Port used when starting the subprocess",
        ge=1,
        le=65535,
    )
    llama_server_np: int = Field(
        default=1,
        description="[llama_server] Parallel generation slots (llama-server -np)",
        ge=1,
    )
    llama_server_health_timeout: int = Field(
        default=300,
        description="[llama_server] Seconds to wait for the server to come up during load",
        ge=5,
    )
    llama_server_mmproj: str | None = Field(
        default=None,
        description=(
            "[llama_server] Multimodal projector (.gguf) path; --mmproj is appended "
            "automatically when set"
        ),
    )

    # vLLM OpenAI-compatible server specific config
    vllm_gpu_memory_utilization: float = Field(
        default=0.8,
        description="[vLLM] --gpu-memory-utilization",
        ge=0.05,
        le=0.99,
    )
    vllm_max_model_len: int | None = Field(
        default=None,
        description="[vLLM] --max-model-len; falls back to n_ctx when omitted",
        ge=1,
    )
    vllm_dtype: str = Field(default="auto", description="[vLLM] --dtype")
    vllm_quantization: str | None = Field(
        default=None,
        description="[vLLM] --quantization, e.g. awq/gptq/fp8",
    )
    vllm_enforce_eager: bool = Field(
        default=False,
        description="[vLLM] Whether to pass --enforce-eager",
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
            "[vLLM] --kv-offloading-size (in GB; under multi-GPU TP this is the total "
            "across all TP ranks, not per GPU)"
        ),
        validation_alias=AliasChoices("vllm_kv_offloading_size", "vllm_swap_space"),
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
    # vLLM multimodal (Vision / Audio / Video) specific config
    # Applies to VLM models such as Gemma 4, Gemma 3n, Qwen-VL, LLaVA
    vllm_mm_image_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max image count in --limit-mm-per-prompt; required, e.g. 1, when a "
            "multimodal model (such as gemma-4-E2B-it) should handle images"
        ),
    )
    vllm_mm_audio_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] Max audio count in --limit-mm-per-prompt; only needed for models with "
            "audio support such as Gemma 4 E2B/E4B"
        ),
    )
    vllm_mm_video_limit: int | None = Field(
        default=None,
        ge=1,
        description="[vLLM] Max video count in --limit-mm-per-prompt",
    )
    vllm_hf_overrides: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "[vLLM] --hf-overrides, forces an override of HuggingFace config.json fields; "
            "e.g. when gemma-4-E2B-it is misdetected as a text-only architecture, set "
            '{"architectures":["Gemma4ForConditionalGeneration"]} to force the multimodal '
            "variant. Accepts a dict or a JSON string"
        ),
    )
    vllm_chat_template: str | None = Field(
        default=None,
        description=(
            "[vLLM] --chat-template, path to a custom chat template file (.jinja); needed "
            "to support /v1/chat/completions when tokenizer_config.json has no "
            "chat_template field (e.g. some base or quantized builds)"
        ),
    )

    @field_validator("vllm_chat_template", mode="before")
    @classmethod
    def _normalize_vllm_chat_template(cls, v: Any) -> Any:  # noqa: ANN401 - pydantic pre-validator accepts arbitrary raw input
        """Normalize an empty chat_template string to None."""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @field_validator("vllm_hf_overrides", mode="before")
    @classmethod
    def _normalize_vllm_hf_overrides(cls, v: Any) -> Any:  # noqa: ANN401 - pydantic pre-validator accepts arbitrary raw input
        """Accept a dict/list or a JSON string; an empty string counts as unset."""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        if isinstance(v, (dict, list)):
            return v
        raise ValueError("vllm_hf_overrides must be a dict, list, JSON string, or None")


class ChatRequest(BaseModel):
    """Chat request."""

    message: str = Field(..., description="User message")
    max_new_tokens: int = Field(default=512, description="Max tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p sampling", ge=0.0, le=1.0)
    top_k: int = Field(default=50, description="Top-k sampling", ge=0)
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty", ge=1.0)
    stream: bool = Field(default=True, description="Whether to stream the response")
    system_prompt: str | None = Field(default=None, description="System prompt")

    # Timeout control
    total_timeout: int = Field(
        default=300,
        description="Total generation timeout in seconds; generation stops once exceeded",
        ge=10,
    )

    # Chat template control
    enable_thinking: bool | None = Field(
        default=True,
        description=(
            "Enable thinking mode (for models that support it, e.g. DeepSeek, QwQ). "
            "None = use the model default"
        ),
    )

    # RAG control
    use_rag: bool = Field(
        default=False, description="Whether to run RAG retrieval and inject context"
    )
    rag_top_k: int = Field(default=3, description="Number of documents RAG returns", ge=1, le=50)
    rag_query: str | None = Field(
        default=None, description="Override the user message as the RAG query"
    )
    rag_include_sources: bool = Field(
        default=True, description="Whether to include sources in the prompt"
    )

    # Hybrid session management
    session_id: str | None = Field(
        default=None, description="Session ID; when set, history can be kept on the backend"
    )
    reset_history: bool = Field(
        default=False,
        description="Whether to reset session history before this request (needs session_id)",
    )
    # Optional: history passed straight from the frontend (takes precedence when set)
    # Structure mirrors OpenAI: [{role: user|assistant|system, content: str}]
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional conversation history; overrides the backend copy when set",
    )
    images: list[str] | None = Field(
        default=None,
        description=(
            "Optional image inputs (only effective on multimodal models). "
            "Each item may be a local file path, an http(s) URL, or data:image/...;base64,..."
        ),
        max_length=8,
    )
    request_id: str | None = Field(
        default=None,
        description=(
            "Optional request ID. Once set it can be used with /inference/stop_generation "
            "to stop exactly this request"
        ),
    )


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible message format."""

    role: str = Field(..., description="Message role, e.g. system/user/assistant")
    content: str | list[dict[str, Any]] | None = Field(
        default="",
        description="Message content; a string, or an OpenAI multimodal content parts list",
    )
    name: str | None = Field(
        default=None, description="Optional name field, common on tool/function messages"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool call list in an assistant message"
    )
    tool_call_id: str | None = Field(
        default=None, description="The tool_call_id this tool message answers"
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = Field(default=None, description="Model name (compatibility field)")
    messages: list[OpenAIChatMessage] = Field(
        ..., min_length=1, description="Multi-turn conversation messages"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    total_timeout: int | None = Field(
        default=300, ge=10, description="Total generation timeout in seconds"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Max tokens to generate",
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "OpenAI-compatible field; approximately mapped when repetition_penalty is not "
            "given explicitly"
        ),
    )
    stream: bool = Field(default=False, description="Whether to stream back over SSE")
    stream_options: dict[str, Any] | None = Field(
        default=None,
        description="OpenAI-compatible stream_options, e.g. {'include_usage': true}",
    )
    user: str | None = Field(
        default=None, description="End-user identifier (can map to session_id)"
    )

    # Backend extension fields below, kept aligned with /inference/chat
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    session_id: str | None = Field(default=None)
    reset_history: bool = Field(default=False)
    enable_thinking: bool | None = Field(default=True)
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Compatible with extra_body.chat_template_kwargs, e.g. {'enable_thinking': false}"
        ),
    )
    use_rag: bool = Field(default=False)
    rag_top_k: int = Field(default=3, ge=1, le=50)
    rag_query: str | None = Field(default=None)
    rag_include_sources: bool = Field(default=True)
    request_id: str | None = Field(default=None)
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="OpenAI tool definition list"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None, description="OpenAI tool_choice setting"
    )


class StopGenerationRequest(BaseModel):
    """Stop generation request."""

    request_id: str | None = Field(
        default=None,
        description="Optional worker request ID. When set, stops exactly that generation",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional conversation session ID. Without request_id, the backend looks up the "
            "session's currently active worker request and stops it"
        ),
    )


class CleanupGenerationMemoryRequest(BaseModel):
    """Cleanup generation memory request."""

    slot: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional slot id. When set, only that slot's cache is cleared; otherwise all "
            "visible slots are cleared"
        ),
    )


class RagAddDocument(BaseModel):
    """Request to add or update a RAG document."""

    doc_id: str | None = Field(default=None, description="Document ID; generated when omitted")
    content: str = Field(..., description="Plain text content")


class TrainingConfig(BaseModel):
    """
    Training config.

    Supports three training methods:
    - full: full-parameter fine-tuning
    - lora: LoRA parameter-efficient fine-tuning
    - qlora: QLoRA quantization + LoRA fine-tuning
    """

    model_name: str = Field(
        ..., description="Model name label (the label from the models registry config)"
    )
    method: TrainingMethod = Field(..., description="Training method: lora / qlora / full")
    dataset_path: str = Field(..., description="Training dataset file path; must be JSON or JSONL")
    output_dir: str = Field(..., description="Output folder path for the fine-tuned model files")
    offload_folder: str | None = Field(
        default="./deepspeed_offload",
        description="Offload folder path; overrides the DeepSpeed JSON config",
    )

    # LoRA/QLoRA specific
    lora_r: int = Field(
        default=8,
        description=(
            "[method = LoRA/QLoRA only] LoRA rank, controls the trainable parameter count. "
            "Higher means more parameters and possibly better results, but slower training"
        ),
    )
    lora_alpha: int = Field(
        default=16,
        description=("[method = LoRA/QLoRA only] LoRA alpha scaling factor, usually 1-2x lora_r"),
    )
    lora_dropout: float = Field(
        default=0.05,
        description="[method = LoRA/QLoRA only] LoRA dropout rate, guards against overfitting",
    )
    lora_target_modules: list[str] | None = Field(
        default=None,
        description=(
            "[method = LoRA/QLoRA only] LoRA target module list, naming the layers LoRA is "
            "applied to. null uses the defaults (e.g. q_proj, k_proj, v_proj, o_proj)"
        ),
    )

    # Training hyperparameters
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        default=1, description="Samples taken per training step (batch size)"
    )
    gradient_accumulation_steps: int = Field(
        default=8, description="Gradients accumulated before each parameter update"
    )
    learning_rate: float = Field(
        default=2e-4, description="Learning rate, controls the size of each parameter update"
    )
    warmup_steps: int = Field(
        default=100, description="Steps to warm up with a smaller learning rate"
    )
    logging_steps: int = Field(
        default=10, description="Steps between training progress reports (loss, step, ...)"
    )
    save_steps: int = Field(default=500, description="Steps between checkpoint saves")
    save_total_limit: int | None = Field(
        default=2, description="Max checkpoints to keep; the oldest are deleted beyond this"
    )
    max_seq_length: int = Field(
        default=2048, description="Max token length during training; longer inputs are truncated"
    )

    # Dataset field configuration - pick one of the three training modes
    text_field: str | None = Field(
        default="text",
        description=(
            "[Training mode 1] Single-field mode: predict later turns from earlier ones. "
            "Names the dataset field, e.g. 'text'. Mutually exclusive with the other modes"
        ),
    )
    prompt_field: str | None = Field(
        default=None,
        description=(
            "[Training mode 2] Two-field mode: separates question from answer. Names the "
            "dataset prompt field, e.g. 'prompt'. Must be used with completion_field"
        ),
    )
    completion_field: str | None = Field(
        default=None,
        description=(
            "[Training mode 2] Two-field mode: separates question from answer. Names the "
            "dataset completion field, e.g. 'completion'. Must be used with prompt_field"
        ),
    )
    messages_field: str | None = Field(
        default=None,
        description=(
            "[Training mode 3] OpenAI chat format mode: the dataset has a messages field "
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
            "Steps between test-set evaluations. Defaults to logging_steps when empty. Only "
            "effective when eval_split_ratio is set"
        ),
    )
    eval_split_seed: int = Field(
        default=42,
        description=(
            "Random seed for the test-set split, so the same data always yields the same split"
        ),
    )
    save_tokenizer: bool = Field(
        default=True, description="Whether to save the tokenizer to the output dir after training"
    )

    # DeepSpeed settings
    use_deepspeed: bool = Field(
        default=False,
        description=(
            "Whether to train with DeepSpeed offload (needs deepspeed_config or "
            "deepspeed_profile when enabled)"
        ),
    )
    deepspeed_config: str | None = Field(
        default=None,
        description=(
            "[Option 1] Full path to a detailed DeepSpeed config file, e.g. "
            "'./my_deepspeed.json'. Use either this or deepspeed_profile"
        ),
    )
    deepspeed_profile: str | None = Field(
        default=None,
        description=(
            "[Option 2] DeepSpeed config profile name, loaded automatically from "
            "service/configs/deepspeed/<profile>.json. Use either this or deepspeed_config"
        ),
    )
    # SFTTrainer specific
    use_sft_trainer: bool = Field(
        default=True,
        description="Whether to train with TRL's SFTTrainer (recommended for instruction tuning)",
    )
    packing: bool = Field(
        default=False,
        description=(
            "[use_sft_trainer=True only] Whether to enable packing (pack several short "
            "sequences into one long sequence for better training throughput)"
        ),
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

    @field_validator(
        "text_field",
        "prompt_field",
        "completion_field",
        "messages_field",
        "deepspeed_config",
        "deepspeed_profile",
        mode="before",
    )
    @classmethod
    def _normalize_optional_str(cls, v: Any) -> Any:  # noqa: ANN401 - pydantic pre-validator accepts arbitrary raw input
        """
        Normalize optional string fields.

        The frontend and other callers often use an empty string "" for "not set"; convert it
        to None to remove the ambiguity. Strings are also stripped.
        """
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @model_validator(mode="after")
    def _validate_dataset_fields(self) -> "TrainingConfig":
        """
        Validate dataset field configuration.

        - prompt_field and completion_field must both be set or both be omitted.
        - messages_field cannot be combined with prompt_field/completion_field.
        - The three modes are mutually exclusive (if none is set, dataset_loader auto-detects).
        """
        has_prompt = bool(self.prompt_field)
        has_completion = bool(self.completion_field)
        has_messages = bool(self.messages_field)

        if has_prompt != has_completion:
            raise ValueError(
                "prompt_field and completion_field must both be set or both be empty/None."
            )
        if has_messages and (has_prompt or has_completion):
            raise ValueError(
                "messages_field cannot be combined with prompt_field/completion_field; "
                "the three training modes are mutually exclusive."
            )
        return self


class DeviceAllocation(BaseModel):
    """Device allocation statistics."""

    summary: str | None = Field(
        default=None, description="Module count per device, e.g. 'cuda:0:30, cpu:10'"
    )
    total_modules: int | None = Field(default=None, description="Total module count of the model")
    layer_lines: list[str] | None = Field(
        default=None, description="Per-layer allocation, e.g. ['model.layers.0 -> cuda:0', ...]"
    )


class ModelStatus(InferenceSharedFields):
    """Model status."""

    loaded: bool = Field(default=False, description="Whether the model is loaded")
    is_loading: bool = Field(default=False, description="Whether the model is loading")
    loading_error: str | None = Field(default=None, description="Load error message")
    quantization: str | None = Field(default=None, description="Quantization type")
    device: str | None = Field(default=None, description="Device")
    memory_usage: dict | None = Field(default=None, description="Memory usage")
    device_allocation: DeviceAllocation | None = Field(
        default=None, description="Actual device allocation stats (only available once loaded)"
    )
    prefill_strategy: str | None = Field(
        default=None, description="[llama_server] Prefill strategy, e.g. slot or cache_prompt"
    )
    llama_capabilities: list[str] | None = Field(
        default=None, description="[llama_server] Capabilities reported by /v1/models"
    )
    slot_restore_summary: dict[str, Any] | None = Field(
        default=None, description="[llama_server] Summary of the slot restore result"
    )


class TrainingLog(BaseModel):
    """Training progress log."""

    timestamp: float = Field(..., description="Timestamp")
    step: int = Field(..., description="Step")
    loss: float = Field(..., description="Loss")
    learning_rate: float | None = Field(None, description="Learning Rate")
    epoch: float | None = Field(None, description="Epoch")
    accuracy: float | None = Field(None, description="Accuracy or Eval Accuracy")
    split: str = Field(
        default="train",
        description="Whether this metric comes from the training set (train) or test set (eval)",
    )


class GPULog(BaseModel):
    """Per-GPU log."""

    index: int = Field(..., description="GPU Index")
    name: str = Field(..., description="GPU Name")
    gpu_util_percent: float = Field(..., description="GPU Util %")
    gpu_memory_used_gb: float = Field(..., description="Used Memory (GB)")
    gpu_memory_total_gb: float = Field(..., description="Total Memory (GB)")
    temperature: float | None = Field(None, description="Temperature (C)")


class ResourceLog(BaseModel):
    """System resource log (field structure matches /system/resources)."""

    timestamp: float = Field(..., description="Timestamp")
    cpu: Optional["CPUInfo"] = Field(default=None, description="CPU/RAM resource info")
    gpu: Optional["GPUResource"] = Field(default=None, description="GPU resource info")
    disk: Optional["DiskResource"] = Field(default=None, description="Disk resource info")


class TrainingHistoryResponse(BaseModel):
    """Training history response."""

    session_id: str
    logs: list[TrainingLog]
    eval_logs: list[TrainingLog] = Field(
        default_factory=list,
        description=(
            "Held-out test set evaluation metrics, kept separate from training metrics; "
            "empty when evaluation is disabled"
        ),
    )


class SystemResourceHistoryResponse(BaseModel):
    """System resource history response."""

    session_id: str
    resources: list[ResourceLog]


class TrainingStatus(BaseModel):
    """Training status."""

    is_training: bool = Field(default=False, description="Whether training is in progress")
    progress: float = Field(default=0.0, description="Training progress (0-1)")
    current_step: int = Field(default=0, description="Current step")
    total_steps: int = Field(default=0, description="Total steps")
    loss: float | None = Field(default=None, description="Current loss")
    current_epoch: float | None = Field(default=0.0, description="Current epoch")
    total_epochs: int | None = Field(default=None, description="Total epochs")
    status: str | None = Field(default=None, description="Coarse status string")
    phase: str | None = Field(
        default=None, description="Fine-grained lifecycle phase (see core.Phase)"
    )
    phase_detail: str | None = Field(
        default=None, description="Human-readable detail for the current phase"
    )
    session_id: str | None = Field(default=None, description="Current session ID")
    job_id: str | None = Field(
        default=None, description="Job ID (alias of session_id; universal key for logs)"
    )
    error: str | None = Field(
        default=None, description="Real error message only (progress lives in `phase`)"
    )
    config: TrainingConfig | None = Field(
        default=None, description="Current or last training config"
    )


class TrainingLogEventsResponse(BaseModel):
    """Structured training-log events for a job (SSE backlog / polling)."""

    job_id: str
    cursor: int = Field(
        default=0, description="Highest event seq returned; pass as `since` next time"
    )
    events: list[dict[str, Any]] = Field(default_factory=list)


class MemoryEstimateRequest(BaseModel):
    """Memory estimate request."""

    model_name: str = Field(..., description="Model name or path")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization type: none, int8, int4, nf4, fp4",
    )
    batch_size: int = Field(default=1, description="Batch size", ge=1, le=32)
    sequence_length: int = Field(default=2048, description="Sequence length", ge=512, le=32768)
    include_activations: bool = Field(
        default=True, description="Whether to include activation memory in the estimate"
    )


class MemoryEstimateResponse(BaseModel):
    """Memory estimate response."""

    model_name: str
    model_size_billions: float
    quantization: str
    memory_breakdown_gb: dict[str, float]
    overhead_details_gb: dict[str, float] | None = None
    recommendations: dict[str, float]
    offload_strategies: list[dict]
    notes: list[str]


class KvCacheType(StrEnum):
    """llama.cpp KV cache data types accepted by -ctk / -ctv."""

    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    Q8_0 = "q8_0"
    Q5_1 = "q5_1"
    Q5_0 = "q5_0"
    Q4_1 = "q4_1"
    Q4_0 = "q4_0"
    IQ4_NL = "iq4_nl"


class GgufEstimateBase(BaseModel):
    """llama.cpp runtime settings shared by all GGUF memory estimation requests."""

    model_path: str = Field(
        default="",
        description=(
            "Path to the GGUF file; for a sharded model the first shard is enough. "
            "May also be supplied as -m inside args."
        ),
    )
    args: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("args", "llama_server_extra_args"),
        description=(
            "Extra llama-server arguments, taken verbatim from the frontend's "
            'InferenceConfig.llama_server_extra_args, e.g. ["-ncmoe","24","-ctk","q8_0"]. '
            'A single string such as "-ngl 20 -c 32768" is accepted too. Every alias is '
            "supported (-ngl/--gpu-layers/--n-gpu-layers), as is --flag=value form and "
            "-ot/--override-tensor. Values given here override the other fields of this "
            "JSON body, matching how llama-server appends these arguments last."
        ),
    )
    n_batch: int = Field(default=2048, description="Logical batch size (-b)", ge=1, le=1048576)
    n_ubatch: int = Field(default=512, description="Physical batch size (-ub)", ge=1, le=1048576)
    n_parallel: int = Field(default=1, description="Parallel sequence count (-np)", ge=1, le=256)
    flash_attn: bool = Field(default=True, description="Whether flash attention is on (-fa)")
    n_gpu: int = Field(default=1, description="Number of GPUs taking part in offload", ge=0, le=16)
    tensor_split: list[float] | None = Field(
        default=None, description="Multi-GPU split ratio (-ts); split evenly when omitted"
    )


class GgufEstimateRequest(GgufEstimateBase):
    """Request for a single-point GGUF memory estimate."""

    n_gpu_layers: int = Field(
        default=-1, description="Layers to offload to the GPU (-ngl); -1 means all"
    )
    n_ctx: int = Field(
        default=0,
        description="Context length (-c); 0 uses the model's trained context length",
        ge=0,
    )
    cache_type_k: KvCacheType = Field(default=KvCacheType.F16, description="K cache type (-ctk)")
    cache_type_v: KvCacheType = Field(default=KvCacheType.F16, description="V cache type (-ctv)")
    n_cpu_moe: int = Field(
        default=0,
        description="Keep the MoE expert weights of the first N layers on the CPU (-ncmoe)",
        ge=0,
    )
    cpu_moe: bool = Field(
        default=False, description="Keep MoE expert weights of every layer on the CPU (--cpu-moe)"
    )
    no_kv_offload: bool = Field(default=False, description="Hold the whole KV cache in host memory")
    swa_full: bool = Field(
        default=False, description="Use a full-size cache on sliding-window models"
    )
    include_per_layer: bool = Field(
        default=True, description="Whether to return the per-layer memory table"
    )
    verify: bool = Field(
        default=False,
        description=(
            "Also call llama-fit-params for exact figures; requires that binary and adds "
            "roughly a second"
        ),
    )


class GgufSweepRequest(GgufEstimateBase):
    """Request for a GGUF feasibility sweep."""

    n_gpu_layers_grid: list[int] | None = Field(
        default=None, description="-ngl values to sweep; a ladder is generated when omitted"
    )
    n_ctx_grid: list[int] | None = Field(
        default=None, description="Context lengths to sweep; common values are used when omitted"
    )
    kv_quant_grid: list[KvCacheType] | None = Field(
        default=None, description="KV cache types to sweep; defaults to [f16, q8_0]"
    )
    gpu_budget_mib: float | None = Field(
        default=None,
        description="GPU memory budget in MiB; currently free VRAM is read when omitted",
        ge=0,
    )
    host_budget_mib: float | None = Field(
        default=None,
        description="Host memory budget in MiB; currently free DRAM is read when omitted",
        ge=0,
    )
    margin_mib: float = Field(
        default=1024, description="Memory to leave free on each device, in MiB", ge=0
    )


class GgufRecommendRequest(GgufEstimateBase):
    """Request for a recommended GGUF configuration."""

    n_ctx: int = Field(
        default=0,
        description="Target context length; 0 uses the model's trained context length",
        ge=0,
    )
    n_ctx_min: int = Field(
        default=4096, description="Lowest context length the search may fall back to", ge=256
    )
    n_ctx_max: int = Field(
        default=0,
        description=(
            "Highest context the search may grow to when spending leftover budget; "
            "0 uses the model's trained context length"
        ),
        ge=0,
    )
    target_utilization: float = Field(
        default=0.9,
        description=(
            "Fraction of the usable budget the result should reach. Does not constrain the "
            "search, only whether the response reports unexplained headroom and why"
        ),
        ge=0.0,
        le=1.0,
    )
    gpu_budget_mib: float | None = Field(
        default=None,
        description="GPU memory budget in MiB; currently free VRAM is read when omitted",
        ge=0,
    )
    host_budget_mib: float | None = Field(
        default=None,
        description="Host memory budget in MiB; currently free DRAM is read when omitted",
        ge=0,
    )
    margin_mib: float = Field(
        default=1024, description="Memory to leave free on each device, in MiB", ge=0
    )
    allow_kv_quant: bool = Field(
        default=True, description="Whether the KV cache may be quantized to make it fit"
    )
    allow_ctx_reduction: bool = Field(
        default=True, description="Whether the context may be shortened to make it fit"
    )
    kv_cache_types: list[KvCacheType] | None = Field(
        default=None,
        description=(
            "KV cache ladder the search walks, first fit wins; overrides allow_kv_quant. "
            "Pass a single type to pin it"
        ),
    )
    verify: bool = Field(
        default=True,
        description=(
            "Cross-check the final recommendation with llama-fit-params; skipped when the "
            "binary is absent"
        ),
    )


class GgufPlanRequest(GgufEstimateBase):
    """Request for a menu of GGUF configurations across GPU budgets and KV cache types."""

    gpu_budgets_mib: list[float] | None = Field(
        default=None,
        description=(
            "GPU budgets in MiB to solve for, one plan each; currently free VRAM is used "
            "when omitted"
        ),
    )
    kv_cache_types: list[KvCacheType] | None = Field(
        default=None,
        description="KV cache types to offer per budget; defaults to [f16, q8_0, q4_0]",
    )
    n_ctx: int = Field(
        default=0,
        description="Pin the context length; 0 lets each candidate grow into its budget",
        ge=0,
    )
    n_ctx_min: int = Field(
        default=4096, description="Lowest context length a candidate may fall back to", ge=256
    )
    n_ctx_max: int = Field(
        default=0,
        description=(
            "Highest context a candidate may grow to; 0 uses the model's trained context length"
        ),
        ge=0,
    )
    host_budget_mib: float | None = Field(
        default=None,
        description="Host memory budget in MiB; currently free DRAM is read when omitted",
        ge=0,
    )
    margin_mib: float = Field(
        default=1024, description="Memory to leave free on each device, in MiB", ge=0
    )
    target_utilization: float = Field(
        default=0.9,
        description="Fraction of each budget a candidate should reach before it is called full",
        ge=0.0,
        le=1.0,
    )
    verify: bool = Field(
        default=True,
        description=(
            "Cross-check every candidate with llama-fit-params; skipped when the binary is "
            "absent. Costs roughly a second per candidate"
        ),
    )


class GgufEstimateResponse(BaseModel):
    """Response for a single-point GGUF memory estimate."""

    source: str
    model_path: str
    settings: dict[str, Any]
    model_info: dict[str, Any]
    memory_breakdown_mib: dict[str, Any]
    placement: dict[str, Any]
    per_layer_mib: list[dict[str, Any]] | None = None
    parsed_args: dict[str, Any] | None = Field(
        default=None,
        description="Parsed args: the settings applied, the flags skipped, and any warnings",
    )
    verification: dict[str, Any] | None = None
    notes: list[str]


class GgufSweepResponse(BaseModel):
    """Response for a GGUF feasibility sweep."""

    source: str
    model_path: str
    model_info: dict[str, Any]
    grid: dict[str, Any]
    budgets_mib: dict[str, Any]
    margin_mib: float
    rows: list[dict[str, Any]]
    truncated: bool
    parsed_args: dict[str, Any] | None = Field(
        default=None,
        description="Parsed args: the settings applied, the flags skipped, and any warnings",
    )


class GgufConfigCheckResponse(BaseModel):
    """Response for a GGUF pre-flight check: does this llama-server configuration fit."""

    fits: bool = Field(
        ..., description="Whether the configuration fits current VRAM, margin included"
    )
    verdict: str = Field(..., description="One of ok / gpu_short / host_short / unknown_budget")
    summary: str = Field(..., description="One-line conclusion, ready to show in the frontend")
    model_path: str
    resolved_settings: dict[str, Any] = Field(
        ...,
        description=(
            "The llama.cpp settings actually in effect, derived from the config plus "
            "llama_server_extra_args"
        ),
    )
    parsed_args: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parsed llama_server_extra_args: the settings applied, the flags skipped, and "
            "any warnings"
        ),
    )
    model_info: dict[str, Any]
    budgets_mib: dict[str, Any]
    memory_breakdown_mib: dict[str, Any]
    placement: dict[str, Any]
    per_layer_mib: list[dict[str, Any]] | None = None
    suggestion: dict[str, Any] | None = Field(
        default=None,
        description=(
            "A workable configuration and llama-server argument string, returned when the "
            "requested one does not fit"
        ),
    )
    verification: dict[str, Any] | None = None
    notes: list[str]


class GgufRecommendResponse(BaseModel):
    """Response for a recommended GGUF configuration."""

    source: str
    model_path: str
    model_info: dict[str, Any]
    budgets_mib: dict[str, Any]
    margin_mib: float
    recommended: dict[str, Any]
    utilization: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "How fully the budget is used: usable_budget_mib, allocated_mib, headroom_mib, "
            "utilization_pct, within_budget, and the correction applied after verification"
        ),
    )
    memory_breakdown_mib: dict[str, Any]
    placement: dict[str, Any]
    host_fits: bool
    llama_server_args: str
    parsed_args: dict[str, Any] | None = Field(
        default=None,
        description="Parsed args: the settings applied, the flags skipped, and any warnings",
    )
    verification: dict[str, Any] | None = None
    notes: list[str]


class GgufPlanResponse(BaseModel):
    """Response holding one plan per GPU budget, each with several candidate settings."""

    source: str
    model_path: str
    model_info: dict[str, Any]
    margin_mib: float
    host_budget_mib: float | None = None
    kv_cache_types: list[str]
    verified: bool = Field(
        default=False, description="Whether the figures were confirmed with llama-fit-params"
    )
    plans: list[dict[str, Any]] = Field(
        description=(
            "One entry per GPU budget: usable_budget_mib, the ranked candidates (best "
            "offload first, then longest context), and the recommended argument string"
        )
    )
    parsed_args: dict[str, Any] | None = Field(
        default=None,
        description="Parsed args: the settings applied, the flags skipped, and any warnings",
    )
    notes: list[str]


# ==================== System Resource Models ====================


class MemoryModule(BaseModel):
    """Memory module info."""

    size: str | None = None
    type: str | None = None
    speed_mhz: int | None = None
    manufacturer: str | None = None


class MemoryInfo(BaseModel):
    """Full memory info (spec and usage combined)."""

    # Spec
    total_gb: float | None = None
    type: str | None = None
    speed_mhz: int | None = None
    modules: list[MemoryModule] = Field(default_factory=list)
    # Usage
    used_gb: float | None = None  # System DRAM used
    cached_gb: float | None = None  # OS Cache / Buffers (often contains mmap models)
    other_used_gb: float | None = None  # Reserved for breakdown / compatibility
    free_gb: float | None = None
    percent: float | None = None  # System DRAM used percent
    system_used_gb: float | None = None  # Total system used

    note: str | None = None


class CPUInfo(BaseModel):
    """Combined CPU info."""

    # Spec
    model: str | None = None
    cores: int | None = None
    threads: int | None = None
    architecture: str | None = None
    max_frequency_mhz: float | None = None

    # Usage
    cpu_util_percent: float | None = None

    # Sub-resources
    dram: MemoryInfo | None = None


class GPUInfo(BaseModel):
    """GPU info (spec and usage combined)."""

    index: int
    name: str
    total_gb: float
    used_gb: float | None = None
    free_gb: float | None = None
    percent: float | None = None
    gpu_util: float | None = None  # GPU Compute Usage %
    temperature: float | None = None


class DiskDevice(BaseModel):
    """Physical disk device (lsblk)."""

    name: str
    size: str | None = None
    model: str | None = None
    type: str | None = None


class DiskMount(BaseModel):
    """Logical mount point (df)."""

    path: str
    total_gb: float | None = None
    used_gb: float | None = None
    free_gb: float | None = None
    percent: float | None = None
    fstype: str | None = None
    folder_size_gb: float | None = None
    read_speed_mbps: float | None = None
    write_speed_mbps: float | None = None
    error: str | None = None


class GPUResource(BaseModel):
    """Combined GPU resource model."""

    available: bool
    gpus: list[GPUInfo]


class DiskResource(BaseModel):
    """Combined disk resource model."""

    devices: list[DiskDevice] | None = None
    mounts: list[DiskMount]
    main: DiskMount | None = None


class SystemResourcesResponse(BaseModel):
    """System resources API response."""

    mode: str  # "spec" or "usage"
    timestamp: str
    cpu: CPUInfo
    gpu: GPUResource
    disk: DiskResource


class ModelConversionRequest(BaseModel):
    """Model conversion request."""

    model_path: str = Field(..., description="HF model path or ID")
    output_path: str | None = Field(
        default=None, description="Output GGUF file path; defaults to the model_path directory"
    )
    outtype: str = Field(
        default="f16", description="Output type, passed straight to the llama.cpp convert script"
    )


class ConversionResponse(BaseModel):
    """Conversion response."""

    job_id: str
    status: str
    message: str
