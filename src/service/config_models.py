"""Configuration models for inference and training."""

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing import Optional, List, Dict, Union, Any
from enum import Enum


class InferenceEngine(str, Enum):
    """推理引擎"""

    TRANSFORMERS = "transformers"
    LLAMA_SERVER = "llama_server"
    VLLM = "vllm"


class QuantizationType(str, Enum):
    """量化類型"""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # QLoRA 使用的 4-bit normal float
    FP4 = "fp4"  # 4-bit float


class TrainingMethod(str, Enum):
    """訓練方法"""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class InferenceSharedFields(BaseModel):
    """推理設定與狀態共用欄位。"""

    model_name: Optional[str] = Field(default=None, description="模型名稱或路徑")
    model_path: Optional[str] = Field(
        default=None,
        description="本地微調後模型的路徑 = output_dir；若提供，優先於 model_name 用於 from_pretrained",
    )
    engine: InferenceEngine = Field(
        default=InferenceEngine.TRANSFORMERS,
        description="推理引擎: transformers (預設), llama_server, vllm",
    )
    quantization: Optional[Union[QuantizationType, str]] = Field(
        default=None,
        description="量化類型: none, int8, int4, nf4, fp4",
    )
    device_map: Optional[Union[str, Dict]] = Field(
        default="auto",
        description="設備映射策略，例如: 'auto', 'cpu', 'cuda:0' 或 {'': 0, 'cpu': 'cpu'}, 'balanced_low_0'",
    )
    model_total_memory: Optional[str] = Field(
        default=None, description="模型總記憶體需求，例如 '15GB'"
    )
    max_memory: Optional[Dict[Union[int, str], str]] = Field(
        default=None, description="最大記憶體配置，例如: {0: '20GB', 'cpu': '50GB'}"
    )
    offload_folder: Optional[str] = Field(
        default=None, description="Offload 資料夾路徑，用於將模型權重卸載到磁碟"
    )

    # GGUF / llama-server 共用快照欄位
    n_gpu_layers: Optional[int] = Field(
        default=None, description="[llama_server] GPU 層數，-1 表示全部"
    )
    n_ctx: Optional[int] = Field(default=None, description="[llama_server] 上下文長度")
    n_batch: Optional[int] = Field(default=None, description="[llama_server] 批次處理大小")
    llama_server_extra_args: Optional[List[str]] = Field(
        default=None,
        description="[llama_server] 啟動參數附加列表，例如 ['--mlock', '--no-mmap']",
    )

    # vLLM 共用欄位
    vllm_gpu_memory_utilization: Optional[float] = Field(
        default=None,
        description="[vLLM] --gpu-memory-utilization",
    )
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        description="[vLLM] --max-model-len，未提供時回退使用 n_ctx",
    )
    vllm_dtype: Optional[str] = Field(default=None, description="[vLLM] --dtype")
    vllm_quantization: Optional[str] = Field(
        default=None,
        description="[vLLM] --quantization，例如 awq/gptq/fp8",
    )
    vllm_enforce_eager: Optional[bool] = Field(
        default=None,
        description="[vLLM] 是否啟用 --enforce-eager",
    )
    vllm_kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="[vLLM] --kv-cache-dtype，例如 auto/fp8_e5m2/fp8_e4m3",
    )
    vllm_cpu_offload_gb: Optional[float] = Field(
        default=None,
        description="[vLLM] --cpu-offload-gb",
    )
    vllm_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="[vLLM] --tensor-parallel-size",
    )
    vllm_max_num_seqs: Optional[int] = Field(
        default=None,
        description="[vLLM] --max-num-seqs",
    )
    vllm_max_num_batched_tokens: Optional[int] = Field(
        default=None,
        description="[vLLM] --max-num-batched-tokens",
    )
    vllm_mm_image_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt image 上限",
    )
    vllm_mm_audio_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt audio 上限",
    )
    vllm_mm_video_limit: Optional[int] = Field(
        default=None,
        description="[vLLM] --limit-mm-per-prompt video 上限",
    )
    vllm_kv_offloading_size: Optional[float] = Field(
        default=None,
        description="[vLLM] --kv-offloading-size (單位 GB，多卡 TP 時這個數字是「所有 TP rank 合計」，不是每卡的量)",
    )
    vllm_hf_overrides: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="[vLLM] --hf-overrides",
    )
    vllm_chat_template: Optional[str] = Field(
        default=None,
        description="[vLLM] --chat-template",
    )


class InferenceConfig(InferenceSharedFields):
    """推理配置 - 直接使用 Hugging Face Transformers 格式

    範例:
    {
        "model_name": "Qwen/Qwen3-4B",
        "quantization": "none",
        "device_map": "auto",
        "model_total_memory": "15GB",
        "max_memory": {"0": "5GB", "cpu": "5GB"},
        "offload_folder": "./offload"
    }
    或
    {
        "model_name": "Qwen/Qwen3-8B",
        "quantization": "none",
        "model_total_memory": "20GB",
        "device_map": "cpu"
    }
    """

    model_name: str = Field(..., description="模型名稱或路徑")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="量化類型: none, int8, int4, nf4, fp4",
    )
    torch_dtype: str = Field(default="auto", description="Torch 資料類型")
    trust_remote_code: bool = Field(default=True, description="信任遠端代碼")
    use_cache: bool = Field(default=True, description="使用 KV cache")

    # GGUF / llama-server 共用配置
    n_gpu_layers: int = Field(
        default=-1, description="[llama_server] GPU 層數，-1 表示全部"
    )
    n_ctx: int = Field(default=4096, description="[llama_server] 上下文長度")
    n_batch: int = Field(default=512, description="[llama_server] 批次處理大小")

    # llama-server 專用配置（OpenAI-compatible API）
    llama_server_url: Optional[str] = Field(
        default=None,
        description="[llama_server] 伺服器基礎 URL，例如 http://127.0.0.1:8080",
    )
    llama_server_api_key: Optional[str] = Field(
        default=None, description="[llama_server] API 金鑰（若服務端需要授權）"
    )
    llama_server_model: Optional[str] = Field(
        default=None,
        description="[llama_server] 請求時使用的模型名稱；未提供時預設使用 model_name",
    )
    llama_server_timeout: int = Field(
        default=300,
        description="[llama_server] 請求逾時秒數",
        ge=10,
    )
    llama_server_auto_start: bool = Field(
        default=True,
        description="[llama_server] 是否由引擎在 load 時自動啟動 llama-server 子程序",
    )
    llama_server_binary: Optional[str] = Field(
        default=None,
        description="[llama_server] llama-server 可執行檔路徑（未提供則使用環境預設）",
    )
    llama_server_host: str = Field(
        default="127.0.0.1", description="[llama_server] 啟動子程序時綁定的 host"
    )
    llama_server_port: int = Field(
        default=5001,
        description="[llama_server] 啟動子程序時使用的 port",
        ge=1,
        le=65535,
    )
    llama_server_np: int = Field(
        default=1,
        description="[llama_server] 平行生成槽位（對應 llama-server -np）",
        ge=1,
    )
    llama_server_health_timeout: int = Field(
        default=300,
        description="[llama_server] load 階段等待服務啟動成功的秒數",
        ge=5,
    )
    llama_server_mmproj: Optional[str] = Field(
        default=None,
        description="[llama_server] 多模態 projector (.gguf) 路徑；提供後會自動加上 --mmproj",
    )

    # vLLM OpenAI-compatible server 專用配置
    vllm_gpu_memory_utilization: float = Field(
        default=0.8,
        description="[vLLM] --gpu-memory-utilization",
        ge=0.05,
        le=0.99,
    )
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        description="[vLLM] --max-model-len，未提供時回退使用 n_ctx",
        ge=1,
    )
    vllm_dtype: str = Field(default="auto", description="[vLLM] --dtype")
    vllm_quantization: Optional[str] = Field(
        default=None,
        description="[vLLM] --quantization，例如 awq/gptq/fp8",
    )
    vllm_enforce_eager: bool = Field(
        default=False,
        description="[vLLM] 是否啟用 --enforce-eager",
    )
    vllm_kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="[vLLM] --kv-cache-dtype，例如 auto/fp8_e5m2/fp8_e4m3",
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
            "[vLLM] --kv-offloading-size (單位 GB，多卡 TP 時這個數字是"
            "「所有 TP rank 合計」，不是每卡的量)"
        ),
        validation_alias=AliasChoices(
            "vllm_kv_offloading_size", "vllm_swap_space"
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
    vllm_max_num_batched_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="[vLLM] --max-num-batched-tokens",
    )
    # vLLM 多模態 (Vision / Audio / Video) 專用配置
    # 適用於 Gemma 4、Gemma 3n、Qwen-VL、LLaVA 等 VLM 模型
    vllm_mm_image_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] --limit-mm-per-prompt 中的 image 數量上限；"
            "載入多模態模型（如 gemma-4-E2B-it）若要處理圖片需設定此值，例如 1"
        ),
    )
    vllm_mm_audio_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[vLLM] --limit-mm-per-prompt 中的 audio 數量上限；"
            "Gemma 4 E2B/E4B 等支援音訊的模型才需設定"
        ),
    )
    vllm_mm_video_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="[vLLM] --limit-mm-per-prompt 中的 video 數量上限",
    )
    vllm_hf_overrides: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "[vLLM] --hf-overrides，用於強制覆寫 HuggingFace config.json 的欄位；"
            "例如當 gemma-4-E2B-it 被誤識為純文字架構時，可設定 "
            '{"architectures":["Gemma4ForConditionalGeneration"]} 強制啟用多模態版本。'
            "可傳入 dict 或 JSON 字串"
        ),
    )
    vllm_chat_template: Optional[str] = Field(
        default=None,
        description=(
            "[vLLM] --chat-template，指定自訂 chat template 檔案路徑（.jinja）；"
            "當 tokenizer_config.json 缺少 chat_template 欄位時（例如某些 base 版本或量化版本）"
            "需提供此參數以支援 /v1/chat/completions"
        ),
    )

    @field_validator("vllm_chat_template", mode="before")
    @classmethod
    def _normalize_vllm_chat_template(cls, v):
        """將空字串 chat_template 統一正規化為 None。"""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @field_validator("vllm_hf_overrides", mode="before")
    @classmethod
    def _normalize_vllm_hf_overrides(cls, v):
        """接受 dict/list 或 JSON 字串；空字串視為未設定。"""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        if isinstance(v, (dict, list)):
            return v
        raise ValueError("vllm_hf_overrides 必須是 dict、list、JSON 字串或 None")


class ChatRequest(BaseModel):
    """聊天請求"""

    message: str = Field(..., description="用戶消息")
    max_new_tokens: int = Field(default=512, description="最大生成 token 數")
    temperature: float = Field(default=0.7, description="溫度參數", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p 採樣", ge=0.0, le=1.0)
    top_k: int = Field(default=50, description="Top-k 採樣", ge=0)
    repetition_penalty: float = Field(default=1.1, description="重複懲罰", ge=1.0)
    stream: bool = Field(default=True, description="是否使用串流")
    system_prompt: Optional[str] = Field(default=None, description="系統提示詞")

    # 超時控制
    total_timeout: int = Field(
        default=300, description="生成總超時時間（秒），超過此時間將停止生成", ge=10
    )

    # Chat template 控制
    enable_thinking: Optional[bool] = Field(
        default=True,
        description="啟用思考模式（適用於支援的模型如 DeepSeek、QwQ）。None=使用模型預設值",
    )

    # RAG 控制
    use_rag: bool = Field(default=False, description="是否啟用 RAG 檢索並注入上下文")
    rag_top_k: int = Field(default=3, description="RAG 檢索返回的文檔數量", ge=1, le=50)
    rag_query: Optional[str] = Field(
        default=None, description="覆蓋用戶消息作為 RAG 查詢"
    )
    rag_include_sources: bool = Field(
        default=True, description="是否在提示中包含來源資訊"
    )

    # 混合式會話管理
    session_id: Optional[str] = Field(
        default=None, description="會話 ID，若提供則可在後端維持歷史"
    )
    reset_history: bool = Field(
        default=False, description="是否在本次請求前重置會話歷史（需配合 session_id）"
    )
    # 可選：前端直接帶入的歷史（若提供則優先生效）
    # 結構與 OpenAI 類似：[{role: user|assistant|system, content: str}]
    history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="可選的對話歷史，若提供將覆蓋後端保存"
    )
    images: Optional[List[str]] = Field(
        default=None,
        description=(
            "可選圖片輸入（僅支援多模態模型時生效）。"
            "每個元素可為本地檔案路徑、http(s) URL，或 data:image/...;base64,..."
        ),
        max_length=8,
    )
    request_id: Optional[str] = Field(
        default=None,
        description="可選請求 ID。提供後可用於 /inference/stop_generation 精準停止單一請求",
    )


class OpenAIChatMessage(BaseModel):
    """OpenAI 相容消息格式"""

    role: str = Field(..., description="消息角色，例如 system/user/assistant")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default="",
        description="消息內容；可為字串，或 OpenAI 多模態 content parts 列表",
    )
    name: Optional[str] = Field(
        default=None, description="可選名稱欄位，常用於 tool/function message"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="assistant 訊息中的工具呼叫列表"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="tool 訊息對應的 tool_call_id"
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI 相容 /v1/chat/completions 請求"""

    model_config = ConfigDict(populate_by_name=True)

    model: Optional[str] = Field(default=None, description="模型名稱（相容欄位）")
    messages: List[OpenAIChatMessage] = Field(
        ..., min_length=1, description="多輪對話消息"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    total_timeout: Optional[int] = Field(
        default=300, ge=10, description="生成總超時時間（秒）"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="最大生成 token 數",
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="OpenAI 相容欄位；若未明確提供 repetition_penalty，將近似映射使用",
    )
    stream: bool = Field(default=False, description="是否以 SSE 串流回傳")
    stream_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenAI 相容 stream_options，例如 {'include_usage': true}",
    )
    user: Optional[str] = Field(
        default=None, description="終端使用者識別（可映射為 session_id）"
    )

    # 以下為後端擴展欄位，保持與 /inference/chat 對齊
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    session_id: Optional[str] = Field(default=None)
    reset_history: bool = Field(default=False)
    enable_thinking: Optional[bool] = Field(default=True)
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="相容 extra_body.chat_template_kwargs，例如 {'enable_thinking': false}",
    )
    use_rag: bool = Field(default=False)
    rag_top_k: int = Field(default=3, ge=1, le=50)
    rag_query: Optional[str] = Field(default=None)
    rag_include_sources: bool = Field(default=True)
    request_id: Optional[str] = Field(default=None)
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="OpenAI tool 定義列表"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="OpenAI tool_choice 設定"
    )


class StopGenerationRequest(BaseModel):
    """停止生成請求"""

    request_id: Optional[str] = Field(
        default=None, description="可選 worker 請求 ID。若提供則精準停止該次生成"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="可選對話 session ID。若未提供 request_id，後端會以 session_id 查找當前活躍 worker 請求並停止",
    )


class CleanupGenerationMemoryRequest(BaseModel):
    """清理生成記憶體請求"""

    slot: Optional[int] = Field(
        default=None,
        ge=0,
        description="可選 slot id。提供時僅清理指定 slot 的 cache；未提供則清理所有可見 slots",
    )


class RagAddDocument(BaseModel):
    """新增或更新 RAG 文檔的請求"""

    doc_id: Optional[str] = Field(
        default=None, description="文檔 ID；若不提供將自動生成"
    )
    content: str = Field(..., description="純文字內容")


class TrainingConfig(BaseModel):
    """訓練配置

    支援三種訓練方法：
    - full: 全參數微調
    - lora: LoRA 參數高效微調
    - qlora: QLoRA 量化 + LoRA 微調
    """

    model_name: str = Field(
        ..., description="模型名稱標籤（從 models registry config 中的 label）"
    )
    method: TrainingMethod = Field(..., description="訓練方法：lora / qlora / full")
    dataset_path: str = Field(
        ..., description="訓練數據集文件路徑，必須是 JSON 或 JSONL 格式"
    )
    output_dir: str = Field(..., description="微調後的模型文件輸出資料夾路徑")
    offload_folder: Optional[str] = Field(
        default="./deepspeed_offload",
        description="Offload 資料夾路徑，會覆蓋 DeepSpeed Json 配置",
    )

    # LoRA/QLoRA specific
    lora_r: int = Field(
        default=8,
        description="【僅 method = LoRA/QLoRA】LoRA rank（秩），控制可訓練參數量。值越大參數越多，訓練效果可能越好但耗時更長",
    )
    lora_alpha: int = Field(
        default=16,
        description="【僅 method = LoRA/QLoRA】LoRA alpha 縮放係數，通常設為 lora_r 的 1-2 倍",
    )
    lora_dropout: float = Field(
        default=0.05,
        description="【僅 method = LoRA/QLoRA】LoRA dropout 比率，用於防止過擬合",
    )
    lora_target_modules: Optional[List[str]] = Field(
        default=None,
        description="【僅 method = LoRA/QLoRA】LoRA 目標模組列表，指定要應用 LoRA 的層。null 則使用預設值（如 q_proj, k_proj, v_proj, o_proj）",
    )

    # Training hyperparameters
    num_train_epochs: int = Field(default=3, description="訓練幾輪（epochs）")
    per_device_train_batch_size: int = Field(
        default=1, description="每次訓練取幾筆資料（batch size）"
    )
    gradient_accumulation_steps: int = Field(
        default=8, description="累積幾次梯度後才更新一次模型參數"
    )
    learning_rate: float = Field(
        default=2e-4, description="學習率，控制每次參數更新的幅度"
    )
    warmup_steps: int = Field(default=100, description="前幾步使用較小學習率進行預熱")
    logging_steps: int = Field(
        default=10, description="每幾步回傳一次訓練進度（loss、step 等）"
    )
    save_steps: int = Field(default=500, description="每幾步儲存一次 checkpoint")
    save_total_limit: Optional[int] = Field(
        default=2, description="最多儲存幾個 checkpoint，超過會自動刪除最舊的"
    )
    max_seq_length: int = Field(
        default=2048, description="訓練時的最大 token 長度，超過會被截斷"
    )

    # Dataset field configuration - 三種訓練模式三擇一
    text_field: Optional[str] = Field(
        default="text",
        description="【訓練模式一】單欄位模式：用前面對話預測後面對話。指定 dataset 中的欄位名稱，例如 'text'。與其他模式三擇一",
    )
    prompt_field: Optional[str] = Field(
        default=None,
        description="【訓練模式二】雙欄位模式：區分提問與回答。指定 dataset 中 prompt 欄位名稱，例如 'prompt'。需搭配 completion_field 使用",
    )
    completion_field: Optional[str] = Field(
        default=None,
        description="【訓練模式二】雙欄位模式：區分提問與回答。指定 dataset 中 completion 欄位名稱，例如 'completion'。需搭配 prompt_field 使用",
    )
    messages_field: Optional[str] = Field(
        default=None,
        description="【訓練模式三】OpenAI chat format 模式：dataset 中含有 messages 欄位（list of {role, content}）。TRL 會自動套用模型的 chat template，並只對 assistant tokens 計算 loss。留空時若 dataset 有 messages 欄位會自動偵測",
    )
    save_tokenizer: bool = Field(
        default=True, description="訓練完成後是否保存 tokenizer 到輸出目錄"
    )

    # DeepSpeed settings
    use_deepspeed: bool = Field(
        default=False,
        description="是否使用 DeepSpeed offload 訓練（啟用後需搭配 deepspeed_config 或 deepspeed_profile）",
    )
    deepspeed_config: Optional[str] = Field(
        default=None,
        description="【方式一】DeepSpeed 詳細配置文件的完整路徑，例如 './my_deepspeed.json'。與 deepspeed_profile 二擇一",
    )
    deepspeed_profile: Optional[str] = Field(
        default=None,
        description="【方式二】DeepSpeed 配置 profile 名稱，會自動從 service/configs/deepspeed/<profile>.json 載入。與 deepspeed_config 二擇一",
    )
    # SFTTrainer specific
    use_sft_trainer: bool = Field(
        default=True,
        description="是否使用 TRL 的 SFTTrainer 進行訓練（推薦用於指令微調）",
    )
    packing: bool = Field(
        default=False,
        description="【僅 use_sft_trainer=True】是否啟用 packing（將多個短序列打包成一個長序列以提高訓練效率）",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="是否啟用 gradient checkpointing（用計算時間換記憶體空間，對大模型訓練至關重要）",
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
    def _normalize_optional_str(cls, v):
        """Normalize optional string fields.

        前端/其他執行檔常用空字串 "" 表示未填；為避免判斷歧義，統一轉為 None。
        同時會對字串做 strip。
        """
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v

    @model_validator(mode="after")
    def _validate_dataset_fields(self):
        """Validate dataset field configuration.

        - prompt_field 與 completion_field 必須同時提供或同時不提供。
        - messages_field 不可與 prompt_field/completion_field 同時設定。
        - 三種模式互斥（但若都未設定，dataset_loader 會自動偵測）。
        """
        has_prompt = bool(self.prompt_field)
        has_completion = bool(self.completion_field)
        has_messages = bool(self.messages_field)

        if has_prompt != has_completion:
            raise ValueError(
                "prompt_field 與 completion_field 必須同時設定或同時為空/None。"
            )
        if has_messages and (has_prompt or has_completion):
            raise ValueError(
                "messages_field 不可與 prompt_field/completion_field 同時設定，三種訓練模式互斥。"
            )
        return self


class DeviceAllocation(BaseModel):
    """設備分配統計資訊"""

    summary: Optional[str] = Field(
        default=None, description="各設備的模組數統計摘要，例如: 'cuda:0:30, cpu:10'"
    )
    total_modules: Optional[int] = Field(default=None, description="模型的總模組數")
    layer_lines: Optional[List[str]] = Field(
        default=None, description="層級分配，例如: ['model.layers.0 -> cuda:0', ...]"
    )


class ModelStatus(InferenceSharedFields):
    """模型狀態"""

    loaded: bool = Field(default=False, description="是否已加載")
    is_loading: bool = Field(default=False, description="是否正在載入")
    loading_error: Optional[str] = Field(default=None, description="載入錯誤訊息")
    quantization: Optional[str] = Field(default=None, description="量化類型")
    device: Optional[str] = Field(default=None, description="設備")
    memory_usage: Optional[Dict] = Field(default=None, description="記憶體使用情況")
    device_allocation: Optional[DeviceAllocation] = Field(
        default=None, description="實際設備分配統計資訊（僅在模型載入後可用）"
    )
    prefill_strategy: Optional[str] = Field(
        default=None, description="[llama_server] 預填策略，例如 slot 或 cache_prompt"
    )
    llama_capabilities: Optional[List[str]] = Field(
        default=None, description="[llama_server] /v1/models 回報的 capabilities"
    )
    slot_restore_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="[llama_server] slot restore 結果摘要"
    )


class TrainingLog(BaseModel):
    """訓練過程日誌"""

    timestamp: float = Field(..., description="時間戳")
    step: int = Field(..., description="Step")
    loss: float = Field(..., description="Loss")
    learning_rate: Optional[float] = Field(None, description="Learning Rate")
    epoch: Optional[float] = Field(None, description="Epoch")
    accuracy: Optional[float] = Field(None, description="Accuracy or Eval Accuracy")


class GPULog(BaseModel):
    """個別 GPU 日誌"""

    index: int = Field(..., description="GPU Index")
    name: str = Field(..., description="GPU Name")
    gpu_util_percent: float = Field(..., description="GPU Util %")
    gpu_memory_used_gb: float = Field(..., description="Used Memory (GB)")
    gpu_memory_total_gb: float = Field(..., description="Total Memory (GB)")
    temperature: Optional[float] = Field(None, description="Temperature (C)")


class ResourceLog(BaseModel):
    """系統資源日誌（欄位結構與 /system/resources 一致）"""

    timestamp: float = Field(..., description="時間戳")
    cpu: Optional["CPUInfo"] = Field(default=None, description="CPU/RAM 資源資訊")
    gpu: Optional["GPUResource"] = Field(default=None, description="GPU 資源資訊")
    disk: Optional["DiskResource"] = Field(default=None, description="Disk 資源資訊")


class TrainingHistoryResponse(BaseModel):
    """訓練歷史紀錄回應"""

    session_id: str
    logs: List[TrainingLog]


class SystemResourceHistoryResponse(BaseModel):
    """系統資源歷史紀錄回應"""

    session_id: str
    resources: List[ResourceLog]


class TrainingStatus(BaseModel):
    """訓練狀態"""

    is_training: bool = Field(default=False, description="是否正在訓練")
    progress: float = Field(default=0.0, description="訓練進度 (0-1)")
    current_step: int = Field(default=0, description="當前步數")
    total_steps: int = Field(default=0, description="總步數")
    loss: Optional[float] = Field(default=None, description="當前損失")
    current_epoch: Optional[float] = Field(default=0.0, description="當前 epoch")
    total_epochs: Optional[int] = Field(default=None, description="總 Epoch 數")
    status: Optional[str] = Field(default=None, description="狀態消息")
    session_id: Optional[str] = Field(default=None, description="當前訓練的 Session ID")
    error: Optional[str] = Field(default=None, description="錯誤訊息（簡短描述）")
    config: Optional[TrainingConfig] = Field(
        default=None, description="當前或最後一次的訓練配置"
    )


class MemoryEstimateRequest(BaseModel):
    """記憶體估計請求"""

    model_name: str = Field(..., description="模型名稱或路徑")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="量化類型: none, int8, int4, nf4, fp4",
    )
    batch_size: int = Field(default=1, description="批次大小", ge=1, le=32)
    sequence_length: int = Field(default=2048, description="序列長度", ge=512, le=32768)
    include_activations: bool = Field(
        default=True, description="是否包含激活值記憶體估計"
    )


class MemoryEstimateResponse(BaseModel):
    """記憶體估計響應"""

    model_name: str
    model_size_billions: float
    quantization: str
    memory_breakdown_gb: Dict[str, float]
    overhead_details_gb: Optional[Dict[str, float]] = None
    recommendations: Dict[str, float]
    offload_strategies: List[Dict]
    notes: List[str]


# ==================== System Resource Models ====================


class MemoryModule(BaseModel):
    """記憶體模組資訊"""

    size: Optional[str] = None
    type: Optional[str] = None
    speed_mhz: Optional[int] = None
    manufacturer: Optional[str] = None


class MemoryInfo(BaseModel):
    """記憶體完整資訊 (整合 Spec 與 Usage)"""

    # Spec
    total_gb: Optional[float] = None
    type: Optional[str] = None
    speed_mhz: Optional[int] = None
    modules: List[MemoryModule] = Field(default_factory=list)
    # Usage
    used_gb: Optional[float] = None  # System DRAM used
    cached_gb: Optional[float] = None  # OS Cache / Buffers (often contains mmap models)
    other_used_gb: Optional[float] = None  # Reserved for breakdown / compatibility
    free_gb: Optional[float] = None
    percent: Optional[float] = None  # System DRAM used percent
    system_used_gb: Optional[float] = None  # Total system used

    note: Optional[str] = None


class CPUInfo(BaseModel):
    """CPU 整合資訊"""

    # Spec
    model: Optional[str] = None
    cores: Optional[int] = None
    threads: Optional[int] = None
    architecture: Optional[str] = None
    max_frequency_mhz: Optional[float] = None

    # Usage
    cpu_util_percent: Optional[float] = None

    # Sub-resources
    dram: Optional[MemoryInfo] = None


class GPUInfo(BaseModel):
    """GPU 資訊 (整合 Spec 與 Usage)"""

    index: int
    name: str
    total_gb: float
    used_gb: Optional[float] = None
    free_gb: Optional[float] = None
    percent: Optional[float] = None
    gpu_util: Optional[float] = None  # GPU Compute Usage %
    temperature: Optional[float] = None


class DiskDevice(BaseModel):
    """實體磁碟裝置 (lsblk)"""

    name: str
    size: Optional[str] = None
    model: Optional[str] = None
    type: Optional[str] = None


class DiskMount(BaseModel):
    """邏輯掛載點 (df)"""

    path: str
    total_gb: Optional[float] = None
    used_gb: Optional[float] = None
    free_gb: Optional[float] = None
    percent: Optional[float] = None
    fstype: Optional[str] = None
    folder_size_gb: Optional[float] = None
    read_speed_mbps: Optional[float] = None
    write_speed_mbps: Optional[float] = None
    error: Optional[str] = None


class GPUResource(BaseModel):
    """GPU 資源整合模型"""

    available: bool
    gpus: List[GPUInfo]


class DiskResource(BaseModel):
    """磁碟資源整合模型"""

    devices: Optional[List[DiskDevice]] = None
    mounts: List[DiskMount]
    main: Optional[DiskMount] = None


class SystemResourcesResponse(BaseModel):
    """系統資源 API 回應"""

    mode: str  # "spec" or "usage"
    timestamp: str
    cpu: CPUInfo
    gpu: GPUResource
    disk: DiskResource


class ModelConversionRequest(BaseModel):
    """模型轉換請求"""

    model_path: str = Field(..., description="HF模型路徑或ID")
    output_path: Optional[str] = Field(
        default=None, description="輸出GGUF檔案路徑，預設與model_path同目錄"
    )
    outtype: str = Field(
        default="f16", description="輸出類型，直接傳給 llama.cpp 轉換腳本"
    )


class ConversionResponse(BaseModel):
    """轉換回應"""

    job_id: str
    status: str
    message: str
