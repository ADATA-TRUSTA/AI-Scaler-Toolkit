"""
PEFT/LoRA Model Loader - 載入 PEFT 微調模型的工具函數
"""
import json
import logging
from pathlib import Path
from typing import Optional

# 嘗試導入 PEFT
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    PeftConfig = None

logger = logging.getLogger(__name__)


def is_peft_model(model_path: str) -> bool:
    """
    檢測路徑是否為 PEFT/LoRA 微調模型
    
    Args:
        model_path: 模型路徑
    
    Returns:
        True 如果是 PEFT 模型（包含 adapter_config.json）
    """
    path = Path(model_path)
    if not path.exists():
        return False
    # PEFT 模型的標誌檔案
    adapter_config = path / "adapter_config.json"
    return adapter_config.exists()


def load_peft_model(model_path: str, base_model, hf_token: Optional[str] = None, **kwargs):
    """
    載入 PEFT/LoRA 微調模型
    
    Args:
        model_path: LoRA adapter 路徑
        base_model: 已載入的基礎模型
        hf_token: HuggingFace token
    
    Returns:
        合併了 LoRA adapter 的模型
    
    Raises:
        RuntimeError: 如果 PEFT 未安裝
        Exception: 如果載入失敗
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library not available. Install with: pip install peft")
    
    logger.info(f"[PEFT] Loading adapters from: {model_path}")

    try:
        # Workaround for accelerate bug with MoE models (e.g. Qwen3.5-MoE):
        # model._no_split_modules may contain nested sets/lists, causing
        # "unhashable type: 'set'" in accelerate's get_balanced_memory().
        # Flatten it to a plain list of strings before PEFT tries to load adapters.
        _no_split = getattr(base_model, "_no_split_modules", None)
        if _no_split is not None:
            flattened = []
            for item in _no_split:
                if isinstance(item, (set, list, tuple)):
                    flattened.extend(str(x) for x in item)
                else:
                    flattened.append(str(item))
            base_model._no_split_modules = flattened

        # 載入 LoRA adapters 並合併到 base model
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            token=hf_token,
            max_memory=kwargs.get("max_memory", None),
        )
        logger.info("[PEFT] Adapters loaded successfully")
        return model
    except Exception as e:
        logger.error(f"[PEFT] Failed to load PEFT model: {e}")
        raise

def read_base_model_name(model_path: str) -> str:
    """
    從 adapter_config.json 讀取 base model 名稱
    
    Args:
        model_path: PEFT 模型路徑
    
    Returns:
        Base model 名稱或路徑
    
    Raises:
        FileNotFoundError: 如果 adapter_config.json 不存在
        ValueError: 如果無法找到 base_model_name_or_path
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"
    
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_path}")
    
    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            
            if not base_model_name:
                raise ValueError("base_model_name_or_path not found in adapter_config.json")
            
            logger.info(f"[PEFT] Base model from config: {base_model_name}")
            return base_model_name
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in adapter_config.json: {e}")
