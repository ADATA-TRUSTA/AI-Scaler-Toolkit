"""
PEFT/LoRA Model Loader - helper functions for loading PEFT fine-tuned models.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Try to import PEFT
try:
    from peft import PeftConfig, PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    PeftConfig = None

if TYPE_CHECKING:
    from peft import PeftModel as PeftModelType
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def is_peft_model(model_path: str) -> bool:
    """
    Check whether a path holds a PEFT/LoRA fine-tuned model.

    Args:
        model_path: Model path

    Returns:
        True if it is a PEFT model (contains adapter_config.json)
    """
    path = Path(model_path)
    if not path.exists():
        return False
    # Marker file of a PEFT model
    adapter_config = path / "adapter_config.json"
    return adapter_config.exists()


def load_peft_model(
    model_path: str,
    base_model: "PreTrainedModel",
    hf_token: str | None = None,
    **kwargs: Any,  # noqa: ANN401 - forwarded verbatim to PeftModel.from_pretrained
) -> "PeftModelType":
    """
    Load a PEFT/LoRA fine-tuned model.

    Args:
        model_path: LoRA adapter path
        base_model: Already-loaded base model
        hf_token: HuggingFace token

    Returns:
        Model with the LoRA adapter attached

    Raises:
        RuntimeError: If PEFT is not installed
        Exception: If loading fails
    """
    # A None PeftModel means PEFT is not installed; check it too so the type narrows
    if not PEFT_AVAILABLE or PeftModel is None:
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

        # Load the LoRA adapters onto the base model
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            token=hf_token,
            max_memory=kwargs.get("max_memory", None),
        )
        logger.info("[PEFT] Adapters loaded successfully")
        return model
    except Exception as e:
        logger.exception(f"[PEFT] Failed to load PEFT model: {e}")
        raise


def read_base_model_name(model_path: str) -> str:
    """
    Read the base model name from adapter_config.json.

    Args:
        model_path: PEFT model path

    Returns:
        Base model name or path

    Raises:
        FileNotFoundError: If adapter_config.json does not exist
        ValueError: If base_model_name_or_path cannot be found
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_path}")

    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

            if not base_model_name:
                raise ValueError("base_model_name_or_path not found in adapter_config.json")

            logger.info(f"[PEFT] Base model from config: {base_model_name}")
            return base_model_name

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in adapter_config.json: {e}") from e
