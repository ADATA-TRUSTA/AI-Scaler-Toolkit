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


def _restore_device_placement_skip_keys(
    model: "PeftModelType", skip_keys: str | list[str] | None
) -> int:
    """
    Re-apply the base model's ``_skip_keys_device_placement`` to accelerate's hooks.

    transformers dispatches an offloaded model with
    ``skip_keys=model._skip_keys_device_placement``; PEFT re-dispatches the wrapped
    model and does not forward that argument, so every ``AlignDevicesHook`` ends up
    with ``skip_keys=None``. accelerate then routes those kwargs through
    ``send_to_device``, which *rebuilds* a dict instead of passing the same object --
    so a dict the layers mutate to talk to each other is silently copied per layer.

    Gemma 4 shares KV across layers through exactly such a dict: layer 22 writes
    ``shared_kv_states[22]`` into its own copy and layer 24 raises ``KeyError: 22``.
    The base model alone works because transformers' dispatch got it right; only
    adding the adapter breaks it. A single-device load carries no hooks, so this is
    a no-op there.

    Returns the number of hooks patched.
    """
    if not skip_keys:
        return 0
    try:
        from accelerate.hooks import AlignDevicesHook, SequentialHook
    except ImportError:  # pragma: no cover - accelerate is a hard dependency
        return 0

    patched = 0
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        candidates = hook.hooks if isinstance(hook, SequentialHook) else [hook]
        for candidate in candidates:
            # Leave an explicitly-set skip list alone; only repair the wiped one.
            if isinstance(candidate, AlignDevicesHook) and not candidate.skip_keys:
                candidate.skip_keys = skip_keys
                patched += 1
    return patched


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

        # Read this before wrapping: PEFT's re-dispatch drops it, and the wrapper
        # does not expose the base model's class attribute reliably.
        skip_keys = getattr(base_model, "_skip_keys_device_placement", None)

        # Load the LoRA adapters onto the base model
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            token=hf_token,
            max_memory=kwargs.get("max_memory", None),
        )
        patched = _restore_device_placement_skip_keys(model, skip_keys)
        if patched:
            logger.info(
                f"[PEFT] Restored skip_keys={skip_keys} on {patched} offload hooks "
                "that PEFT's re-dispatch cleared"
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
