"""
QLoRA combinations that cannot work must be refused, not silently mishandled.

Two of them, both verified against the installed stack:

1. **QLoRA + ZeRO-3.** `from_pretrained` skips zero.Init for a quantized load
   (`modeling_utils.py:1481`: `needs_zero3_init = is_deepspeed_zero3_enabled()
   and not _is_quantized and ...`), so none of the requested parameter offload
   happens; bitsandbytes then rewrites our `device_map=None` to
   `{"": cuda:N}` (`quantizer_bnb_4bit.py:102-105`), putting the whole 4-bit
   model on one GPU. The caller asked for offload and gets an OOM that looks
   like the profile was merely too ambitious.

2. **QLoRA + `multimodal_scope=text_and_bridge`.** Scope B trains the bridge in
   full via peft `modules_to_save`, but under QLoRA the bridge is quantized too,
   and peft calls `requires_grad_(True)` on a uint8 tensor. It fails before
   training, but the message is about dtypes rather than the combination.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.config_models import MultimodalScope, TrainingConfig, TrainingMethod  # noqa: E402
from training_fixtures import training_config_kwargs  # noqa: E402

pytestmark = pytest.mark.finetune


def _config(**overrides) -> dict:
    """Minimal valid kwargs for a qlora run; shared in tests/training_fixtures.py."""
    return training_config_kwargs("qlora", **overrides)


# ------------------------------------------------------------- scope B


def test_qlora_with_a_trainable_bridge_is_rejected():
    with pytest.raises(ValueError, match="cannot be combined with"):
        TrainingConfig(**_config(multimodal_scope=MultimodalScope.TEXT_AND_BRIDGE))


def test_the_message_names_both_ways_out():
    """A refusal is only useful if it says what to do instead."""
    with pytest.raises(ValueError) as excinfo:
        TrainingConfig(**_config(multimodal_scope=MultimodalScope.TEXT_AND_BRIDGE))
    message = str(excinfo.value)
    assert "method='lora'" in message
    assert MultimodalScope.TEXT_ONLY in message


def test_qlora_with_the_default_scope_is_fine():
    cfg = TrainingConfig(**_config())
    assert cfg.multimodal_scope is MultimodalScope.TEXT_ONLY


def test_lora_can_still_train_the_bridge():
    """The guard must be about quantization, not about scope B itself."""
    cfg = TrainingConfig(
        **_config(method="lora", multimodal_scope=MultimodalScope.TEXT_AND_BRIDGE)
    )
    assert cfg.multimodal_scope is MultimodalScope.TEXT_AND_BRIDGE


def test_full_finetune_can_still_train_the_bridge():
    cfg = TrainingConfig(
        **_config(method="full", multimodal_scope=MultimodalScope.TEXT_AND_BRIDGE)
    )
    assert cfg.method is TrainingMethod.FULL


# ------------------------------------------------------------- ZeRO-3


def test_qlora_under_zero3_is_refused_before_loading_anything(monkeypatch):
    """
    Checked at the real runtime condition rather than from the config, because
    `use_deepspeed` alone does not tell you whether ZeRO-3 ended up active.
    """
    import transformers.integrations.deepspeed as ds

    from service.training.core.model_loader import ModelLoader

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: True)
    loader = ModelLoader(
        TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="zero3_offload_cpu_cpu")),
        None,
    )

    with pytest.raises(ValueError, match="cannot be combined with DeepSpeed ZeRO-3"):
        loader._prepare_qlora_model()


def test_the_zero3_refusal_offers_both_alternatives(monkeypatch):
    import transformers.integrations.deepspeed as ds

    from service.training.core.model_loader import ModelLoader

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: True)
    loader = ModelLoader(
        TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="zero3_offload_cpu_cpu")),
        None,
    )

    with pytest.raises(ValueError) as excinfo:
        loader._prepare_qlora_model()
    message = str(excinfo.value)
    assert "method='lora'" in message, "should point at LoRA + ZeRO-3 for oversized models"
    assert "use_deepspeed=false" in message, "should point at plain QLoRA"


def test_the_guard_does_not_fire_without_zero3(monkeypatch):
    """
    QLoRA on its own must stay usable; the guard is about the combination.

    Stops at the load itself (no such model locally), which is already past the
    point the guard would have raised.
    """
    import transformers.integrations.deepspeed as ds

    from service.training.core.model_loader import ModelLoader

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: False)
    loader = ModelLoader(TrainingConfig(**_config(model_name="definitely/not-a-real-model")), None)

    with pytest.raises(Exception) as excinfo:
        loader._prepare_qlora_model()
    assert "ZeRO-3" not in str(excinfo.value)
