"""Persist trained models, tokenizers, and finetuned-model registry entries."""

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import torch

from ...config_models import TrainingConfig, TrainingMethod
from ...model_registry import FinetunedModelInfo, model_registry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin, Trainer

logger = logging.getLogger(__name__)


def _is_distributed() -> bool:
    """True when a process group is up, i.e. more than this one rank may be here."""
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def save_training_results(
    trainer: "Trainer",
    tokenizer: "PreTrainedTokenizerBase | ProcessorMixin",
    config: TrainingConfig,
) -> None:
    """Save model, tokenizer, and update registry."""

    # Force offline mode to prevent connection attempts to HF Hub during save
    # This is critical for offline environments where base model is already local
    original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Branch on what DeepSpeed ACTUALLY did, not on what the config asked for.
    # `use_deepspeed` can be true while `_resolve_deepspeed_config` returned None
    # -- neither deepspeed_config nor deepspeed_profile set, or the profile file
    # missing, both of which only warn. Training then runs without DeepSpeed, and
    # taking the gather path here calls `get_rank()` on an uninitialised process
    # group, which raises and loses the entire run's adapter at the last step.
    # `CustomTrainer` already gets this right by checking `args.deepspeed`.
    deepspeed_active = (
        getattr(trainer.args, "deepspeed", None) is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if getattr(config, "use_deepspeed", False) and not deepspeed_active:
        logger.warning(
            "[ModelSaver] use_deepspeed=True but DeepSpeed is not active for this run "
            "(no resolvable deepspeed_config/deepspeed_profile). Saving via the standard "
            "path instead of the ZeRO-3 gather."
        )

    try:
        # Save model
        # Special handling for DeepSpeed LoRA/QLoRA
        if deepspeed_active and config.method in [
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
        ]:
            logger.info(
                "[ModelSaver] Saving LoRA adapter only (bypassing full DeepSpeed checkpoint)..."
            )

            os.makedirs(str(config.output_dir), exist_ok=True)

            model_to_save: Any = trainer.model
            while hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module

            import deepspeed  # pyright: ignore[reportMissingImports]  # linux-only optional dep

            from .zero3_utils import drain_zero3_inflight

            trainable_params = [p for p in model_to_save.parameters() if p.requires_grad]

            try:
                # Complete any in-flight ZeRO-3 all-gathers before gathering the
                # adapter, else __exit__'s re-partition hits "param in flight".
                drain_zero3_inflight(trainer)
                with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        model_to_save.save_pretrained(str(config.output_dir))

                        # Save training args
                        try:
                            # save_to_json exists at runtime but is missing from stubs.
                            cast(Any, trainer.args).save_to_json(
                                os.path.join(config.output_dir, "training_args.json")
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.exception(f"[ModelSaver] Failed to save final LoRA adapter: {e}")
                raise
            finally:
                # In a `finally`: if rank 0 raised inside the gather (disk full,
                # permission), its peers are already parked in barrier() and
                # would sit there for the whole ddp_timeout -- half an hour of
                # apparent hang before the launcher tears the tree down.
                try:
                    torch.distributed.barrier()
                except Exception as barrier_err:
                    logger.warning(f"[ModelSaver] barrier after save failed: {barrier_err}")

        else:
            # Full finetuning or non-Deepspeed LoRA/QLoRA: use the standard save path
            trainer.save_model()

        # Everything below writes to shared paths, so only one rank may do it.
        # Under DeepSpeed every rank enters this function (the gather above is a
        # collective and needs them all), which meant N processes concurrently
        # truncating the same tokenizer.json and appending N duplicate registry
        # entries.
        if _is_distributed() and torch.distributed.get_rank() != 0:
            return

        if config.save_tokenizer:
            tokenizer.save_pretrained(str(config.output_dir))

        # record last finetuned model
        try:
            info = {
                "base_model_name": config.model_name,
                "method": config.method.value,
                "output_dir": str(config.output_dir),
            }
            from ...settings import PROJECT_ROOT

            # Anchored to the project root: this was cwd-relative, so a worker
            # started from anywhere else wrote the marker into a stray directory.
            cfg_dir = PROJECT_ROOT / "service" / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            info_path = cfg_dir / "last_finetuned_model.json"
            info_path.write_text(
                __import__("json").dumps(info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            reg_info = FinetunedModelInfo(
                base_model_name=info["base_model_name"],
                method=info["method"],
                output_dir=info["output_dir"],
                label=str(config.output_dir),
            )
            model_registry.add_finetuned(reg_info)
        except Exception as reg_err:
            logger.warning(f"[ModelSaver] Failed to update model registry: {reg_err}")

    finally:
        # Restore original HF_HUB_OFFLINE state
        if original_hf_hub_offline is None:
            del os.environ["HF_HUB_OFFLINE"]
        else:
            os.environ["HF_HUB_OFFLINE"] = original_hf_hub_offline
