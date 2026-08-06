"""
SFTTrainer runner module.

This module handles the initialization and execution of SFTTrainer from the `trl` library.
It is used by `training_process.py` when `use_sft_trainer` is enabled in the configuration.
"""

import os
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed  # explicit submodule import so `torch.distributed` resolves
from peft import LoraConfig
from transformers import TrainingArguments

from ..config_models import TrainingConfig
from ..settings import configure_logging

if TYPE_CHECKING:
    from datasets import Dataset

    # trl stub does not re-export these public symbols.
    from trl import SFTTrainer  # pyright: ignore[reportPrivateImportUsage]

logger = configure_logging(__name__)


def run_sft_training(
    training_config: TrainingConfig,
    model: Any,  # noqa: ANN401 - base or PEFT model, dynamically loaded
    tokenizer: Any,  # noqa: ANN401 - tokenizer or multimodal processor, dynamically loaded
    dataset: "Dataset",
    peft_config: LoraConfig | None = None,
    training_args: TrainingArguments | None = None,
    data_collator: Any = None,  # noqa: ANN401 - TRL/transformers collator, dynamically loaded
    eval_dataset: "Dataset | None" = None,
) -> "SFTTrainer":
    """
    Initialize and return an SFTTrainer instance.

    Args:
        training_config: The training configuration object.
        model: The model to train (can be base model or PEFT model).
        tokenizer: The tokenizer to use.
        dataset: The training dataset (raw, not tokenized).
        peft_config: Optional LoRA configuration.
        training_args: Training arguments.
        data_collator: Optional data collator.

    Returns:
        SFTTrainer instance.
    """
    try:
        # trl stub does not re-export these public symbols.
        from trl import SFTConfig, SFTTrainer  # pyright: ignore[reportPrivateImportUsage]
    except ImportError as exc:
        raise ImportError(
            "Please install 'trl' library to use SFTTrainer: pip install trl"
        ) from exc

    # Define CustomSFTTrainer to override save_model
    class CustomSFTTrainer(SFTTrainer):
        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            """
            Override save_model to bypass DeepSpeed's full checkpoint saving when using LoRA.
            This prevents OOM (if gather=True) and FileExists errors (if gather=False).
            """
            if output_dir is None:
                # self.args.output_dir is typed str | None; it is always set here.
                output_dir = cast(str, self.args.output_dir)

            # Check if we are using DeepSpeed and LoRA
            is_deepspeed = self.args.deepspeed is not None

            # Check if model is PEFT
            from peft import PeftModel

            # self.model may be wrapped (DDP/DeepSpeed); unwrap dynamically.
            model: Any = self.model
            while hasattr(model, "module"):
                model = model.module
            is_peft = isinstance(model, PeftModel)

            if is_deepspeed and is_peft:
                logger.info(
                    f"[CustomSFTTrainer] Saving LoRA adapter to {output_dir} (Bypassing DeepSpeed checkpoint)"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Force offline mode to prevent connection attempts to HF Hub during save
                # Critical for offline environments
                original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"

                # Fix: use DeepSpeed GatheredParameters so ZeRO-3 params are gathered.
                # Key to avoiding size mismatch on 32B models, and safe from OOM
                # because only the adapter is gathered.
                import deepspeed  # pyright: ignore[reportMissingImports]  # Linux-only optional dep

                from .core.zero3_utils import drain_zero3_inflight

                trainable_params = [p for p in model.parameters() if p.requires_grad]

                try:
                    # Complete any in-flight ZeRO-3 all-gathers before gathering the
                    # adapter, else __exit__'s re-partition hits "param in flight".
                    drain_zero3_inflight(self)
                    with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            # Save adapter
                            model.save_pretrained(output_dir)

                            # Save tokenizer
                            # In newer transformers, self.tokenizer was renamed to self.processing_class
                            _tokenizer = getattr(self, "processing_class", None) or getattr(
                                self, "tokenizer", None
                            )
                            if _tokenizer is not None:
                                _tokenizer.save_pretrained(output_dir)

                            # Save training args (to_json_string is the current
                            # transformers API; save_to_json no longer exists).
                            try:
                                args_path = os.path.join(output_dir, "training_args.json")
                                with open(args_path, "w", encoding="utf-8") as f:
                                    f.write(self.args.to_json_string())
                            except Exception as e:
                                logger.warning(
                                    f"[CustomSFTTrainer] Failed to write training_args.json: {e}"
                                )

                    torch.distributed.barrier()
                except Exception as e:
                    logger.exception(f"[CustomSFTTrainer] Failed to save LoRA adapter: {e}")
                    raise
                finally:
                    # Restore original HF_HUB_OFFLINE state
                    if original_hf_hub_offline is None:
                        del os.environ["HF_HUB_OFFLINE"]
                    else:
                        os.environ["HF_HUB_OFFLINE"] = original_hf_hub_offline
            else:
                # Fallback to default behavior
                super().save_model(output_dir, _internal_call)

    logger.info("[SFTRunner] Starting SFTTrainer setup...")

    # formatting_func is intentionally kept as None.
    # In TRL 0.26.x, `completion_only_loss=True` is NOT compatible with a formatting_func.
    formatting_func = None
    dataset_text_field = None

    # Determine dataset mode
    # messages_field can name the column explicitly, or the messages column is auto-detected
    messages_col = training_config.messages_field or "messages"
    is_chat_format = messages_col in dataset.column_names
    is_prompt_completion = bool(training_config.prompt_field and training_config.completion_field)

    if is_chat_format and not is_prompt_completion:
        # Mode 3: OpenAI chat format — TRL applies tokenizer.apply_chat_template automatically.
        # dataset_text_field must NOT be set so TRL uses the messages column directly.
        dataset_text_field = None
        logger.info(
            "[SFTRunner] Dataset mode=chat/messages; Loss mode=completion_only; "
            "TRL will apply tokenizer chat template"
        )
    elif is_prompt_completion:
        # Mode 2: prompt + completion column pair
        prompt_col = training_config.prompt_field
        completion_col = training_config.completion_field
        logger.info(
            "[SFTRunner] Dataset mode=prompt+completion; Loss mode=completion_only "
            f"(prompt_field='{prompt_col}', completion_field='{completion_col}')"
        )

        # Rename columns when user config uses different names.
        rename_map = {}
        if prompt_col != "prompt":
            rename_map[prompt_col] = "prompt"
        if completion_col != "completion":
            rename_map[completion_col] = "completion"
        if rename_map:
            missing = [k for k in rename_map.keys() if k not in dataset.column_names]
            if missing:
                raise ValueError(
                    f"Prompt/completion columns not found in dataset. Missing={missing}, available={dataset.column_names}"
                )
            logger.info(f"[SFTRunner] Renaming dataset columns for TRL: {rename_map}")
            dataset = dataset.rename_columns(rename_map)
    else:
        # Mode 1: single text column
        dataset_text_field = training_config.text_field or "text"
        logger.info(
            f"[SFTRunner] Dataset mode=text; Loss mode=full_sequence; dataset_text_field='{dataset_text_field}'"
        )

    # If packing is enabled, we might not need data_collator as SFTTrainer handles it,
    # but if provided, we pass it.

    # Note: SFTTrainer automatically handles PEFT if peft_config is provided.
    # However, if the model is already a PeftModel (from get_peft_model),
    # passing peft_config again might be redundant or cause issues depending on trl version.
    # But usually passing peft_config to SFTTrainer is for it to call get_peft_model itself.
    # In training_process.py, get_peft_model is called before.
    # If model is already PeftModel, we should probably pass peft_config=None to SFTTrainer
    # to avoid double wrapping, OR we pass the base model and peft_config to SFTTrainer.

    # Let's check if model is PeftModel
    from peft import PeftModel

    if isinstance(model, PeftModel):
        logger.info("[SFTRunner] Model is already a PeftModel, passing it directly to SFTTrainer")
        # If model is already PeftModel, we don't pass peft_config to SFTTrainer
        # to avoid it trying to wrap it again or create a new adapter.
        pass_peft_config = None
    else:
        pass_peft_config = peft_config

    # Create SFTConfig
    # We assume training_args is provided (it is in training_process.py)
    sft_config_kwargs: dict[str, Any]
    if training_args:
        sft_config_kwargs = training_args.to_dict()
    else:
        # Fallback if training_args is None (should not happen in current flow)
        sft_config_kwargs = {
            "output_dir": training_config.output_dir,
        }

    # Add SFT specific args
    if dataset_text_field:
        sft_config_kwargs["dataset_text_field"] = dataset_text_field

    # Map max_seq_length to max_length for SFTConfig
    sft_config_kwargs["max_length"] = training_config.max_seq_length
    sft_config_kwargs["packing"] = training_config.packing

    # Vision-language path: TRL switches to DataCollatorForVisionLanguageModeling
    # when processing_class is a ProcessorMixin and the dataset carries images.
    # It rejects packing outright for VLMs, so fail here with the reason rather
    # than letting TRL raise a bare ValueError deeper in setup.
    is_vision_dataset = any(f in dataset.column_names for f in ("images", "image"))
    if is_vision_dataset:
        try:
            from transformers import ProcessorMixin

            has_processor = isinstance(tokenizer, ProcessorMixin)
        except ImportError:
            has_processor = False
        if not has_processor:
            raise ValueError(
                "Dataset contains images but no multimodal processor was loaded for "
                f"model '{training_config.model_name}'. Image training requires a "
                "checkpoint with an AutoProcessor."
            )
        if training_config.packing:
            raise ValueError(
                "packing=True is not supported for vision-language training. "
                "Set packing=false in the training config."
            )
        logger.info(
            "[SFTRunner] Vision-language dataset detected; TRL will use "
            "DataCollatorForVisionLanguageModeling with the processor"
        )

    # Loss masking policy:
    # - chat format: only compute loss on assistant tokens (TRL handles masking)
    # - prompt+completion mode: only compute loss on completion tokens
    # - text mode: compute loss on the full sequence
    completion_only_loss = bool(is_chat_format or is_prompt_completion)

    # DataCollatorForVisionLanguageModeling only masks the prompt for
    # prompt/completion datasets; with `messages` it raises on
    # completion_only_loss and computes loss over the whole sequence. TRL's
    # usual fallback for chat data, assistant_only_loss, is rejected for VLMs
    # too, so on this path full-sequence loss is the only option available.
    if is_vision_dataset and completion_only_loss and not is_prompt_completion:
        completion_only_loss = False
        logger.warning(
            "[SFTRunner] Vision-language training with 'messages' format computes loss "
            "over the whole sequence, including the user turn -- TRL supports neither "
            "completion_only_loss nor assistant_only_loss here. Use prompt/completion "
            "fields instead if the prompt must be masked out."
        )

    sft_config_kwargs["completion_only_loss"] = completion_only_loss
    logger.info(
        f"[SFTRunner] SFTConfig.completion_only_loss={sft_config_kwargs['completion_only_loss']}"
    )

    # Initialize SFTConfig
    sft_config = SFTConfig(**sft_config_kwargs)

    trainer = CustomSFTTrainer(
        model=model,  # pyright: ignore[reportArgumentType]  # dynamically loaded base/PEFT model
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=pass_peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("[SFTRunner] SFTTrainer initialized successfully")

    if is_vision_dataset:
        from .core.model_loader import (
            MMTokenTypeAlignmentCollator,
            resolve_image_token_id,
            verify_images_reach_the_model,
        )

        # PeftModel proxies `.config` through to the wrapped model.
        # A loaded HF model always exposes `.config` (dynamic attr -> Any).
        model_config: Any = getattr(model, "config", None)
        image_token_id = resolve_image_token_id(model_config)

        # Keep mm_token_type_ids aligned with input_ids before the guard runs,
        # so verification collates the same batch training will (Qwen3.5-VL).
        if image_token_id is not None and not isinstance(
            trainer.data_collator, MMTokenTypeAlignmentCollator
        ):
            video_token_id = getattr(model_config, "video_token_id", None)
            trainer.data_collator = MMTokenTypeAlignmentCollator(
                trainer.data_collator, image_token_id, video_token_id
            )

        verify_images_reach_the_model(
            trainer.data_collator,
            [dataset[i] for i in range(min(2, len(dataset)))],
            model_config,
            tokenizer,
        )
        logger.info("[SFTRunner] Verified image tokens reach the model")

    return trainer
