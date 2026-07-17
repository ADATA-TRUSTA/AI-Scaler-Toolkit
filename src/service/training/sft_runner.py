"""
SFTTrainer runner module.

This module handles the initialization and execution of SFTTrainer from the `trl` library.
It is used by `training_process.py` when `use_sft_trainer` is enabled in the configuration.
"""

from typing import Optional, Dict, Any, List, Union
import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from ..config_models import TrainingConfig
from ..settings import configure_logging

logger = configure_logging(__name__)

def run_sft_training(
    training_config: TrainingConfig,
    model,
    tokenizer,
    dataset,
    peft_config: Optional[LoraConfig] = None,
    training_args: Optional[TrainingArguments] = None,
    data_collator=None,
):
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
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        raise ImportError("Please install 'trl' library to use SFTTrainer: pip install trl")

    # Define CustomSFTTrainer to override save_model
    class CustomSFTTrainer(SFTTrainer):
        def save_model(self, output_dir=None, _internal_call=False):
            """
            Override save_model to bypass DeepSpeed's full checkpoint saving when using LoRA.
            This prevents OOM (if gather=True) and FileExists errors (if gather=False).
            """
            if output_dir is None:
                output_dir = self.args.output_dir
            
            # Check if we are using DeepSpeed and LoRA
            is_deepspeed = self.args.deepspeed is not None
            
            # Check if model is PEFT
            from peft import PeftModel
            model = self.model
            while hasattr(model, 'module'):
                model = model.module
            is_peft = isinstance(model, PeftModel)
            
            if is_deepspeed and is_peft:
                logger.info(f"[CustomSFTTrainer] Saving LoRA adapter to {output_dir} (Bypassing DeepSpeed checkpoint)")
                os.makedirs(output_dir, exist_ok=True)
                
                # Force offline mode to prevent connection attempts to HF Hub during save
                # Critical for offline environments
                original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                
                # 修正: 使用 DeepSpeed GatheredParameters 確保 ZeRO-3 參數被收集
                # 這是解決 32B 模型 size mismatch 的關鍵，且因為只收集 adapter，不會 OOM
                import deepspeed
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                
                try:
                    with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            # Save adapter
                            model.save_pretrained(output_dir)
                            
                            # Save tokenizer
                            # In newer transformers, self.tokenizer was renamed to self.processing_class
                            _tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
                            if _tokenizer is not None:
                                _tokenizer.save_pretrained(output_dir)
                            
                            # Save training args
                            try:
                                self.args.save_to_json(os.path.join(output_dir, "training_args.json"))
                            except Exception:
                                pass
                    
                    torch.distributed.barrier()
                except Exception as e:
                    logger.error(f"[CustomSFTTrainer] Failed to save LoRA adapter: {e}")
                    raise e
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
    # messages_field 可以明確指定欄位名稱，或自動偵測 dataset 中的 messages 欄位
    messages_col = training_config.messages_field or "messages"
    is_chat_format = messages_col in dataset.column_names
    is_prompt_completion = bool(training_config.prompt_field and training_config.completion_field)

    if is_chat_format and not is_prompt_completion:
        # 模式三：OpenAI chat format — TRL applies tokenizer.apply_chat_template automatically.
        # dataset_text_field must NOT be set so TRL uses the messages column directly.
        dataset_text_field = None
        logger.info(
            "[SFTRunner] Dataset mode=chat/messages; Loss mode=completion_only; "
            "TRL will apply tokenizer chat template"
        )
    elif is_prompt_completion:
        # 模式二：prompt + completion 雙欄位
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
        # 模式一：單一 text 欄位
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

    # Loss masking policy:
    # - chat format: only compute loss on assistant tokens (TRL handles masking)
    # - prompt+completion mode: only compute loss on completion tokens
    # - text mode: compute loss on the full sequence
    sft_config_kwargs["completion_only_loss"] = bool(is_chat_format or is_prompt_completion)
    logger.info(
        f"[SFTRunner] SFTConfig.completion_only_loss={sft_config_kwargs['completion_only_loss']}"
    )
    
    # Initialize SFTConfig
    sft_config = SFTConfig(**sft_config_kwargs)

    trainer = CustomSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=pass_peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("[SFTRunner] SFTTrainer initialized successfully")
    return trainer
