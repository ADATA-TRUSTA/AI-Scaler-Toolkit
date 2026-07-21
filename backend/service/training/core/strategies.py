import os
import logging
import torch
from abc import ABC, abstractmethod
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
from ...config_models import TrainingConfig, TrainingMethod
from ..sft_runner import run_sft_training

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """
    Custom Trainer to override save_model for DeepSpeed + LoRA compatibility.
    """
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
        model = self.model
        while hasattr(model, 'module'):
            model = model.module
        is_peft = isinstance(model, PeftModel)
        
        if is_deepspeed and is_peft:
            logger.info(f"[CustomTrainer] Saving LoRA adapter to {output_dir} (Bypassing DeepSpeed checkpoint)")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                import deepspeed
                # 找出所有 trainable parameters (即 LoRA adapters)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                
                # 使用 GatheredParameters 將參數收集到 rank 0
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
                
                # 必須同步
                torch.distributed.barrier()

            except Exception as e:
                logger.error(f"[CustomTrainer] Failed to save LoRA adapter with DeepSpeed gather: {e}")
                raise e
        else:
            # Fallback to default behavior
            super().save_model(output_dir, _internal_call)


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""
    
    def __init__(self, config: TrainingConfig, deepspeed_config=None):
        self.config = config
        self.deepspeed_config = deepspeed_config

    @abstractmethod
    def prepare_trainer(self, model, tokenizer, dataset, training_args=None) -> Trainer:
        pass

    def preprocess_dataset(self, dataset, tokenizer):
        """Optional preprocessing step before model loading."""
        return dataset

    def get_training_args(self) -> TrainingArguments:
        # 使用 bf16 代替 fp16 以避免梯度衝突
        use_fp16 = False
        use_bf16 = True
        optim = "adamw_torch"

        # Gradient checkpointing: trades compute for memory by recomputing
        # activations during backward pass instead of storing them.
        # Critical for large models (e.g. 35B MoE) to fit in GPU memory.
        use_gc = getattr(self.config, "gradient_checkpointing", True)

        # DeepSpeed ZeRO-3 requires use_reentrant=True because it manages
        # parameter partitioning/gathering during forward. With use_reentrant=False,
        # the recomputation during backward finds offloaded (shape=[0]) tensors
        # instead of the original parameter shapes, causing CheckpointError.
        # Without DeepSpeed, use_reentrant=False is preferred (PEFT compatible).
        gc_use_reentrant = True if self.deepspeed_config else False
        gc_kwargs = {"use_reentrant": gc_use_reentrant} if use_gc else None

        return TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit or 2,
            fp16=use_fp16,
            bf16=use_bf16,
            optim=optim,
            gradient_checkpointing=use_gc,
            gradient_checkpointing_kwargs=gc_kwargs,
            deepspeed=self.deepspeed_config,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )


class SFTStrategy(TrainingStrategy):
    """Strategy for Supervised Fine-Tuning (SFT)."""
    
    def prepare_trainer(self, model, tokenizer, dataset, training_args=None) -> Trainer:
        logger.info("[SFTStrategy] Using SFTTrainer")
        if training_args is None:
            training_args = self.get_training_args()
        return run_sft_training(
            training_config=self.config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_args=training_args,
        )


class CausalLMStrategy(TrainingStrategy):
    """Strategy for standard Causal Language Modeling (Text Completion)."""
    
    def prepare_trainer(self, model, tokenizer, dataset, training_args=None) -> Trainer:
        logger.info("[CausalLMStrategy] Using CustomTrainer for Causal LM")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        if training_args is None:
            training_args = self.get_training_args()
        
        return CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

    def preprocess_dataset(self, dataset, tokenizer):
        """Tokenize dataset for Causal LM."""

        max_len = int(getattr(self.config, "max_seq_length", 1024) or 1024)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        prompt_col = getattr(self.config, "prompt_field", None)
        completion_col = getattr(self.config, "completion_field", None)
        is_prompt_completion = bool(prompt_col and completion_col)

        if is_prompt_completion:
            logger.info(
                "[CausalLMStrategy] Dataset mode=prompt+completion; Loss mode=completion_only (mask prompt labels=-100) "
                f"(prompt_field='{prompt_col}', completion_field='{completion_col}')"
            )
        else:
            field_name = getattr(self.config, "text_field", None) or "text"
            logger.info(
                f"[CausalLMStrategy] Dataset mode=text; Loss mode=full_sequence (labels=input_ids) (text_field='{field_name}')"
            )

        # Keep the same separator behavior as the previous implementation: prompt + "\n" + completion
        sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]

        def _tokenize_fn(examples):
            if not isinstance(examples, dict):
                examples = {k: [v] for k, v in (examples or {}).items()}

            # Determine batch size
            first_key = next(iter(examples)) if examples else None
            batch_size = len(examples[first_key]) if first_key else 0

            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []

            if is_prompt_completion:
                prompts = examples.get(prompt_col) or [""] * batch_size
                completions = examples.get(completion_col) or [""] * batch_size

                for i in range(batch_size):
                    prompt_text = prompts[i] if i < len(prompts) else ""
                    completion_text = completions[i] if i < len(completions) else ""
                    prompt_text = str(prompt_text) if prompt_text is not None else ""
                    completion_text = str(completion_text) if completion_text is not None else ""

                    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

                    ids = p_ids + sep_ids + c_ids
                    labels = ([-100] * (len(p_ids) + len(sep_ids))) + c_ids

                    # Truncate
                    ids = ids[:max_len]
                    labels = labels[:max_len]

                    # Pad
                    attn = [1] * len(ids)
                    if len(ids) < max_len:
                        pad_n = max_len - len(ids)
                        ids = ids + ([pad_id] * pad_n)
                        attn = attn + ([0] * pad_n)
                        labels = labels + ([-100] * pad_n)

                    input_ids_batch.append(ids)
                    attention_mask_batch.append(attn)
                    labels_batch.append(labels)

            else:
                # Text mode: full-sequence loss
                field_name = getattr(self.config, "text_field", None) or "text"
                texts = examples.get(field_name)
                if texts is None:
                    # fallback: build from any existing "text" column
                    texts = examples.get("text")
                if texts is None:
                    texts = [""] * batch_size

                texts = [str(t) if t is not None else "" for t in texts]
                toks = tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors=None,
                )
                input_ids_batch = toks["input_ids"]
                attention_mask_batch = toks["attention_mask"]
                labels_batch = [ids.copy() for ids in toks["input_ids"]]

            return {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
            }

        logger.info(f"[CausalLMStrategy] Tokenizing {len(dataset)} examples...")
        dataset = dataset.map(
            _tokenize_fn, 
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )
        logger.info("[CausalLMStrategy] Tokenization completed")
        
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        return dataset


class StrategyFactory:
    @staticmethod
    def get_strategy(config: TrainingConfig, deepspeed_config=None) -> TrainingStrategy:
        if getattr(config, "use_sft_trainer", True):
            return SFTStrategy(config, deepspeed_config)
        else:
            return CausalLMStrategy(config, deepspeed_config)
