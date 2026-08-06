"""Training strategy implementations for SFT and Causal LM fine-tuning."""

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftModel
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from ...config_models import TrainingConfig
from ..sft_runner import run_sft_training

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """
    Custom Trainer to override save_model for DeepSpeed + LoRA compatibility.
    """

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """
        Override save_model to bypass DeepSpeed's full checkpoint saving when using LoRA.
        This prevents OOM (if gather=True) and FileExists errors (if gather=False).
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        assert output_dir is not None  # TrainingArguments always resolves output_dir  # noqa: S101

        # Check if we are using DeepSpeed and LoRA
        is_deepspeed = self.args.deepspeed is not None

        # Check if model is PEFT
        model: Any = self.model
        while hasattr(model, "module"):
            model = model.module
        is_peft = isinstance(model, PeftModel)

        if is_deepspeed and is_peft:
            logger.info(
                f"[CustomTrainer] Saving LoRA adapter to {output_dir} (Bypassing DeepSpeed checkpoint)"
            )
            os.makedirs(output_dir, exist_ok=True)

            try:
                import deepspeed  # pyright: ignore[reportMissingImports]  # linux-only optional dep

                from .zero3_utils import drain_zero3_inflight

                # Collect all trainable parameters (i.e. the LoRA adapters)
                trainable_params = [p for p in model.parameters() if p.requires_grad]

                # Complete any in-flight ZeRO-3 all-gathers before gathering the
                # adapter, else __exit__'s re-partition hits "param in flight".
                drain_zero3_inflight(self)
                # Use GatheredParameters to gather the params onto rank 0
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

                        # Save training args
                        try:
                            _args_path = os.path.join(output_dir, "training_args.json")
                            # save_to_json exists at runtime but is missing from stubs.
                            cast(Any, self.args).save_to_json(_args_path)
                        except Exception:
                            pass

                # Synchronization is required
                torch.distributed.barrier()

            except Exception as e:
                logger.exception(
                    f"[CustomTrainer] Failed to save LoRA adapter with DeepSpeed gather: {e}"
                )
                raise
        else:
            # Fallback to default behavior
            super().save_model(output_dir, _internal_call)


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""

    def __init__(
        self, config: TrainingConfig, deepspeed_config: dict[str, Any] | str | None = None
    ) -> None:
        self.config = config
        self.deepspeed_config = deepspeed_config

    @abstractmethod
    def prepare_trainer(
        self,
        model: "PreTrainedModel | PeftModel",
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin",
        dataset: "Dataset",
        training_args: TrainingArguments | None = None,
        eval_dataset: "Dataset | None" = None,
    ) -> Trainer:
        """Build and return a configured Trainer for this strategy."""

    def preprocess_dataset(
        self, dataset: "Dataset", tokenizer: "PreTrainedTokenizerBase"
    ) -> "Dataset":
        """Optional preprocessing step before model loading."""
        return dataset

    def get_training_args(self) -> TrainingArguments:
        """Assemble the TrainingArguments for this run."""
        # Use bf16 instead of fp16 to avoid gradient conflicts
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

        # Periodic evaluation on the held-out split. Only enabled when a split
        # ratio is configured; training_process pairs this with an eval_dataset,
        # so the two must agree (eval_strategy="steps" with no eval_dataset errors).
        eval_kwargs = {}
        if getattr(self.config, "eval_split_ratio", None):
            eval_steps = getattr(self.config, "eval_steps", None) or self.config.logging_steps
            eval_kwargs = {
                "eval_strategy": "steps",
                "eval_steps": eval_steps,
                "per_device_eval_batch_size": self.config.per_device_train_batch_size,
            }

        return TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
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
            **eval_kwargs,
        )


class SFTStrategy(TrainingStrategy):
    """Strategy for Supervised Fine-Tuning (SFT)."""

    def prepare_trainer(
        self,
        model: "PreTrainedModel | PeftModel",
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin",
        dataset: "Dataset",
        training_args: TrainingArguments | None = None,
        eval_dataset: "Dataset | None" = None,
    ) -> Trainer:
        """Build an SFTTrainer via the shared SFT runner."""
        logger.info("[SFTStrategy] Using SFTTrainer")
        if training_args is None:
            training_args = self.get_training_args()
        return run_sft_training(
            training_config=self.config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_args=training_args,
            eval_dataset=eval_dataset,
        )


class CausalLMStrategy(TrainingStrategy):
    """Strategy for standard Causal Language Modeling (Text Completion)."""

    def prepare_trainer(
        self,
        model: "PreTrainedModel | PeftModel",
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin",
        dataset: "Dataset",
        training_args: TrainingArguments | None = None,
        eval_dataset: "Dataset | None" = None,
    ) -> Trainer:
        """Build a CustomTrainer configured for Causal LM training."""
        logger.info("[CausalLMStrategy] Using CustomTrainer for Causal LM")

        # Use default_data_collator (plain stack) instead of
        # DataCollatorForLanguageModeling(mlm=False). The latter discards the
        # precomputed "labels" and regenerates them from input_ids (masking only
        # pad_token_id), which silently wipes out the prompt/completion masking
        # done in preprocess_dataset. preprocess_dataset already pads every
        # sequence to max_seq_length and builds labels with -100 over prompt and
        # pad positions, so a plain collate preserves that masking exactly.
        data_collator = default_data_collator

        if training_args is None:
            training_args = self.get_training_args()

        return CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    def preprocess_dataset(
        self, dataset: "Dataset", tokenizer: "PreTrainedTokenizerBase"
    ) -> "Dataset":
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

        def _encode_ids(text: str) -> list[int]:
            # Tokenizer stubs type ["input_ids"] loosely; it is a list[int] here.
            return cast(list[int], tokenizer(text, add_special_tokens=False)["input_ids"])

        # Keep the same separator behavior as the previous implementation: prompt + "\n" + completion
        sep_ids = _encode_ids("\n")

        def _tokenize_fn(examples: dict[str, Any]) -> dict[str, list[list[int]]]:
            if not isinstance(examples, dict):
                # datasets passes a LazyBatch (not a dict subclass) for
                # batched map. Materialize it to a plain dict of columns.
                # Do NOT wrap each value in an extra list: with batched=True the
                # value is already the column list for the batch; wrapping it
                # double-nests the batch and tokenizes each column's repr
                # (e.g. "['abc']") instead of the actual text.
                examples = dict(examples) if examples else {}

            # Determine batch size
            first_key = next(iter(examples)) if examples else None
            batch_size = len(examples[first_key]) if first_key else 0

            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []

            if is_prompt_completion:
                # is_prompt_completion is only True when both fields are set.
                assert prompt_col is not None and completion_col is not None  # noqa: S101
                prompts = examples.get(prompt_col) or [""] * batch_size
                completions = examples.get(completion_col) or [""] * batch_size

                for i in range(batch_size):
                    prompt_text = prompts[i] if i < len(prompts) else ""
                    completion_text = completions[i] if i < len(completions) else ""
                    prompt_text = str(prompt_text) if prompt_text is not None else ""
                    completion_text = str(completion_text) if completion_text is not None else ""

                    p_ids = _encode_ids(prompt_text)
                    c_ids = _encode_ids(completion_text)

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
                input_ids_batch = cast(list[list[int]], toks["input_ids"])
                attention_mask_batch = cast(list[list[int]], toks["attention_mask"])
                # Mask padding positions to -100 so loss is not computed over
                # pad tokens (attention_mask does not exclude tokens from loss).
                labels_batch = [
                    [tok if mask == 1 else -100 for tok, mask in zip(ids, attn, strict=True)]
                    for ids, attn in zip(input_ids_batch, attention_mask_batch, strict=True)
                ]

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
    """Factory selecting the training strategy from the config."""

    @staticmethod
    def get_strategy(
        config: TrainingConfig, deepspeed_config: dict[str, Any] | str | None = None
    ) -> TrainingStrategy:
        """Return the SFT or Causal LM strategy per config.use_sft_trainer."""
        if getattr(config, "use_sft_trainer", True):
            return SFTStrategy(config, deepspeed_config)
        else:
            return CausalLMStrategy(config, deepspeed_config)
