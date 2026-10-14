"""Unit tests for CausalLMStrategy label masking end-to-end.

CausalLMStrategy.preprocess_dataset builds `labels` with -100 over prompt and
pad positions (completion-only loss). This masking is only effective if the
data collator preserves the precomputed labels. The previous collator,
DataCollatorForLanguageModeling(mlm=False), silently discarded them by cloning
input_ids and masking only pad_token_id, which wiped out the prompt mask.

These tests verify:
  1. preprocess_dataset produces correctly masked labels.
  2. The collator wired in prepare_trainer preserves those labels.
  3. Regression guard: the old collator would have destroyed the prompt mask.
"""

import datasets
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, default_data_collator

# The datasets torch formatter probes torchvision's video decoder on row access;
# some torchvision builds lack VideoReader, which is irrelevant to int tensors.
# Disable the probe so set_format("torch") works in this test environment.
datasets.config.TORCHVISION_AVAILABLE = False

# The stub tokenizer is not stably hashable, so datasets may reuse a stale
# on-disk map() cache. Disable caching so every run recomputes from source.
datasets.disable_caching()

from service.config_models import TrainingConfig
from service.training.core import strategies
from service.training.core.strategies import CausalLMStrategy


PAD_ID = 0
EOS_ID = 1


import pytest

pytestmark = pytest.mark.finetune


class StubTokenizer:
    """Minimal char-level tokenizer covering the two call patterns used by
    preprocess_dataset: single-string (add_special_tokens=False) and batched
    with padding='max_length'. Each char maps to its ordinal (all >= 10 here,
    so none collide with PAD_ID/EOS_ID)."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(text, str):
            return {"input_ids": [ord(c) for c in text]}

        # Batched (text mode)
        seqs = [[ord(c) for c in str(t)] for t in text]
        if truncation and max_length is not None:
            seqs = [s[:max_length] for s in seqs]
        input_ids, attention_mask = [], []
        for s in seqs:
            attn = [1] * len(s)
            if padding == "max_length" and max_length is not None and len(s) < max_length:
                pad_n = max_length - len(s)
                s = s + [PAD_ID] * pad_n
                attn = attn + [0] * pad_n
            input_ids.append(s)
            attention_mask.append(attn)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pad(self, features, return_tensors=None, pad_to_multiple_of=None, **kwargs):
        # preprocess_dataset already pads every sequence to max_seq_length, so
        # padding here is just a stack. Used only by the old-collator regression
        # guard below (DataCollatorForLanguageModeling calls tokenizer.pad).
        batch = {}
        for key in features[0]:
            vals = [f[key] for f in features]
            batch[key] = torch.stack(
                [v if torch.is_tensor(v) else torch.tensor(v) for v in vals]
            )
        return batch


def _strategy(**overrides):
    base = dict(
        model_name="demo-model",
        method="lora",
        dataset_path="dataset/dataset.jsonl",
        output_dir="runtime_data/fine_tuned/demo-model",
        use_sft_trainer=False,
    )
    base.update(overrides)
    return CausalLMStrategy(TrainingConfig(**base))


def _collate(dataset):
    """Feed the formatted dataset through the exact collator prepare_trainer
    uses, returning the batched dict of tensors."""
    features = [dataset[i] for i in range(len(dataset))]
    return default_data_collator(features)


def test_preprocess_masks_prompt_and_pad_in_prompt_completion_mode():
    strategy = _strategy(prompt_field="prompt", completion_field="completion", max_seq_length=8)
    ds = Dataset.from_dict({"prompt": ["ab"], "completion": ["cd"]})
    out = strategy.preprocess_dataset(ds, StubTokenizer())

    labels = _collate(out)["labels"][0].tolist()
    # ids: [a,b] + [\n] + [c,d] -> prompt=pos0-2, completion=pos3-4, pad=pos5-7
    assert labels[:3] == [-100, -100, -100], "prompt/sep must be masked"
    assert labels[3:5] == [ord("c"), ord("d")], "completion must be preserved"
    assert labels[5:] == [-100, -100, -100], "pad must be masked"


def test_preprocess_masks_pad_in_text_mode():
    strategy = _strategy(text_field="text", max_seq_length=6)
    ds = Dataset.from_dict({"text": ["abc"]})
    out = strategy.preprocess_dataset(ds, StubTokenizer())

    batch = _collate(out)
    labels = batch["labels"][0].tolist()
    attn = batch["attention_mask"][0].tolist()
    # Real tokens preserved, pad positions (attn==0) masked to -100.
    assert labels[:3] == [ord("a"), ord("b"), ord("c")]
    assert labels[3:] == [-100, -100, -100]
    assert attn == [1, 1, 1, 0, 0, 0]


def test_old_collator_would_destroy_prompt_mask_regression_guard():
    """Documents why the collator was swapped: DataCollatorForLanguageModeling
    regenerates labels from input_ids and only masks pad_token_id, so the prompt
    mask (-100 over real prompt tokens) is lost."""
    strategy = _strategy(prompt_field="prompt", completion_field="completion", max_seq_length=8)
    ds = Dataset.from_dict({"prompt": ["ab"], "completion": ["cd"]})
    out = strategy.preprocess_dataset(ds, StubTokenizer())

    old = DataCollatorForLanguageModeling(tokenizer=StubTokenizer(), mlm=False)
    features = [out[i] for i in range(len(out))]
    labels = old.torch_call(features)["labels"][0].tolist()
    # Prompt positions come back as real token ids, NOT -100 -> masking lost.
    assert labels[:3] == [ord("a"), ord("b"), ord("\n")]


def test_prepare_trainer_wires_label_preserving_collator(monkeypatch):
    """The collator handed to the Trainer must be the label-preserving
    default_data_collator, not one that regenerates labels."""
    captured = {}

    class RecordingTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(strategies, "CustomTrainer", RecordingTrainer)

    strategy = _strategy(text_field="text")
    sentinel_args = object()
    strategy.prepare_trainer(
        model=None, tokenizer=StubTokenizer(), dataset=None, training_args=sentinel_args
    )

    assert captured["data_collator"] is default_data_collator
