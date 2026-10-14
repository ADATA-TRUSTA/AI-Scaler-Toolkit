"""Unit tests for service.training.core.strategies.StrategyFactory.

The factory selects a training strategy from `config.use_sft_trainer` and
threads the optional deepspeed_config through to the strategy instance.
Constructing a strategy only stores attributes (no model/trainer is built),
so this stays a lightweight unit test.
"""

from service.config_models import TrainingConfig
from service.training.core.strategies import (
    StrategyFactory,
    SFTStrategy,
    CausalLMStrategy,
)


import pytest

pytestmark = pytest.mark.finetune


def _config(**overrides):
    base = dict(
        model_name="demo-model",
        method="lora",
        dataset_path="dataset/dataset.jsonl",
        output_dir="runtime_data/fine_tuned/demo-model",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def test_sft_trainer_true_returns_sft_strategy():
    strategy = StrategyFactory.get_strategy(_config(use_sft_trainer=True))
    assert isinstance(strategy, SFTStrategy)


def test_sft_trainer_false_returns_causal_lm_strategy():
    strategy = StrategyFactory.get_strategy(_config(use_sft_trainer=False))
    assert isinstance(strategy, CausalLMStrategy)


def test_default_config_uses_sft_strategy():
    # use_sft_trainer defaults to True in TrainingConfig.
    strategy = StrategyFactory.get_strategy(_config())
    assert isinstance(strategy, SFTStrategy)


def test_config_and_deepspeed_config_are_threaded_through():
    config = _config(use_sft_trainer=True)
    deepspeed_config = {"zero_optimization": {"stage": 3}}
    strategy = StrategyFactory.get_strategy(config, deepspeed_config)
    assert strategy.config is config
    assert strategy.deepspeed_config == deepspeed_config


def test_deepspeed_config_defaults_to_none():
    strategy = StrategyFactory.get_strategy(_config())
    assert strategy.deepspeed_config is None
