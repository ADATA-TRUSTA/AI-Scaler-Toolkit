"""
JSON -> pydantic config construction.

create_training_config used to maintain a hand-written allowlist of optional
fields, so anything added to TrainingConfig was silently dropped until someone
remembered to extend the list. It now forwards the whole dict; this module is
what keeps that true.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_scaler_console.config_models import TrainingConfig
from ai_scaler_console.helpers.config_loader import (
    create_inference_config,
    create_training_config,
    load_and_create_config,
    load_and_create_training_config,
)


def test_base_model_is_normalised_to_model_name() -> None:
    config = create_inference_config({"base_model": "Qwen/Qwen3-4B", "engine": "vllm"})

    assert config.model_name == "Qwen/Qwen3-4B"


def test_explicit_model_name_wins_over_base_model() -> None:
    config = create_inference_config({"base_model": "a", "model_name": "b"})

    assert config.model_name == "b"


def test_engine_specific_fields_are_forwarded() -> None:
    config = create_inference_config(
        {"base_model": "m", "engine": "llama_server", "n_gpu_layers": 10, "n_ctx": 4096}
    )

    assert config.n_gpu_layers == 10
    assert config.n_ctx == 4096


def test_missing_config_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_and_create_config("configs/inference/does-not-exist.json")


@pytest.mark.parametrize(
    "path",
    [
        "configs/inference/llama.json",
        "configs/inference/vllm.json",
        "configs/inference/transformers.json",
        "configs/inference/transformers_offload.json",
    ],
)
def test_every_shipped_inference_config_loads(path: str) -> None:
    config = load_and_create_config(path)

    assert config.model_name
    assert config.engine


def test_training_config_forwards_every_key_it_is_given() -> None:
    """No allowlist: a field on the model must arrive with its value intact."""
    raw = {
        "model_name": "Qwen/Qwen3-4B",
        "method": "lora",
        "dataset_path": "data.jsonl",
        "output_dir": "out",
        "lora_r": 64,
        "num_train_epochs": 7,
        "learning_rate": 0.000123,
        "packing": True,
    }
    config = create_training_config(raw)

    assert config.lora_r == 64
    assert config.num_train_epochs == 7
    assert config.learning_rate == pytest.approx(0.000123)
    assert config.packing is True


@pytest.mark.parametrize(
    "path",
    [
        "configs/finetune/training_llama_3.1_70b_lora.json",
        "configs/finetune/training_qwen3_4b_lora_fasttest.json",
    ],
)
def test_shipped_training_configs_lose_nothing(path: str) -> None:
    """Every key in the file must survive into the model with the same value."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dumped = load_and_create_training_config(path).model_dump()

    known = set(TrainingConfig.model_fields)
    unknown = [k for k in raw if k not in known]
    assert not unknown, f"{path} has keys the model does not define: {unknown}"

    changed = {k: (raw[k], dumped[k]) for k in raw if dumped[k] != raw[k]}
    assert not changed, f"{path} values altered in transit: {changed}"
