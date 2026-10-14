"""
Config loader helper for loading and creating model configurations.
"""

import json
from typing import Any

from ai_scaler_console.config_models import InferenceConfig, TrainingConfig


def load_config_from_json(config_path: str) -> dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Configuration file path

    Returns:
        Dict[str, Any]: Loaded configuration dictionary

    Raises:
        FileNotFoundError: When the configuration file does not exist
        json.JSONDecodeError: When JSON format is incorrect
    """
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def create_inference_config(
    raw_config: dict[str, Any],
) -> InferenceConfig:
    """
    Create InferenceConfig based on raw configuration.

    The JSON files under `configs/inference/` are aligned with the
    Trusta-AST-Frontend's load-model payload, so we just normalize
    `base_model` -> `model_name` and forward all remaining engine-specific
    fields (transformers / llama_server / vllm) directly to the Pydantic model
    for validation.

    Args:
        raw_config: Raw configuration loaded from JSON.

    Returns:
        Created inference configuration object.
    """
    config = dict(raw_config)
    if "base_model" in config and "model_name" not in config:
        config["model_name"] = config.pop("base_model")
    return InferenceConfig(**config)


def load_and_create_config(
    config_path: str,
) -> InferenceConfig:
    """
    Load configuration file and create InferenceConfig.

    Args:
        config_path (str): Configuration file path
    Returns:
        InferenceConfig: Created inference configuration object
    """
    # Load configuration
    raw_config = load_config_from_json(config_path)

    # Create InferenceConfig
    return create_inference_config(raw_config)


def create_training_config(
    raw_config: dict[str, Any],
) -> TrainingConfig:
    """
    Create TrainingConfig based on raw configuration.

    Forwards every key straight to the Pydantic model, the same way
    :func:`create_inference_config` does, so a field added to TrainingConfig
    works immediately. The previous hand-maintained allowlist of optional fields
    silently dropped anything not listed on it.

    Args:
        raw_config (Dict[str, Any]): Raw configuration loaded from JSON

    Returns:
        TrainingConfig: Created training configuration object
    """
    return TrainingConfig(**raw_config)


def load_and_create_training_config(
    config_path: str,
) -> TrainingConfig:
    """
    Load training configuration file and create TrainingConfig.

    Args:
        config_path (str): Training configuration file path

    Returns:
        TrainingConfig: Created training configuration object
    """
    # Load configuration
    raw_config = load_config_from_json(config_path)

    # Create TrainingConfig
    return create_training_config(raw_config)
