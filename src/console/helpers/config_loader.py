"""
Config loader helper for loading and creating model configurations
"""

import json
from typing import Dict, Any

from config_models import InferenceConfig, TrainingConfig


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file

    Args:
        config_path (str): Configuration file path

    Returns:
        Dict[str, Any]: Loaded configuration dictionary

    Raises:
        FileNotFoundError: When the configuration file does not exist
        json.JSONDecodeError: When JSON format is incorrect
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_inference_config(
    raw_config: Dict[str, Any],
) -> InferenceConfig:
    """Create InferenceConfig based on raw configuration.

    The JSON files under `infer_model_configs/` are aligned with the
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
    """Load configuration file and create InferenceConfig

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
    raw_config: Dict[str, Any],
) -> TrainingConfig:
    """Create TrainingConfig based on raw configuration

    Args:
        raw_config (Dict[str, Any]): Raw configuration loaded from JSON

    Returns:
        TrainingConfig: Created training configuration object
    """
    # Build basic parameters
    config_params = {
        "model_name": raw_config["model_name"],
        "method": raw_config["method"],
        "dataset_path": raw_config["dataset_path"],
        "output_dir": raw_config["output_dir"],
    }

    # Add optional parameters (if present in raw_config)
    optional_fields = [
        "offload_folder",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "max_seq_length",
        "text_field",
        "prompt_field",
        "completion_field",
        "save_tokenizer",
        "use_deepspeed",
        "deepspeed_config",
        "deepspeed_profile",
        "use_sft_trainer",
        "packing",
    ]
    for field in optional_fields:
        if field in raw_config:
            config_params[field] = raw_config[field]

    return TrainingConfig(**config_params)


def load_and_create_training_config(
    config_path: str,
) -> TrainingConfig:
    """Load training configuration file and create TrainingConfig

    Args:
        config_path (str): Training configuration file path

    Returns:
        TrainingConfig: Created training configuration object
    """
    # Load configuration
    raw_config = load_config_from_json(config_path)

    # Create TrainingConfig
    return create_training_config(raw_config)
