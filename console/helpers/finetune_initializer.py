"""
Fine-tune Initializer
Handles the initialization process for fine-tuning tasks: load settings, prepare configuration, create client
"""

import os
import sys
import json
from typing import Tuple
from ai_client import AIClient
from config_models import AppSettings, TrainingConfig
from helpers.config_loader import load_and_create_training_config
from helpers.default_backend import load_default_backend_url


def load_app_settings(settings_path: str) -> AppSettings:
    """Load application settings file"""
    with open(settings_path, "r", encoding="utf-8") as f:
        settings_data = json.load(f)
    if "backend_url" not in settings_data:
        settings_data["backend_url"] = load_default_backend_url()
    return AppSettings(**settings_data)


def initialize_finetune(settings_path: str) -> Tuple[AIClient, TrainingConfig]:
    """
    Initialize fine-tune environment

    Returns:
        Tuple[AIClient, TrainingConfig]: (client, training_config)
    """
    # Load application settings
    try:
        app_settings = load_app_settings(settings_path)
        print(f"✓ Loaded settings from: {settings_path}")
        print(f"  Backend URL: {app_settings.backend_url}")
        print(f"  Fine-tune Config: {app_settings.finetune_config_path}")
        print()
    except FileNotFoundError:
        print(f"✗ Settings file not found: {settings_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        sys.exit(1)

    # Initialize AI Client
    client = AIClient(app_settings.backend_url)

    # Prepare training configuration
    if not app_settings.finetune_config_path:
        print("✗ finetune_config_path is not specified in settings file")
        sys.exit(1)

    finetune_config_path = app_settings.finetune_config_path
    if not os.path.isabs(finetune_config_path):
        finetune_config_path = os.path.join(
            os.path.dirname(__file__), "..", finetune_config_path
        )

    try:
        training_config = load_and_create_training_config(finetune_config_path)
        print(f"✓ Loaded training config from: {finetune_config_path}")
        print(f"  Model: {training_config.model_name}")
        print(f"  Method: {training_config.method}")
        print(f"  Dataset: {training_config.dataset_path}")
        print(f"  Output: {training_config.output_dir}")
        print(f"  Epochs: {training_config.num_train_epochs}")
        print()
    except FileNotFoundError:
        print(f"✗ Training config file not found: {finetune_config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to load training config: {e}")
        sys.exit(1)

    return client, training_config
