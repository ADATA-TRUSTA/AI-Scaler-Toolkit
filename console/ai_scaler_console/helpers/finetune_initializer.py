"""
Fine-tune Initializer
Handles the initialization process for fine-tuning tasks: load settings, prepare configuration, create client.
"""

import json
import sys

from ai_scaler_console.client import AIClient
from ai_scaler_console.config_models import AppSettings, TrainingConfig
from ai_scaler_console.helpers.config_loader import load_and_create_training_config
from ai_scaler_console.settings import BACKEND_URL


def load_app_settings(settings_path: str) -> AppSettings:
    """Load application settings file."""
    with open(settings_path, encoding="utf-8") as f:
        settings_data = json.load(f)
    if not settings_data.get("backend_url"):
        settings_data["backend_url"] = BACKEND_URL
    return AppSettings(**settings_data)


def initialize_finetune(settings_path: str) -> tuple[AIClient, TrainingConfig]:
    """
    Initialize fine-tune environment.

    Returns:
        Tuple[AIClient, TrainingConfig]: (client, training_config)
    """
    # Load application settings
    try:
        app_settings = load_app_settings(settings_path)
        print(f"[OK] Loaded settings from: {settings_path}")
        print(f"  Backend URL: {app_settings.backend_url}")
        print(f"  Fine-tune Config: {app_settings.finetune_config_path}")
        print()
    except FileNotFoundError:
        print(f"[FAIL] Settings file not found: {settings_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Failed to load settings: {e}")
        sys.exit(1)

    # Initialize AI Client
    client = AIClient(app_settings.backend_url)

    # Prepare training configuration
    if not app_settings.finetune_config_path:
        print("[FAIL] finetune_config_path is not specified in settings file")
        sys.exit(1)

    # Resolved against the current directory, like every other config path the
    # examples take. Resolving it against the package directory instead sent
    # `configs/...` to `ai_scaler_console/configs/...`, which does not exist.
    finetune_config_path = app_settings.finetune_config_path

    try:
        training_config = load_and_create_training_config(finetune_config_path)
        print(f"[OK] Loaded training config from: {finetune_config_path}")
        print(f"  Model: {training_config.model_name}")
        print(f"  Method: {training_config.method}")
        print(f"  Dataset: {training_config.dataset_path}")
        print(f"  Output: {training_config.output_dir}")
        print(f"  Epochs: {training_config.num_train_epochs}")
        print()
    except FileNotFoundError:
        print(f"[FAIL] Training config file not found: {finetune_config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Failed to load training config: {e}")
        sys.exit(1)

    return client, training_config
