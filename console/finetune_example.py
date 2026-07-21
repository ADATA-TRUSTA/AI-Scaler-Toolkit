"""
AST Console AI API Client - Fine-tune Usage Example
Three main steps for fine-tune training process:
1. Check training status
2. Start training
3. Monitor training progress
"""

import sys
import argparse

from helpers.training_handler import (
    check_training_status,
    start_training,
    monitor_training_progress,
)
from helpers.finetune_initializer import initialize_finetune


if __name__ == "__main__":
    # ==================== Initialize Settings ====================
    parser = argparse.ArgumentParser(
        description="AST Console AI Client Fine-tune Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="app_settings/app_settings_finetune.json",
        help="Application settings file path (default: app_settings/app_settings_finetune.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force interrupt if training is already in progress (default: False)",
    )
    args = parser.parse_args()

    # Initialize Fine-tune environment (load settings, prepare configuration, create client)
    client, training_config = initialize_finetune(args.settings)

    # ==================== Step 1: Check Training Status ====================
    check_training_status(client, force=args.force)

    # ==================== Step 2: Start Training ====================
    try:
        start_training(client, training_config)
    except Exception as e:
        print(f"\n✗ Training start failed: {e}")
        sys.exit(1)

    # ==================== Step 3: Monitor Training Progress ====================
    try:
        monitor_training_progress(client)
    except Exception as e:
        print(f"\n✗ Training monitoring failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Fine-tune Training Completed")
    print("=" * 60)
