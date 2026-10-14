"""
AST Console AI API Client - Fine-tune Usage Example
Three main steps for fine-tune training process:
1. Check training status
2. Start training
3. Monitor training progress.
"""

import argparse
import sys

from ai_scaler_console.helpers.finetune_initializer import initialize_finetune
from ai_scaler_console.helpers.training_handler import (
    check_training_status,
    monitor_training_log,
    monitor_training_progress,
    start_training,
)


def main() -> None:
    """Check status, start fine-tune training, then follow it to completion."""
    # ==================== Initialize Settings ====================
    parser = argparse.ArgumentParser(
        description="AST Console AI Client Fine-tune Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="configs/finetune/run.json",
        help="Application settings file path (default: configs/finetune/run.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force interrupt if training is already in progress (default: False)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        default=False,
        help="Poll /training/status instead of following the SSE training log",
    )
    parser.add_argument(
        "--show-resource",
        action="store_true",
        default=False,
        help="Include per-second GPU/DRAM/SSD samples when following the log",
    )
    args = parser.parse_args()

    # Initialize Fine-tune environment (load settings, prepare configuration, create client)
    client, training_config = initialize_finetune(args.settings)

    with client:
        # ==================== Step 1: Check Training Status ====================
        check_training_status(client, force=args.force)

        # ==================== Step 2: Start Training ====================
        try:
            job_id = start_training(client, training_config)
        except Exception as e:
            print(f"\n[FAIL] Training start failed: {e}")
            sys.exit(1)

        # ==================== Step 3: Monitor Training Progress ====================
        # The SSE log is pushed per event and is the only place held-out eval
        # metrics appear; polling is the fallback when the backend returned no
        # job id, or when --poll is given.
        try:
            if job_id and not args.poll:
                monitor_training_log(client, job_id, show_resource=args.show_resource)
            else:
                monitor_training_progress(client)
        except Exception as e:
            print(f"\n[FAIL] Training monitoring failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Fine-tune Training Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
