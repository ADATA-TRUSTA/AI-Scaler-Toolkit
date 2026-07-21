"""
Training Handler
Handles the checking, starting, and monitoring processes for model fine-tuning training
"""

import sys
import time
from ai_client import AIClient
from config_models import TrainingConfig


def check_training_status(client: AIClient, force: bool = False) -> None:
    """
    Check if training is currently in progress
    If yes and force=True, stop current training
    If yes and force=False, exit program
    """
    print("\n" + "=" * 60)
    print("Checking current training status...")
    print("=" * 60)

    try:
        status = client.get_training_status()
        if status.is_training:
            print("⚠ Training is already in progress")
            print(f"  Current Step: {status.current_step}/{status.total_steps}")
            if status.current_epoch and status.total_epochs:
                print(
                    f"  Current Epoch: {status.current_epoch:.2f}/{status.total_epochs}"
                )

            if force:
                print("\n🔄 Force mode enabled - stopping current training...")
                try:
                    client.stop_training()
                    print("✓ Current training stopped successfully")
                    time.sleep(2)  # Wait a while to ensure training completely stops
                except Exception as e:
                    print(f"✗ Failed to stop training: {e}")
                    sys.exit(1)
            else:
                print("\n✗ Cannot start new training while another is in progress")
                print("  Use --force flag to stop current training and start new one")
                sys.exit(1)
        else:
            print("✓ No training currently in progress")
    except Exception as e:
        print(f"⚠ Failed to check training status: {e}")
        print("  Proceeding to start training...")


def start_training(client: AIClient, training_config: TrainingConfig) -> None:
    """Start training"""
    print("\n" + "=" * 60)
    print("Starting Fine-tune Training...")
    print("=" * 60)

    try:
        client.start_training(training_config)
        print("✓ Training started successfully!")
        print()
    except Exception as e:
        print(f"✗ Failed to start training: {e}")
        raise


def monitor_training_progress(client: AIClient) -> None:
    """
    Monitor training progress, continuously display training status until completion or failure
    """
    print("\n" + "=" * 60)
    print("Monitoring Training Progress...")
    print("=" * 60)
    print()

    # Track last displayed state to avoid repeating same information
    last_step = -1
    last_status_msg = ""

    while True:
        try:
            # Get training status
            status = client.get_training_status()

            # If not training, check training result
            if not status.is_training:
                print("\n" + "-" * 60)
                if status.error:
                    print(f"✗ Training failed: {status.error}")
                    raise Exception(f"Training failed: {status.error}")
                else:
                    print("✓ Training completed!")
                    print(f"  Status: {status.status or 'No status message'}")
                    print(f"  Final Step: {status.current_step}/{status.total_steps}")
                    if status.loss is not None:
                        print(f"  Final Loss: {status.loss:.4f}")
                break

            # Training in progress, display progress (only when updated)
            current_step = status.current_step
            status_msg = status.status or ""

            # Display if step is updated or status message has changed
            if current_step != last_step or status_msg != last_status_msg:
                progress_pct = status.progress * 100
                print("⏳ Training in progress...")
                print(
                    f"   Progress: {progress_pct:.1f}% | Step: {current_step}/{status.total_steps}",
                    end="",
                )

                if status.current_epoch is not None and status.total_epochs is not None:
                    print(
                        f" | Epoch: {status.current_epoch:.2f}/{status.total_epochs}",
                        end="",
                    )

                if status.loss is not None:
                    print(f" | Loss: {status.loss:.4f}", end="")

                print()  # Newline

                if status_msg:
                    print(f"   Status: {status_msg}")

                # Update tracking variables
                last_step = current_step
                last_status_msg = status_msg

            # Wait a while before checking again (avoid overly frequent queries)
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\n⚠ Training monitoring interrupted by user")
            print("Note: Training may still be running on the server")
            sys.exit(0)
        except Exception as e:
            if "Training failed" in str(e):
                # This is a training failure error, raise directly
                raise
            # Other errors, retry
            print(f"\n✗ Error while monitoring training: {e}")
            print("Will retry in 5 seconds...")
            time.sleep(5)

    print("=" * 60)
