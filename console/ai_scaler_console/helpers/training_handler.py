"""
Training Handler
Handles the checking, starting, and monitoring processes for model fine-tuning training.
"""

import sys
import time

from ai_scaler_console.client import AIClient
from ai_scaler_console.config_models import (
    TrainingConfig,
    TrainingEvent,
    TrainingEventType,
    TrainingPhase,
    TrainingStatus,
)
from ai_scaler_console.exceptions import AIClientError, TrainingFailedError


def _format_status(status: TrainingStatus) -> str:
    """
    Compose the one-line status text.

    The backend carries lifecycle progress in `phase` / `phase_detail`; `status`
    is the coarse fallback for backends that do not report a phase.
    """
    parts = [part for part in (status.phase, status.phase_detail) if part]
    return " - ".join(parts) if parts else (status.status or "")


def check_training_status(client: AIClient, force: bool = False) -> None:
    """
    Check whether training is currently in progress.

    If it is and force=True, stop the current training. If it is and
    force=False, exit the program.
    """
    print("\n" + "=" * 60)
    print("Checking current training status...")
    print("=" * 60)

    try:
        status = client.get_training_status()
        if status.is_training:
            print("[WARN] Training is already in progress")
            print(f"  Current Step: {status.current_step}/{status.total_steps}")
            if status.current_epoch and status.total_epochs:
                print(f"  Current Epoch: {status.current_epoch:.2f}/{status.total_epochs}")

            if force:
                print("\n[..] Force mode enabled - stopping current training...")
                try:
                    client.stop_training()
                    print("[OK] Current training stopped successfully")
                    time.sleep(2)  # Wait a while to ensure training completely stops
                except Exception as e:
                    print(f"[FAIL] Failed to stop training: {e}")
                    sys.exit(1)
            else:
                print("\n[FAIL] Cannot start new training while another is in progress")
                print("  Use --force flag to stop current training and start new one")
                sys.exit(1)
        else:
            print("[OK] No training currently in progress")
    except AIClientError as e:
        # A network-level failure carries no status code, and means the very next
        # call would fail for the same reason. Saying "proceeding" there just
        # delays the same error by one step; anything the backend actually
        # answered is worth continuing past.
        if e.status_code is None:
            print(f"[FAIL] Cannot reach the backend: {e}")
            sys.exit(1)
        print(f"[WARN] Failed to check training status: {e}")
        print("  Proceeding to start training...")
    except Exception as e:
        print(f"[WARN] Failed to check training status: {e}")
        print("  Proceeding to start training...")


def start_training(client: AIClient, training_config: TrainingConfig) -> str | None:
    """
    Start training and return the job id, when the backend reports one.

    The backend nests it as `result.session_id`; it is the key every
    `/training/{job_id}/log*` endpoint is addressed by. None means the response
    did not carry one, so callers must fall back to polling `/training/status`.
    """
    print("\n" + "=" * 60)
    print("Starting Fine-tune Training...")
    print("=" * 60)

    try:
        response = client.start_training(training_config)
        print("[OK] Training started successfully!")
        print()
        try:
            job_id = (response.json().get("result") or {}).get("session_id")
        except ValueError:
            job_id = None
        return str(job_id) if job_id else None
    except Exception as e:
        print(f"[FAIL] Failed to start training: {e}")
        raise


def monitor_training_progress(client: AIClient) -> None:
    """Poll and display training status until it completes or fails."""
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
                    print(f"[FAIL] Training failed: {status.error}")
                    raise TrainingFailedError(f"Training failed: {status.error}")
                else:
                    print("[OK] Training completed!")
                    print(f"  Status: {_format_status(status) or 'No status message'}")
                    print(f"  Final Step: {status.current_step}/{status.total_steps}")
                    if status.loss is not None:
                        print(f"  Final Loss: {status.loss:.4f}")
                break

            # Training in progress, display progress (only when updated)
            current_step = status.current_step
            status_msg = _format_status(status)

            # Display if step is updated or status message has changed
            if current_step != last_step or status_msg != last_status_msg:
                progress_pct = status.progress * 100
                print("[..] Training in progress...")
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
            print("\n\n[WARN] Training monitoring interrupted by user")
            print("Note: Training may still be running on the server")
            sys.exit(0)
        except TrainingFailedError:
            # A real training failure, not a polling hiccup — do not retry.
            raise
        except Exception as e:
            print(f"\n[FAIL] Error while monitoring training: {e}")
            print("Will retry in 5 seconds...")
            time.sleep(5)

    print("=" * 60)


def _render_event(event: TrainingEvent) -> str | None:
    """Render one training-log event as a console line, or None to skip it."""
    if event.type == TrainingEventType.STAGE:
        detail = f" - {event.msg}" if event.msg else ""
        return f"[phase] {event.phase}{detail}"

    if event.type == TrainingEventType.METRIC:
        data = event.data or {}
        bits = [f"step {data.get('step')}"]
        if data.get("total") is not None:
            bits[0] += f"/{data['total']}"
        for key, label in (("loss", "loss"), ("learning_rate", "lr"), ("epoch", "epoch")):
            if data.get(key) is not None:
                bits.append(f"{label}={data[key]}")
        # Eval points are interleaved into the same stream; tag them so they are not
        # mistaken for the training curve.
        if event.is_eval_metric:
            tag = "[eval|final]" if data.get("final") else "[eval]"
            return f"{tag} " + " ".join(bits)
        return "[metric] " + " ".join(bits)

    if event.type == TrainingEventType.ERROR:
        data = event.data or {}
        oom = " (OOM)" if data.get("is_oom") else ""
        return f"[error]{oom} {data.get('error') or event.msg or 'unknown error'}"

    if event.type == TrainingEventType.INFO:
        return f"[info] {event.msg}" if event.msg else None

    # `resource` samples arrive every second; too noisy for a console log.
    return None


def monitor_training_log(client: AIClient, job_id: str, show_resource: bool = False) -> None:
    """
    Follow a training job's structured log over SSE until it reaches a terminal event.

    Preferred over monitor_training_progress(): the backend pushes each lifecycle
    phase, metric, and error as it happens instead of being sampled every few
    seconds, and held-out eval metrics are only visible here.

    Raises TrainingFailedError when the job ends in an error or a failed phase.
    """
    print("\n" + "=" * 60)
    print(f"Following Training Log (job {job_id})...")
    print("=" * 60)
    print()

    last_progress = -1.0
    try:
        for event in client.stream_training_log(job_id):
            if event.type == TrainingEventType.RESOURCE and not show_resource:
                continue

            line = _render_event(event)
            if line:
                print(line)

            # The final eval metric carries no progress; guarding keeps the last
            # real value instead of blanking it at the very end.
            progress = (event.data or {}).get("progress")
            if progress is not None and progress != last_progress:
                last_progress = progress
                print(f"   Progress: {progress * 100:.1f}%")

            if event.is_terminal:
                if event.type == TrainingEventType.ERROR or event.phase == TrainingPhase.ERROR:
                    details = (event.data or {}).get("error") or event.msg or "unknown error"
                    raise TrainingFailedError(f"Training failed: {details}")
                print("\n" + "-" * 60)
                print(f"[OK] Training finished: {event.phase}")

    except KeyboardInterrupt:
        print("\n\n[WARN] Log following interrupted by user")
        print("Note: Training may still be running on the server")
        sys.exit(0)

    print("=" * 60)
