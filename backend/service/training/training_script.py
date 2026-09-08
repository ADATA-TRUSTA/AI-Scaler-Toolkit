"""
Standalone training entry point launched by `deepspeed --num_gpus N`.

Hybrid status IPC
-----------------
Primary   : stdout pipe
    Rank 0 prints ``__STATUS__:<json>\\n`` lines.  The parent worker reads
    them in real time via a background thread — zero polling lag, no external
    service required.

Backup    : Redis (terminal states only)
    For ``saved`` and ``error`` states, rank 0 also writes to Redis *after*
    printing to stdout.  The parent worker reads Redis **once** after the
    process exits to recover the terminal state in case the process was killed
    before its stdout buffer was fully drained (e.g. SIGKILL, OOM killer).

Redis is also used for persisting metrics history (best-effort).

Usage (launched by _run_deepspeed_subprocess, not called directly):
    deepspeed --num_gpus N service/training/training_script.py \\
        --config /tmp/ds_train_cfg_xxx.json \\
        --session-id <uuid> \\
        [--ds-config /tmp/ds_config_override_xxx.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# DeepSpeed launcher sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* — do NOT override.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure the backend project root is importable.
# File layout: service/training/training_script.py  →  parents[2] = project root
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from service.config_models import TrainingConfig  # noqa: E402
from service.settings import REDIS_DB, REDIS_HOST, REDIS_PORT, configure_logging  # noqa: E402
from service.training.core import (  # noqa: E402
    EventType,
    Phase,
    build_resource_snapshot,
    build_training_components,
    log_mem,
    save_training_results,
    start_memory_sampler,
)
from service.utils.token_utils import load_hf_token  # noqa: E402

logger = configure_logging(__name__)

_RANK = int(os.environ.get("RANK", "0"))
_IS_RANK0 = _RANK == 0

# Must match the constant in training_process._run_deepspeed_subprocess.
_STATUS_PREFIX = "__STATUS__:"
_TERMINAL_STATES = frozenset({"saved", "error"})


def main() -> None:
    """Parse CLI args and run the DeepSpeed training job for one config."""
    parser = argparse.ArgumentParser(description="DeepSpeed multi-GPU training entry point")
    parser.add_argument("--config", required=True, help="TrainingConfig JSON file path")
    parser.add_argument("--session-id", required=True, dest="session_id")
    parser.add_argument(
        "--ds-config",
        dest="ds_config",
        default=None,
        help="Already-resolved DeepSpeed config path (handled by launcher)",
    )
    # DeepSpeed may inject additional args; ignore them.
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)
    training_config = TrainingConfig(**cfg_dict)
    session_id = args.session_id
    ds_config_path: str | None = args.ds_config

    # Redis: optional, used for (a) terminal-state backup and (b) metrics history.
    redis_client = None
    _status_redis_key = f"training:mgr_status:{session_id}"
    _key_metrics = f"training:history:{session_id}:metrics"
    try:
        import redis as _redis_lib

        redis_client = _redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
        redis_client.ping()
    except Exception:
        redis_client = None

    def _push_status(payload: dict) -> None:
        """
        Send a status update.

        Primary path  : stdout — immediate, synchronous, no polling lag.
        Backup path   : Redis — written only for terminal states (saved/error)
                        so the parent can recover if the process is killed
                        before stdout is fully read.

        Progress-style statuses are rank 0 only, but ERRORS are reported from
        every rank. A data-parallel OOM fires on whichever rank is unluckiest,
        and gating those on rank 0 meant the parent saw only
        "deepspeed exited with return code N" with `is_oom=False`: the real
        traceback reached the server log and never `writer.error`,
        `/training/error_details`, or the OOM-specific handling. That made the
        most likely multi-GPU failure the worst-reported one.
        """
        is_error = payload.get("status") == "error"
        if not _IS_RANK0 and not is_error:
            return
        if not _IS_RANK0:
            # Say which rank, or two ranks failing looks like one confused one.
            payload = {**payload, "rank": _RANK}

        # 1. stdout — always, for all statuses
        print(f"{_STATUS_PREFIX}{json.dumps(payload)}", flush=True)

        # 2. Redis — only for terminal states (crash-safe backup)
        if redis_client and payload.get("status") in _TERMINAL_STATES:
            try:
                redis_client.set(_status_redis_key, json.dumps(payload))
            except Exception:
                pass  # Redis failure must never abort training

    def _emit_event(
        etype: str, phase: str | None = None, msg: str | None = None, data: dict | None = None
    ) -> None:
        """
        Emit a structured log event over stdout (rank 0 only).

        The parent worker is the single file writer; it persists these to the
        job's events.jsonl. Stage events also drive the manager's phase field.
        """
        if not _IS_RANK0:
            return
        payload: dict = {"type": etype}
        if phase is not None:
            payload["phase"] = phase
        if msg is not None:
            payload["msg"] = msg
        if data is not None:
            payload["data"] = data
        print(f"{_STATUS_PREFIX}{json.dumps(payload)}", flush=True)

    def _emit_stage(phase: str, detail: str | None = None) -> None:
        _emit_event("stage", phase=phase, msg=detail)

    hf_token = load_hf_token()
    if hf_token:
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    # Resource sampler (rank 0 only), started before loading so the user gets
    # GPU/DRAM/SSD feedback during the load phase. Emitted as resource events
    # that the parent worker persists to events.jsonl.
    _sampler = (None, None)

    try:
        _push_status({"status": "initializing"})

        if _IS_RANK0:

            def _on_mem_sample(probe: dict) -> None:
                try:
                    _emit_event("resource", data=build_resource_snapshot(training_config, probe))
                except Exception:
                    pass

            _sample_interval = float(os.getenv("TRAINING_MEM_SAMPLE_INTERVAL", "5.0"))
            _sampler = start_memory_sampler(interval=_sample_interval, on_sample=_on_mem_sample)

        # The shared setup sequence -- same one training_process.py runs for a
        # single GPU. This used to be hand-duplicated here and drifted: image
        # training and the eval split were both missing.
        components = build_training_components(
            training_config,
            hf_token,
            ds_config_path,
            on_stage=_emit_stage,
            # _emit_event is already rank-0 guarded internally.
            on_note=lambda msg, payload: _emit_event(EventType.INFO, msg=msg, data=payload),
        )
        strategy = components.strategy
        training_args = components.training_args
        model = components.model
        if _IS_RANK0:
            log_mem("after_model_load")
        _emit_stage(Phase.PREPARING_TRAINER)
        trainer = strategy.prepare_trainer(
            model,
            components.processing_class,
            components.dataset,
            training_args,
            eval_dataset=components.eval_dataset,
        )
        if _IS_RANK0:
            log_mem("after_trainer_prepare")

        _emit_stage(Phase.TRAINING)
        _push_status({"status": "running"})

        _total_steps = trainer.state.max_steps if trainer.state else 0

        def _log_progress() -> None:
            if not _IS_RANK0:
                return
            try:
                state = trainer.state
                if not state:
                    return
                step = int(state.global_step or 0)
                max_steps = int(state.max_steps or _total_steps or 1)
                last_log = state.log_history[-1] if state.log_history else {}
                if not isinstance(last_log, dict):
                    last_log = {}

                # `on_log` fires for train and for eval; the Trainer prefixes eval
                # entries with `eval_`. Taking `loss or eval_loss` folded the two
                # curves into one, so an evaluation point landed in the training
                # history and read as a sudden jump. Split them, exactly as the
                # single-GPU path in training_process.py does -- these payloads are
                # consumed by the same writer and the same UI, so they have to carry
                # the same keys.
                is_eval = last_log.get("eval_loss") is not None
                split = "eval" if is_eval else "train"
                if is_eval:
                    loss_raw = last_log.get("eval_loss")
                    acc = last_log.get("eval_accuracy") or last_log.get("eval_mean_token_accuracy")
                else:
                    loss_raw = last_log.get("loss")
                    acc = last_log.get("mean_token_accuracy")

                loss_val: float | None = None
                if loss_raw is not None:
                    try:
                        loss_val = float(loss_raw)
                    except (TypeError, ValueError):
                        loss_val = None

                progress = step / max_steps if max_steps else 0.0

                metric_payload = {
                    "timestamp": time.time(),
                    "step": step,
                    "total": max_steps,
                    "progress": progress,
                    "loss": loss_val,
                    "learning_rate": last_log.get("learning_rate"),
                    "epoch": last_log.get("epoch"),
                    "accuracy": acc,
                    "split": split,
                }

                # Structured metric event — the parent worker persists it to the
                # job log and the metrics history (single rpush path via the writer).
                _emit_event("metric", phase=Phase.TRAINING, data=metric_payload)

                # Coarse progress for the manager's status fields.
                _push_status({"status": "progress", **metric_payload})
            except Exception:
                pass

        from transformers import (
            TrainerCallback,
            TrainerControl,
            TrainerState,
            TrainingArguments,
        )

        class _ProgressCallback(TrainerCallback):
            def on_log(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs: object,
            ) -> None:
                _log_progress()
                if _IS_RANK0:
                    log_mem("step")

        trainer.add_callback(_ProgressCallback())

        # The memory sampler is already running (started before loading).
        trainer.train()

        # Generation check, before the save and while the trained model is still
        # the live engine -- same placement and reasoning as the single-GPU path in
        # training_process.py: eval_loss uses the same collator and the same
        # completion_only_loss as training, so it cannot detect a model that
        # converged on a degenerate rule ("answer Simon for any person"); only
        # asking it a question can.
        #
        # Without this, `generation_check_path` was silently a no-op for
        # num_gpus > 1: the multi-GPU branch of training_process.py returns as
        # soon as this subprocess exits, long before its own generation check
        # runs, so the field did nothing on exactly the path used for the models
        # where a degenerate rule costs the most to discover late.
        #
        # Runs on EVERY rank, not just rank 0. Generating drives a forward pass,
        # and under ZeRO-3 each one is a collective: a rank that skipped it would
        # leave the others blocked in an all-gather while it went on to the save
        # -- the same constraint the save below documents. Reporting is still
        # rank 0 only, which _emit_event enforces internally.
        _check_path = getattr(training_config, "generation_check_path", None)
        if _check_path:
            # Imported outside the try: an ImportError from our own package is a
            # bug worth surfacing, not something to downgrade to "check skipped",
            # and the handler below needs collective_world_size to be bound.
            from service.training.core.generation_check import (
                collective_world_size,
                load_probes,
                run_generation_check,
            )

            try:
                _verdict = run_generation_check(
                    trainer.model, components.processing_class, load_probes(_check_path)
                )
                _emit_event(EventType.INFO, msg="generation check", data=_verdict)
                if _verdict["failed"]:
                    logger.warning(
                        f"[TrainingScript] rank={_RANK} generation check: "
                        f"{_verdict['passed']}/{_verdict['total']} probes passed -- "
                        "the loss curve alone would not have shown this"
                    )
            except Exception as _check_err:
                # On one process the verdict is a report, not a gate: the adapter
                # is still worth saving, so the run carries on.
                #
                # Across ranks it has to fail. Whatever went wrong left this rank
                # outside a collective the others are still inside, so continuing
                # to save_training_results() -- itself a collective -- would strand
                # them until NCCL's watchdog fires with a timeout that names
                # neither this rank nor this cause. Raising hands it to the
                # handler below, which reports the error with is_oom set and exits
                # non-zero so the launcher tears the whole job down.
                if collective_world_size() > 1:
                    raise
                logger.warning(
                    f"[TrainingScript] rank={_RANK} generation check failed: {_check_err}"
                )
                _emit_event(
                    EventType.INFO,
                    msg="generation check skipped",
                    data={"reason": str(_check_err)},
                )

        # All ranks must participate in save (ZeRO-3 GatheredParameters requires it).
        # processing_class, not the tokenizer: an image run must save the
        # processor, or the output has no processor_config.json for inference.
        save_training_results(trainer, components.processing_class, training_config)

        # Terminal state: stdout first, then Redis backup.
        _push_status({"status": "saved"})
        logger.info(f"[TrainingScript] rank={_RANK} finished successfully")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"[TrainingScript] rank={_RANK} error: {exc}\n{tb}")
        error_str = str(exc).strip() or tb.strip().splitlines()[-1]
        is_oom = "out of memory" in error_str.lower() or (
            "cuda" in error_str.lower() and "oom" in error_str.lower()
        )
        # Terminal error state: stdout + Redis backup.
        _push_status(
            {
                "status": "error",
                "error": error_str,
                "traceback": tb,
                "is_oom": is_oom,
            }
        )
        sys.exit(1)

    finally:
        _s_thread, _s_stop = _sampler
        if _s_stop is not None:
            try:
                _s_stop.set()
                assert _s_thread is not None  # sampler returns thread+stop together  # noqa: S101
                _s_thread.join(timeout=5.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
