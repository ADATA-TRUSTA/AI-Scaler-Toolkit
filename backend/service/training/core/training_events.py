"""
Structured training-log event schema and lifecycle phases.

This is the pure-data layer for the per-job training log. It defines:
  - ``Phase``      : the fine-grained training lifecycle (queued → training → completed).
  - ``EventType``  : the kinds of events written to ``events.jsonl``.
  - helpers to map a phase to the coarse ``current_status`` and to format an
    event into a human-readable ``training.log`` line.

Replaces the previous practice of stuffing progress strings into the ``error``
channel. Everything is keyed by ``job_id`` so the queue (Phase B) and RBAC
(Phase C) layers can hang off the same spine without reworking this.
"""

from __future__ import annotations

import time


class Phase:
    """Ordered fine-grained training lifecycle phases (string values)."""

    PENDING = "pending"  # job created, not yet dispatched   (reserved, Phase B)
    QUEUED = "queued"  # waiting for a free GPU             (reserved, Phase B)
    RESOLVING_DEEPSPEED = "resolving_deepspeed"
    LAUNCHING_WORKERS = "launching_workers"  # multi-GPU deepspeed subprocess
    LOADING_TOKENIZER = "loading_tokenizer"
    LOADING_DATASET = "loading_dataset"
    LOADING_MODEL = "loading_model"
    PREPARING_TRAINER = "preparing_trainer"
    TRAINING = "training"
    SAVING = "saving"
    MERGING_LORA = "merging_lora"
    CONVERTING_GGUF = "converting_gguf"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    CANCELLED = "cancelled"  # cancelled while queued             (reserved, Phase B)


# Display order for progress UIs (terminal states excluded).
PHASE_ORDER = [
    Phase.PENDING,
    Phase.QUEUED,
    Phase.RESOLVING_DEEPSPEED,
    Phase.LAUNCHING_WORKERS,
    Phase.LOADING_TOKENIZER,
    Phase.LOADING_DATASET,
    Phase.LOADING_MODEL,
    Phase.PREPARING_TRAINER,
    Phase.TRAINING,
    Phase.SAVING,
    Phase.MERGING_LORA,
    Phase.CONVERTING_GGUF,
]

# Default human-readable label per phase.
PHASE_LABELS = {
    Phase.PENDING: "Waiting to be dispatched",
    Phase.QUEUED: "Queued, waiting for a free GPU",
    Phase.RESOLVING_DEEPSPEED: "Resolving DeepSpeed config",
    Phase.LAUNCHING_WORKERS: "Launching multi-GPU training subprocess",
    Phase.LOADING_TOKENIZER: "Loading tokenizer",
    Phase.LOADING_DATASET: "Loading dataset",
    Phase.LOADING_MODEL: "Loading model",
    Phase.PREPARING_TRAINER: "Preparing trainer",
    Phase.TRAINING: "Training",
    Phase.SAVING: "Saving training results",
    Phase.MERGING_LORA: "Merging LoRA weights",
    Phase.CONVERTING_GGUF: "Converting to GGUF / quantizing",
    Phase.COMPLETED: "Completed",
    Phase.STOPPED: "Stopped",
    Phase.ERROR: "Error",
    Phase.CANCELLED: "Cancelled",
}

# Phases that end the job — used to close SSE streams.
TERMINAL_PHASES = frozenset({Phase.COMPLETED, Phase.STOPPED, Phase.ERROR, Phase.CANCELLED})


class EventType:
    """Kinds of records in ``events.jsonl``."""

    STAGE = "stage"  # lifecycle transition
    RESOURCE = "resource"  # GPU / DRAM / SSD snapshot
    METRIC = "metric"  # trainer step metrics (loss, lr, ...)
    ERROR = "error"  # a real error / traceback
    INFO = "info"  # misc info (env snapshot, audit, ...)


# Map a fine-grained phase to the coarse legacy ``current_status`` string.
_PHASE_TO_STATUS = {
    Phase.PENDING: "starting",
    Phase.QUEUED: "starting",
    Phase.RESOLVING_DEEPSPEED: "initializing",
    Phase.LAUNCHING_WORKERS: "initializing",
    Phase.LOADING_TOKENIZER: "initializing",
    Phase.LOADING_DATASET: "initializing",
    Phase.LOADING_MODEL: "initializing",
    Phase.PREPARING_TRAINER: "initializing",
    Phase.TRAINING: "running",
    Phase.SAVING: "running",
    Phase.MERGING_LORA: "running",
    Phase.CONVERTING_GGUF: "running",
    Phase.COMPLETED: "completed",
    Phase.STOPPED: "stopped",
    Phase.ERROR: "error",
    Phase.CANCELLED: "stopped",
}


def phase_to_status(phase: str) -> str:
    """Coarse status for a phase (idle/starting/initializing/running/...)."""
    return _PHASE_TO_STATUS.get(phase, "running")


def format_event_line(ev: dict) -> str:
    """Render one event as a human-readable ``training.log`` line."""
    ts = ev.get("ts", time.time())
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    etype = str(ev.get("type", "")).upper()
    phase = ev.get("phase")
    head = f"{stamp} [{etype}]"
    if phase:
        head += f" {phase}"
    msg = ev.get("msg")
    data = ev.get("data")

    if etype == EventType.METRIC.upper() and isinstance(data, dict):
        parts = []
        # Tag the line as eval so the test-set curve is distinguishable in the
        # human-readable log; a plain training metric stays unprefixed.
        split = data.get("split", "train")
        if split == "eval":
            parts.append("[eval]" + ("[final]" if data.get("final") else ""))
        parts.extend(
            f"{k}={data[k]}"
            for k in ("step", "total", "loss", "learning_rate", "epoch", "accuracy")
            if data.get(k) is not None
        )
        return f"{head} | " + " ".join(parts)

    if etype == EventType.RESOURCE.upper() and isinstance(data, dict):
        gpu = data.get("gpu") or {}
        bits = [
            f"GPU{g.get('index')}={g.get('used_gb')}/{g.get('total_gb')}GB"
            for g in gpu.get("gpus") or []
        ]
        dram = (data.get("cpu") or {}).get("dram") or {}
        if dram.get("used_gb") is not None:
            bits.append(f"DRAM={dram.get('used_gb')}/{dram.get('total_gb')}GB")
        disk = (data.get("disk") or {}).get("main") or {}
        if disk:
            bits.append(f"SSD r/w={disk.get('read_speed_mbps')}/{disk.get('write_speed_mbps')}MB/s")
        proc = data.get("proc") or {}
        if proc.get("vmrss_mb") is not None:
            bits.append(f"VmRSS={proc.get('vmrss_mb')}MB")
        return f"{head} | " + " ".join(bits)

    line = head
    if msg:
        line += f" | {msg}"
    return line
