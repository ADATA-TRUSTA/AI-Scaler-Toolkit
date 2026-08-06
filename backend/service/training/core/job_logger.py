"""
Per-job structured training log writer + reader helpers.

Each training job gets its own directory under ``LOG_DIR/training/{job_id}/``::

    events.jsonl   structured event stream (SSE source + persistent record)
    training.log   human-readable mirror
    meta.json      config snapshot + owner + final result

``JobLogWriter`` is the *single writer* (lives in the training worker process).
The FastAPI process only *reads* these files (they're on shared disk), so SSE
and history work regardless of which uvicorn worker launched the job — this is
the main enabler for future multi-worker / queue (Phase B) and RBAC (Phase C).

Design notes:
  - thread-safe: the memory-sampler thread and the main thread both emit.
  - Redis stays the primary for metrics/resources *history* (back-compat) and
    also holds a live ``training:status:{job_id}`` key for cross-process status.
  - ``owner`` is reserved (``None`` in Phase A); Phase C fills it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from ...settings import LOG_DIR
from ...utils.system_monitor import system_monitor
from .training_events import (
    PHASE_LABELS,
    TERMINAL_PHASES,
    EventType,
    Phase,
    format_event_line,
    phase_to_status,
)

logger = logging.getLogger(__name__)


def job_log_base_dir() -> Path:
    """Return the base directory holding all per-job training logs."""
    return Path(LOG_DIR) / "training"


def job_log_dir(job_id: str) -> Path:
    """Return the log directory for a single training job."""
    return job_log_base_dir() / job_id


def events_file(job_id: str) -> Path:
    """Return the path to a job's structured event stream file."""
    return job_log_dir(job_id) / "events.jsonl"


def meta_file(job_id: str) -> Path:
    """Return the path to a job's metadata file."""
    return job_log_dir(job_id) / "meta.json"


def read_events(job_id: str, since_seq: int = 0) -> list[dict[str, Any]]:
    """Return events with ``seq > since_seq`` from ``events.jsonl`` (best-effort)."""
    path = events_file(job_id)
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    # partial / mid-flush line — skip, it'll be read next time
                    continue
                if int(ev.get("seq", 0)) > since_seq:
                    out.append(ev)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[JobLog] failed reading events for %s: %s", job_id, e)
    return out


def read_meta(job_id: str) -> dict[str, Any] | None:
    """Read and return a job's metadata, or None if unavailable."""
    path = meta_file(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_terminal_event(ev: dict[str, Any]) -> bool:
    """Return True if the event marks a terminal job state."""
    if ev.get("type") == EventType.ERROR:
        return True
    if ev.get("type") == EventType.STAGE and ev.get("phase") in TERMINAL_PHASES:
        return True
    return False


def build_resource_snapshot(
    training_config: Any = None,  # noqa: ANN401 - duck-typed training config, accessed via getattr
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble a GPU/DRAM/SSD + process-memory snapshot.

    Reuses ``system_monitor`` for the rich node-level numbers and merges the
    lightweight ``/proc`` probe (VmRSS/USS/cuda) for DeepSpeed offload attribution.
    Shape of cpu/gpu/disk matches the existing ``ResourceLog`` so the history
    endpoint keeps working unchanged.
    """
    ts = time.time()
    cpu_payload: dict | None = None
    gpu_payload: dict | None = None
    disk_payload: dict | None = None

    try:
        cpu_info = system_monitor.get_cpu_resource("usage", force_by_process=True)
        if cpu_info:
            dram_payload = None
            if cpu_info.dram and cpu_info.dram.system_used_gb is not None:
                dram_payload = {
                    "total_gb": cpu_info.dram.total_gb,
                    "used_gb": cpu_info.dram.used_gb,
                }
            cpu_payload = {
                "cpu_util_percent": cpu_info.cpu_util_percent
                if cpu_info.cpu_util_percent is not None
                else 0.0,
                "dram": dram_payload,
            }
    except Exception:
        pass

    try:
        gpu_info = system_monitor.get_gpu_resource("usage", force_by_process=True)
        if gpu_info:
            gpus_payload = [
                {
                    "index": g.index,
                    "name": g.name,
                    "gpu_util": g.gpu_util if g.gpu_util is not None else 0.0,
                    "used_gb": g.used_gb if g.used_gb is not None else 0.0,
                    "total_gb": g.total_gb,
                    "temperature": g.temperature,
                }
                for g in gpu_info.gpus or []
            ]
            gpu_payload = {
                "available": bool(getattr(gpu_info, "available", False)),
                "gpus": gpus_payload,
            }
    except Exception:
        pass

    try:
        offload_path = getattr(training_config, "offload_folder", None) if training_config else None
        use_ds = (
            bool(getattr(training_config, "use_deepspeed", False)) if training_config else False
        )
        should_calc = bool(offload_path) and use_ds
        disk_info = system_monitor.get_disk_resource(
            "usage",
            path=offload_path if (offload_path and use_ds) else "/",
            calc_size=should_calc,
        )
        if disk_info and disk_info.main:
            m = disk_info.main
            disk_payload = {
                "mounts": [],
                "main": {
                    "total_gb": m.total_gb,
                    "path": m.path,
                    "percent": m.percent,
                    "read_speed_mbps": m.read_speed_mbps or 0.0,
                    "write_speed_mbps": m.write_speed_mbps or 0.0,
                    "folder_size_gb": getattr(m, "folder_size_gb", None),
                },
            }
    except Exception:
        pass

    snap: dict[str, Any] = {
        "timestamp": ts,
        "cpu": cpu_payload,
        "gpu": gpu_payload,
        "disk": disk_payload,
    }
    if probe:
        snap["proc"] = {
            "vmrss_mb": probe.get("vmrss_mb"),
            "vmhwm_mb": probe.get("vmhwm_mb"),
            "uss_mb": probe.get("uss_mb"),
            "cached_mb": probe.get("cached_mb"),
            "mem_available_mb": probe.get("mem_available_mb"),
        }
        snap["cuda"] = {
            "alloc_mb": probe.get("cuda_alloc_mb"),
            "reserved_mb": probe.get("cuda_reserved_mb"),
        }
    return snap


class JobLogWriter:
    """Single-writer, thread-safe per-job log emitter."""

    def __init__(
        self,
        job_id: str,
        base_dir: Path | None = None,
        redis_client: Any = None,  # noqa: ANN401 - optional duck-typed Redis client
        owner: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.job_id = job_id
        self.owner = owner  # reserved for Phase C (RBAC); None in Phase A
        self.base_dir = Path(base_dir) if base_dir else job_log_base_dir()
        self.dir = self.base_dir / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.log_path = self.dir / "training.log"
        self.meta_path = self.dir / "meta.json"

        self.redis_client = redis_client
        self._key_metrics = f"training:history:{job_id}:metrics"
        self._key_eval_metrics = f"training:history:{job_id}:eval_metrics"
        self._key_resources = f"training:history:{job_id}:resources"
        self._status_key = f"training:status:{job_id}"

        self._lock = threading.Lock()
        self._seq = 0
        self._cur_phase: str | None = None
        self._closed = False

        self._events_f = open(self.events_path, "a", encoding="utf-8")
        self._log_f = open(self.log_path, "a", encoding="utf-8")

        self._write_meta(status=Phase.PENDING, config=config)
        self._retention_cleanup()

    # ── internal ────────────────────────────────────────────────────────────
    def _emit(
        self,
        etype: str,
        phase: str | None = None,
        msg: str | None = None,
        data: dict | None = None,
    ) -> dict:
        with self._lock:
            self._seq += 1
            ev: dict[str, Any] = {
                "ts": time.time(),
                "seq": self._seq,
                "job_id": self.job_id,
                "type": etype,
            }
            if self.owner is not None:
                ev["owner"] = self.owner
            if phase is not None:
                ev["phase"] = phase
            if msg is not None:
                ev["msg"] = msg
            if data is not None:
                ev["data"] = data
            try:
                self._events_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._events_f.flush()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[JobLog] write events.jsonl failed: %s", e)
            try:
                self._log_f.write(format_event_line(ev) + "\n")
                self._log_f.flush()
            except Exception:
                pass
            return ev

    def _update_live_status(self, phase: str, detail: str | None) -> None:
        if not self.redis_client:
            return
        try:
            self.redis_client.set(
                self._status_key,
                json.dumps(
                    {
                        "job_id": self.job_id,
                        "phase": phase,
                        "phase_detail": detail,
                        "status": phase_to_status(phase),
                        "updated_at": time.time(),
                    }
                ),
            )
        except Exception:
            pass

    def _write_meta(
        self,
        status: str,
        config: dict | None = None,
        result: dict | None = None,
        **extra: object,
    ) -> None:
        existing: dict[str, Any] = {}
        try:
            if self.meta_path.exists():
                existing = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        meta = dict(existing)
        meta.setdefault("created_at", time.time())
        meta["job_id"] = self.job_id
        meta["owner"] = self.owner
        meta["status"] = status
        meta["updated_at"] = time.time()
        if config is not None and "config" not in meta:
            meta["config"] = config
        if result is not None:
            meta["result"] = result
        meta.update(extra)
        try:
            self.meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[JobLog] write meta.json failed: %s", e)

    def _retention_cleanup(self) -> None:
        """Opt-in: keep only the newest N job dirs (set TRAINING_LOG_RETENTION>0)."""
        try:
            keep = int(os.getenv("TRAINING_LOG_RETENTION", "0") or "0")
        except ValueError:
            keep = 0
        if keep <= 0:
            return
        try:
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            for d in dirs[keep:]:
                if d.resolve() == self.dir.resolve():
                    continue
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    # ── public API ──────────────────────────────────────────────────────────
    def stage(self, phase: str, msg: str | None = None) -> dict:
        """Emit a stage/phase transition event and update live status."""
        if msg is None:
            msg = PHASE_LABELS.get(phase, phase)
        self._cur_phase = phase
        ev = self._emit(EventType.STAGE, phase=phase, msg=msg)
        self._update_live_status(phase, msg)
        return ev

    def resource(self, snapshot: dict, phase: str | None = None) -> dict:
        """Emit a resource snapshot event and persist it to history."""
        ev = self._emit(EventType.RESOURCE, phase=phase or self._cur_phase, data=snapshot)
        if self.redis_client:
            try:
                self.redis_client.rpush(self._key_resources, json.dumps(snapshot))
            except Exception:
                pass
        return ev

    def metric(self, data: dict, phase: str | None = None) -> dict:
        """Emit a training metric event and persist valid losses to history."""
        ev = self._emit(
            EventType.METRIC, phase=phase or self._cur_phase or Phase.TRAINING, data=data
        )
        # Persist to the metrics history only when a valid loss is present,
        # matching the existing TrainingLog semantics (avoids loss=0 noise).
        # Eval metrics go to their own list so the test-set curve is retrievable
        # independently of the training curve.
        if self.redis_client and data.get("loss") is not None:
            split = data.get("split", "train")
            key = self._key_eval_metrics if split == "eval" else self._key_metrics
            try:
                self.redis_client.rpush(
                    key,
                    json.dumps(
                        {
                            "timestamp": data.get("timestamp", time.time()),
                            "step": data.get("step"),
                            "loss": data.get("loss"),
                            "learning_rate": data.get("learning_rate"),
                            "epoch": data.get("epoch"),
                            "accuracy": data.get("accuracy"),
                            "split": split,
                        }
                    ),
                )
            except Exception:
                pass
        return ev

    def info(self, msg: str, data: dict | None = None) -> dict:
        """Emit an informational event."""
        return self._emit(EventType.INFO, phase=self._cur_phase, msg=msg, data=data)

    def error(self, error: str, traceback: str = "", is_oom: bool = False) -> dict:
        """Emit an error event and update live status."""
        self._cur_phase = Phase.ERROR
        ev = self._emit(
            EventType.ERROR,
            phase=Phase.ERROR,
            msg=str(error),
            data={"error": str(error), "traceback": traceback, "is_oom": bool(is_oom)},
        )
        self._update_live_status(Phase.ERROR, str(error))
        return ev

    def finalize(self, status: str, result: dict | None = None) -> None:
        """Write the final job status and result to meta.json."""
        self._write_meta(status=status, result=result, finished_at=time.time())

    def close(self) -> None:
        """Close the underlying log file handles."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for f in (self._events_f, self._log_f):
                try:
                    f.close()
                except Exception:
                    pass
