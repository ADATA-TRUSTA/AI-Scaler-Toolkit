"""
Lightweight memory attribution probe for DeepSpeed ZeRO-3 NVMe-offload runs.

Goal: distinguish *real* process resident memory (VmRSS — DeepSpeed's mandatory
NVMe swap/staging buffers) from *reclaimable* OS page cache (`Cached`, produced
by buffered NVMe offload I/O). This tells us whether the large DRAM usage seen
during full-param FT is a code bug or inherent DeepSpeed offload behaviour.

Pure stdlib reads of `/proc` — no new dependency, zero side effects on training.
`psutil` is used only if already present (for USS, the truest private-memory
number); its absence is silently tolerated.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_KIB_TO_MB = 1.0 / 1024.0  # /proc values are in kB


def _read_proc_status_mb() -> dict:
    """Return this process's VmRSS / VmHWM (peak RSS) in MB from /proc/self/status."""
    out: dict = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, val, *_ = line.split()
                    # key like 'VmRSS:'  val like '12345'(kB)
                    out[key.rstrip(":")] = round(int(val) * _KIB_TO_MB, 1)
    except Exception:
        pass
    return out


def _read_meminfo_mb() -> dict:
    """Return system-wide Cached / MemAvailable / Dirty in MB from /proc/meminfo."""
    wanted = {"Cached", "MemAvailable", "Dirty", "MemFree"}
    out: dict = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in wanted:
                    val = rest.strip().split()[0]  # kB
                    out[key] = round(int(val) * _KIB_TO_MB, 1)
    except Exception:
        pass
    return out


def _read_uss_mb() -> float | None:
    """USS (unique set size) in MB via psutil, if available; else None."""
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_full_info().uss * _KIB_TO_MB / 1024.0, 1)
    except Exception:
        return None


def _cuda_mb() -> dict:
    """Allocated / reserved CUDA memory in MB, if torch+CUDA available."""
    out: dict = {}
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_alloc"] = round(torch.cuda.memory_allocated() / (1024.0**2), 1)
            out["cuda_reserved"] = round(torch.cuda.memory_reserved() / (1024.0**2), 1)
    except Exception:
        pass
    return out


def probe_snapshot() -> dict:
    """
    Return a lightweight structured memory snapshot (MB) for event logging.

    Pure /proc + torch.cuda — no heavy deps. The worker enriches this with the
    node-level GPU/DRAM/SSD numbers from system_monitor (see build_resource_snapshot).
    """
    status = _read_proc_status_mb()
    mem = _read_meminfo_mb()
    cuda = _cuda_mb()
    return {
        "vmrss_mb": status.get("VmRSS"),
        "vmhwm_mb": status.get("VmHWM"),
        "uss_mb": _read_uss_mb(),
        "cached_mb": mem.get("Cached"),
        "mem_available_mb": mem.get("MemAvailable"),
        "dirty_mb": mem.get("Dirty"),
        "cuda_alloc_mb": cuda.get("cuda_alloc"),
        "cuda_reserved_mb": cuda.get("cuda_reserved"),
    }


def log_mem(tag: str) -> None:
    """
    Emit one INFO line snapshotting RSS vs page cache vs CUDA, labelled by `tag`.

    All values in MB. The key comparison for attribution:
      - VmRSS  = real private process memory (DeepSpeed swap/staging buffers)
      - Cached = reclaimable OS page cache (buffered NVMe offload I/O)
    """
    status = _read_proc_status_mb()
    mem = _read_meminfo_mb()
    cuda = _cuda_mb()
    uss = _read_uss_mb()

    parts = [f"[MEMPROBE:{tag}]"]
    if "VmRSS" in status:
        parts.append(f"VmRSS={status['VmRSS']:.0f}MB")
    if "VmHWM" in status:
        parts.append(f"VmHWM(peak)={status['VmHWM']:.0f}MB")
    if uss is not None:
        parts.append(f"USS={uss:.0f}MB")
    parts.extend(f"{k}={mem[k]:.0f}MB" for k in ("Cached", "MemAvailable", "Dirty") if k in mem)
    parts.extend(f"{k}={cuda[k]:.0f}MB" for k in ("cuda_alloc", "cuda_reserved") if k in cuda)

    logger.info(" ".join(parts))


def start_memory_sampler(
    interval: float = 10.0,
    on_sample: Callable[[dict], None] | None = None,
) -> tuple[threading.Thread, threading.Event]:
    """
    Start a daemon thread that samples memory every `interval` seconds.

    On each tick it logs the ``[MEMPROBE:sampler]`` line (as before) and, if
    ``on_sample`` is given, calls ``on_sample(probe_snapshot())`` so the caller
    can route a structured snapshot into the per-job event log.

    Returns (thread, stop_event). Call stop_event.set() then thread.join(timeout)
    to stop. Caller is responsible for only starting this on rank 0 / main process.
    """
    stop_event = threading.Event()

    def _tick() -> None:
        log_mem("sampler")
        if on_sample is not None:
            try:
                on_sample(probe_snapshot())
            except Exception:
                pass  # sampling must never disturb training

    def _run() -> None:
        # Emit one immediately so we always have a baseline even for short runs.
        _tick()
        while not stop_event.wait(interval):
            _tick()

    thread = threading.Thread(target=_run, name="mem-probe-sampler", daemon=True)
    thread.start()
    return thread, stop_event
