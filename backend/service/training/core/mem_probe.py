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
            # torch tracks the high-water mark itself, which catches transient
            # spikes between samples that polling would miss entirely -- the
            # all-gather peaks that decide whether a ZeRO-3 run fits. The
            # pipeline already called reset_peak_memory_stats() and then never
            # read these back, so the peak was being reset and discarded.
            out["cuda_alloc_peak"] = round(torch.cuda.max_memory_allocated() / (1024.0**2), 1)
            out["cuda_reserved_peak"] = round(torch.cuda.max_memory_reserved() / (1024.0**2), 1)
    except Exception:
        pass
    return out


def _disk_mb(path: str | None) -> dict:
    """
    SSD footprint: bytes this process has written, plus the offload dir's size.

    `write_bytes` from /proc/self/io is cumulative for the process, so a caller
    wanting "written during this run" should difference it against a baseline.
    The directory size is what actually matters for capacity planning of NVMe
    offload, and it is the number nothing was reporting.
    """
    out: dict = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("read_bytes", "write_bytes"):
                    out[f"io_{key}"] = round(int(rest.strip()) / (1024.0**2), 1)
    except Exception:
        pass
    if path:
        try:
            from pathlib import Path

            root = Path(path)
            if root.is_dir():
                total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
                out["offload_dir"] = round(total / (1024.0**2), 1)
        except Exception:
            pass
    return out


def probe_snapshot(offload_path: str | None = None) -> dict:
    """
    Return a lightweight structured memory snapshot (MB) for event logging.

    Pure /proc + torch.cuda — no heavy deps. The worker enriches this with the
    node-level GPU/DRAM/SSD numbers from system_monitor (see build_resource_snapshot).

    ``offload_path`` adds the on-disk size of the DeepSpeed NVMe offload
    directory, which is the SSD number that matters for capacity planning.
    """
    status = _read_proc_status_mb()
    mem = _read_meminfo_mb()
    cuda = _cuda_mb()
    disk = _disk_mb(offload_path)
    return {
        "vmrss_mb": status.get("VmRSS"),
        "vmhwm_mb": status.get("VmHWM"),
        "uss_mb": _read_uss_mb(),
        "cached_mb": mem.get("Cached"),
        "mem_available_mb": mem.get("MemAvailable"),
        "dirty_mb": mem.get("Dirty"),
        "cuda_alloc_mb": cuda.get("cuda_alloc"),
        "cuda_reserved_mb": cuda.get("cuda_reserved"),
        "cuda_alloc_peak_mb": cuda.get("cuda_alloc_peak"),
        "cuda_reserved_peak_mb": cuda.get("cuda_reserved_peak"),
        "io_read_mb": disk.get("io_read_bytes"),
        "io_write_mb": disk.get("io_write_bytes"),
        "offload_dir_mb": disk.get("offload_dir"),
    }


class MemoryAggregator:
    """
    Accumulate snapshots into per-metric average and peak.

    Sampling produced a stream of point-in-time dicts and nothing ever reduced
    them, so "how much memory did this run actually use" could not be answered
    from the recorded data at all. Averages come from the samples; peaks prefer
    a kernel- or torch-tracked high-water mark where one exists (``VmHWM`` for
    DRAM, ``max_memory_allocated`` for GPU) because those catch spikes between
    samples that polling cannot see.
    """

    # metric -> the authoritative high-water-mark field, when one exists
    _TRUE_PEAKS = {
        "vmrss_mb": "vmhwm_mb",
        "cuda_alloc_mb": "cuda_alloc_peak_mb",
        "cuda_reserved_mb": "cuda_reserved_peak_mb",
    }
    # Cumulative counters: a mean over them is meaningless, only the delta is.
    _CUMULATIVE = {"io_read_mb", "io_write_mb"}

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._maxes: dict[str, float] = {}
        self._first: dict[str, float] = {}
        self.samples = 0

    def add(self, snapshot: dict) -> None:
        """Fold one ``probe_snapshot()`` into the running totals."""
        self.samples += 1
        for key, value in snapshot.items():
            if not isinstance(value, int | float):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1
            self._maxes[key] = max(self._maxes.get(key, value), value)
            self._first.setdefault(key, value)

    def summary(self) -> dict:
        """``{metric: {"avg": x, "peak": y}}`` plus deltas for cumulative counters."""
        out: dict = {"samples": self.samples}
        if not self.samples:
            return out
        for key, total in self._sums.items():
            if key.endswith("_peak_mb") or key == "vmhwm_mb":
                continue  # reported as the peak of their base metric instead
            if key in self._CUMULATIVE:
                out[key] = {"total": round(self._maxes[key] - self._first[key], 1)}
                continue
            peak_key = self._TRUE_PEAKS.get(key)
            peak = self._maxes.get(peak_key) if peak_key else None
            out[key] = {
                "avg": round(total / self._counts[key], 1),
                "peak": round(peak if peak is not None else self._maxes[key], 1),
            }
        return out

    def format_line(self) -> str:
        """One-line human summary, for a test report or a log tail."""
        s = self.summary()
        if not s.get("samples"):
            return "[MEM] no samples"
        parts = [f"[MEM] samples={s['samples']}"]
        for label, key in (
            ("GPU", "cuda_alloc_mb"),
            ("GPUres", "cuda_reserved_mb"),
            ("DRAM", "vmrss_mb"),
        ):
            if key in s:
                parts.append(f"{label} avg={s[key]['avg']:.0f}MB peak={s[key]['peak']:.0f}MB")
        if "offload_dir_mb" in s:
            parts.append(
                f"SSD avg={s['offload_dir_mb']['avg']:.0f}MB peak={s['offload_dir_mb']['peak']:.0f}MB"
            )
        if "io_write_mb" in s:
            parts.append(f"SSDwritten={s['io_write_mb']['total']:.0f}MB")
        return "  ".join(parts)


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
    offload_path: str | None = None,
    aggregator: MemoryAggregator | None = None,
) -> tuple[threading.Thread, threading.Event]:
    """
    Start a daemon thread that samples memory every `interval` seconds.

    On each tick it logs the ``[MEMPROBE:sampler]`` line (as before) and, if
    ``on_sample`` is given, calls ``on_sample(probe_snapshot())`` so the caller
    can route a structured snapshot into the per-job event log.

    Pass an ``aggregator`` to accumulate every sample into running
    average/peak figures; ``offload_path`` adds the NVMe offload directory's
    size to each snapshot.

    Returns (thread, stop_event). Call stop_event.set() then thread.join(timeout)
    to stop. Caller is responsible for only starting this on rank 0 / main process.
    """
    stop_event = threading.Event()

    def _tick() -> None:
        log_mem("sampler")
        snapshot = None
        if on_sample is not None or aggregator is not None:
            try:
                snapshot = probe_snapshot(offload_path)
            except Exception:
                return  # sampling must never disturb training
        if aggregator is not None and snapshot is not None:
            try:
                aggregator.add(snapshot)
            except Exception:
                pass
        if on_sample is not None and snapshot is not None:
            try:
                on_sample(snapshot)
            except Exception:
                pass

    def _run() -> None:
        # Emit one immediately so we always have a baseline even for short runs.
        _tick()
        while not stop_event.wait(interval):
            _tick()

    thread = threading.Thread(target=_run, name="mem-probe-sampler", daemon=True)
    thread.start()
    return thread, stop_event
