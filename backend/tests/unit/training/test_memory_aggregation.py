"""
Runs must record how much memory they actually used.

The sampler produced a stream of point-in-time snapshots and nothing ever
reduced them, so "average and peak GPU/DRAM/SSD for this run" could not be
answered from the recorded data. Worse, `reset_peak_memory_stats()` was called
and `max_memory_allocated()` never read -- the GPU high-water mark was reset and
then discarded -- and there was no disk metric at all.

Peaks prefer a kernel- or torch-tracked high-water mark over the sampled
maximum, because a spike between two samples is invisible to polling and those
spikes (ZeRO-3 all-gathers) are exactly what decides whether a run fits.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.mem_probe import MemoryAggregator, probe_snapshot  # noqa: E402

pytestmark = pytest.mark.finetune


def test_empty_aggregator_reports_no_samples():
    assert MemoryAggregator().summary() == {"samples": 0}


def test_average_and_peak_are_both_reported():
    agg = MemoryAggregator()
    for value in (100.0, 300.0, 200.0):
        agg.add({"vmrss_mb": value})

    summary = agg.summary()

    assert summary["samples"] == 3
    assert summary["vmrss_mb"]["avg"] == 200.0
    assert summary["vmrss_mb"]["peak"] == 300.0


def test_kernel_high_water_mark_wins_over_the_sampled_maximum():
    """
    VmHWM is the kernel's own peak RSS. If a spike happened between samples,
    the sampled max understates it -- and understating peak memory is how a run
    looks like it fits when it does not.
    """
    agg = MemoryAggregator()
    agg.add({"vmrss_mb": 100.0, "vmhwm_mb": 900.0})
    agg.add({"vmrss_mb": 200.0, "vmhwm_mb": 900.0})

    summary = agg.summary()

    assert summary["vmrss_mb"]["avg"] == 150.0
    assert summary["vmrss_mb"]["peak"] == 900.0, "must use the kernel's peak, not max(samples)"
    assert "vmhwm_mb" not in summary, "the raw peak field is folded in, not reported twice"


def test_torch_gpu_peak_wins_over_the_sampled_maximum():
    agg = MemoryAggregator()
    agg.add({"cuda_alloc_mb": 1000.0, "cuda_alloc_peak_mb": 7000.0})
    agg.add({"cuda_alloc_mb": 2000.0, "cuda_alloc_peak_mb": 7000.0})

    summary = agg.summary()

    assert summary["cuda_alloc_mb"]["avg"] == 1500.0
    assert summary["cuda_alloc_mb"]["peak"] == 7000.0
    assert "cuda_alloc_peak_mb" not in summary


def test_cumulative_io_counters_report_a_delta_not_a_mean():
    """
    /proc/self/io counters only grow. Averaging them is meaningless; what the
    caller wants is how much this run wrote.
    """
    agg = MemoryAggregator()
    agg.add({"io_write_mb": 500.0})
    agg.add({"io_write_mb": 900.0})
    agg.add({"io_write_mb": 1700.0})

    summary = agg.summary()

    assert summary["io_write_mb"] == {"total": 1200.0}
    assert "avg" not in summary["io_write_mb"]


def test_non_numeric_fields_are_ignored():
    """probe_snapshot yields None for anything unavailable on this platform."""
    agg = MemoryAggregator()
    agg.add({"vmrss_mb": 100.0, "uss_mb": None, "note": "text"})

    summary = agg.summary()

    assert summary["vmrss_mb"]["avg"] == 100.0
    assert "uss_mb" not in summary
    assert "note" not in summary


def test_snapshot_reports_gpu_dram_and_ssd_fields():
    """The three the user asked for must all be present as keys."""
    snapshot = probe_snapshot()

    for field in ("vmrss_mb", "cached_mb", "mem_available_mb"):
        assert field in snapshot, f"DRAM field {field} missing"
    for field in ("cuda_alloc_mb", "cuda_alloc_peak_mb", "cuda_reserved_peak_mb"):
        assert field in snapshot, f"GPU field {field} missing"
    for field in ("io_read_mb", "io_write_mb"):
        assert field in snapshot, f"SSD field {field} missing"


def test_snapshot_measures_the_offload_directory(tmp_path):
    """The SSD number that matters for NVMe offload is the directory's size."""
    (tmp_path / "swap").mkdir()
    (tmp_path / "swap" / "chunk.bin").write_bytes(b"\0" * (3 * 1024 * 1024))

    snapshot = probe_snapshot(offload_path=str(tmp_path))

    assert snapshot["offload_dir_mb"] == pytest.approx(3.0, abs=0.2)


def test_snapshot_without_an_offload_path_omits_the_directory_size():
    assert probe_snapshot()["offload_dir_mb"] is None


def test_format_line_mentions_each_resource():
    agg = MemoryAggregator()
    agg.add(
        {
            "vmrss_mb": 100.0,
            "vmhwm_mb": 150.0,
            "cuda_alloc_mb": 2000.0,
            "cuda_alloc_peak_mb": 3000.0,
            "offload_dir_mb": 500.0,
            "io_write_mb": 10.0,
        }
    )
    agg.add({"io_write_mb": 60.0})

    line = agg.format_line()

    assert "GPU avg=" in line and "peak=3000MB" in line
    assert "DRAM avg=" in line
    assert "SSD avg=" in line
    assert "SSDwritten=50MB" in line


def test_sampler_feeds_the_aggregator():
    """End to end: the thread must actually accumulate."""
    from service.training.core.mem_probe import start_memory_sampler

    agg = MemoryAggregator()
    thread, stop = start_memory_sampler(interval=0.05, aggregator=agg)
    try:
        deadline = __import__("time").time() + 1.0
        while agg.samples < 2 and __import__("time").time() < deadline:
            __import__("time").sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert agg.samples >= 2, "sampler should have accumulated several snapshots"
    assert "vmrss_mb" in agg.summary()
