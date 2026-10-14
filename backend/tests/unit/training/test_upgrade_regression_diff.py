"""
The upgrade harness's diff logic.

`scripts/upgrade_regression.py` compares two runs of the offload matrix across a
dependency bump. Its whole value is catching changes that still pass: offload
silently stopping, the loss curve shifting, memory doubling. A diff that misses
those is worse than none, because it grants confidence it has not earned.

These cover the comparison itself, which is pure functions over recorded JSON --
no GPU, no training.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.finetune


def _load_module():
    """Import the script by path; scripts/ is not a package."""
    path = PROJECT_ROOT / "scripts" / "upgrade_regression.py"
    spec = importlib.util.spec_from_file_location("upgrade_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_module()


def _entry(**overrides) -> dict:
    base = {
        "case": "demo",
        "status": "passed",
        "partition": {"managed": 400, "partitioned": 400, "resident": 0},
        "losses": [2.5, 2.1, 1.8],
        "memory": {
            "cuda_alloc_mb": {"avg": 1000.0, "peak": 4000.0},
            "vmrss_mb": {"avg": 5000.0, "peak": 8000.0},
        },
    }
    base.update(overrides)
    return base


def test_identical_runs_report_nothing(harness):
    assert harness._compare_case("demo", _entry(), _entry()) == []


def test_offload_silently_stopping_is_flagged(harness):
    """
    The one that matters most: everything else still passes when ZeRO-3 stops
    sharding, so only the resident count reveals it.
    """
    old = _entry()
    new = _entry(partition={"managed": 400, "partitioned": 0, "resident": 400})

    findings = harness._compare_case("demo", old, new)

    assert any("OFFLOAD" in f for f in findings)
    assert any("stopped offloading" in f for f in findings)


def test_recovering_offload_is_reported_without_the_alarm(harness):
    """resident going 400 -> 0 is a change worth seeing, but it is good news."""
    old = _entry(partition={"managed": 400, "partitioned": 0, "resident": 400})
    new = _entry()

    findings = harness._compare_case("demo", old, new)

    assert any("OFFLOAD" in f for f in findings)
    assert not any("stopped offloading" in f for f in findings)


def test_status_change_is_flagged_with_the_error(harness):
    old = _entry()
    new = _entry(status="failed", error="OutOfMemoryError: CUDA out of memory")

    findings = harness._compare_case("demo", old, new)

    assert any("STATUS" in f and "passed -> failed" in f for f in findings)
    assert any("OutOfMemoryError" in f for f in findings)


def test_a_meaningful_loss_shift_is_flagged(harness):
    """Fixed data and seed: step-1 loss moving means the pipeline changed."""
    findings = harness._compare_case("demo", _entry(), _entry(losses=[9.0, 2.1, 1.8]))
    assert any("LOSS" in f for f in findings)


def test_loss_jitter_is_not_flagged(harness):
    findings = harness._compare_case("demo", _entry(), _entry(losses=[2.6, 2.1, 1.8]))
    assert not any("LOSS" in f for f in findings)


def test_memory_regression_is_flagged_with_a_percentage(harness):
    new = _entry(
        memory={
            "cuda_alloc_mb": {"avg": 1000.0, "peak": 8000.0},  # doubled
            "vmrss_mb": {"avg": 5000.0, "peak": 8000.0},
        }
    )

    findings = harness._compare_case("demo", _entry(), new)

    gpu = [f for f in findings if "GPU peak" in f]
    assert gpu, "a doubled GPU peak must be reported"
    assert "+100%" in gpu[0]


def test_small_memory_moves_are_tolerated_as_allocator_noise(harness):
    new = _entry(
        memory={
            "cuda_alloc_mb": {"avg": 1000.0, "peak": 4300.0},  # +7.5%
            "vmrss_mb": {"avg": 5000.0, "peak": 8000.0},
        }
    )
    assert harness._compare_case("demo", _entry(), new) == []


def test_a_failed_case_still_carries_comparable_numbers(harness):
    """
    Failures record partition state and memory too -- that is what attributes a
    regression, and it is exactly when you most want the numbers.
    """
    old = _entry(status="failed", error="old boom")
    new = _entry(status="failed", error="new boom")

    # Same status, so no STATUS line; but memory/offload are still compared.
    assert harness._compare_case("demo", old, new) == []


def test_load_keeps_the_last_record_per_case(harness, tmp_path):
    """A re-run appends; the newest attempt is the one that counts."""
    path = tmp_path / "latest.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"case": "a", "status": "failed"}) + "\n")
        f.write(json.dumps({"case": "a", "status": "passed"}) + "\n")
        f.write(json.dumps({"case": "b", "status": "passed"}) + "\n")

    records = harness._load(path)

    assert records["a"]["status"] == "passed"
    assert set(records) == {"a", "b"}


def test_load_of_a_missing_file_is_empty(harness, tmp_path):
    assert harness._load(tmp_path / "nope.jsonl") == {}


def test_peak_handles_a_missing_metric(harness):
    assert harness._peak(_entry(), "offload_dir_mb") is None
    assert harness._peak(_entry(), "cuda_alloc_mb") == 4000.0
