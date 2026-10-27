"""
Size cap for vLLM's disk (fs) KV offload tier.

vLLM never evicts from that tier, so the service measures it and deletes the least
recently used blocks: on a schedule while a capped vLLM model is loaded, and on
demand through /inference/kv_cache/disk/cleanup. The layout faked here is vLLM's:

    <root>/<model>_<12 hex>_r<rank>/<3 hex>/<2 hex>_g<group>/<hash>.bin
"""

import os
import time
from threading import Lock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import service.app as app_module
from service.config_models import InferenceConfig
from service.inference import kv_disk_tier
from service.inference.kv_disk_tier import (
    GIB,
    LOW_WATER_FRACTION,
    STALE_TEMP_SECONDS,
    DiskTierJanitor,
    DiskTierNotConfigured,
    disk_tier_usage,
    evict_disk_tier,
)
from service.model_manager import ModelManager

pytestmark = pytest.mark.inference

RANK = "Qwen3.5-35B_0123456789ab_r0"
BLOCK = 1000


def _write(path, size: int, when: float) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    os.utime(path, (when, when))
    return str(path)


def _block(root, block_hash: str, when: float, groups=(0,), rank: str = RANK) -> list[str]:
    """One block: the same hash under every KV cache group."""
    return [
        _write(
            root / rank / block_hash[:3] / f"{block_hash[3:5]}_g{g}" / f"{block_hash}.bin",
            BLOCK,
            when,
        )
        for g in groups
    ]


def _hash(i: int) -> str:
    return f"{i:016x}"


# --------------------------------------------------------------------------
# Measuring and evicting
# --------------------------------------------------------------------------


def test_usage_counts_blocks_across_groups_and_ignores_foreign_files(tmp_path):
    now = time.time()
    _block(tmp_path, _hash(1), now, groups=(0, 1))
    _block(tmp_path, _hash(2), now)
    _write(tmp_path / "Qwen3.5-35B_0123456789ab" / "config.json", 50, now)
    _write(tmp_path / RANK / "abc" / "de_g0" / "notes.txt", 50, now)
    _write(tmp_path / "unrelated" / "big.bin", 5000, now)

    usage = disk_tier_usage(str(tmp_path))

    assert usage == {"total_bytes": 3 * BLOCK, "blocks": 2, "block_files": 3, "temp_files": 0}


def test_usage_of_a_missing_directory_is_empty(tmp_path):
    assert disk_tier_usage(str(tmp_path / "absent"))["total_bytes"] == 0


def test_eviction_takes_the_least_recently_used_block_whole(tmp_path):
    now = time.time()
    old = _block(tmp_path, _hash(1), now - 300, groups=(0, 1))
    mid = _block(tmp_path, _hash(2), now - 200, groups=(0, 1))
    new = _block(tmp_path, _hash(3), now - 100, groups=(0, 1))

    result = evict_disk_tier(str(tmp_path), target_bytes=4 * BLOCK)

    assert not any(os.path.exists(p) for p in old)
    assert all(os.path.exists(p) for p in mid + new)
    assert result == {"deleted_files": 2, "freed_bytes": 2 * BLOCK, "remaining_bytes": 4 * BLOCK}


def test_a_recent_read_keeps_an_old_block(tmp_path):
    """atime counts as use, so a block written long ago but read lately survives."""
    now = time.time()
    read_lately = _block(tmp_path, _hash(1), now - 300)
    os.utime(read_lately[0], (now - 10, now - 300))
    untouched = _block(tmp_path, _hash(2), now - 200)

    evict_disk_tier(str(tmp_path), target_bytes=BLOCK)

    assert os.path.exists(read_lately[0])
    assert not os.path.exists(untouched[0])


def test_target_zero_empties_the_tier_but_keeps_the_run_config(tmp_path):
    now = time.time()
    for i in range(3):
        _block(tmp_path, _hash(i), now)
    run_config = _write(tmp_path / "Qwen3.5-35B_0123456789ab" / "config.json", 50, now)

    result = evict_disk_tier(str(tmp_path), target_bytes=0)

    assert result["remaining_bytes"] == 0
    assert disk_tier_usage(str(tmp_path))["blocks"] == 0
    assert os.path.exists(run_config)
    # Emptied shard directories go; the rank directory stays for vLLM to reuse.
    assert os.listdir(tmp_path / RANK) == []


def test_only_stale_temp_files_are_removed(tmp_path):
    now = time.time()
    group = tmp_path / RANK / "abc" / "de_g0"
    stale = _write(group / f"{_hash(1)}.bin_123.tmp", BLOCK, now - STALE_TEMP_SECONDS - 1)
    in_flight = _write(group / f"{_hash(2)}.bin_456.tmp", BLOCK, now - 5)

    result = evict_disk_tier(str(tmp_path), target_bytes=10 * BLOCK, now=now)

    assert not os.path.exists(stale)
    assert os.path.exists(in_flight)
    assert result["deleted_files"] == 1


def test_a_block_deleted_under_the_sweep_is_not_an_error(tmp_path, monkeypatch):
    """vLLM (or another sweep) may remove a file between the scan and the delete."""
    now = time.time()
    paths = _block(tmp_path, _hash(1), now)
    real_remove = os.remove

    def vanish_first(path):
        real_remove(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(kv_disk_tier.os, "remove", vanish_first)

    evict_disk_tier(str(tmp_path), target_bytes=0)

    assert not os.path.exists(paths[0])


# --------------------------------------------------------------------------
# Janitor
# --------------------------------------------------------------------------


def test_janitor_trims_to_the_low_water_mark_once_over_the_cap(tmp_path):
    now = time.time()
    for i in range(10):
        _block(tmp_path, _hash(i), now - 100 + i)
    janitor = DiskTierJanitor(str(tmp_path), max_bytes=8 * BLOCK, interval_s=3600)

    janitor.check_once()

    assert janitor.last_usage_bytes == int(8 * BLOCK * LOW_WATER_FRACTION) // BLOCK * BLOCK
    assert disk_tier_usage(str(tmp_path))["blocks"] == 7


def test_janitor_leaves_a_tier_under_the_cap_alone(tmp_path):
    now = time.time()
    for i in range(3):
        _block(tmp_path, _hash(i), now)
    janitor = DiskTierJanitor(str(tmp_path), max_bytes=8 * BLOCK, interval_s=3600)

    janitor.check_once()

    assert janitor.last_usage_bytes == 3 * BLOCK
    assert disk_tier_usage(str(tmp_path))["blocks"] == 3


def test_janitor_checks_on_start_and_stops(tmp_path):
    now = time.time()
    for i in range(4):
        _block(tmp_path, _hash(i), now)
    janitor = DiskTierJanitor(str(tmp_path), max_bytes=2 * BLOCK, interval_s=3600).start()
    try:
        deadline = time.time() + 5
        while janitor.last_usage_bytes is None and time.time() < deadline:
            time.sleep(0.01)
    finally:
        janitor.stop()

    assert disk_tier_usage(str(tmp_path))["total_bytes"] <= 2 * BLOCK
    assert not janitor._thread.is_alive()


def test_a_failed_sweep_does_not_end_the_janitor(tmp_path, monkeypatch):
    calls = []

    def flaky(root):
        calls.append(root)
        if len(calls) == 1:
            raise OSError("transient")
        return {"total_bytes": 0, "blocks": 0, "block_files": 0, "temp_files": 0}

    monkeypatch.setattr(kv_disk_tier, "disk_tier_usage", flaky)
    janitor = DiskTierJanitor(str(tmp_path), max_bytes=BLOCK, interval_s=0.01).start()
    try:
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        janitor.stop()

    assert len(calls) >= 2


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def _vllm_cfg(tmp_path, **overrides) -> InferenceConfig:
    fields = {
        "engine": "vllm",
        "model_name": "demo",
        "vllm_kv_offloading_size": 4,
        "vllm_kv_offloading_disk_path": str(tmp_path),
    }
    fields.update(overrides)
    return InferenceConfig(**fields)


def test_a_cap_without_a_disk_tier_is_refused():
    with pytest.raises(ValidationError, match="vllm_kv_offloading_disk_path is not set"):
        InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_disk_max_gb=10)


@pytest.mark.parametrize("bad", [0, -1])
def test_the_cap_must_be_positive(tmp_path, bad):
    with pytest.raises(ValidationError):
        _vllm_cfg(tmp_path, vllm_kv_offloading_disk_max_gb=bad)


# --------------------------------------------------------------------------
# ModelManager wiring
# --------------------------------------------------------------------------


class _Worker:
    current_status = "idle"

    def __init__(self, loaded: bool = False) -> None:
        self.loaded = loaded
        self.unloaded = False

    def is_loaded(self) -> bool:
        return self.loaded

    def get_status(self) -> dict:
        return {"loaded": self.loaded, "is_loading": False}

    def unload_model(self) -> None:
        self.unloaded = True

    def stop_process(self) -> None:
        pass


def _manager(monkeypatch, worker: _Worker | None = None) -> ModelManager:
    """A ModelManager that never spawns the worker process or writes the config file."""
    manager = object.__new__(ModelManager)
    manager.inference_process = worker or _Worker()
    manager.config = None
    manager.pending_config = None
    manager._kv_disk_janitor = None
    manager._sleeping = False
    manager._async_client = None
    manager._stale_async_clients = []
    monkeypatch.setattr(ModelManager, "_lock", Lock())
    monkeypatch.setattr(manager, "_save_config", lambda config: None)
    monkeypatch.setattr(manager, "_close_async_clients_best_effort", lambda: None)
    monkeypatch.setattr("service.model_manager.settle_managed_port", lambda config: None)
    return manager


def test_a_capped_vllm_load_starts_the_janitor_and_unload_stops_it(tmp_path, monkeypatch):
    manager = _manager(monkeypatch)
    manager.prepare_config(_vllm_cfg(tmp_path, vllm_kv_offloading_disk_max_gb=2))
    janitor = manager._kv_disk_janitor
    try:
        assert janitor is not None
        assert janitor.root == str(tmp_path)
        assert janitor.max_bytes == 2 * GIB
    finally:
        manager.inference_process.loaded = True
        manager.unload_model()

    assert manager._kv_disk_janitor is None
    assert not janitor._thread.is_alive()


def test_a_failed_load_does_not_leave_the_janitor_running(tmp_path, monkeypatch):
    """unload with nothing loaded returns early, but must still stop the janitor."""
    manager = _manager(monkeypatch)
    manager.prepare_config(_vllm_cfg(tmp_path, vllm_kv_offloading_disk_max_gb=2))
    janitor = manager._kv_disk_janitor

    manager.unload_model()

    assert manager._kv_disk_janitor is None
    assert not janitor._thread.is_alive()


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {},  # disk tier without a cap
        {"engine": "llama_server", "vllm_kv_offloading_disk_path": None},
    ],
)
def test_no_janitor_without_a_capped_vllm_tier(tmp_path, monkeypatch, config_kwargs):
    manager = _manager(monkeypatch)
    if config_kwargs.get("engine") == "llama_server":
        config = InferenceConfig(engine="llama_server", model_name="demo")
    else:
        config = _vllm_cfg(tmp_path, **config_kwargs)

    manager.prepare_config(config)

    assert manager._kv_disk_janitor is None


def test_usage_and_cleanup_work_without_a_cap(tmp_path, monkeypatch):
    now = time.time()
    for i in range(3):
        _block(tmp_path, _hash(i), now - 10 + i)
    manager = _manager(monkeypatch)
    manager.config = _vllm_cfg(tmp_path)

    usage = manager.kv_disk_usage()
    result = manager.kv_disk_cleanup(target_gb=0)

    assert usage["path"] == str(tmp_path)
    assert usage["max_gb"] is None
    assert usage["total_bytes"] == 3 * BLOCK
    assert result["remaining_bytes"] == 0


def test_cleanup_goes_through_the_janitor_and_updates_status(tmp_path, monkeypatch):
    now = time.time()
    for i in range(3):
        _block(tmp_path, _hash(i), now)
    manager = _manager(monkeypatch)
    manager.config = _vllm_cfg(tmp_path, vllm_kv_offloading_disk_max_gb=1)
    manager._kv_disk_janitor = DiskTierJanitor(str(tmp_path), GIB, 3600)  # not started

    manager.kv_disk_cleanup(target_gb=0)

    assert manager._kv_disk_janitor.last_usage_bytes == 0


@pytest.mark.parametrize(
    "config",
    [
        None,
        InferenceConfig(engine="llama_server", model_name="demo"),
        InferenceConfig(engine="vllm", model_name="demo"),
    ],
)
def test_no_disk_tier_is_reported_as_such(monkeypatch, config):
    manager = _manager(monkeypatch)
    manager.config = config

    with pytest.raises(DiskTierNotConfigured):
        manager.kv_disk_usage()
    with pytest.raises(DiskTierNotConfigured):
        manager.kv_disk_cleanup()


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

# As this machine: other machines only get inference (see test_remote_access.py).
client = TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


def _not_configured(*args, **kwargs):
    raise DiskTierNotConfigured("No vLLM model is loaded or loading")


def test_endpoints_answer_409_without_a_disk_tier(monkeypatch):
    monkeypatch.setattr(app_module.model_manager, "kv_disk_usage", _not_configured)
    monkeypatch.setattr(app_module.model_manager, "kv_disk_cleanup", _not_configured)

    assert client.get("/inference/kv_cache/disk").status_code == 409
    assert client.post("/inference/kv_cache/disk/cleanup").status_code == 409


def test_usage_endpoint_reports_the_measurement(monkeypatch):
    monkeypatch.setattr(
        app_module.model_manager,
        "kv_disk_usage",
        lambda: {
            "path": "/nvme/kv",
            "max_gb": 50.0,
            "total_bytes": 123,
            "blocks": 2,
            "block_files": 4,
            "temp_files": 0,
        },
    )

    response = client.get("/inference/kv_cache/disk")

    assert response.status_code == 200
    assert response.json()["total_bytes"] == 123
    assert response.json()["max_gb"] == 50.0


def test_cleanup_endpoint_passes_the_target_and_defaults_to_empty(monkeypatch):
    seen = []

    def cleanup(target_gb):
        seen.append(target_gb)
        return {"path": "/nvme/kv", "deleted_files": 1, "freed_bytes": 9, "remaining_bytes": 0}

    monkeypatch.setattr(app_module.model_manager, "kv_disk_cleanup", cleanup)

    assert client.post("/inference/kv_cache/disk/cleanup").status_code == 200
    assert (
        client.post("/inference/kv_cache/disk/cleanup", json={"target_gb": 20}).status_code == 200
    )
    assert seen == [0.0, 20.0]


def test_cleanup_endpoint_rejects_a_negative_target(monkeypatch):
    monkeypatch.setattr(app_module.model_manager, "kv_disk_cleanup", _not_configured)

    response = client.post("/inference/kv_cache/disk/cleanup", json={"target_gb": -1})

    assert response.status_code == 422


def test_status_reports_the_cap_and_last_measured_usage(tmp_path, monkeypatch):
    manager = _manager(monkeypatch)
    manager._last_async_error = None
    manager.config = _vllm_cfg(tmp_path, vllm_kv_offloading_disk_max_gb=5)
    janitor = DiskTierJanitor(str(tmp_path), 5 * GIB, 3600)
    janitor.last_usage_bytes = 42
    manager._kv_disk_janitor = janitor
    monkeypatch.setattr(
        manager,
        "inference_process",
        SimpleNamespace(get_status=lambda: {"loaded": True}, stop_process=lambda: None),
    )

    status = manager.get_status()

    assert status["vllm_kv_offloading_disk_max_gb"] == 5
    assert status["vllm_kv_disk_usage_bytes"] == 42
