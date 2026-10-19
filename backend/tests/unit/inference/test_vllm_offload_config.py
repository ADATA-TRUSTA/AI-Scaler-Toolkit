"""
Expert-weight offload and DRAM/NVMe KV offload: config fields, flags and preflight.

The fields map onto vLLM flags as follows:

    vllm_cpu_offload_gb           --cpu-offload-gb
    vllm_cpu_offload_params       --cpu-offload-params <segment>...
    vllm_kv_offloading_size       --kv-offloading-size + --enable-prefix-caching
    vllm_kv_offloading_disk_path  --kv-transfer-config (TieringOffloadingSpec, fs tier)

Most of what is tested here is a setup that would load and quietly do nothing:
offload params with no budget, a disk tier with no DRAM tier in front of it, or
an extra arg that silently overrides a field.
"""

import json
import os

import pytest
from pydantic import ValidationError

from service.config_models import (
    InferenceConfig,
    InferenceSharedFields,
    ModelStatus,
    VllmConfigFields,
)
from service.inference.engines import vllm_engine
from service.inference.engines.vllm_engine import VllmEngine

pytestmark = pytest.mark.inference


def _engine() -> VllmEngine:
    return VllmEngine.__new__(VllmEngine)


def _cfg(**overrides) -> InferenceConfig:
    return InferenceConfig(engine="vllm", model_name="demo", **overrides)


def _moe_cfg(tmp_path, **overrides) -> InferenceConfig:
    """The shape of the measured Qwen3.5-35B setup, pointed at a temp directory."""
    fields = {
        "vllm_cpu_offload_gb": 32,
        "vllm_cpu_offload_params": ["experts"],
        "vllm_kv_offloading_size": 30,
        "vllm_kv_offloading_disk_path": str(tmp_path / "kv"),
    }
    fields.update(overrides)
    return _cfg(**fields)


def _flag_values(args: list[str], flag: str) -> list[str]:
    """The values following ``flag``, up to the next ``--`` flag."""
    i = args.index(flag) + 1
    out = []
    while i < len(args) and not args[i].startswith("--"):
        out.append(args[i])
        i += 1
    return out


# ==================== vllm_cpu_offload_params: normalization ====================


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["experts"], ["experts"]),
        ("experts", ["experts"]),
        ("experts, shared_expert", ["experts", "shared_expert"]),
        ("experts shared_expert", ["experts", "shared_expert"]),
        ([" experts ", "", "experts", "w2_weight"], ["experts", "w2_weight"]),
    ],
)
def test_cpu_offload_params_accepts_a_list_or_a_form_string(raw, expected):
    config = _cfg(vllm_cpu_offload_gb=8, vllm_cpu_offload_params=raw)
    assert config.vllm_cpu_offload_params == expected


@pytest.mark.parametrize("empty", [None, [], "", "  ", [" ", ""]])
def test_an_empty_cpu_offload_params_means_no_restriction(empty):
    config = _cfg(vllm_cpu_offload_gb=8, vllm_cpu_offload_params=empty)
    assert config.vllm_cpu_offload_params is None


def test_a_non_list_cpu_offload_params_is_a_422():
    with pytest.raises(ValidationError, match="vllm_cpu_offload_params"):
        _cfg(vllm_cpu_offload_gb=8, vllm_cpu_offload_params={"experts": True})


def test_cpu_offload_params_without_a_budget_is_rejected():
    """It only picks which weights the budget goes to; with 0 GB nothing moves."""
    with pytest.raises(ValidationError, match="vllm_cpu_offload_gb is 0"):
        _cfg(vllm_cpu_offload_params=["experts"])


def test_an_empty_cpu_offload_params_needs_no_budget():
    assert _cfg(vllm_cpu_offload_params=[]).vllm_cpu_offload_params is None


# ==================== vllm_kv_offloading_disk_path: normalization ====================


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_disk_path_means_no_disk_tier(blank):
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=blank)
    assert config.vllm_kv_offloading_disk_path is None


def test_a_relative_disk_path_is_resolved_against_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path="kv_cache")
    assert config.vllm_kv_offloading_disk_path == str(tmp_path / "kv_cache")


def test_a_home_relative_disk_path_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path="~/kv")
    assert config.vllm_kv_offloading_disk_path == str(tmp_path / "kv")


def test_a_non_string_disk_path_is_a_422():
    with pytest.raises(ValidationError, match="vllm_kv_offloading_disk_path"):
        _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=123)


@pytest.mark.parametrize("size", [None, 0, 0.0])
def test_a_disk_tier_without_a_dram_tier_is_rejected(size):
    """Disk loads go through DRAM; with no DRAM tier vLLM builds no connector at all."""
    with pytest.raises(ValidationError, match="needs vllm_kv_offloading_size > 0"):
        _cfg(vllm_kv_offloading_size=size, vllm_kv_offloading_disk_path="/tmp/kv")


def test_the_legacy_swap_space_alias_still_sets_the_dram_tier():
    """vllm_swap_space predates the rename; a stored config must keep loading."""
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_swap_space=16)
    assert config.vllm_kv_offloading_size == 16


# ==================== conflicts with vllm_server_extra_args ====================


@pytest.mark.parametrize(
    "extra",
    [
        ["--kv-transfer-config", "{}"],
        ["--kv-transfer-config={}"],
    ],
)
def test_an_extra_kv_transfer_config_would_replace_the_disk_tier(tmp_path, extra):
    with pytest.raises(ValidationError, match="--kv-transfer-config replaces"):
        _moe_cfg(tmp_path, vllm_server_extra_args=extra)


def test_an_extra_kv_transfer_config_alone_is_still_allowed():
    """Without the disk field it is the only connector config, not a conflicting one."""
    config = _cfg(vllm_server_extra_args=["--kv-transfer-config", "{}"])
    assert config.vllm_server_extra_args == ["--kv-transfer-config", "{}"]


def test_extra_cpu_offload_params_would_override_the_field(tmp_path):
    with pytest.raises(ValidationError, match="--cpu-offload-params overrides"):
        _moe_cfg(tmp_path, vllm_server_extra_args=["--cpu-offload-params", "w2_weight"])


def test_disabling_prefix_caching_leaves_kv_offload_nothing_to_do():
    with pytest.raises(ValidationError, match="--no-enable-prefix-caching"):
        _cfg(vllm_kv_offloading_size=4, vllm_server_extra_args=["--no-enable-prefix-caching"])


def test_disabling_prefix_caching_without_kv_offload_is_allowed():
    config = _cfg(vllm_server_extra_args=["--no-enable-prefix-caching"])
    assert config.vllm_server_extra_args == ["--no-enable-prefix-caching"]


def test_every_conflict_is_reported_at_once(tmp_path):
    extra = ["--kv-transfer-config", "{}", "--cpu-offload-params", "x", "--no-enable-prefix-caching"]
    with pytest.raises(ValidationError) as err:
        _moe_cfg(tmp_path, vllm_server_extra_args=extra)
    message = str(err.value)
    assert "--kv-transfer-config" in message
    assert "--cpu-offload-params" in message
    assert "--no-enable-prefix-caching" in message


# ==================== flags ====================


def test_cpu_offload_gb_is_always_passed():
    args = _engine()._core_args(_cfg())
    assert _flag_values(args, "--cpu-offload-gb") == ["0.0"]


def test_cpu_offload_params_follow_the_budget_with_every_segment():
    config = _cfg(vllm_cpu_offload_gb=32, vllm_cpu_offload_params=["experts", "shared_expert"])
    args = _engine()._core_args(config)
    assert _flag_values(args, "--cpu-offload-gb") == ["32.0"]
    assert _flag_values(args, "--cpu-offload-params") == ["experts", "shared_expert"]


def test_no_cpu_offload_params_flag_without_the_field():
    assert "--cpu-offload-params" not in _engine()._core_args(_cfg(vllm_cpu_offload_gb=8))


@pytest.mark.parametrize("size", [None, 0.0])
def test_no_kv_offload_flags_without_a_dram_tier(size):
    args = _engine()._perf_args(_cfg(vllm_kv_offloading_size=size))
    assert "--kv-offloading-size" not in args
    assert "--enable-prefix-caching" not in args
    assert "--kv-transfer-config" not in args


def test_a_dram_tier_turns_prefix_caching_on_and_adds_no_disk_tier():
    args = _engine()._perf_args(_cfg(vllm_kv_offloading_size=12))
    assert _flag_values(args, "--kv-offloading-size") == ["12.0"]
    assert "--enable-prefix-caching" in args
    assert "--kv-transfer-config" not in args


def test_the_disk_tier_config_names_only_the_tiering_spec(tmp_path):
    """
    No kv_connector, no kv_role, no cpu_bytes_to_use: vLLM supplies those from
    --kv-offloading-size and merges them into this config. Setting them here as
    well would be a second, competing description of the same connector.
    """
    config = _moe_cfg(tmp_path)
    args = _engine()._perf_args(config)
    tier = json.loads(_flag_values(args, "--kv-transfer-config")[0])
    assert tier == {
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "secondary_tiers": [{"type": "fs", "root_dir": str(tmp_path / "kv")}],
        }
    }


def test_vllm_merges_the_two_flags_into_one_tiered_connector(tmp_path):
    """The emitted flags, run through vLLM's own merge step, give the tested setup."""
    pytest.importorskip("vllm")
    from vllm.config import CacheConfig, KVTransferConfig, VllmConfig

    config = _moe_cfg(tmp_path)
    args = _engine()._perf_args(config)
    tier = json.loads(_flag_values(args, "--kv-transfer-config")[0])
    size_gb = float(_flag_values(args, "--kv-offloading-size")[0])

    merged = object.__new__(VllmConfig)
    object.__setattr__(merged, "cache_config", CacheConfig(kv_offloading_size=size_gb))
    object.__setattr__(merged, "kv_transfer_config", KVTransferConfig(**tier))
    VllmConfig._post_init_kv_transfer_config(merged)

    ktc = merged.kv_transfer_config
    assert ktc.kv_connector == "OffloadingConnector"
    assert ktc.kv_role == "kv_both"
    extra = ktc.kv_connector_extra_config
    assert extra["spec_name"] == "TieringOffloadingSpec"
    assert extra["secondary_tiers"] == [{"type": "fs", "root_dir": str(tmp_path / "kv")}]
    assert int(extra["cpu_bytes_to_use"]) == 30 * 1024**3


def test_the_user_can_still_append_unrelated_extra_args(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_vllm_port", lambda: 8100)
    monkeypatch.setattr(engine, "_template_args", lambda config: [])
    monkeypatch.setattr(engine, "_mm_args", lambda config: [])
    config = _moe_cfg(tmp_path, vllm_server_extra_args=["--max-num-seqs", "32"])
    cmd = engine._build_server_cmd(config)
    assert cmd[-2:] == ["--max-num-seqs", "32"]
    assert "--kv-transfer-config" in cmd
    assert "--cpu-offload-params" in cmd


# ==================== preflight: the disk tier ====================


def test_preflight_does_nothing_without_a_disk_tier(monkeypatch):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
    VllmEngine._prepare_kv_disk_tier(_cfg(vllm_kv_offloading_size=4))


def test_preflight_creates_a_nested_disk_tier_directory(tmp_path):
    target = tmp_path / "a" / "b" / "kv"
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=str(target))
    VllmEngine._prepare_kv_disk_tier(config)
    assert target.is_dir()


def test_preflight_accepts_an_existing_directory(tmp_path):
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=str(tmp_path))
    VllmEngine._prepare_kv_disk_tier(config)


@pytest.mark.parametrize("value", ["1", "true", "YES", " True "])
def test_preflight_refuses_the_simple_connector_that_has_no_disk_tier(tmp_path, monkeypatch, value):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", value)
    config = _moe_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="VLLM_USE_SIMPLE_KV_OFFLOAD"):
        VllmEngine._prepare_kv_disk_tier(config)


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_preflight_allows_the_simple_connector_flag_when_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", value)
    VllmEngine._prepare_kv_disk_tier(_moe_cfg(tmp_path))


def test_preflight_rejects_a_path_that_is_a_file(tmp_path):
    blocker = tmp_path / "kv"
    blocker.write_text("not a directory")
    config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=str(blocker))
    with pytest.raises(RuntimeError, match="cannot be created"):
        VllmEngine._prepare_kv_disk_tier(config)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_preflight_rejects_an_unwritable_directory(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        config = _cfg(vllm_kv_offloading_size=4, vllm_kv_offloading_disk_path=str(locked))
        with pytest.raises(RuntimeError, match="not writable"):
            VllmEngine._prepare_kv_disk_tier(config)
    finally:
        locked.chmod(0o700)


def test_the_runtime_check_runs_the_disk_tier_preflight(tmp_path, monkeypatch):
    engine = _engine()
    for name in ("_resolve_vllm_server_dir", "_resolve_launch_prefixes", "_resolve_vllm_port"):
        monkeypatch.setattr(engine, name, lambda: None)
    monkeypatch.setattr(engine, "_repair_vllm_nvidia_runtime", lambda project_dir: None)
    config = _moe_cfg(tmp_path)
    engine._validate_runtime_environment(config)
    assert (tmp_path / "kv").is_dir()


# ==================== /inference/status carries every vLLM field ====================


def test_status_reports_every_vllm_field_as_loaded(tmp_path, monkeypatch):
    """
    get_status lists its fields by hand. A new vLLM field that is left out there
    loads correctly but can never be read back, which is exactly what a frontend
    showing the current offload settings would trip over.
    """
    from service.model_manager import ModelManager

    class _Loaded:
        def get_status(self):
            return {"loaded": True, "is_loading": False}

        def stop_process(self):
            """ModelManager tears its worker down on garbage collection."""

    # ModelManager is a process-wide singleton: __new__ hands back the instance the
    # app uses. Patched through monkeypatch so it is put back -- assigned directly,
    # the stub outlived this test and every later /health call in the run failed on
    # '_Loaded' object has no attribute 'is_loaded'.
    manager = ModelManager.__new__(ModelManager)
    monkeypatch.setattr(manager, "inference_process", _Loaded(), raising=False)
    monkeypatch.setattr(manager, "pending_config", None, raising=False)
    monkeypatch.setattr(manager, "config", _moe_cfg(tmp_path), raising=False)

    status = manager.get_status()
    for name in VllmConfigFields.model_fields:
        assert name in status, f"{name} is missing from get_status()"
        assert status[name] == getattr(manager.config, name), name
    ModelStatus(**status)


def test_the_status_shape_declares_every_vllm_field():
    missing = set(VllmConfigFields.model_fields) - set(InferenceSharedFields.model_fields)
    assert missing == set()
