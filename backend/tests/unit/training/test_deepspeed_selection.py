"""
Asking for DeepSpeed and silently not getting it must be impossible.

Three ways a run could previously end up with `use_deepspeed=True` but no
DeepSpeed, each ending in a lost adapter:

  * neither `deepspeed_config` nor `deepspeed_profile` set -> `_resolve_deepspeed_config`
    returned None and only warned
  * the named profile file missing -> same, warning only
  * the profile lookup being cwd-relative -> not found from any other directory

In all three, TrainingArguments got `deepspeed=None` so ZeRO-3 was off and the
whole model went to the GPU -- and then `save_training_results` took the ZeRO-3
gather branch anyway (it keyed off the config flag), calling
`torch.distributed.get_rank()` on an uninitialised process group. The run
trained to completion and threw the adapter away at the last step.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.config_models import TrainingConfig, TrainingMethod  # noqa: E402
from training_fixtures import training_config_kwargs  # noqa: E402
from service.training.core.model_saver import save_training_results  # noqa: E402

pytestmark = pytest.mark.finetune


def _resolved(cfg: TrainingConfig) -> str:
    from service.training.training_process import _resolve_deepspeed_config

    path = _resolve_deepspeed_config(cfg)
    assert path is not None
    return path


def _config(**overrides) -> dict:
    """Minimal valid kwargs for a lora run; shared in tests/training_fixtures.py."""
    return training_config_kwargs("lora", **overrides)


# ------------------------------------------------------------- config validation


def test_deepspeed_without_a_config_or_profile_is_rejected():
    """The caller asked for offload; giving them none is never what they meant."""
    with pytest.raises(ValueError, match="requires either 'deepspeed_config'"):
        TrainingConfig(**_config(use_deepspeed=True))


@pytest.mark.parametrize(
    "selector",
    [{"deepspeed_profile": "zero3_offload_cpu_cpu"}, {"deepspeed_config": "./my_ds.json"}],
    ids=["profile", "explicit-config"],
)
def test_deepspeed_with_either_selector_is_accepted(selector):
    cfg = TrainingConfig(**_config(use_deepspeed=True, **selector))
    assert cfg.use_deepspeed is True


def test_deepspeed_off_needs_no_selector():
    """The common case must stay frictionless."""
    assert TrainingConfig(**_config(use_deepspeed=False)).use_deepspeed is False


def test_empty_string_selector_is_still_rejected():
    """'' is normalised to None by _normalize_optional_str, so it must not satisfy the rule."""
    with pytest.raises(ValueError, match="requires either 'deepspeed_config'"):
        TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="  "))


# ------------------------------------------------------------- profile resolution


def test_missing_profile_raises_instead_of_silently_disabling_deepspeed():
    from service.training.training_process import _resolve_deepspeed_config

    cfg = TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="no_such_profile"))
    with pytest.raises(ValueError, match="not found"):
        _resolve_deepspeed_config(cfg)


def test_missing_profile_error_lists_what_is_available():
    """A wrong profile name is a typo far more often than a missing file."""
    from service.training.training_process import _resolve_deepspeed_config

    cfg = TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="zero3_offload_cpu_cpuu"))
    with pytest.raises(ValueError, match="zero3_offload_cpu_cpu"):
        _resolve_deepspeed_config(cfg)


def test_profile_lookup_is_anchored_to_the_project_root(monkeypatch, tmp_path):
    """Resolution must not depend on the worker's cwd."""
    from service.training.training_process import _resolve_deepspeed_config

    monkeypatch.chdir(tmp_path)
    cfg = TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="zero3_offload_cpu_cpu"))
    resolved = _resolve_deepspeed_config(cfg)

    assert resolved is not None, "a real profile must resolve from any working directory"
    assert Path(resolved).is_file()


def test_traversal_in_a_profile_name_is_still_rejected(tmp_path):
    from service.training.training_process import _resolve_deepspeed_config

    cfg = TrainingConfig(**_config(use_deepspeed=True, deepspeed_profile="../../etc/passwd"))
    with pytest.raises(ValueError, match="must not contain path separators"):
        _resolve_deepspeed_config(cfg)


# ------------------------------------------------------------- nvme path resolution


def test_shipped_profiles_carry_no_machine_specific_paths():
    """
    The profiles used to hardcode one developer's mount points, which went
    stale: a UUID that is no longer mounted and a renamed project directory.
    Every nvme profile then crashed at zero.Init on any other machine.
    """
    import json

    from service.settings import DEEPSPEED_NVME_PATH_SENTINEL, PROJECT_ROOT

    profiles = sorted((PROJECT_ROOT / "service" / "configs" / "deepspeed").glob("*.json"))
    assert profiles, "expected shipped deepspeed profiles"
    for path in profiles:
        zero_opt = json.loads(path.read_text()).get("zero_optimization", {})
        for section in ("offload_optimizer", "offload_param"):
            sec = zero_opt.get(section)
            if isinstance(sec, dict) and sec.get("device") == "nvme":
                assert sec.get("nvme_path") == DEEPSPEED_NVME_PATH_SENTINEL, (
                    f"{path.name}:{section} must ship the sentinel, not a real path"
                )


@pytest.mark.parametrize(
    "profile", ["zero3_offload_cpu_disk", "zero3_offload_disk_cpu", "zero3_offload_disk_disk"]
)
def test_offload_folder_wins_for_every_nvme_profile(profile, tmp_path):
    import json

    cfg = TrainingConfig(
        **_config(
            use_deepspeed=True, deepspeed_profile=profile, offload_folder=str(tmp_path / "nvme")
        )
    )
    zero_opt = json.loads(Path(_resolved(cfg)).read_text())["zero_optimization"]

    targets = {
        sec["nvme_path"]
        for key in ("offload_optimizer", "offload_param")
        if isinstance(sec := zero_opt.get(key), dict) and sec.get("device") == "nvme"
    }
    assert targets == {str((tmp_path / "nvme").resolve())}


def test_sentinel_falls_back_to_the_configured_default(monkeypatch, tmp_path):
    """With offload_folder explicitly cleared, the DEEPSPEED_NVME_DIR setting is used."""
    import json

    import service.training.training_process as tp

    monkeypatch.setattr(tp, "DEEPSPEED_NVME_DIR", str(tmp_path / "fallback"))
    cfg = TrainingConfig(
        **_config(
            use_deepspeed=True, deepspeed_profile="zero3_offload_disk_disk", offload_folder=None
        )
    )
    zero_opt = json.loads(Path(tp._resolve_deepspeed_config(cfg)).read_text())["zero_optimization"]

    assert zero_opt["offload_param"]["nvme_path"] == str((tmp_path / "fallback").resolve())


def test_a_relative_offload_folder_is_anchored_to_the_project_root(monkeypatch, tmp_path):
    """
    `offload_folder` defaults to the relative './deepspeed_offload'. Resolving
    that against the cwd would put gigabytes wherever the worker started -- and
    the resolver CLEARS that directory before each run.
    """
    import json

    from service.settings import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)
    cfg = TrainingConfig(
        **_config(
            use_deepspeed=True,
            deepspeed_profile="zero3_offload_disk_disk",
            offload_folder="./deepspeed_offload",
        )
    )
    zero_opt = json.loads(Path(_resolved(cfg)).read_text())["zero_optimization"]

    resolved = zero_opt["offload_param"]["nvme_path"]
    assert resolved == str((PROJECT_ROOT / "deepspeed_offload").resolve())
    assert not resolved.startswith(str(tmp_path)), "must not follow the cwd"


def test_an_uncreatable_offload_directory_fails_loudly(tmp_path):
    """
    This used to be a warning; DeepSpeed then repeated the same mkdir unguarded
    and the run died with a bare OSError from library internals.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")

    cfg = TrainingConfig(
        **_config(
            use_deepspeed=True,
            deepspeed_profile="zero3_offload_disk_disk",
            offload_folder=str(blocker / "child"),
        )
    )
    with pytest.raises(ValueError, match="offload_folder|DEEPSPEED_NVME_DIR"):
        _resolved(cfg)


# ------------------------------------------------------------- save path


class _RecordingTrainer:
    """Minimal trainer stand-in; records whether the standard save ran."""

    def __init__(self, deepspeed):
        self.args = SimpleNamespace(deepspeed=deepspeed)
        self.model = SimpleNamespace(parameters=lambda: iter([]))
        self.saved_standard = False

    def save_model(self) -> None:
        self.saved_standard = True


def test_save_uses_the_standard_path_when_deepspeed_never_initialised(tmp_path, caplog):
    """
    The bug: keyed off `config.use_deepspeed`, so a run that fell back to plain
    training still took the ZeRO-3 gather and died on get_rank().
    """
    cfg = TrainingConfig(
        **_config(
            use_deepspeed=True,
            deepspeed_profile="zero3_offload_cpu_cpu",
            output_dir=str(tmp_path),
            method=TrainingMethod.LORA,
            save_tokenizer=False,
        )
    )
    trainer = _RecordingTrainer(deepspeed=None)  # DeepSpeed did not actually initialise

    with caplog.at_level("WARNING"):
        save_training_results(trainer, SimpleNamespace(save_pretrained=lambda _p: None), cfg)

    assert trainer.saved_standard, "must fall back to trainer.save_model(), not the ZeRO-3 gather"
    assert "not active for this run" in caplog.text, "the fallback must be visible, not silent"
