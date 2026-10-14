"""
The fine-tune entry point must load the settings this repo actually ships.

`test_configs.py` already asserts that `run.json` names a training config that
exists on disk, but it reads the JSON directly. That left room for the loader to
resolve the same path against a different base directory and fail anyway, which
is what happened: `configs/...` was joined onto the package directory and became
`ai_scaler_console/configs/...`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_scaler_console.helpers.finetune_initializer import initialize_finetune

SHIPPED_SETTINGS = "configs/finetune/run.json"


def test_shipped_settings_load_without_a_backend() -> None:
    """`uv run examples/finetune/train.py` gets this far before it touches the network."""
    client, training_config = initialize_finetune(SHIPPED_SETTINGS)

    with client:
        expected = json.loads(Path(SHIPPED_SETTINGS).read_text(encoding="utf-8"))
        referenced = json.loads(Path(expected["finetune_config_path"]).read_text(encoding="utf-8"))
        assert training_config.model_name == referenced["model_name"]


def test_a_missing_training_config_exits_instead_of_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The examples report a bad path and stop; they do not surface a traceback."""
    settings = tmp_path / "run.json"
    settings.write_text(
        json.dumps({"finetune_config_path": "configs/finetune/does_not_exist.json"}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        initialize_finetune(str(settings))

    assert exit_info.value.code == 1
    assert "Training config file not found" in capsys.readouterr().out
