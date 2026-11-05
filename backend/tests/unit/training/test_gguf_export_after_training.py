"""
A failed GGUF export after a successful training run says the weights are saved.

The single-GPU path used to let the export's exception fall through to the generic
handler, so a run whose weights were on disk was reported as a failed training --
and the alternative a user reaches for then is hours of training again. Both paths
now finish through _finish_with_gguf_export.
"""

import queue

import pytest

import service.training.training_process as training_process_module
from service.config_models import TrainingConfig
from service.training.core.training_events import Phase

pytestmark = pytest.mark.finetune


class _Writer:
    """Records what the job log would have been told."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.finalized: list[tuple[str, dict | None]] = []

    def error(self, message: str, **kwargs) -> None:
        self.errors.append(message)

    def finalize(self, phase: str, result: dict | None = None) -> None:
        self.finalized.append((phase, result))


def _config() -> TrainingConfig:
    return TrainingConfig(
        model_name="demo-model",
        method="lora",
        dataset_path="dataset/dataset.jsonl",
        output_dir="runtime_data/fine_tuned/demo-model",
    )


def _finish(monkeypatch, convert):
    monkeypatch.setattr(training_process_module, "_convert_training_output_to_q4_k_m", convert)
    writer, status_q, stages = _Writer(), queue.Queue(), []
    training_process_module._finish_with_gguf_export(
        _config(),
        writer=writer,  # type: ignore[arg-type]
        status_q=status_q,  # type: ignore[arg-type]
        emit_stage=lambda phase, detail=None: stages.append(phase),
    )
    return writer, status_q.get_nowait(), stages


def test_a_failed_export_says_the_weights_are_saved_and_what_to_run(monkeypatch):
    def _convert(config, status_callback=None):
        status_callback("convert to gguf")
        raise RuntimeError("convert_hf_to_gguf.py not found")

    writer, status, stages = _finish(monkeypatch, _convert)

    assert status["status"] == "error"
    assert "weights were saved to runtime_data/fine_tuned/demo-model" in status["error"]
    assert "convert_hf_to_gguf.py not found" in status["error"]
    assert "POST /config/models/convert" in status["error"]
    assert writer.errors == [status["error"]]
    assert writer.finalized == [(Phase.ERROR, None)]
    assert stages == [Phase.CONVERTING_GGUF]


def test_a_successful_export_completes_with_the_gguf_paths(monkeypatch):
    def _convert(config, status_callback=None):
        status_callback("merge fine-tune")
        return {
            "quantized_output_path": "demo-q4.gguf",
            "quantization_type": "Q4_K_M",
            "mmproj_output_path": None,
        }

    writer, status, stages = _finish(monkeypatch, _convert)

    expected = {"gguf_path": "demo-q4.gguf", "gguf_quantization": "Q4_K_M", "mmproj_path": None}
    assert status == {"status": "completed", **expected}
    assert writer.finalized == [(Phase.COMPLETED, expected)]
    assert stages == [Phase.MERGING_LORA, Phase.COMPLETED]
