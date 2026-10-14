import pytest
from pydantic import ValidationError

from service.config_models import InferenceConfig, QuantizationType, TrainingConfig
from service.model_manager import ModelManager


pytestmark = pytest.mark.models


def test_model_manager_is_singleton():
    assert ModelManager() is ModelManager()


def test_model_manager_status_uses_pending_config_while_loading():
    manager = ModelManager()
    original_process = manager.inference_process
    original_config = manager.config
    original_pending = manager.pending_config

    class FakeInferenceProcess:
        current_status = "loading"

        def get_status(self):
            return {
                "loaded": False,
                "is_loading": True,
                "loading_error": None,
                "error_type": None,
                "is_oom": False,
                "device": "cpu",
                "process_alive": True,
                "device_allocation": None,
            }

    manager.inference_process = FakeInferenceProcess()
    manager.config = None
    manager.pending_config = InferenceConfig(
        model_name="pending-model",
        quantization=QuantizationType.NONE,
        model_total_memory="1GB",
        device_map="cpu",
    )

    try:
        status = manager.get_status()
    finally:
        manager.inference_process = original_process
        manager.config = original_config
        manager.pending_config = original_pending

    assert status["loaded"] is False
    assert status["is_loading"] is True
    assert status["model_name"] == "pending-model"
    assert status["device_map"] == "cpu"


def test_training_config_requires_prompt_and_completion_together():
    with pytest.raises(ValidationError):
        TrainingConfig(
            model_name="demo-model",
            method="lora",
            dataset_path="dataset/dataset.jsonl",
            output_dir="runtime_data/fine_tuned/demo-model",
            prompt_field="prompt",
        )


def test_training_config_normalizes_optional_deepspeed_fields():
    config = TrainingConfig(
        model_name="demo-model",
        method="lora",
        dataset_path="dataset/dataset.jsonl",
        output_dir="runtime_data/fine_tuned/demo-model",
        deepspeed_profile=" zero3_cpu ",
        deepspeed_config=" ",
    )

    assert config.deepspeed_profile == "zero3_cpu"
    assert config.deepspeed_config is None
