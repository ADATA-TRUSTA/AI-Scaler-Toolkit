import json

from fastapi.testclient import TestClient

import service.app as app_module
from service import platform_support
from service.config_models import TrainingConfig, TrainingStatus
import service.training.training_process as training_process_module


client = TestClient(app_module.app)


import pytest

pytestmark = pytest.mark.finetune


@pytest.fixture(autouse=True)
def _on_linux(monkeypatch):
    """
    Report Linux for every test in this module.

    /training/start refuses a non-Linux host with 501 (see service/platform_support.py),
    and these tests are about the request contract rather than the platform — pinning
    it keeps them asserting the same thing whichever host runs the suite. The refusal
    itself is covered in tests/unit/test_platform_support.py.
    """
    monkeypatch.setattr(platform_support.platform, "system", lambda: "Linux")


def _training_payload(**overrides):
    payload = {
        "model_name": "demo-model",
        "method": "lora",
        "dataset_path": "dataset/dataset.jsonl",
        "output_dir": "runtime_data/fine_tuned/demo-model",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
    }
    payload.update(overrides)
    return payload


def test_training_start_returns_current_contract(monkeypatch):
    captured = {}

    def _fake_start_training(config):
        captured["config"] = config
        return {"job_id": "train-123"}

    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    monkeypatch.setattr(
        app_module.training_manager,
        "start_training",
        _fake_start_training,
    )

    response = client.post("/training/start", json=_training_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Training started"
    assert data["result"] == {"job_id": "train-123"}
    assert captured["config"].model_name == "demo-model"


def test_training_start_accepts_num_gpu_alias(monkeypatch):
    captured = {}

    def _fake_start_training(config):
        captured["config"] = config
        return {"job_id": "train-123"}

    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    monkeypatch.setattr(
        app_module.training_manager,
        "start_training",
        _fake_start_training,
    )

    response = client.post("/training/start", json=_training_payload(num_gpu=2))

    assert response.status_code == 200
    assert captured["config"].num_gpus == 2
    assert response.json()["config"]["num_gpus"] == 2


def test_training_status_returns_serialized_status(monkeypatch):
    config = TrainingConfig(**_training_payload())
    status = TrainingStatus(
        is_training=True,
        progress=0.5,
        current_step=10,
        total_steps=20,
        loss=0.123,
        current_epoch=1.0,
        total_epochs=3,
        status="running",
        session_id="session-1",
        error=None,
        config=config,
    )

    monkeypatch.setattr(app_module.training_manager, "get_status", lambda: status)

    response = client.get("/training/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_training"] is True
    assert data["current_step"] == 10
    assert data["config"]["model_name"] == "demo-model"


def test_training_stop_delegates_to_manager(monkeypatch):
    monkeypatch.setattr(
        app_module.training_manager,
        "stop_training",
        lambda: {"stopped": True},
    )

    response = client.post("/training/stop")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"] == {"stopped": True}


def test_training_force_cleanup_gpu_resets_worker(monkeypatch):
    called = {"cleanup": 0}

    def _fake_cleanup():
        called["cleanup"] += 1

    monkeypatch.setattr(
        training_process_module.training_process_manager,
        "cleanup",
        _fake_cleanup,
    )

    response = client.post("/training/force_cleanup_gpu")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert called["cleanup"] == 1


def test_delete_model_succeeds_when_files_already_missing(monkeypatch, tmp_path):
    """
    Deleting a model whose files are already gone still removes it.

    Files are deleted in the background: the request answers 202 with the entry
    marked "deleting", and the entry leaves the registry only once the worker is
    done. A path that no longer exists counts as done, not as a failure -- the
    files being gone is the outcome the delete was asked for.
    """
    missing_path = tmp_path / "fine_tuned" / "missing-model"
    registry_path = tmp_path / "models_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "base_models": [],
                "finetuned_models": [
                    {
                        "base_model_name": "Qwen/Qwen3-4B",
                        "method": "lora",
                        "output_dir": str(missing_path),
                        "label": "missing-model",
                    }
                ],
                "llama_gguf_models": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module.model_registry, "path", registry_path)

    # TestClient runs background tasks before returning, so the worker has
    # already finished by the time the response is inspected.
    response = client.delete("/config/models/missing-model", params={"delete_files": True})

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "deleting"
    assert data["label"] == "missing-model"
    assert data["path"] == str(missing_path)
    assert json.loads(registry_path.read_text(encoding="utf-8"))["finetuned_models"] == []


def test_delete_model_accepts_normalized_finetuned_output_dir(monkeypatch, tmp_path):
    registry_path = tmp_path / "models_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "base_models": [],
                "finetuned_models": [
                    {
                        "base_model_name": "Qwen/Qwen3-4B",
                        "method": "lora",
                        "output_dir": "./finetune_output/qwen3_4b_lora",
                        "label": "./finetune_output/qwen3_4b_lora",
                    }
                ],
                "llama_gguf_models": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module.model_registry, "path", registry_path)

    response = client.delete("/config/models/finetune_output/qwen3_4b_lora")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["label"] == "finetune_output/qwen3_4b_lora"
    assert data["registry_removed"] is True

    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert updated_registry["finetuned_models"] == []
