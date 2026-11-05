"""
What POST /config/models/download checks before it queues a task, and what it keeps.

A download used to be accepted whatever it named, failing in the background where a
caller that does not poll never looks, and every finished task stayed listed until
the service restarted.
"""

import httpx
import huggingface_hub
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import RepositoryNotFoundError

import service.app as app_module
import service.download_manager as download_manager_module
from service.download_manager import DownloadManager, DownloadTask

pytestmark = pytest.mark.models

# As this machine: see test_remote_access.py.
client = TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


@pytest.fixture
def started(monkeypatch):
    """Record start_download calls instead of downloading; the registry is empty."""
    calls = []

    def _fake_start(**kwargs):
        calls.append(kwargs)
        return "task-1"

    monkeypatch.setattr(app_module.download_manager, "start_download", _fake_start)
    monkeypatch.setattr(
        app_module.model_registry,
        "list_models",
        lambda: {"base_models": [], "finetuned_models": [], "llama_gguf_models": []},
    )
    return calls


def _hub(monkeypatch, outcome):
    """Make HfApi.repo_exists return `outcome`, or raise it when it is an exception."""

    def _repo_info(self, repo_id, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        if not outcome:
            request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
            raise RepositoryNotFoundError("404", response=httpx.Response(404, request=request))
        return object()

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", _repo_info)


def test_a_model_the_hub_does_not_have_is_refused_before_a_task_starts(monkeypatch, started):
    _hub(monkeypatch, False)

    response = client.post("/config/models/download", json={"model_id": "Qwen/Qwen9-typo"})

    assert response.status_code == 400
    assert "Qwen/Qwen9-typo" in response.json()["detail"]
    assert started == []


def test_a_malformed_model_id_is_refused(monkeypatch, started):
    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)

    response = client.post("/config/models/download", json={"model_id": "not a/valid//id"})

    assert response.status_code == 400
    assert started == []


def test_offline_the_download_is_queued_as_before(monkeypatch, started):
    _hub(monkeypatch, ConnectionError("no route to host"))

    response = client.post("/config/models/download", json={"model_id": "Qwen/Qwen3-4B"})

    assert response.status_code == 200
    assert response.json()["task_id"] == "task-1"
    assert len(started) == 1


def test_an_existing_model_is_queued(monkeypatch, started):
    _hub(monkeypatch, True)

    response = client.post("/config/models/download", json={"model_id": "Qwen/Qwen3-4B"})

    assert response.status_code == 200
    assert started[0]["label"] == "qwen3_4b"


def test_a_gguf_without_a_label_is_labelled_with_its_file_name(monkeypatch, started):
    _hub(monkeypatch, True)

    response = client.post(
        "/config/models/download",
        json={"model_id": "Qwen/Qwen3-4B-GGUF", "filename": "Q4_K_M/Qwen3-4B-Q4_K_M.gguf"},
    )

    assert response.status_code == 200
    assert response.json()["label"] == "Qwen3-4B-Q4_K_M.gguf"
    assert started[0]["filename"] == "Q4_K_M/Qwen3-4B-Q4_K_M.gguf"


def test_a_gguf_keeps_the_label_it_was_given(monkeypatch, started):
    _hub(monkeypatch, True)

    response = client.post(
        "/config/models/download",
        json={
            "model_id": "Qwen/Qwen3-4B-GGUF",
            "filename": "Qwen3-4B-Q4_K_M.gguf",
            "label": "qwen3-q4",
        },
    )

    assert response.json()["label"] == "qwen3-q4"


def _finished(task_id: str, start: float, status: str = "completed") -> DownloadTask:
    return DownloadTask(task_id=task_id, model_id="m/x", label="x", status=status, start_time=start)


def test_only_the_newest_finished_tasks_are_kept(monkeypatch):
    manager = DownloadManager()
    manager.tasks.clear()
    monkeypatch.setattr(manager, "_download_worker", lambda *args: None)
    kept = download_manager_module._FINISHED_TASKS_KEPT
    statuses = ("completed", "failed", "cancelled")
    for i in range(kept + 20):
        manager.tasks[f"old-{i}"] = _finished(f"old-{i}", float(i), statuses[i % 3])
    manager.tasks["running"] = _finished("running", -1.0, "running")

    new_id = manager.start_download(model_id="m/new", label="new")

    finished = [t for t in manager.tasks.values() if t.status in statuses]
    assert len(finished) == kept
    assert min(t.start_time for t in finished) == 20.0
    assert "running" in manager.tasks
    assert new_id in manager.tasks
    manager.tasks.clear()
