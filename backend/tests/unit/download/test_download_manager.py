import threading
from pathlib import Path

import huggingface_hub
import pytest

import service.download_manager as download_manager_module
from service.download_manager import DownloadManager, DownloadTask

pytestmark = pytest.mark.models


def _register_task(manager: DownloadManager, *, task_id: str) -> DownloadTask:
    task = DownloadTask(
        task_id=task_id,
        model_id="unsloth/gemma-3-27b-it-GGUF",
        label="gemma_3_27b_it_gguf",
        status="pending",
        start_time=0.0,
        progress="Initialized",
    )
    manager.tasks[task_id] = task
    return task


def test_download_worker_downloads_split_gguf_parts_and_mmproj(monkeypatch, tmp_path):
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-split")

    selected_file = "BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf"
    expected_files = [
        "BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf",
        "BF16/gemma-3-27b-it-BF16-00002-of-00002.gguf",
        "BF16/mmproj-model-f16.gguf",
    ]
    calls = []
    registry_call = {}

    def _fake_hf_hub_download(
        repo_id,
        filename,
        cache_dir=None,
        token=None,
        resume_download=True,
        force_download=False,
        **kwargs,
    ):
        # **kwargs because the worker also passes tqdm_class, the progress bar it
        # aborts a transfer from when a download is cancelled. A double that lists the
        # arguments it expects fails the moment the caller passes a real one.
        calls.append(filename)
        path = tmp_path / filename.replace("/", "__")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-gguf-data")
        return str(path)

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, token=None: list(expected_files),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)
    monkeypatch.setattr(
        download_manager_module.model_registry,
        "add_llama_gguf_model",
        lambda **kwargs: registry_call.update(kwargs),
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename=selected_file,
    )

    assert task.status == "completed"
    assert calls[:3] == expected_files
    assert registry_call["label"] == task.label
    assert registry_call["base_model_name"] == task.model_id
    assert registry_call["filename"] == selected_file
    assert registry_call["local_path"].endswith("BF16__gemma-3-27b-it-BF16-00001-of-00002.gguf")


def test_download_worker_marks_failed_task_when_hf_download_errors(monkeypatch):
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-fail")

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, token=None: [])
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("invalid filename request")),
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename="BF16/",
    )

    assert task.status == "failed"
    assert "invalid filename request" in task.error


def test_cancelling_stops_before_the_next_file_and_registers_nothing(monkeypatch, tmp_path):
    """A stop asked for part way through leaves the registry alone."""
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-cancel")
    manager._cancels[task.task_id] = threading.Event()

    files = ["a.safetensors", "b.safetensors", "c.safetensors"]
    fetched = []
    registered = {}

    def _fake_hf_hub_download(repo_id, filename, **kwargs):
        fetched.append(filename)
        # Asked to stop while the first file is in flight.
        manager.cancel(task.task_id)
        path = tmp_path / filename
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, token=None: files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)
    monkeypatch.setattr(
        download_manager_module.model_registry,
        "add_base_model",
        lambda **kwargs: registered.update(kwargs),
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert task.progress == "Cancelled"
    # It stopped rather than working through the rest of the repository.
    assert fetched == ["a.safetensors"]
    # Half a repository is not a model to offer anyone.
    assert registered == {}


def test_cancelling_during_the_last_file_is_not_lost(monkeypatch, tmp_path):
    """The check after the loop, without which a late stop registered the model anyway."""
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-cancel-last")
    manager._cancels[task.task_id] = threading.Event()

    registered = {}

    def _fake_hf_hub_download(repo_id, filename, **kwargs):
        # Only the final file is interrupted, so every per-file check has passed.
        if filename == "last.safetensors":
            manager.cancel(task.task_id)
        path = tmp_path / filename
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, token=None: ["first.safetensors", "last.safetensors"],
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(
        download_manager_module.model_registry,
        "add_base_model",
        lambda **kwargs: registered.update(kwargs),
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert registered == {}


def test_a_task_reports_its_size_before_fetching_anything(monkeypatch, tmp_path):
    """The count and total that tell a one-file download from a whole repository."""
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-size")

    seen = {}

    def _fake_hf_hub_download(repo_id, filename, **kwargs):
        # By the first fetch the task must already know what it is in for.
        seen["total_files"] = task.total_files
        seen["total_bytes"] = task.total_bytes
        path = tmp_path / filename
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo_id, token=None: ["one.gguf", "two.gguf"]
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)
    monkeypatch.setattr(
        DownloadManager, "_repo_size", staticmethod(lambda model_id, files, hf_token: 2048)
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(
        download_manager_module.model_registry, "add_base_model", lambda **kwargs: None
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename=None,
    )

    assert seen == {"total_files": 2, "total_bytes": 2048}
    assert task.completed_files == 2


def _fake_cache_writer(cache_root: Path, model_id: str, on_download=None):
    """A hf_hub_download double that lays out the cache the way the real one does.

    A blob per file plus a snapshot entry pointing at it, so a test can tell what a
    cancel removed from what it left alone.
    """
    from huggingface_hub.file_download import repo_folder_name

    name = repo_folder_name(repo_id=model_id, repo_type="model")
    folder = cache_root / name
    locks = cache_root / ".locks" / name

    def _download(repo_id, filename, **kwargs):
        # huggingface_hub takes a per-file lock in a directory of its own, named
        # after the repository. Left behind, it is an empty directory carrying the
        # name of the model that was just cancelled.
        locks.mkdir(parents=True, exist_ok=True)
        (locks / f"{filename.replace('/', '__')}.lock").write_bytes(b"")
        blob = folder / "blobs" / filename.replace("/", "__")
        snapshot = folder / "snapshots" / "main" / filename
        for target in (blob, snapshot):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"weights")
        # The per-attempt fragment. The real one is unlinked in a finally, but a
        # transfer ended part way leaves one behind often enough -- a killed process
        # is all it takes -- and it is the thing a cancel most obviously must not keep.
        (folder / "blobs" / f"{filename.replace('/', '__')}.abcd1234.incomplete").write_bytes(
            b"half"
        )
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        (folder / "refs" / "main").write_text("main")
        if on_download is not None:
            on_download(filename)
        return str(snapshot)

    return folder, locks, _download


def test_cancelling_removes_what_the_download_added(monkeypatch, tmp_path):
    """Cancel means gone: nothing is registered, so nothing is left on the disk.

    What it leaves is anything that was already cached. Asking again for a model you
    already have and then cancelling must not take the copy you had.
    """
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-discard")
    manager._cancels[task.task_id] = threading.Event()

    folder, locks, download = _fake_cache_writer(
        tmp_path, task.model_id, on_download=lambda name: manager.cancel(task.task_id)
    )
    kept = folder / "blobs" / "already-here"
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"a model the user already had")

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo_id, token=None: ["a.bin", "b.bin"]
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        DownloadManager, "_repo_size", staticmethod(lambda model_id, files, hf_token: None)
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=str(tmp_path),
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert kept.read_bytes() == b"a model the user already had"
    assert not (folder / "blobs" / "a.bin").exists()
    assert not (folder / "blobs" / "a.bin.abcd1234.incomplete").exists()
    assert not (folder / "snapshots").exists()
    assert not (folder / "refs" / "main").exists()
    assert not locks.exists()


def test_cancelling_a_first_download_leaves_no_repository_behind(monkeypatch, tmp_path):
    """Nothing was there before, so afterwards the repository itself is gone.

    Without the directory pruning this passed the file assertions and still left an
    empty models--... tree in the cache for a model the user never got.
    """
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-discard-all")
    manager._cancels[task.task_id] = threading.Event()

    fetched = []

    def _after(name):
        fetched.append(name)
        if len(fetched) == 2:
            manager.cancel(task.task_id)

    folder, locks, download = _fake_cache_writer(tmp_path, task.model_id, on_download=_after)

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, token=None: ["a.bin", "b.bin", "c.bin"],
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        DownloadManager, "_repo_size", staticmethod(lambda model_id, files, hf_token: None)
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=str(tmp_path),
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert fetched == ["a.bin", "b.bin"]
    assert not folder.exists()
    assert not locks.exists()


def test_a_second_task_on_the_same_model_stops_the_cleanup(monkeypatch, tmp_path):
    """Two tasks, one repository: cancelling one must not delete the other's bytes.

    The cache is per repository, not per task, so "what this download added" cannot be
    told apart from what a concurrent one added. Leaving it alone is the safe half.
    """
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-one")
    manager._cancels[task.task_id] = threading.Event()
    other = _register_task(manager, task_id="task-two")
    other.status = "running"

    folder, locks, download = _fake_cache_writer(
        tmp_path, task.model_id, on_download=lambda name: manager.cancel(task.task_id)
    )

    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, token=None: ["a.bin"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        DownloadManager, "_repo_size", staticmethod(lambda model_id, files, hf_token: None)
    )

    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=str(tmp_path),
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert (folder / "blobs" / "a.bin").exists()
    assert locks.exists()


def test_a_stop_that_has_been_asked_for_is_visible_to_everyone(monkeypatch, tmp_path):
    """
    The status says "cancelling" from the moment the request lands.

    The button used to be driven by what one browser remembered, so a second screen --
    or the same screen after a refresh -- was still offered Stop on a task that was
    already stopping, and pressing it again was the only way to find out. Every client
    reads the status, so that is where a stop in progress has to show.
    """
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-stopping")
    task.status = "running"
    manager._cancels[task.task_id] = threading.Event()

    assert manager.cancel(task.task_id) is True
    assert task.status == "cancelling"
    assert task.progress == "Cancelling..."

    # And a second press is turned away rather than starting the work again.
    assert manager.cancel(task.task_id) is False


def test_a_stop_asked_for_while_the_task_was_still_pending_is_not_undone(monkeypatch, tmp_path):
    """The worker marks itself running; a stop from before that must survive it."""
    manager = DownloadManager()
    manager.tasks.clear()
    task = _register_task(manager, task_id="task-stop-early")
    manager._cancels[task.task_id] = threading.Event()

    registered = {}
    monkeypatch.setattr(download_manager_module, "load_hf_token", lambda: None)
    monkeypatch.setattr(
        download_manager_module.model_registry,
        "add_base_model",
        lambda **kwargs: registered.update(kwargs),
    )

    manager.cancel(task.task_id)
    manager._download_worker(
        task.task_id,
        task.model_id,
        task.label,
        cache_dir=None,
        force_download=False,
        filename=None,
    )

    assert task.status == "cancelled"
    assert registered == {}
