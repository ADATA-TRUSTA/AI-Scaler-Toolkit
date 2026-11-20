"""Regressions: a load during an unload is refused, and get_status never imports torch."""

import asyncio
import importlib.abc
import sys
import threading
import time

import pytest
from fastapi import HTTPException

from service import app as app_module
from service.config_models import InferenceConfig
from service.model_manager import ModelManager, ModelUnloadingError


class _SlowUnloadWorker:
    """An engine that takes a long time to shut down."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def get_status(self):
        return {"loaded": True, "is_loading": False}

    def unload_model(self):
        self.started.set()
        self.release.wait(10)


@pytest.fixture()
def manager(monkeypatch):
    m = ModelManager()
    old = (m.inference_process, m.config, m.pending_config)
    yield m
    m.inference_process, m.config, m.pending_config = old
    m._unloading = 0


def test_a_load_during_an_unload_is_refused_without_waiting(manager, monkeypatch):
    worker = _SlowUnloadWorker()
    monkeypatch.setattr(manager, "inference_process", worker)
    unloader = threading.Thread(target=manager.unload_model)
    unloader.start()
    try:
        assert worker.started.wait(5)
        assert manager.is_unloading()

        started = time.monotonic()
        with pytest.raises(ModelUnloadingError):
            manager.prepare_config(InferenceConfig(model_name="x"))
        # It must not queue behind the unload's lock: that is the event loop stalling.
        assert time.monotonic() - started < 1.0
    finally:
        worker.release.set()
        unloader.join(10)
    assert not manager.is_unloading()


def test_the_load_endpoint_answers_409_not_already_loaded(manager, monkeypatch):
    monkeypatch.setattr(manager, "is_unloading", lambda: True)
    monkeypatch.setattr(manager, "is_loaded", lambda: True)  # still reads as loaded
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_module.load_model(InferenceConfig(model_name="x")))
    assert exc.value.status_code == 409
    assert "unload" in exc.value.detail.lower()


class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch":
            raise AssertionError("get_status imported torch on the event loop")
        return None


def test_get_status_does_not_import_torch(manager, monkeypatch):
    class Worker:
        def get_status(self):
            return {"loaded": True, "is_loading": False}

    monkeypatch.setattr(manager, "inference_process", Worker())
    for name in [n for n in sys.modules if n == "torch" or n.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_NoTorch(), *sys.meta_path])
    status = manager.get_status()
    assert "memory_usage" not in status
