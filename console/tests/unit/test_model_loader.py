"""
Model load orchestration.

The polling loop used to raise a bare Exception and throw away
status.loading_error, so a failed load told you nothing about why. It now
raises ModelLoadFailedError carrying the backend's own message.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_scaler_console import ModelLoadFailedError
from ai_scaler_console.config_models import ModelStatus
from ai_scaler_console.helpers import model_loader


class FakeClient:
    """Returns a scripted sequence of statuses, then repeats the last one."""

    def __init__(self, statuses: list[dict[str, Any]]) -> None:
        self._statuses = statuses
        self.calls = 0
        self.unloaded = 0

    def get_status(self) -> ModelStatus:
        index = min(self.calls, len(self._statuses) - 1)
        self.calls += 1
        return ModelStatus(**self._statuses[index])

    def unload_model(self) -> None:
        self.unloaded += 1

    def load_model(self, config) -> None:  # noqa: ANN001 - test double
        pass


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loops poll on a timer; tests should not wait for it."""
    monkeypatch.setattr(model_loader.time, "sleep", lambda _: None)


def test_waits_until_loading_finishes() -> None:
    client = FakeClient(
        [
            {"is_loading": True, "loaded": False},
            {"is_loading": True, "loaded": False},
            {"is_loading": False, "loaded": False},
        ]
    )

    model_loader.check_and_wait_for_loading(client)  # type: ignore[arg-type]

    assert client.calls == 3


def test_existing_model_is_unloaded_first() -> None:
    client = FakeClient([{"loaded": True, "model_name": "old"}])

    model_loader.unload_existing_model(client)  # type: ignore[arg-type]

    assert client.unloaded == 1


def test_nothing_is_unloaded_when_no_model_is_loaded() -> None:
    client = FakeClient([{"loaded": False}])

    model_loader.unload_existing_model(client)  # type: ignore[arg-type]

    assert client.unloaded == 0


def test_monitoring_returns_once_the_model_is_loaded() -> None:
    client = FakeClient(
        [
            {"is_loading": True, "loaded": False},
            {"is_loading": False, "loaded": True},
        ]
    )

    model_loader.monitor_loading_progress(client)  # type: ignore[arg-type]

    assert client.calls == 2


def test_finished_but_not_loaded_raises_with_the_backend_message() -> None:
    client = FakeClient(
        [{"is_loading": False, "loaded": False, "loading_error": "CUDA OOM at layer 12"}]
    )

    with pytest.raises(ModelLoadFailedError) as excinfo:
        model_loader.monitor_loading_progress(client)  # type: ignore[arg-type]

    assert "CUDA OOM at layer 12" in str(excinfo.value)


def test_failure_without_a_backend_message_still_raises_clearly() -> None:
    client = FakeClient([{"is_loading": False, "loaded": False, "loading_error": None}])

    with pytest.raises(ModelLoadFailedError) as excinfo:
        model_loader.monitor_loading_progress(client)  # type: ignore[arg-type]

    assert "no error reported" in str(excinfo.value)


def test_model_load_failed_is_catchable_as_ai_client_error() -> None:
    """Existing broad handlers must keep working."""
    from ai_scaler_console import AIClientError

    assert issubclass(ModelLoadFailedError, AIClientError)


def test_unload_propagates_backend_failures() -> None:
    class Failing(FakeClient):
        def unload_model(self) -> None:
            raise RuntimeError("backend said no")

    with pytest.raises(RuntimeError, match="backend said no"):
        model_loader.unload_model(Failing([{"loaded": True}]))  # type: ignore[arg-type]
