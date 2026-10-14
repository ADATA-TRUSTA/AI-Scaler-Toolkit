"""
Fine-tune monitoring: a real failure must stop, a polling hiccup must retry.

The loop used to tell those apart with ``if "Training failed" in str(e)``, so
rewording a message would have silently turned a fatal error into an infinite
retry. It now dispatches on TrainingFailedError.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_scaler_console import TrainingFailedError
from ai_scaler_console.config_models import TrainingStatus
from ai_scaler_console.helpers import training_handler


class FakeClient:
    """Yields scripted training statuses; a status may instead be an exception."""

    def __init__(self, statuses: list[Any]) -> None:
        self._statuses = statuses
        self.calls = 0
        self.stopped = 0

    def get_training_status(self) -> TrainingStatus:
        index = min(self.calls, len(self._statuses) - 1)
        self.calls += 1
        item = self._statuses[index]
        if isinstance(item, Exception):
            raise item
        return TrainingStatus(**item)

    def stop_training(self) -> None:
        self.stopped += 1

    def start_training(self, config) -> None:  # noqa: ANN001 - test double
        pass


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_handler.time, "sleep", lambda _: None)


def test_completion_ends_the_monitoring_loop() -> None:
    client = FakeClient(
        [
            {"is_training": True, "current_step": 1, "total_steps": 2, "progress": 0.5},
            {"is_training": False, "current_step": 2, "total_steps": 2, "progress": 1.0},
        ]
    )

    training_handler.monitor_training_progress(client)  # type: ignore[arg-type]

    assert client.calls == 2


def test_reported_failure_raises_training_failed_error() -> None:
    client = FakeClient([{"is_training": False, "error": "loss became NaN"}])

    with pytest.raises(TrainingFailedError) as excinfo:
        training_handler.monitor_training_progress(client)  # type: ignore[arg-type]

    assert "loss became NaN" in str(excinfo.value)


def test_a_real_failure_is_not_retried() -> None:
    """The regression guard: TrainingFailedError must escape the retry branch."""
    client = FakeClient([{"is_training": False, "error": "boom"}])

    with pytest.raises(TrainingFailedError):
        training_handler.monitor_training_progress(client)  # type: ignore[arg-type]

    assert client.calls == 1, "a real failure was retried instead of raised"


def test_transient_polling_errors_are_retried() -> None:
    client = FakeClient(
        [
            ConnectionError("backend hiccup"),
            {"is_training": False, "current_step": 1, "total_steps": 1, "progress": 1.0},
        ]
    )

    training_handler.monitor_training_progress(client)  # type: ignore[arg-type]

    assert client.calls == 2, "the transient error was not retried"


def test_a_message_that_merely_mentions_training_failure_is_still_transient() -> None:
    """Dispatch is by type now, so wording cannot change control flow."""
    client = FakeClient(
        [
            RuntimeError("Training failed to reach the server"),
            {"is_training": False, "current_step": 1, "total_steps": 1, "progress": 1.0},
        ]
    )

    training_handler.monitor_training_progress(client)  # type: ignore[arg-type]

    assert client.calls == 2


def test_training_failed_is_catchable_as_ai_client_error() -> None:
    from ai_scaler_console import AIClientError

    assert issubclass(TrainingFailedError, AIClientError)


def test_status_check_passes_when_nothing_is_training() -> None:
    client = FakeClient([{"is_training": False}])

    training_handler.check_training_status(client)  # type: ignore[arg-type]

    assert client.stopped == 0


def test_force_stops_an_in_progress_run() -> None:
    client = FakeClient([{"is_training": True, "current_step": 5, "total_steps": 10}])

    training_handler.check_training_status(client, force=True)  # type: ignore[arg-type]

    assert client.stopped == 1


def test_without_force_an_in_progress_run_exits() -> None:
    client = FakeClient([{"is_training": True, "current_step": 5, "total_steps": 10}])

    with pytest.raises(SystemExit) as excinfo:
        training_handler.check_training_status(client, force=False)  # type: ignore[arg-type]

    assert excinfo.value.code == 1
    assert client.stopped == 0


def test_an_unreachable_backend_stops_here_instead_of_starting() -> None:
    """`/training/start` would fail for the same reason; saying "proceeding" misleads."""
    from ai_scaler_console import AIClientError

    client = FakeClient([AIClientError("Network error connecting to http://127.0.0.1:8000")])

    with pytest.raises(SystemExit) as excinfo:
        training_handler.check_training_status(client)  # type: ignore[arg-type]

    assert excinfo.value.code == 1


def test_a_backend_that_answered_badly_still_proceeds() -> None:
    """A 500 on the status probe says nothing about whether training can start."""
    from ai_scaler_console import AIClientError

    client = FakeClient([AIClientError("boom", status_code=500, response_content="nope")])

    training_handler.check_training_status(client)  # type: ignore[arg-type]

    assert client.stopped == 0
