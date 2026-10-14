"""
Structured training log: SSE parsing, resume, and terminal handling.

The wire format is pinned against service/app.py's stream_training_log(): each
frame is `id: <seq>` / `event: <type>` / `data: <json>`, idle connections get a
bare `: keepalive` comment, and the stream closes after a terminal event.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx2
import pytest

from ai_scaler_console.client import AIClient
from ai_scaler_console.config_models import TrainingEvent, TrainingEventType, TrainingPhase
from ai_scaler_console.exceptions import AIClientError

BASE_URL = "http://backend.test"


def sse_frame(seq: int, etype: str, **fields: object) -> str:
    """Build one SSE frame exactly as the backend emits it."""
    payload = {"seq": seq, "ts": 1750640000.0, "job_id": "job1", "type": etype, **fields}
    return f"id: {seq}\nevent: {etype}\ndata: {json.dumps(payload)}\n\n"


def client_for(handler: object) -> AIClient:
    """An AIClient whose transport is the given MockTransport handler."""
    client = AIClient(base_url=BASE_URL)
    client.session = httpx2.Client(transport=httpx2.MockTransport(handler))  # type: ignore[arg-type]
    return client


def test_stream_parses_frames_and_stops_at_the_terminal_event() -> None:
    body = (
        sse_frame(1, "stage", phase="queued")
        + sse_frame(2, "metric", data={"step": 10, "total": 100, "loss": 1.23, "split": "train"})
        + sse_frame(3, "stage", phase="completed")
        # The backend closes after a terminal event; anything past it must be ignored.
        + sse_frame(4, "stage", phase="training")
    )

    with client_for(lambda request: httpx2.Response(200, text=body)) as client:
        events = list(client.stream_training_log("job1"))

    assert [e.seq for e in events] == [1, 2, 3]
    assert events[-1].is_terminal
    assert events[1].type == TrainingEventType.METRIC


def test_stream_skips_keepalive_comment_frames() -> None:
    """`: keepalive` carries no event; a naive parser would emit a bogus record."""
    body = (
        ": keepalive\n\n"
        + sse_frame(1, "stage", phase="training")
        + ": keepalive\n\n"
        + sse_frame(2, "stage", phase="completed")
    )

    with client_for(lambda request: httpx2.Response(200, text=body)) as client:
        events = list(client.stream_training_log("job1"))

    assert [e.seq for e in events] == [1, 2]


def test_stream_treats_404_as_fatal_and_does_not_retry() -> None:
    """
    A missing job log directory is not a retryable drop.

    The body is an iterator, not bytes, so the response is a real unbuffered
    stream. With a buffered body this passes even when the error body is read
    after the stream context closes — which raises StreamClosed against a live
    backend and masks the 404.
    """
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(
            404,
            headers={"Content-Type": "application/json"},
            content=iter([b'{"detail": "Training job log not found"}']),
        )

    with client_for(handler) as client, pytest.raises(AIClientError) as excinfo:
        list(client.stream_training_log("job1", startup_timeout=0))

    assert excinfo.value.status_code == 404
    assert len(calls) == 1


def test_stream_resumes_from_the_last_seq_after_a_drop() -> None:
    """A mid-stream network error reconnects with Last-Event-ID, without replaying."""
    attempts: list[str | None] = []

    body_first = sse_frame(1, "stage", phase="training") + sse_frame(2, "metric", data={"step": 5})

    def flaky(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request.headers.get("last-event-id"))
        if len(attempts) == 1:
            # Deliver two frames, then fail while the caller is still reading.
            return httpx2.Response(200, content=_break_after(body_first))
        return httpx2.Response(200, text=sse_frame(3, "stage", phase="completed"))

    with client_for(flaky) as client:
        events = list(client.stream_training_log("job1"))

    assert [e.seq for e in events] == [1, 2, 3]
    # Second attempt must resume past what was already delivered.
    assert attempts == [None, "2"]


def _break_after(body: str) -> Iterator[bytes]:
    """A byte stream that yields the body, then fails as a dropped connection."""

    def gen() -> Iterator[bytes]:
        yield body.encode()
        raise httpx2.ReadError("connection dropped")

    return gen()


def test_stream_gives_up_after_max_reconnects() -> None:
    def always_drops(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadError("connection dropped")

    with client_for(always_drops) as client, pytest.raises(AIClientError, match="reconnect"):
        list(client.stream_training_log("job1", max_reconnects=2))


def test_backlog_endpoint_parses_events_and_cursor() -> None:
    payload = {
        "job_id": "job1",
        "cursor": 2,
        "events": [
            {"seq": 1, "type": "stage", "phase": "training"},
            {"seq": 2, "type": "metric", "data": {"step": 1, "split": "eval", "final": True}},
        ],
    }

    with client_for(lambda request: httpx2.Response(200, json=payload)) as client:
        result = client.get_training_log("job1", since=0)

    assert result.cursor == 2
    assert result.events[1].is_eval_metric


def test_error_details_reports_a_clean_run() -> None:
    payload = {"has_error": False, "message": "No error occurred"}

    with client_for(lambda request: httpx2.Response(200, json=payload)) as client:
        details = client.get_training_error_details()

    assert details.has_error is False
    assert details.error_traceback is None


@pytest.mark.parametrize(
    ("etype", "phase", "terminal"),
    [
        ("stage", TrainingPhase.TRAINING, False),
        ("stage", TrainingPhase.COMPLETED, True),
        ("stage", TrainingPhase.STOPPED, True),
        ("stage", TrainingPhase.CANCELLED, True),
        ("error", None, True),
        ("metric", None, False),
    ],
)
def test_terminal_detection_matches_the_backend(
    etype: str, phase: str | None, terminal: bool
) -> None:
    """Mirrors job_logger.is_terminal_event()."""
    assert TrainingEvent(seq=1, type=etype, phase=phase).is_terminal is terminal


def test_unknown_event_kind_does_not_break_the_stream() -> None:
    """A future event type must pass through rather than fail the whole run."""
    body = sse_frame(1, "audit", msg="something new") + sse_frame(2, "stage", phase="completed")

    with client_for(lambda request: httpx2.Response(200, text=body)) as client:
        events = list(client.stream_training_log("job1"))

    assert [e.type for e in events] == ["audit", "stage"]


def test_stream_waits_out_the_startup_404_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `/training/start` returns the job id before the worker creates the log dir.

    Attaching immediately therefore gets a 404 for a second or two. Treating that
    as fatal aborts monitoring the moment a real training run starts.
    """
    monkeypatch.setattr("ai_scaler_console.client.time.sleep", lambda _: None)
    calls: list[int] = []

    def slow_to_appear(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx2.Response(404, content=iter([b'{"detail": "not found"}']))
        return httpx2.Response(200, text=sse_frame(1, "stage", phase="completed"))

    with client_for(slow_to_appear) as client:
        events = list(client.stream_training_log("job1", startup_timeout=30))

    assert [e.seq for e in events] == [1]
    assert len(calls) == 3


def test_stream_stops_waiting_once_events_have_arrived(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first event a 404 is real (the log was deleted), not startup lag."""
    monkeypatch.setattr("ai_scaler_console.client.time.sleep", lambda _: None)
    calls: list[int] = []

    def gone_midway(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(
                200, content=_break_after(sse_frame(1, "stage", phase="training"))
            )
        return httpx2.Response(404, content=iter([b'{"detail": "not found"}']))

    with client_for(gone_midway) as client, pytest.raises(AIClientError) as excinfo:
        list(client.stream_training_log("job1", startup_timeout=30))

    assert excinfo.value.status_code == 404
    assert len(calls) == 2
