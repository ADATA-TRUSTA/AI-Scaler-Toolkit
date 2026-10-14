"""
End-to-end checks against a running backend.

Deselected by default; run with:

    uv run pytest --integration

Requires a backend reachable at BACKEND_URL. Tests that need inference skip
themselves when no model is loaded, so the transport-level checks still run
against a bare backend.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest

from ai_scaler_console import AIClient, settings
from ai_scaler_console.helpers.chat import load_chat_settings, run_prompt
from ai_scaler_console.helpers.parallel_ui import ParallelChatApp, WorkerPane
from ai_scaler_console.openai_client import create_client

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> Iterator[AIClient]:
    with AIClient(timeout=60) as ai:
        yield ai


@pytest.fixture(scope="module")
def _backend_is_up() -> None:
    try:
        httpx2.get(f"{settings.BACKEND_URL}/inference/status", timeout=10)
    except httpx2.RequestError as exc:
        pytest.skip(f"no backend at {settings.BACKEND_URL}: {exc}")


@pytest.fixture(scope="module")
def loaded_model(client: AIClient, _backend_is_up: None) -> str:
    status = client.get_status()
    if not status.loaded:
        pytest.skip("no model loaded; load one first (examples/model/load.py)")
    return status.model_name or "unknown"


def test_status_endpoint_parses(client: AIClient, _backend_is_up: None) -> None:
    status = client.get_status()

    assert isinstance(status.loaded, bool)
    assert status.engine is not None


def test_model_list_endpoint_parses(client: AIClient, _backend_is_up: None) -> None:
    models = client.get_model_list()

    assert isinstance(models.base_models, list)
    assert isinstance(models.finetuned_models, list)
    assert isinstance(models.llama_gguf_models, list)


def test_training_status_endpoint_parses(client: AIClient, _backend_is_up: None) -> None:
    assert isinstance(client.get_training_status().is_training, bool)


def test_unknown_endpoint_raises_a_client_error_with_a_status(
    client: AIClient, _backend_is_up: None
) -> None:
    """The httpx2 error mapping, exercised against a real server."""
    from ai_scaler_console import AIClientError

    with pytest.raises(AIClientError) as excinfo:
        client._get("/definitely/not/an/endpoint")

    assert excinfo.value.status_code is not None


def test_streaming_chat_returns_tokens_and_usage(
    loaded_model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_obj = load_chat_settings("configs/chat/batch.json")
    streaming = type(settings_obj)(**{**settings_obj.__dict__, "stream": True})

    run_prompt(create_client(timeout=300), streaming, "Say hello in five words.")

    out = capsys.readouterr().out
    assert "Assistant:" in out
    assert "[Stats] TTFT:" in out
    assert "[Stats] usage:" in out


def test_non_streaming_chat_returns_a_message(
    loaded_model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_obj = load_chat_settings("configs/chat/batch.json")
    non_streaming = type(settings_obj)(**{**settings_obj.__dict__, "stream": False})

    run_prompt(create_client(timeout=300), non_streaming, "Say hello in five words.")

    out = capsys.readouterr().out
    assert "Assistant:" in out
    assert "[Stats] usage:" in out


@pytest.mark.asyncio
async def test_parallel_workers_all_complete(loaded_model: str, repo_root: Path) -> None:
    chat_settings = load_chat_settings("configs/chat/parallel_8.json")
    app = ParallelChatApp(
        worker_script=repo_root / "examples" / "chat" / "parallel.py",
        settings_path="configs/chat/parallel_8.json",
        prompts=chat_settings.prompts,
        model_name=chat_settings.model_name,
    )

    async with app.run_test(size=(220, 70)) as pilot:
        for _ in range(600):
            await pilot.pause(0.5)
            statuses = [str(p._status.render()) for p in app.query(WorkerPane)]
            if statuses and all(("DONE" in s or "FAILED" in s) for s in statuses):
                break

        panes = list(app.query(WorkerPane))
        statuses = [str(p._status.render()) for p in panes]
        outputs = [str(p._output.render()) for p in panes]

    assert all("DONE" in s for s in statuses), statuses
    assert all("[Stats] TTFT:" in o for o in outputs)
    assert all("[Stats] usage:" in o for o in outputs)

    # Each pane ran its own prompt, not a shared one.
    for prompt, output in zip(chat_settings.prompts, outputs, strict=True):
        assert prompt[:30] in output

    ttfts = [float(m.group(1)) for o in outputs if (m := re.search(r"TTFT: ([\d.]+)s", o))]
    assert len(ttfts) == 8
    assert all(0 < t < 600 for t in ttfts), ttfts


def test_error_details_endpoint_parses(client: AIClient, _backend_is_up: None) -> None:
    """During a normal run the backend reports has_error=False rather than erroring."""
    details = client.get_training_error_details()

    assert isinstance(details.has_error, bool)
    if not details.has_error:
        assert details.error_traceback is None


def test_missing_job_log_is_a_404_not_a_hang(client: AIClient, _backend_is_up: None) -> None:
    """
    A job whose log directory does not exist must surface as a fatal 404.

    The SSE client treats 404 as terminal rather than a retryable drop; getting
    that wrong turns a typo'd job id into five reconnect attempts.
    """
    from ai_scaler_console import AIClientError

    with pytest.raises(AIClientError) as excinfo:
        client.get_training_log("no-such-job-abc123")

    assert excinfo.value.status_code == 404


def test_streaming_a_missing_job_log_fails_fast(client: AIClient, _backend_is_up: None) -> None:
    """Same contract on the SSE path, which opens the connection differently."""
    from ai_scaler_console import AIClientError

    with pytest.raises(AIClientError) as excinfo:
        list(client.stream_training_log("no-such-job-abc123", startup_timeout=0))

    assert excinfo.value.status_code == 404


def test_training_log_round_trips_for_a_real_job(client: AIClient, _backend_is_up: None) -> None:
    """
    Replay the most recent job's log, when the backend has one.

    Exercises the parts a mocked transport cannot: the real envelope keys, the
    real `seq` ordering, and that `cursor` advances past `since`.
    """
    status = client.get_training_status()
    job_id = status.job_id or status.session_id
    if not job_id:
        pytest.skip("backend has no current or recent training job")

    backlog = client.get_training_log(job_id)
    if not backlog.events:
        pytest.skip(f"job {job_id} has no recorded events")

    assert backlog.job_id == job_id
    assert backlog.cursor == backlog.events[-1].seq
    assert [e.seq for e in backlog.events] == sorted(e.seq for e in backlog.events)
    assert all(e.type for e in backlog.events)

    # `since` must exclude what was already delivered.
    tail = client.get_training_log(job_id, since=backlog.events[0].seq)
    assert all(e.seq > backlog.events[0].seq for e in tail.events)
