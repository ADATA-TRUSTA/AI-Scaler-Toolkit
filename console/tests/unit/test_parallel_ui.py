"""
Textual pane grid: spawn, stream, decode, and report.

Runs the app headless with the workers pinned at an unreachable address, so
every one of them fails to connect. That is the point: the failure still has to
travel out of the subprocess, through the incremental UTF-8 decoder, into the
right pane, and end in a terminal status. If that path works when the workers
fail, it works when they succeed -- and pinning the address keeps the result the
same whether or not a backend happens to be running locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_scaler_console.helpers.chat import load_chat_settings
from ai_scaler_console.helpers.parallel_ui import (
    ParallelChatApp,
    WorkerPane,
    default_columns,
)

SETTINGS = "configs/chat/parallel_8.json"


@pytest.mark.parametrize(
    ("panes", "expected"),
    [(1, 1), (2, 2), (4, 2), (8, 3), (9, 3), (16, 4), (17, 5)],
)
def test_default_columns_is_roughly_square(panes: int, expected: int) -> None:
    assert default_columns(panes) == expected


def build_app(repo_root: Path) -> ParallelChatApp:
    settings = load_chat_settings(SETTINGS)
    return ParallelChatApp(
        worker_script=repo_root / "examples" / "chat" / "parallel.py",
        settings_path=SETTINGS,
        prompts=settings.prompts,
        model_name=settings.model_name,
    )


@pytest.fixture
def unreachable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Point the workers at a port nothing can be listening on.

    Without this the outcome depends on whether a backend happens to be running
    on the developer's machine: the workers succeed, and a test written around
    the failure path fails for the wrong reason. Workers inherit os.environ, and
    python-dotenv does not override variables that are already set, so this wins
    over any local .env.
    """
    # Port 9 is discard/unassigned; a connection there is refused immediately.
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:9")


@pytest.mark.asyncio
async def test_grid_is_built_from_the_prompt_list(repo_root: Path) -> None:
    app = build_app(repo_root)

    async with app.run_test(size=(200, 60)):
        panes = list(app.query(WorkerPane))

        assert len(panes) == 8
        assert app.query_one("#panes").styles.grid_size_columns == 3

        titles = [str(p.border_title) for p in panes]
        assert titles[0].startswith("1. ")
        assert app.prompts[0][:12] in titles[0]


@pytest.mark.asyncio
async def test_workers_stream_into_their_own_panes_and_reach_a_terminal_status(
    repo_root: Path,
    unreachable_backend: None,
) -> None:
    app = build_app(repo_root)

    async with app.run_test(size=(200, 60)) as pilot:
        # The OpenAI SDK retries connection failures, so give it room.
        for _ in range(160):
            await pilot.pause(0.25)
            statuses = [str(p._status.render()) for p in app.query(WorkerPane)]
            if statuses and all(("DONE" in s or "FAILED" in s) for s in statuses):
                break

        panes = list(app.query(WorkerPane))
        statuses = [str(p._status.render()) for p in panes]
        outputs = [str(p._output.render()) for p in panes]

        assert len(statuses) == 8
        assert all(("DONE" in s or "FAILED" in s) for s in statuses), statuses

        # Every pane received its worker's stdout.
        assert all(o.strip() for o in outputs)
        assert all("Worker " in o for o in outputs)

        # With no backend the connection failure has to be visible, not silent.
        assert all(("Error" in o or "ERROR" in o) for o in outputs)

        assert app._finished == 8


@pytest.mark.asyncio
async def test_pane_output_is_not_parsed_as_textual_markup(repo_root: Path) -> None:
    """
    Bracketed worker text must survive.

    Static renders markup by default, which silently ate the "[Stats] ..."
    lines until markup=False was set.
    """
    app = build_app(repo_root)

    async with app.run_test(size=(200, 60)):
        pane = next(iter(app.query(WorkerPane)))
        pane.append("[Stats] TTFT: 0.283s\n[not-a-tag] plain text\n")

        rendered = str(pane._output.render())
        assert "[Stats] TTFT: 0.283s" in rendered
        assert "[not-a-tag] plain text" in rendered


@pytest.mark.asyncio
async def test_pane_buffer_is_bounded(repo_root: Path) -> None:
    from ai_scaler_console.helpers import parallel_ui

    app = build_app(repo_root)

    async with app.run_test(size=(200, 60)):
        pane = next(iter(app.query(WorkerPane)))
        pane.append("x" * (parallel_ui.MAX_PANE_CHARS + 5_000))

        assert len(pane._buffer) == parallel_ui.MAX_PANE_CHARS
