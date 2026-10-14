"""
Quitting the pane grid mid-run must not orphan the worker subprocesses.

Textual cancels its own workers on shutdown, but the processes those workers
spawned are not its children to clean up. Without on_unmount killing them,
pressing q returned to the shell with N python processes still streaming from
the backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ai_scaler_console.helpers.chat import load_chat_settings
from ai_scaler_console.helpers.parallel_ui import ParallelChatApp

SETTINGS = "configs/chat/parallel_8.json"


@pytest.fixture
def unreachable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep the workers alive-but-failing regardless of the local machine.

    The test needs workers still running when it quits. Against a real backend
    with a small model they can finish first, which would make the test pass
    vacuously; against no backend at all the retry backoff keeps them busy.
    """
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:9")


@pytest.mark.asyncio
async def test_quitting_mid_run_reaps_every_worker(
    repo_root: Path, unreachable_backend: None
) -> None:
    settings = load_chat_settings(SETTINGS)
    app = ParallelChatApp(
        worker_script=repo_root / "examples" / "chat" / "parallel.py",
        settings_path=SETTINGS,
        prompts=settings.prompts,
        model_name=settings.model_name,
    )

    async with app.run_test(size=(200, 60)) as pilot:
        for _ in range(80):
            await pilot.pause(0.25)
            if len(app._processes) == len(settings.prompts):
                break

        spawned = dict(app._processes)
        assert len(spawned) == 8, "workers did not all spawn"

        alive = [i for i, p in spawned.items() if p.returncode is None]
        assert alive, "nothing was still running; the test would prove nothing"

        await pilot.press("q")

    # Leaving the context unmounts the app, which must have killed them.
    orphans = []
    for index, process in spawned.items():
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                orphans.append(index)

    assert not orphans, f"workers still running after quit: {orphans}"
    assert all(p.returncode is not None for p in spawned.values())
