"""
Textual terminal UI for watching N chat workers run side by side.

Lives in the package rather than in ``examples/chat/parallel.py`` so the worker
subprocesses never import a TUI framework just to send one chat request --
worker start-up cost would otherwise skew the time-to-first-token numbers this
view exists to show.

Runs in any terminal, including over SSH, which the previous PyQt6 version could
not do without X11 forwarding.
"""

from __future__ import annotations

import asyncio
import codecs
import math
import os
import sys
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Label, Static

# Keep each pane's buffer bounded; workers are capped by max_tokens anyway.
MAX_PANE_CHARS = 200_000


def default_columns(pane_count: int) -> int:
    """Pick a roughly square grid for the given number of panes."""
    return max(1, math.ceil(math.sqrt(pane_count)))


class WorkerPane(Vertical):
    """One bordered pane: a status line plus the worker's streamed output."""

    def __init__(self, index: int, prompt: str) -> None:
        super().__init__(id=f"pane-{index}")
        self.index = index
        self.prompt = prompt
        self._buffer = ""
        self.border_title = f"{index + 1}. {prompt}"

        # Built here rather than in compose so they can be held as attributes.
        # markup=False on the output: worker text is arbitrary. Left on, Textual
        # would parse "[Stats] TTFT: ..." (and any bracketed text the model
        # emits) as a style tag and swallow it.
        self._status = Label("PENDING", classes="status pending")
        self._output = Static("", classes="output", markup=False)
        self._scroll = VerticalScroll(self._output)

    def compose(self) -> ComposeResult:
        """Build the pane: a status line above a scrollable output area."""
        yield self._status
        yield self._scroll

    def set_status(self, text: str, state: str) -> None:
        """Update the status line. ``state`` selects the colour class."""
        self._status.update(text)
        self._status.set_classes(f"status {state}")

    def append(self, text: str) -> None:
        """Append streamed text, keeping the view pinned to the newest output."""
        self._buffer += text
        if len(self._buffer) > MAX_PANE_CHARS:
            self._buffer = self._buffer[-MAX_PANE_CHARS:]
        # Widgets are held directly rather than re-queried: append runs once per
        # 1KB of streamed output per pane, and a DOM walk per chunk across 16
        # panes is real work for nothing.
        self._output.update(self._buffer)
        self._scroll.scroll_end(animate=False)


class ParallelChatApp(App[None]):
    """Grid of panes, one per prompt, each streaming one worker subprocess."""

    CSS = """
    Grid#panes {
        grid-gutter: 1;
        padding: 0 1;
    }
    WorkerPane {
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        height: 100%;
    }
    .status {
        height: 1;
        padding: 0 1;
    }
    .status.pending  { color: $text-muted; }
    .status.running  { color: $accent; }
    .status.done     { color: $success; }
    .status.failed   { color: $error; }
    .output {
        padding: 0 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        worker_script: Path,
        settings_path: str,
        prompts: tuple[str, ...],
        model_name: str,
        columns: int | None = None,
    ) -> None:
        super().__init__()
        self.worker_script = worker_script
        self.settings_path = settings_path
        self.prompts = prompts
        self.model_name = model_name
        self.columns = columns or default_columns(len(prompts))
        self._finished = 0
        self._panes: dict[int, WorkerPane] = {}
        self._processes: dict[int, asyncio.subprocess.Process] = {}

    def compose(self) -> ComposeResult:
        """Build the pane grid, one pane per prompt."""
        yield Header(show_clock=True)
        with Grid(id="panes"):
            for index, prompt in enumerate(self.prompts):
                pane = WorkerPane(index, prompt)
                self._panes[index] = pane
                yield pane
        yield Footer()

    def on_mount(self) -> None:
        """Size the grid and start every worker."""
        self.title = "Parallel Chat"
        self.sub_title = (
            f"{self.settings_path}  |  {len(self.prompts)} prompts  |  {self.model_name}"
        )
        self.query_one("#panes", Grid).styles.grid_size_columns = self.columns
        for index in range(len(self.prompts)):
            self.stream_worker(index)

    def _pane(self, index: int) -> WorkerPane:
        return self._panes[index]

    def on_unmount(self) -> None:
        """
        Kill any worker still running.

        Textual cancels its own workers on shutdown, but the subprocesses they
        spawned are not its children to clean up -- quitting mid-run would
        otherwise leave N orphaned processes still streaming from the backend.
        """
        for process in self._processes.values():
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:  # already gone
                    pass

    @work
    async def stream_worker(self, index: int) -> None:
        """Spawn one worker subprocess and stream its output into its pane."""
        pane = self._pane(index)
        pane.set_status("RUNNING", "running")
        started_at = time.perf_counter()

        # Pin the worker's stdout encoding. Writing to a pipe, Python would
        # otherwise pick the locale encoding (cp950 on a zh-TW Windows box),
        # which both garbles non-ASCII model output under the UTF-8 decoder
        # below and can crash the worker outright on an unencodable character.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                # -u so the worker's partial token writes reach us immediately
                # instead of sitting in a block-buffered pipe.
                "-u",
                str(self.worker_script),
                "--worker",
                "--settings",
                self.settings_path,
                "--prompt-index",
                str(index),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except OSError as exc:
            pane.append(f"\n[ERROR] Failed to spawn worker: {exc}\n")
            pane.set_status("FAILED", "failed")
            self._mark_finished()
            return

        # Tracked so on_unmount can kill it if the user quits mid-run.
        self._processes[index] = process

        if process.stdout is None:  # pragma: no cover - stdout=PIPE guarantees a stream
            pane.append("\n[ERROR] Worker produced no stdout stream\n")
            pane.set_status("FAILED", "failed")
            self._mark_finished()
            return

        # Tokens arrive mid-line, so decode incrementally to survive a multi-byte
        # character split across two reads.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await process.stdout.read(1024)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                pane.append(text)

        tail = decoder.decode(b"", final=True)
        if tail:
            pane.append(tail)

        return_code = await process.wait()
        elapsed = time.perf_counter() - started_at
        if return_code == 0:
            pane.set_status(f"DONE  {elapsed:.1f}s", "done")
        else:
            pane.set_status(f"FAILED  exit {return_code}  {elapsed:.1f}s", "failed")
        self._mark_finished()

    def _mark_finished(self) -> None:
        self._finished += 1
        if self._finished == len(self.prompts):
            self.sub_title = f"{self.sub_title}  |  all workers finished"
