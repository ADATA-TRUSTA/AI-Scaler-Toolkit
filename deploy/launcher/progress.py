"""Real progress for the two steps that take real time.

Setup has six phases, but only two of them are long: installing the Python packages
and downloading the GPU runtime. Between them they are most of the wait, and neither
prints anything useful -- uv turns its progress bar off when its output is a pipe, and
the llama installer reports only when it finishes.

So progress is measured from the file system instead: how many packages have landed in
the environment, and how much of the runtime archive has arrived. Both are things that
are actually happening, which is the point. A step with nothing measurable says so and
shows elapsed time rather than inventing a number.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import applog, config, state

log = applog.get("progress")

# Packages installed by a successful run, used as the denominator until a real one is
# recorded. Measured on a CUDA install; the first run on any machine corrects it.
# `uv sync` prints "Resolved 286 packages" but installs far fewer -- resolution covers
# the whole graph including alternatives -- so the resolved count is not the total.
#
# Two numbers because the two platforms install different things: Linux gets the vllm
# extra as well, which more than doubles the tree. One number for both meant a first
# Linux run counted past its own denominator and showed "275 of ~143" -- a progress
# bar that is full and still going is worse than no progress bar.
EXPECTED_PACKAGES = 143 if config.IS_WINDOWS else 319

POLL_SECONDS = 1.0


@dataclass(frozen=True)
class Progress:
    phase: str
    label: str
    # None means this step has nothing measurable. The interface shows an
    # indeterminate bar rather than a fabricated percentage.
    fraction: float | None
    detail: str
    elapsed: int


def installed_packages() -> int:
    """Packages present in the environment, counted by their metadata directories.

    This reads a directory that uv is actively writing hundreds of entries into, so
    the scan can fail part-way -- on Windows a directory that disappears mid-iteration
    raises rather than being skipped. A failed count is reported as "no reading yet",
    never as an exception: the progress display is not worth interrupting an install
    for, and a single bad poll should not end the measurement.
    """
    try:
        site = config.venv_dir() / ("Lib/site-packages" if config.IS_WINDOWS else "lib")
        if not site.is_dir():
            return 0
        if config.IS_WINDOWS:
            return sum(1 for _ in site.glob("*.dist-info"))
        return sum(1 for _ in site.glob("*/site-packages/*.dist-info"))
    except OSError:
        return 0


def expected_packages() -> int:
    recorded = state.get("package_count")
    try:
        recorded = int(recorded)
    except (TypeError, ValueError):
        recorded = 0
    return recorded if recorded > 0 else EXPECTED_PACKAGES


def _sync_progress() -> tuple[float | None, str]:
    done = installed_packages()
    total = expected_packages()
    if done <= 0:
        return None, "resolving dependencies"
    # The estimate can be low on a different accelerator or a newer lock file. Cap the
    # bar rather than let it read past full, and keep counting in the text.
    fraction = min(done / total, 0.99)
    approx = "" if state.get("package_count") else "~"
    return fraction, f"{done} of {approx}{total} packages"


def _llama_download_bytes() -> int:
    """The installer stages its archive in a per-run temporary directory."""
    temp = Path(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")))
    if not temp.is_dir():
        return 0
    total = 0
    try:
        for staging in temp.glob("llama-release-*"):
            for archive in staging.glob("*.zip"):
                total += archive.stat().st_size
    except OSError:
        return 0
    return total


def _llama_progress() -> tuple[float | None, str]:
    downloaded = _llama_download_bytes()
    if downloaded <= 0:
        return None, "contacting the download server"
    # No Content-Length is available here, so this reports what has arrived rather
    # than a percentage of an unknown whole.
    return None, f"{downloaded / 1024 / 1024:.0f} MB downloaded"


SOURCES: dict[str, Callable[[], tuple[float | None, str]]] = {
    "sync": _sync_progress,
    "llama": _llama_progress,
}


class Meter:
    """Polls the current phase's progress source and reports it on a timer."""

    def __init__(self, report: Callable[[Progress], None]) -> None:
        self._report = report
        self._phase = ""
        self._label = ""
        self._phase_started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="progress", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str, label: str) -> None:
        with self._lock:
            self._phase = phase
            self._label = label
            self._phase_started = time.monotonic()
        self._emit()

    def _emit(self) -> None:
        with self._lock:
            phase, label, started = self._phase, self._label, self._phase_started
        if not phase:
            return
        # Reading the source is inside the guard, not outside it. It was outside, and
        # an OSError from scanning site-packages while uv wrote to it escaped this
        # method, ended the polling thread, and left the display frozen on whatever it
        # had last said -- which looked exactly like the install being stuck on
        # "Installing Python packages" while it was in fact several steps further on.
        try:
            source = SOURCES.get(phase)
            fraction, detail = source() if source else (None, "")
            self._report(Progress(phase, label, fraction, detail, int(time.monotonic() - started)))
        except Exception:
            log.debug("progress poll failed", exc_info=True)

    def _loop(self) -> None:
        # Belt and braces: _emit already swallows its own failures, but this thread
        # must outlive anything unexpected. Losing it costs the user the only signal
        # that the install is still moving.
        while not self._stop.wait(POLL_SECONDS):
            try:
                self._emit()
            except Exception:
                log.debug("progress loop iteration failed", exc_info=True)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def record_package_count() -> None:
    """Remember what a complete install looks like on this machine.

    The first run has only the shipped estimate to work with; every run after it has a
    number measured here, so a reinstall shows a bar that is right rather than close.
    """
    count = installed_packages()
    if count > 0:
        state.set("package_count", count)
        log.info("recorded package count: %d", count)
