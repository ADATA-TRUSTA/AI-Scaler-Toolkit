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
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import applog, config, state

log = applog.get("progress")

POLL_SECONDS = 1.0

# The denominator for the package count is what uv itself says this sync will install,
# asked with the same extras setup_env announces just before it syncs. It used to be a
# shipped constant, measured once on one machine, and every other accelerator ran past
# it: "275 of ~143" on Linux, "182 of ~143" on an Intel Windows laptop. A dry run against
# an environment that does not exist yet counts the whole install -- 182, 193 and 266 on
# Windows XPU, Linux XPU and Linux CUDA with vLLM, each equal to what then landed.
_SYNC_ANNOUNCED = re.compile(r"^\[setup_env\] uv sync (?P<extras>(?:--extra \S+ ?)+)")
_WOULD_INSTALL = re.compile(r"Would install (\d+) packages?")
# The dry run needs an interpreter, and on a first install the real sync is what
# fetches it, seconds in. Downloads stay off here so the two never fetch it twice.
_PLAN_ATTEMPTS = 12
_PLAN_RETRY_SECONDS = 5.0

_planned: int | None = None
_planned_extras: tuple[str, ...] = ()
_plan_lock = threading.Lock()


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


def reset_plan() -> None:
    global _planned, _planned_extras
    with _plan_lock:
        _planned, _planned_extras = None, ()


def plan_from_line(line: str, environment: dict[str, str]) -> None:
    """Start counting what the sync setup_env has just announced will install.

    Counted in the background: the sync itself starts at once, and the bar shows the
    installed count alone until the total is known.
    """
    global _planned_extras
    announced = _SYNC_ANNOUNCED.match(line)
    if not announced:
        return
    extras = tuple(announced.group("extras").split())
    with _plan_lock:
        # Linux syncs twice with the same extras (causal-conv1d is built in between).
        if extras == _planned_extras:
            return
        _planned_extras = extras
    threading.Thread(target=_count_planned, args=(extras, environment),
                     name="package-plan", daemon=True).start()


def _count_planned(extras: tuple[str, ...], environment: dict[str, str]) -> None:
    global _planned
    uv = config.uv_binary()
    if uv is None:
        return
    with tempfile.TemporaryDirectory(prefix="package-plan-") as scratch:
        # An environment that does not exist, so the dry run counts the whole install
        # rather than what the real sync, running alongside, has not reached yet.
        env = {**environment, "UV_PROJECT_ENVIRONMENT": str(Path(scratch) / "env"),
               "UV_PYTHON_DOWNLOADS": "never"}
        for _ in range(_PLAN_ATTEMPTS):
            try:
                result = subprocess.run([str(uv), "sync", "--dry-run", "--frozen", *extras],
                                        cwd=config.backend_dir(), env=env, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace",
                                        timeout=60, creationflags=config.subprocess_flags())
                found = _WOULD_INSTALL.search(result.stdout + result.stderr)
                if result.returncode == 0 and found:
                    with _plan_lock:
                        if _planned_extras == extras:
                            _planned = int(found.group(1))
                    log.info("planned packages for %s: %s", " ".join(extras), found.group(1))
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
            time.sleep(_PLAN_RETRY_SECONDS)
    log.info("could not count the planned packages; showing the installed count alone")


def _recorded_packages() -> int:
    try:
        return max(int(state.get("package_count")), 0)
    except (TypeError, ValueError):
        return 0


def _sync_progress() -> tuple[float | None, str]:
    done = installed_packages()
    if done <= 0:
        return None, "resolving dependencies"
    with _plan_lock:
        planned = _planned
    # What a previous install on this machine ended with stands in until uv has
    # answered; a newer lock file can make it short, so past it the total is dropped
    # rather than shown as a bar that is full and still going.
    total = planned or _recorded_packages()
    if total <= 0 or done > total:
        return None, f"{done} packages installed"
    return min(done / total, 0.99), f"{done} of {total} packages"


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
