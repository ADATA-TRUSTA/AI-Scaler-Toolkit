"""First-run provisioning: call the backend's setup_env and show what it is doing.

This module deliberately does very little. setup_env already detects the accelerator,
runs uv sync with the right extras, verifies an XPU install by actually computing on
it, installs the pinned llama binaries, fetches the GGUF conversion tooling and records
LLAMA_SERVER_BINARY. Reimplementing any of that here would mean maintaining a second
copy that drifts from the backend's within a release or two.

So the job is: write the .env keys the launcher owns, run the script, stream its output
to the progress window, and translate its `[setup_env]` lines into something coarse
enough for a person to read.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import applog, config, env_file, progress, state

log = applog.get("provision")

# The setup_env process while one is running, so quitting can stop it.
#
# Without this it outlived the window: closing the application ended the window loop
# and left provisioning running to completion in a process with no interface, which
# then started a service -- while a second launch, seeing the guard released, started
# its own. Two copies provisioning and two services on one port, from one click.
#
# Stopping it is safe because setup is resumable: state records that setup began, and
# setup_env installs only what is missing, so the next run carries on rather than
# starting over.
_running: subprocess.Popen | None = None
_cancelled = False


class Cancelled(RuntimeError):
    """Setup was stopped on purpose. Not a failure, and not worth reporting as one."""


def cancel() -> bool:
    """Stop a setup that is in progress. True if there was one."""
    global _cancelled
    process = _running
    if process is None or process.poll() is not None:
        return False
    _cancelled = True
    log.info("cancelling setup_env (pid %s)", process.pid)

    # The tree, not the process. setup_env is a shell script whose actual work happens
    # in children -- uv resolving and downloading, curl fetching a runtime, git doing a
    # sparse checkout. Killing only the shell leaves those writing into a virtual
    # environment nobody is watching any more, which is how a half-written environment
    # that reports itself as installed gets made.
    if config.IS_WINDOWS:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=20,
                           creationflags=config.subprocess_flags())
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log.warning("setup_env did not stop within 10s; killing it")
        with contextlib.suppress(OSError):
            process.kill()
    return True

MARKER = re.compile(r"^\[setup_env\]\s*(?P<message>.*)$")
ACCEL_SUMMARY = re.compile(r"^\s*Accelerator\s*:\s*(?P<accel>\S+)")
SUCCESS = "Environment setup complete"


@dataclass(frozen=True)
class Phase:
    key: str
    label: str


# Display order. setup_env fetches the conversion tools before the llama runtime on
# Windows and the other way round on Linux, so the interface tracks which phases have
# been *seen* rather than assuming this order holds.
#
# "detect" is not here. It finishes in well under a second -- it is a check for
# nvidia-smi -- so as a step it only ever appeared already complete, which reads as
# one that was skipped.
PHASES: tuple[Phase, ...] = (
    Phase("sync", "Installing Python packages"),
    Phase("xpu", "Checking the GPU can compute"),
    Phase("convert", "Fetching conversion tools"),
    Phase("llama", "Downloading the GPU runtime"),
    Phase("done", "Finishing up"),
)

# Phases that only happen on some accelerators. Showing one that will not run leaves
# a permanent gap in the middle of the row, which looks like something broke rather
# than like a step that does not apply here.
ACCELERATOR_ONLY: dict[str, str] = {"xpu": "xpu"}

ACCEL_MARKER = re.compile(r"^accelerator=(?P<accel>\S+)")


def phases_for(accelerator: str) -> tuple[Phase, ...]:
    """The steps this machine will actually take."""
    return tuple(
        phase for phase in PHASES
        if phase.key not in ACCELERATOR_ONLY or ACCELERATOR_ONLY[phase.key] == accelerator
    )

# Substrings of setup_env's own output, in the order it emits them. Matching on its
# messages keeps this readable, and a message it stops printing shows up as a phase
# that never advances rather than as a crash.
PHASE_HINTS: tuple[tuple[str, str], ...] = (
    ("uv sync", "sync"),
    ("verifying the XPU install", "xpu"),
    ("skipping the XPU check", "convert"),
    # Conversion tooling: resolving the pinned commit is the first thing it prints.
    ("resolves to", "convert"),
    ("sparse checkout of the convert tooling", "convert"),
    ("fetching the convert scripts", "convert"),
    # Conversion tooling finished, so the runtime is next. Matching the *end* of one
    # phase to the start of the next is what keeps the display honest when a phase is
    # skipped entirely.
    ("convert tooling ready", "llama"),
    ("convert tooling already present", "llama"),
    ("torch ships", "llama"),
    ("publishes llama-", "llama"),
    ("downloading llama-", "llama"),
    ("downloading the official install", "llama"),
    ("llama backend=", "llama"),
    # Runtime in place; all that is left is recording where it went.
    ("installed:", "done"),
    ("llama binary already present", "done"),
    ("skipping llama", "done"),
    ("recorded LLAMA_SERVER_BINARY", "done"),
    ("llama install failed", "done"),
)


def label_for(key: str) -> str:
    return next((p.label for p in PHASES if p.key == key), key)


def phase_for(message: str) -> str | None:
    lowered = message.lower()
    for hint, key in PHASE_HINTS:
        if hint.lower() in lowered:
            return key
    return None


def disk_free_gb(path: Path) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / (1024**3)


def check_disk() -> None:
    """Stop before starting rather than halfway through a multi-gigabyte download."""
    free = disk_free_gb(config.data_dir())
    if free < config.MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Only {free:.1f} GB free where the environment is installed; "
            f"at least {config.MIN_FREE_DISK_GB} GB is needed "
            "(virtual environment ~3.7 GB, GPU runtime ~2.7 GB, llama a few hundred MB)."
        )


def build_command() -> list[str]:
    script = config.setup_env_script()
    log.info("setup_env script: %s (exists: %s)", script, script.is_file())
    if not script.is_file():
        raise RuntimeError(f"provisioning script not found: {script}")
    if config.IS_WINDOWS:
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    return ["bash", str(script)]


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    # uv would otherwise create .venv inside the payload directory. Keeping it in the
    # per-user data directory means a reinstall does not re-download it and the program
    # directory does not grow by gigabytes after setup.
    env["UV_PROJECT_ENVIRONMENT"] = str(config.venv_dir())
    env["HF_HOME"] = str(config.hf_home())
    # Where setup_env finds the Visual C++ runtime to place beside torch and llama.
    # Passed rather than derived: the backend project is also run straight from a
    # checkout, where there is no payload and no need for one.
    runtime = config.vcruntime_dir()
    if config.IS_WINDOWS and runtime.is_dir():
        env["TRUSTA_VCRUNTIME_DIR"] = str(runtime)
    # The uv the payload carries. Without it setup_env downloads astral.sh's install
    # script and runs it, which is fine on a developer's machine and not something to
    # do on a user's behalf. Also puts it on PATH, because uv re-executes itself by
    # name in places and a bare "uv" has to resolve to the same binary.
    uv = config.uv_binary()
    if uv is not None:
        env["TRUSTA_UV"] = str(uv)
        env["PATH"] = str(uv.parent) + os.pathsep + env.get("PATH", "")
    if not config.IS_WINDOWS:
        # setup_env leaves an ordinary desktop's system packages alone unless it is asked,
        # which is the right default for a script someone runs by hand but the wrong one
        # here: the user launched an installer, so installing what it needs to build
        # (cmake, git, build-essential, the Vulkan headers) is the job. Without this the
        # tools stay missing, llama.cpp is never built, and the only trace is one line in
        # the log telling a user with no terminal open to run apt themselves. An explicit
        # setting in the environment still wins, including 0 to forbid it.
        env.setdefault("TRUSTA_APT_INSTALL", "1")
    # Every default that setup_env exposes is left alone. It picks better than a guess
    # here would: cuda when nvidia-smi answers, xpu otherwise, and it proves an XPU
    # install by running a real GEMM plus a training step before declaring success.
    return env


def run(on_line: Callable[[str], None] | None = None,
        on_phase: Callable[[str, str], None] | None = None,
        on_progress: Callable[[progress.Progress], None] | None = None,
        on_phases: Callable[[tuple[Phase, ...]], None] | None = None) -> str:
    """Run provisioning to completion. Returns the accelerator setup_env settled on.

    Raises RuntimeError with the tail of the output if it fails; the full log is kept
    on disk either way, because "provisioning failed" with no detail is unactionable.
    """
    log.info("provisioning requested")

    free = disk_free_gb(config.data_dir())
    log.info("disk free at %s: %.1f GB (need %s GB)", config.data_dir(), free, config.MIN_FREE_DISK_GB)
    check_disk()

    port = state.service_port()
    written = env_file.ensure(port)
    log.info("env keys written: %s", ", ".join(written) if written else "none needed")

    config.log_dir().mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir() / "provision.log"

    command = build_command()
    environment = build_env()
    log.info("command    : %s", " ".join(command))
    log.info("cwd        : %s", config.backend_dir())
    log.info("UV_PROJECT_ENVIRONMENT=%s", environment.get("UV_PROJECT_ENVIRONMENT"))
    log.info("HF_HOME=%s", environment.get("HF_HOME"))
    log.info("output goes to %s", log_path)

    meter = progress.Meter(on_progress) if on_progress else None

    try:
        return _run_setup(command, environment, log_path, meter, on_line, on_phase, on_phases)
    finally:
        if meter:
            meter.close()


def _run_setup(command, environment, log_path, meter, on_line, on_phase, on_phases=None) -> str:
    seen_phase: str | None = None
    accelerator = ""
    tail: list[str] = []

    global _running, _cancelled
    _cancelled = False
    with log_path.open("w", encoding="utf-8") as log_file:
        log.info("starting setup_env")
        process = subprocess.Popen(
            command,
            cwd=config.backend_dir(),  # setup_env resolves the project root from its own location
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # Otherwise a PowerShell console sits open for the whole install.
            creationflags=config.subprocess_flags(),
        )
        assert process.stdout is not None
        _running = process
        for raw in process.stdout:
            line = raw.rstrip("\n")
            log_file.write(line + "\n")
            log_file.flush()  # so the log is readable while a long install is still running
            tail.append(line)
            del tail[:-40]
            if on_line:
                on_line(line)

            summary = ACCEL_SUMMARY.match(line)
            if summary:
                accelerator = summary.group("accel")

            marker = MARKER.match(line)
            if not marker:
                continue
            message = marker.group("message")

            # The accelerator is the first thing setup_env prints, which is early
            # enough to publish a step list with the steps this machine will skip
            # already left out.
            accel_line = ACCEL_MARKER.match(message)
            if accel_line and on_phases:
                accelerator = accel_line.group("accel")
                on_phases(phases_for(accelerator))

            phase = phase_for(message)
            if phase and phase != seen_phase:
                seen_phase = phase
                log.info("phase -> %s", phase)
                if meter:
                    meter.set_phase(phase, label_for(phase))
                if on_phase:
                    label = next((p.label for p in PHASES if p.key == phase), phase)
                    on_phase(phase, label)

        code = process.wait()
        _running = None
    log.info("setup_env exited with %s", code)

    if _cancelled:
        # We stopped it. Reporting a non-zero exit as a failure would put
        # "Provisioning failed" in the log for someone who pressed Quit, which is the
        # first thing anyone reads when something is actually wrong.
        raise Cancelled("setup was cancelled")

    if code != 0:
        raise RuntimeError(
            f"Provisioning failed (exit {code}). Full log: {log_path}\n\n" + "\n".join(tail[-15:])
        )

    if not any(SUCCESS in line for line in tail):
        # An exit code of zero without the completion banner means the script ended
        # somewhere unexpected. Treating that as success is how a half-provisioned
        # install reaches a user.
        raise RuntimeError(f"Provisioning ended without reporting completion. Full log: {log_path}")

    # The banner is what setup_env says, not what it did. It has said "Environment
    # setup complete" over an environment with no packages in it: `uv sync` failed,
    # nothing checked its exit code, and the script carried on to the end. That has
    # been fixed in the script, but "provisioned" is this application's claim to its
    # user, so it is checked here against the one artefact that has to exist.
    interpreter = config.venv_dir() / ("Scripts/python.exe" if config.IS_WINDOWS else "bin/python")
    if not interpreter.is_file():
        raise RuntimeError(
            "Setup reported success but installed no Python environment "
            f"({interpreter} is not there). Full log: {log_path}\n\n"
            + "\n".join(tail[-15:])
        )

    progress.record_package_count()
    state.save({"provisioned": True, "accelerator": accelerator or None})
    log.info("provisioning complete, accelerator=%s", accelerator or "unknown")
    return accelerator


# What setup's tools print when the network is what failed -- uv, curl.exe, the
# PowerShell cmdlets and Python's urllib, in their own words. Matched in the output of
# a failed run, so a connection lost halfway through setup is told as that rather than
# as a wall of log; the check before setup covers the machine that was never online.
_NETWORK_MARKERS = (
    "failed to download",
    "request failed after",
    "error sending request",
    "dns error",
    "failed to lookup address",
    "temporary failure in name resolution",
    "name or service not known",
    "could not resolve host",
    "the remote name could not be resolved",
    "no such host is known",
    "network is unreachable",
    "connection timed out",
    "the operation has timed out",
    "connection refused",
    "connection reset",
)


def looks_like_network_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _NETWORK_MARKERS)


def is_provisioned() -> bool:
    """Trust the recorded flag, but only alongside the artefacts it claims exist."""
    if not state.get("provisioned"):
        return False
    return (config.venv_dir() / ("Scripts" if config.IS_WINDOWS else "bin")).is_dir()


if __name__ == "__main__":
    try:
        accel = run(on_line=lambda line: print(line, flush=True),
                    on_phase=lambda _key, label: print(f"== {label}", flush=True))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"provisioned, accelerator={accel or 'unknown'}")
