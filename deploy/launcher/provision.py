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

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import applog, config, env_file, progress, state

log = applog.get("provision")

MARKER = re.compile(r"^\[setup_env\]\s*(?P<message>.*)$")
ACCEL_SUMMARY = re.compile(r"^\s*Accelerator\s*:\s*(?P<accel>\S+)")
SUCCESS = "Environment setup complete"


@dataclass(frozen=True)
class Phase:
    key: str
    label: str


PHASES: tuple[Phase, ...] = (
    Phase("detect", "Detecting your GPU"),
    Phase("sync", "Installing Python packages"),
    Phase("xpu", "Checking the GPU can compute"),
    Phase("llama", "Downloading the GPU runtime"),
    Phase("convert", "Fetching conversion tools"),
    Phase("done", "Finishing up"),
)

# Substrings of setup_env's own output, in the order it emits them. Matching on its
# messages keeps this readable, and a message it stops printing shows up as a phase
# that never advances rather than as a crash.
PHASE_HINTS: tuple[tuple[str, str], ...] = (
    ("accelerator=", "detect"),
    ("uv sync", "sync"),
    ("verifying the XPU install", "xpu"),
    ("skipping the XPU check", "llama"),
    ("llama backend=", "llama"),
    ("downloading the official install", "llama"),
    ("llama installed", "convert"),
    ("llama binary already present", "convert"),
    ("skipping llama", "convert"),
    ("sparse checkout of the convert tooling", "convert"),
    ("fetching the convert scripts", "convert"),
    ("convert tooling ready", "done"),
    ("convert tooling already present", "done"),
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
    # Every default that setup_env exposes is left alone. It picks better than a guess
    # here would: cuda when nvidia-smi answers, xpu otherwise, and it proves an XPU
    # install by running a real GEMM plus a training step before declaring success.
    return env


def run(on_line: Callable[[str], None] | None = None,
        on_phase: Callable[[str, str], None] | None = None,
        on_progress: Callable[[progress.Progress], None] | None = None) -> str:
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
        return _run_setup(command, environment, log_path, meter, on_line, on_phase)
    finally:
        if meter:
            meter.close()


def _run_setup(command, environment, log_path, meter, on_line, on_phase) -> str:
    seen_phase: str | None = None
    accelerator = ""
    tail: list[str] = []

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
        )
        assert process.stdout is not None
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
    log.info("setup_env exited with %s", code)

    if code != 0:
        raise RuntimeError(
            f"Provisioning failed (exit {code}). Full log: {log_path}\n\n" + "\n".join(tail[-15:])
        )

    if not any(SUCCESS in line for line in tail):
        # An exit code of zero without the completion banner means the script ended
        # somewhere unexpected. Treating that as success is how a half-provisioned
        # install reaches a user.
        raise RuntimeError(f"Provisioning ended without reporting completion. Full log: {log_path}")

    progress.record_package_count()
    state.save({"provisioned": True, "accelerator": accelerator or None})
    log.info("provisioning complete, accelerator=%s", accelerator or "unknown")
    return accelerator


def is_provisioned() -> bool:
    """Trust the recorded flag, but only alongside the artefacts it claims exist."""
    if not state.get("provisioned"):
        return False
    return (config.venv_dir() / ("Scripts" if config.IS_WINDOWS else "bin")).is_dir()


def iter_phases() -> Iterator[Phase]:
    return iter(PHASES)


if __name__ == "__main__":
    try:
        accel = run(on_line=lambda line: print(line, flush=True),
                    on_phase=lambda _key, label: print(f"== {label}", flush=True))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"provisioned, accelerator={accel or 'unknown'}")
