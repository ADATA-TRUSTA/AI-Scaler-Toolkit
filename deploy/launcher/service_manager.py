"""Own the backend service process: start it, find it, adopt it, stop it.

Two rules shape everything here.

Closing the window does not stop the service. Training runs for hours, and a window
close that killed it would throw that away; only "Quit" in the tray ends the process.

A port is never changed silently. If something else holds the port we ask, because
moving off it without saying so leaves the user with a working app on an address they
did not choose -- and a stale one written into their config.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from . import applog, config, env_file, state

log = applog.get("service")

PortStatus = Literal["free", "ours", "foreign"]

SERVICE_LOG = "service.log"

# Fields only this backend's /health returns. Used to tell our own leftover process
# from an unrelated service that happens to sit on the port.
FINGERPRINT = ("status", "training_supported", "supported_engines")


def is_port_free(port: int, host: str = config.SERVICE_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def next_free_port(start: int, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        if is_port_free(port):
            return port
    raise RuntimeError(f"no free port found between {start} and {start + attempts - 1}")


def health(port: int, timeout: float = 2.0) -> dict | None:
    url = f"{config.service_url(port)}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def probe(port: int) -> PortStatus:
    """Who, if anyone, is on this port."""
    if is_port_free(port):
        return "free"
    body = health(port)
    if body and all(key in body for key in FINGERPRINT):
        return "ours"
    return "foreign"


def venv_python() -> Path:
    venv = config.venv_dir()
    return venv / ("Scripts/python.exe" if config.IS_WINDOWS else "bin/python")


def _log_tail(path: Path, lines: int = 12) -> str:
    """Put the end of the service log in the error itself.

    "The service did not start" with nothing else is unactionable, and the person
    reading it may not know a log file exists.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"(no output captured; see {path})"
    if not content:
        return f"(it produced no output; see {path})"
    return f"Last lines of {path}:\n" + "\n".join(content[-lines:])


class ServiceManager:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_handle = None

    @property
    def port(self) -> int:
        return state.service_port()

    def status(self) -> PortStatus:
        return probe(self.port)

    def is_running(self) -> bool:
        return self.status() == "ours"

    def start(self, wait_sec: int = 900, on_wait: Callable[[int], None] | None = None) -> None:  # noqa: D401
        """Start the service, or adopt one that is already answering.

        Adoption is the common case after an unclean shutdown, and it is safe: the
        fingerprint check has already established that what is on the port is ours.

        The wait is generous on purpose. A first start imports torch and transformers
        against a cold file cache, on Windows with a virus scanner reading every
        newly written extension for the first time, and the backend spends most of a
        minute timing out on an optional Redis connection before it binds. Two minutes
        looked like plenty and was not: it gave up on a service that was starting
        normally and would have been ready shortly after.

        So the deadline is a backstop, not the real condition -- what actually matters
        is whether the process is still alive, which is checked every second.
        """
        status = self.status()
        log.info("start requested on port %s, current status: %s", self.port, status)
        if status == "ours":
            log.info("a healthy service is already there; adopting it")
            return
        if status == "foreign":
            raise RuntimeError(
                f"Port {self.port} is in use by another program. "
                "Choose a different port in Settings; it will not be changed for you."
            )

        python = venv_python()
        if not python.is_file():
            raise RuntimeError(f"Python environment not found at {config.venv_dir()}; setup has not run.")

        env = {
            **env_file.read_keys(env_file.env_path()),
            "PYTHONPATH": str(config.backend_dir()),
            "HF_HOME": str(config.hf_home()),
            "LOG_DIR": str(config.log_dir()),
        }
        import os

        merged = os.environ.copy()
        merged.update({k: v for k, v in env.items() if v})

        # Its output goes to a file, not to nowhere. Discarding it meant a service that
        # failed to start left no explanation at all.
        config.log_dir().mkdir(parents=True, exist_ok=True)
        service_log = config.log_dir() / SERVICE_LOG
        log.info("spawning %s -m service.app (cwd=%s)", python, config.backend_dir())
        log.info("service output -> %s", service_log)

        creation = subprocess.CREATE_NO_WINDOW if config.IS_WINDOWS else 0
        self._log_handle = service_log.open("w", encoding="utf-8", errors="replace")
        self._process = subprocess.Popen(
            [str(python), "-m", "service.app"],
            cwd=config.backend_dir(),
            env=merged,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )

        started = time.monotonic()
        deadline = started + wait_sec
        last_report = 0.0
        while time.monotonic() < deadline:
            if self.status() == "ours":
                elapsed = time.monotonic() - started
                log.info("service healthy after %.0fs", elapsed)
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"The service stopped while starting (exit code {self._process.returncode}).\n\n"
                    + _log_tail(service_log)
                )
            elapsed = time.monotonic() - started
            if on_wait and elapsed - last_report >= 5:
                last_report = elapsed
                on_wait(int(elapsed))
            time.sleep(1.0)

        raise RuntimeError(
            f"The service did not answer within {wait_sec}s, and is still running.\n\n"
            + _log_tail(service_log)
        )

    def stop(self, timeout: float = 20.0) -> None:
        """Only called from Quit. Everything else leaves the service running."""
        if self._process is None or self._process.poll() is not None:
            self._process = None
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def restart(self) -> None:
        self.stop()
        # An adopted process is not ours to terminate by handle; wait for the port.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and self.status() == "ours":
            time.sleep(0.5)
        self.start()


def set_port(port: int) -> None:
    """Change the service port everywhere it is recorded.

    Both writes are required. The frontend reads apiBaseUrl from config.json at runtime
    and hands it to its HTTP client, and that file is written at packaging time -- so a
    port change that only touches .env leaves the UI talking to the old address, which
    looks exactly like the backend being down.
    """
    state.set("service_port", port)
    env_file.set_port(port)

    config_path = config.frontend_config()
    if not config_path.is_file():
        raise RuntimeError(f"frontend config not found at {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["apiBaseUrl"] = config.service_url(port)
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_first_run_port(prompt) -> int:  # noqa: D401
    """Settle on a port before .env is written.

    prompt(requested, suggested) -> int is called only when the default is taken, and
    its answer is used as given; the caller decides how to ask.
    """
    default = config.DEFAULT_SERVICE_PORT
    if is_port_free(default):
        log.info("port %s is free; using it", default)
        return default
    suggested = next_free_port(default + 1)
    log.warning("port %s is taken; asking whether to use %s instead", default, suggested)
    chosen = int(prompt(default, suggested))
    if not is_port_free(chosen):
        raise RuntimeError(f"Port {chosen} is also in use.")
    return chosen
