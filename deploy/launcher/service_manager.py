"""Own the backend service process: start it, find it, adopt it, stop it.

Two rules shape everything here.

Closing the window does not stop the service. Training runs for hours, and a window
close that killed it would throw that away; only "Quit" in the tray ends the process.

A port is never changed silently. If something else holds the port we ask, because
moving off it without saying so leaves the user with a working app on an address they
did not choose -- and a stale one written into their config.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import threading
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


def listening_pid(port: int, host: str = config.SERVICE_HOST) -> int | None:
    """Which process is listening on this port, if any.

    Needed because a launcher can adopt a service it did not spawn, and then has no
    handle to stop it with. netstat rather than a library: this runs inside a frozen
    build where every dependency is weight, and the answer is one line of output.
    """
    if not config.IS_WINDOWS:
        return None
    try:
        result = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                capture_output=True, text=True, timeout=30,
                                creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f"{host}:{port}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1] == needle and parts[3].upper() == "LISTENING":
            with contextlib.suppress(ValueError):
                return int(parts[4])
    return None


def terminate_pid(pid: int) -> bool:
    """Ask a process to go, then insist. Someone else's service is still someone's."""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                       capture_output=True, timeout=20,
                       creationflags=config.subprocess_flags())
        time.sleep(1.0)
        if listening_pid_alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=20,
                           creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError):
        log.warning("could not stop pid %s", pid)
        return False
    return True


def listening_pid_alive(pid: int) -> bool:
    try:
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                capture_output=True, text=True, timeout=20,
                                creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError):
        return False
    return str(pid) in result.stdout


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
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()

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

        # Every start, not just the first. A reinstall, an upgrade over the top, or
        # anyone deleting the file leaves the backend reading .env.example instead --
        # which does not fall back to code defaults, it applies the example's values.
        written = env_file.ensure(self.port)
        if written:
            log.info("restored missing .env keys: %s", ", ".join(written))

        env = {
            **env_file.read_keys(env_file.env_path()),
            "PYTHONPATH": str(config.backend_dir()),
            "HF_HOME": str(config.hf_home()),
            "LOG_DIR": str(config.log_dir()),
            # Passed explicitly, and last, so the file cannot move the service off the
            # port we are about to poll. That is not hypothetical: a rebuilt payload
            # lost its .env, the backend fell back to the example's port 8000, and the
            # launcher waited on 8571 for a service that was running perfectly well
            # somewhere else.
            "SERVICE_HOST": config.SERVICE_HOST,
            "SERVICE_PORT": str(self.port),
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

        creation = config.subprocess_flags()
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
            # A port that is bound but silent is normal here -- uvicorn listens before
            # it finishes starting. A port that *answers* without the keys we identify
            # ourselves by is not: it will still be answering in fifteen minutes, and
            # waiting them out ends in "the service did not answer" about a service
            # that answered every time. Say what is actually wrong instead.
            body = health(self.port)
            if body is not None:
                missing = [key for key in FINGERPRINT if key not in body]
                if missing:
                    raise RuntimeError(
                        f"Something is answering on port {self.port}, but it is not "
                        f"this application's backend -- its health response is missing "
                        f"{', '.join(missing)}.\n\nEither another program is using the "
                        "port, or this installation's backend is a different version "
                        "from its launcher."
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

    def watch(self, on_lost: Callable[[str], None], interval: float = 10.0,
              on_tick: Callable[[PortStatus], None] | None = None) -> None:
        """Notice if the service goes away after it started.

        Starting it is checked carefully and then never looked at again, which means a
        backend that dies -- out of memory during a load, or an unhandled crash -- is
        invisible: the tray still says it is running and the window simply stops
        answering. Nothing here restarts it; being told is the point, because an
        automatic restart of something that just crashed usually crashes again and
        buries the reason.

        on_tick receives the status on every poll, whatever it is. This loop is already
        asking the question the tray's status line displays, so anything that wants to
        stay in step with it can do so here for free rather than probing again.
        """
        if self._watch_thread is not None:
            return

        def loop() -> None:
            while not self._watch_stop.wait(interval):
                status = self.status()
                if on_tick is not None:
                    with contextlib.suppress(Exception):
                        on_tick(status)
                if status == "ours":
                    continue
                # One missed probe is not a death: the service blocks its event loop
                # briefly while a model loads. Confirm before saying anything.
                if self._watch_stop.wait(3.0) or self.status() == "ours":
                    continue

                code = self._process.poll() if self._process is not None else None
                detail = (f"it exited with code {code}" if code is not None
                          else "it stopped answering")
                log.warning("the service is no longer responding: %s", detail)
                with contextlib.suppress(Exception):
                    on_lost(detail)
                return

        self._watch_stop.clear()
        self._watch_thread = threading.Thread(target=loop, name="service-watch", daemon=True)
        self._watch_thread.start()
        log.info("watching the service every %.0fs", interval)

    def unwatch(self) -> None:
        self._watch_stop.set()
        self._watch_thread = None

    def stop(self, timeout: float = 20.0) -> None:
        """Stop the service, whether or not this process is the one that started it.

        Terminating by process handle covers only the service we spawned. Adoption is
        a normal path -- a launcher that starts while a healthy service is already
        there takes it over rather than fighting for the port -- and an adopted service
        has no handle here. Stopping it was therefore a no-op that reported success:
        "Stop backend" left the backend running, and Quit, whose docstring calls itself
        the only path that stops the service, left it running too.

        Waiting for the port afterwards matters as well. uvicorn keeps the socket bound
        while it shuts down, so a status probe in that window finds something listening
        that does not answer /health and calls it a foreign program -- which is what
        the tray displayed straight after a clean stop.
        """
        self.unwatch()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

        if not self._wait_for_port(5):
            owner = listening_pid(self.port)
            if owner is not None:
                log.info("port %s is held by pid %s, which this launcher did not "
                         "start; stopping it", self.port, owner)
                terminate_pid(owner)
                self._wait_for_port(10)

        if not is_port_free(self.port):
            log.warning("port %s is still occupied after stopping the service", self.port)

    def _wait_for_port(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if is_port_free(self.port):
                return True
            time.sleep(0.2)
        return is_port_free(self.port)

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
