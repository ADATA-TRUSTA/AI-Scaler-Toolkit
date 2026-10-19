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
import os
import signal
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

# A service that stops sooner than this never got as far as starting. Uvicorn binds in
# a few seconds even on a cold cache; the long part of a start is importing torch,
# which happens after.
EARLY_EXIT_SECONDS = 15.0

# Fields only this backend's /health returns. Used to tell our own leftover process
# from an unrelated service that happens to sit on the port.
FINGERPRINT = ("status", "training_supported", "supported_engines")


def is_port_free(port: int, host: str = config.SERVICE_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _listening_pid_linux(port: int) -> int | None:
    """The same answer as netstat below, from /proc.

    /proc/net/tcp lists every socket with the inode that owns it, and /proc/<pid>/fd
    holds a symlink named socket:[inode] for each socket a process has open; joining
    the two identifies the listener. ss and lsof answer the same question, and neither
    is guaranteed to be installed -- ss is missing from minimal images and lsof from
    most of them -- whereas /proc is Linux itself.

    Matched on the port alone. The service binds the loopback address, but a
    hand-started backend on the same port may have bound 0.0.0.0, and that is still
    the process standing between this launcher and the port it needs.
    """
    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            lines = Path(f"/proc/net/{name}").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A is TCP_LISTEN
                continue
            _, _, local_port = fields[1].rpartition(":")
            try:
                if int(local_port, 16) == port:
                    inodes.add(fields[9])
            except ValueError:
                continue
    if not inodes:
        return None

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for descriptor in (entry / "fd").iterdir():
                target = os.readlink(descriptor)
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return int(entry.name)
        except OSError:
            continue  # gone, or someone else's process
    return None


def listening_pid(port: int) -> int | None:
    """Which process is listening on this port, if any.

    Needed because a launcher can adopt a service it did not spawn, and then has no
    handle to stop it with. netstat rather than a library: this runs inside a frozen
    build where every dependency is weight, and the answer is one line of output.

    This returned None on Linux, which made every caller a no-op there: stopping an
    adopted service did nothing and said nothing, so Quit -- which calls itself the
    only path that stops the service -- left the backend running, exactly the fault
    the Windows side of this was written to fix.
    """
    if not config.IS_WINDOWS:
        return _listening_pid_linux(port)
    try:
        result = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                capture_output=True, text=True, timeout=30,
                                creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    # Matched on the port alone, as the Linux path above already was. Matching the
    # address too meant looking for 127.0.0.1:8571 and missing our own service the
    # moment it was opened to the network, because netstat then reports 0.0.0.0:8571
    # -- so an adopted service could not be found, and stopping it did nothing.
    needle = f":{port}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(needle) and parts[3].upper() == "LISTENING":
            with contextlib.suppress(ValueError):
                return int(parts[4])
    return None


def terminate_pid(pid: int) -> bool:
    """Ask a process to go, then insist. Someone else's service is still someone's."""
    if not config.IS_WINDOWS:
        # SIGTERM, then SIGKILL, to the process group so that the engine subprocesses
        # the backend spawned (llama-server, vLLM) go with it rather than being
        # reparented and left holding the GPU.
        #
        # Only when the service leads that group, which a service this launcher
        # spawned does (start_new_session). Otherwise the group is whoever started
        # the launcher's, and it is not ours to signal: a service from a launcher
        # that autostarted at login shares gnome-session's group, and signalling it
        # took the whole desktop session down with it -- the user logged out, seen on
        # the Linux test VM when an upgrade restarted a service an older launcher had
        # started. That service gets the signal alone, and stops its own engines.
        try:
            group: int | None = os.getpgid(pid)
            if group != pid or group == os.getpgid(0):
                group = None
        except OSError:
            group = None

        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            try:
                if group is not None:
                    os.killpg(group, signal_number)
                else:
                    os.kill(pid, signal_number)
            except ProcessLookupError:
                return True  # already gone
            except PermissionError:
                # Not ours to stop. The fingerprint says the service on this port is
                # this backend, which is not the same as saying this user started it --
                # on a shared machine, or after a system-wide install, it can belong to
                # someone else. Reporting that as a successful stop made Quit claim to
                # have stopped a service that was still holding the port, and the next
                # start adopted it again.
                log.warning("pid %s belongs to another user; it cannot be stopped from "
                            "here", pid)
                return False
            except OSError:
                log.warning("could not signal pid %s", pid)
                return False
            time.sleep(1.0 if signal_number == signal.SIGTERM else 0.2)
            if not listening_pid_alive(pid):
                return True
        log.warning("could not stop pid %s", pid)
        return False

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
    if not config.IS_WINDOWS:
        # /proc rather than signal 0: a process this launcher did not spawn is not its
        # child, so there is no reaping, and a zombie would answer signal 0 as alive.
        return Path(f"/proc/{pid}").is_dir()

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


def settle_venv_home() -> None:
    """Point the environment at the interpreter directory, not at a link to it.

    uv installs Python as cpython-3.12.13-... and puts a minor-version alias beside it,
    cpython-3.12-..., which on Windows is a junction. That alias is what it records in
    pyvenv.cfg, so starting the service traverses a mount point -- which a process
    running under the redirection-trust mitigation is refused.

    This is a tidy-up, not the fix for that. It cannot be: os.readlink reads the link's
    target rather than following it, which is the only reason it works at all here --
    an earlier version used Path.resolve(), which has to traverse the junction, so in
    the one situation that needed it the call raised and this returned having done
    nothing. What keeps the mitigation away from us is hardening.step_out_of_the_way().

    Resolving it changes nothing else: the alias and its target are the same
    interpreter. Done on every start because uv writes the file again whenever it
    syncs, and it is cheap.
    """
    config_file = config.venv_dir() / "pyvenv.cfg"
    try:
        # utf-8-sig: a byte-order mark would otherwise attach itself to the first key,
        # which is `home`, and this would quietly find nothing to do.
        lines = config_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return

    changed = False
    for index, line in enumerate(lines):
        key, _, value = line.partition("=")
        if key.strip() != "home":
            continue
        recorded = value.strip()
        try:
            target = os.readlink(recorded)
        except (OSError, ValueError):
            # Not a link, or unreadable. Either way there is nothing to settle.
            return
        # A junction's target comes back in verbatim form, \\?\C:\..., which is not
        # something to write into a configuration file other tools read.
        for prefix in ("\\\\?\\", "\\??\\"):
            if target.startswith(prefix):
                target = target[len(prefix):]
                break
        resolved = Path(target)
        # Only when it is still an interpreter directory -- Windows keeps python.exe at
        # the top, other platforms keep it under bin/. A resolve that lands somewhere
        # unexpected is not something to write into an environment.
        looks_right = any((resolved / name).is_file()
                          for name in ("python.exe", "bin/python3", "bin/python"))
        if str(resolved) == recorded or not looks_right:
            return
        log.info("pyvenv.cfg home %s is a link; recording %s", recorded, resolved)
        lines[index] = f"home = {resolved}"
        changed = True

    if changed:
        with contextlib.suppress(OSError):
            config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    def _spawn(self, python: Path, environment: dict, service_log: Path) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(OSError):
                self._log_handle.close()
        log.info("spawning %s -m service.app (cwd=%s)", python, config.backend_dir())
        self._log_handle = service_log.open("w", encoding="utf-8", errors="replace")
        self._process = subprocess.Popen(
            [str(python), "-m", "service.app"],
            cwd=config.backend_dir(),
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=config.subprocess_flags(),
            # Its own process group, so that terminate_pid can stop it together with
            # the engines it spawns and nothing else -- see there.
            start_new_session=not config.IS_WINDOWS,
        )
        state.set("service_version", config.app_version())

    def status(self) -> PortStatus:
        return probe(self.port)

    def is_running(self) -> bool:
        return self.status() == "ours"

    # How long a service whose process is still alive may go without answering before
    # it counts as gone. A busy one misses probes: it blocks its event loop while a
    # model loads, and on a loaded machine /health can take longer than the probe
    # waits. One miss used to be enough -- the release's acceptance run reported the
    # service "stopped by itself" while it was answering requests, and a 20 GB model
    # load had the window say the service was lost.
    UNANSWERED_GRACE_SEC = 60.0

    def has_stopped(self, unanswered_since: float) -> bool:
        """Whether a service that has not answered since `unanswered_since` is gone.

        Gone at once if the process we started has exited or the port is free;
        otherwise only after UNANSWERED_GRACE_SEC without an answer. An adopted
        service has no process to look at, so only the time and the port count.
        """
        if self._process is not None and self._process.poll() is not None:
            return True
        if self.status() == "free":
            return True
        return time.monotonic() - unanswered_since >= self.UNANSWERED_GRACE_SEC

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
        settle_venv_home()

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
            #
            # bind_host, not SERVICE_HOST. This used to pass the constant, which is
            # 127.0.0.1 and always wins over the .env -- so the switch that opens the
            # service to the network wrote SERVICE_HOST=0.0.0.0 into the file, the
            # launcher logged "bound to 0.0.0.0", and the service went on listening on
            # loopback alone. Nothing caught it because every test until now was from
            # this machine, where loopback answers.
            "SERVICE_HOST": config.bind_host(),
            "SERVICE_PORT": str(self.port),
            # Only in the environment, never written to the .env: the key lives in the
            # launcher's state, and a copy in a second file is one that goes stale when
            # it is changed in Settings. The process environment wins over the .env,
            # which the backend loads without override.
            "SERVICE_API_KEY": config.service_api_key(),
            # Other computers get the inference API under /v1/ and nothing else. "all"
            # also opens model loading, downloads, fine-tuning and RAG, whose requests
            # carry file paths and launch arguments; that is for someone running the
            # backend from source behind protection of their own, not a switch in this
            # application. Passed here, not left to the .env, so an edited .env cannot
            # widen it: the backend loads the .env without overriding the environment.
            "SERVICE_REMOTE_ACCESS": "inference",
        }
        merged = os.environ.copy()
        merged.update({k: v for k, v in env.items() if v})
        # Set even when empty. Skipped like the others, a SERVICE_API_KEY left in the
        # launcher's own environment would reach the service and open it with a key
        # nobody here chose.
        merged["SERVICE_API_KEY"] = env["SERVICE_API_KEY"]

        # The pip-installed CUDA toolkit, for vLLM's first-use kernel compilation --
        # see config.wheel_cuda_home. In the environment rather than the .env: it is
        # worked out from the disk on every start, so a rebuilt environment with a
        # different CUDA major cannot leave a stale path behind, which a key written
        # once and never rewritten would. A CUDA_HOME someone set, in either place,
        # wins.
        cuda_home = config.wheel_cuda_home()
        if cuda_home is not None and not merged.get("CUDA_HOME"):
            merged["CUDA_HOME"] = str(cuda_home)
            merged["PATH"] = str(cuda_home / "bin") + os.pathsep + merged.get("PATH", "")
            log.info("CUDA_HOME=%s (the toolkit the cuda extra installed)", cuda_home)

        # Intel only: setup_env.sh links libze_loader.so into the environment when
        # libze-dev is absent, because oneCCL opens that unversioned name and training
        # fails without it while inference works. On LD_LIBRARY_PATH here, as
        # run_service.sh does, so the training process the service spawns inherits it.
        ze_links = config.venv_dir() / "lib" / "trusta-ze"
        if not config.IS_WINDOWS and ze_links.is_dir():
            current = merged.get("LD_LIBRARY_PATH", "")
            merged["LD_LIBRARY_PATH"] = str(ze_links) + (os.pathsep + current if current else "")
            log.info("LD_LIBRARY_PATH += %s (libze_loader.so without libze-dev)", ze_links)

        # Its output goes to a file, not to nowhere. Discarding it meant a service that
        # failed to start left no explanation at all.
        config.log_dir().mkdir(parents=True, exist_ok=True)
        service_log = config.log_dir() / SERVICE_LOG
        log.info("service output -> %s", service_log)

        self._spawn(python, merged, service_log)

        started = time.monotonic()
        deadline = started + wait_sec
        last_report = 0.0
        retried = False
        while time.monotonic() < deadline:
            if self.status() == "ours":
                elapsed = time.monotonic() - started
                log.info("service healthy after %.0fs", elapsed)
                return
            if self._process.poll() is not None:
                # A service that dies this fast never began: it failed to spawn, or it
                # failed on an import. One of those has been seen to come and go --
                # Windows refusing to traverse the link to the interpreter, right after
                # an upgrade, on a machine where the very next attempt worked. Trying
                # once more turns a dead end into a pause. A backend that is genuinely
                # broken fails the same way twice, a second apart, and reports both.
                if not retried and time.monotonic() - started < EARLY_EXIT_SECONDS:
                    retried = True
                    first = _log_tail(service_log)
                    log.warning("the service stopped %.0fs in (exit %s); trying once more\n%s",
                                time.monotonic() - started, self._process.returncode, first)
                    self._spawn(python, merged, service_log)
                    started = time.monotonic()
                    continue
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
                # while a model loads. Keep asking until it answers or has_stopped().
                unanswered_since = time.monotonic()
                answered = False
                while not self._watch_stop.wait(3.0):
                    if self.status() == "ours":
                        answered = True
                        break
                    if self.has_stopped(unanswered_since):
                        break
                if answered or self._watch_stop.is_set():
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
