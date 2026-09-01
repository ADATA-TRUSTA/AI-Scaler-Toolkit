"""Tray application: first-run setup, then a window onto the existing WebUI.

Division of labour, matching what a desktop AI tool is expected to do: the WebUI owns
the work (training and inference settings), and the tray owns the environment (the
service, its port, optional components, updates).

Threading is not incidental here. pywebview must own the main thread on Windows, so
the tray icon runs on a worker thread and all of the first-run sequencing happens in
the callable passed to webview.start().
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import (
    applog,
    autostart,
    component_manager,
    config,
    env_file,
    preflight,
    provision,
    service_manager,
    state,
    updater,
)

UI_DIR = config.resource_root() / "launcher" / "ui"

log = applog.get("main")


def _icon_path() -> Path | None:
    """The icon packaging chose, under the fixed name it writes.

    Falling back to the frontend package keeps a source checkout looking right, but the
    packaged build must use app.ico: only one frontend package survives into a payload,
    and its icon is not necessarily the brand the application is meant to show.
    """
    packaged = config.payload_dir() / "app.ico"
    if packaged.is_file():
        return packaged
    for package in ("dist", "dist_client"):
        directory = config.payload_dir() / "frontend" / package
        if not directory.is_dir():
            continue
        for icon in sorted(directory.glob("*.ico")):
            return icon
    return None


class LineBuffer:
    """Collect provisioning output and push it to the page in batches.

    evaluate_js is a blocking round-trip to the webview -- the first one has to wait
    for the page to finish loading, and every one after that still costs a hop. uv sync
    emits thousands of lines, so pushing one at a time makes the window unresponsive
    for the whole install, which looks exactly like a hang and is nearly as bad.
    """

    def __init__(self, flush: Callable[[str], None], interval: float = 0.3) -> None:
        self._flush = flush
        self._interval = interval
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="log-flush", daemon=True)
        self._thread.start()

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            # Bound the buffer: a burst should not turn into one enormous evaluate_js.
            if len(self._lines) > 500:
                del self._lines[:-500]

    def _drain(self) -> None:
        with self._lock:
            if not self._lines:
                return
            batch = self._lines
            self._lines = []
        # Losing progress text must never stop provisioning.
        with contextlib.suppress(Exception):
            self._flush("\n".join(batch))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._drain()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()


class SettingsApi:
    """Bridge exposed to the settings page. Every method is called from JavaScript."""

    def __init__(self, launcher: Launcher) -> None:
        self._launcher = launcher
        self._update: updater.UpdateStatus | None = None

    def get_state(self) -> dict[str, Any]:
        stored = state.load()
        return {
            "autostart": autostart.is_enabled(),
            "service_port": stored["service_port"],
            "installed_version": stored["installed_version"],
            "accelerator": stored["accelerator"],
            "platform": sys.platform,
        }

    def set_autostart(self, enabled: bool) -> dict[str, Any]:
        try:
            autostart.set_enabled(bool(enabled))
            state.set("autostart", bool(enabled))
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def set_port(self, port: int) -> dict[str, Any]:
        try:
            port = int(port)
        except (TypeError, ValueError):
            return {"ok": False, "error": "That is not a port number."}
        if not 1024 <= port <= 65535:
            return {"ok": False, "error": "Choose a port between 1024 and 65535."}

        current = state.service_port()
        if port != current and not service_manager.is_port_free(port):
            return {"ok": False, "error": f"Port {port} is already in use by something else."}

        try:
            service_manager.set_port(port)
            self._launcher.service.restart()
            self._launcher.reload_window()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def components(self) -> list[dict[str, Any]]:
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "installed": c.installed,
                "command": c.command,
                "url": c.url,
                "note": c.note,
            }
            for c in component_manager.load()
        ]

    def check_update(self) -> dict[str, Any]:
        self._update = updater.check()
        return {
            "current": self._update.current,
            "latest": self._update.latest,
            "available": self._update.available,
        }

    def open_release_page(self) -> None:
        if self._update:
            updater.open_release_page(self._update)

    def open_url(self, url: str) -> None:
        if str(url).startswith(("http://", "https://")):
            webbrowser.open(url)


class Launcher:
    def __init__(self) -> None:
        self.service = service_manager.ServiceManager()
        self.window: Any = None
        self.settings_window: Any = None
        self.tray: Any = None
        self._quitting = threading.Event()

    # ---------------------------------------------------------------- window

    def _js(self, script: str) -> None:
        """Push a UI update, tolerating a window that is mid-navigation or gone."""
        if self.window is None:
            return
        with contextlib.suppress(Exception):
            self.window.evaluate_js(script)

    def reload_window(self) -> None:
        if self.window is not None:
            self.window.load_url(config.service_url(self.service.port))

    def show_window(self) -> None:
        if self.window is not None:
            self.window.show()

    def open_in_browser(self) -> None:
        """The same interface, in the system browser.

        Both entry points reach the full WebUI; this one exists for people who want it
        in a real browser tab, with their own extensions and zoom.
        """
        webbrowser.open(config.service_url(self.service.port))

    def open_settings(self) -> None:
        import webview

        if self.settings_window is not None:
            try:
                self.settings_window.show()
                return
            except Exception:
                self.settings_window = None

        self.settings_window = webview.create_window(
            f"{config.APP_TITLE} — Settings",
            str(UI_DIR / "settings.html"),
            js_api=SettingsApi(self),
            width=680,
            height=760,
        )

    def _on_closing(self) -> bool:
        """Hide instead of exiting.

        A training run can last hours. Closing the window is how people put an
        application away, not how they abandon a job, so only Quit stops the service.
        """
        if self._quitting.is_set():
            return True
        if self.window is not None:
            self.window.hide()
        return False

    # ------------------------------------------------------------- first run

    def _report_blocking(self, report: preflight.Report) -> None:
        lines = ["This machine is missing something setup needs:", ""]
        for check in report.blocking:
            lines.append(f"  - {check.label}: {check.detail}")
            if check.remedy:
                lines.append(f"      {check.remedy}")
        self._js(f"window.showError({_js_string(chr(10).join(lines))})")

    def _choose_port(self, requested: int, suggested: int) -> int:
        """Ask rather than move quietly.

        Silently taking a different port leaves someone with a working application on
        an address they never chose, and a stale one written into their configuration.
        """
        import webview

        answer = webview.windows[0].create_confirmation_dialog(
            "Port in use",
            f"Port {requested} is already taken. Use {suggested} instead?",
        )
        return suggested if answer else requested

    def bootstrap(self) -> None:
        """Everything between launch and a usable window. Runs off the main thread."""
        try:
            log.info("bootstrap: preflight")
            report = preflight.run()
            if not report.ok:
                self._report_blocking(report)
                return

            provisioned = provision.is_provisioned()
            log.info("bootstrap: already provisioned = %s", provisioned)

            if not provisioned:
                phases = [{"key": p.key, "label": p.label} for p in provision.iter_phases()]
                log.info("bootstrap: sending %d phases to the page", len(phases))
                self._js(f"window.setPhases({_js_json(phases)})")

                log.info("bootstrap: resolving the service port")
                port = service_manager.resolve_first_run_port(self._choose_port)
                state.set("service_port", port)
                env_file.ensure(port)
                if port != config.DEFAULT_SERVICE_PORT:
                    # The frontend's address is baked in at packaging time, so a
                    # non-default port has to be written there too or the UI loads and
                    # then cannot reach the backend.
                    log.info("bootstrap: rewriting the frontend config for port %s", port)
                    service_manager.set_port(port)

                buffer = LineBuffer(lambda text: self._js(f"window.appendLog({_js_string(text)})"))
                try:
                    log.info("bootstrap: handing over to provisioning")
                    provision.run(
                        on_line=buffer.add,
                        on_phase=lambda key, _label: self._js(f"window.setPhase({_js_string(key)})"),
                    )
                finally:
                    buffer.close()

            log.info("bootstrap: starting the service")
            self.service.start()
            log.info("bootstrap: loading the interface")
            self.reload_window()
            log.info("bootstrap: done")

        except Exception as exc:
            log.exception("bootstrap failed")
            self._js(f"window.showError({_js_string(str(exc))})")

    # ------------------------------------------------------------------ tray

    def build_tray(self) -> Any:
        import pystray
        from PIL import Image

        icon_file = _icon_path()
        image = Image.open(icon_file) if icon_file else Image.new("RGB", (64, 64), (47, 111, 237))

        def backend_status(_item: Any) -> str:
            return {
                "ours": "Backend: running",
                "free": "Backend: stopped",
                "foreign": "Backend: port taken by another program",
            }[self.service.status()]

        menu = pystray.Menu(
            pystray.MenuItem("Open window", lambda: self.show_window(), default=True),
            pystray.MenuItem("Open full interface in browser", lambda: self.open_in_browser()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(backend_status, None, enabled=False),
            pystray.MenuItem("Start backend", lambda: self._guard(self.service.start)),
            pystray.MenuItem("Restart backend", lambda: self._guard(self.service.restart)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", lambda: self.open_settings()),
            pystray.MenuItem("Check for updates", lambda: self._check_updates()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self.quit()),
        )
        return pystray.Icon(config.APP_ID, image, config.APP_TITLE, menu)

    def _guard(self, action: Any) -> None:
        try:
            action()
        except Exception as exc:
            self._js(f"window.showError({_js_string(str(exc))})")

    def _check_updates(self) -> None:
        status = updater.check()
        if status.available:
            updater.open_release_page(status)

    # ------------------------------------------------------------------ exit

    def quit(self) -> None:
        """The only path that stops the service."""
        self._quitting.set()
        try:
            self.service.stop()
        finally:
            if self.tray is not None:
                self.tray.stop()
            if self.window is not None:
                with contextlib.suppress(Exception):
                    self.window.destroy()


def _js_string(value: str) -> str:
    import json

    return json.dumps(value)


def _js_json(value: Any) -> str:
    import json

    return json.dumps(value)


def diagnose() -> int:
    """Walk the startup sequence without opening a window, and say what it found.

    A windowed build has no console and a hang leaves nothing behind, so this exists to
    answer "where did it stop" from a terminal -- and to be runnable on a user's machine
    when the application itself will not start.

    It stops short of actually provisioning: it reports the command it would run.
    """
    applog.banner()
    print(f"log file: {applog.log_path()}\n")

    print("-- preflight " + "-" * 55)
    report = preflight.run()
    print(preflight.describe(report.checks))
    print("verdict:", "ready" if report.ok else
          "BLOCKED by " + ", ".join(c.label for c in report.blocking))

    print("\n-- state " + "-" * 59)
    for key, value in sorted(state.load().items()):
        print(f"  {key:<18} {value}")
    print(f"  {'is_provisioned()':<18} {provision.is_provisioned()}")

    print("\n-- ports " + "-" * 59)
    port = state.service_port()
    print(f"  configured         {port}")
    print(f"  status             {service_manager.probe(port)}")
    print(f"  venv python        {service_manager.venv_python()} "
          f"({'present' if service_manager.venv_python().is_file() else 'absent'})")

    print("\n-- .env " + "-" * 60)
    env_path = env_file.env_path()
    print(f"  path               {env_path} ({'exists' if env_path.is_file() else 'missing'})")
    existing = env_file.read_keys(env_path)
    for key, value in sorted(env_file.desired(port).items()):
        mark = "set" if key in existing else "WOULD ADD"
        print(f"  {key:<18} {mark:<10} {existing.get(key, value)}")

    print("\n-- provisioning " + "-" * 52)
    try:
        command = provision.build_command()
        print(f"  command            {' '.join(command)}")
        print(f"  cwd                {config.backend_dir()}")
        environment = provision.build_env()
        for key in ("UV_PROJECT_ENVIRONMENT", "HF_HOME"):
            print(f"  {key:<18} {environment.get(key)}")
        print(f"  disk free          {provision.disk_free_gb(config.data_dir()):.1f} GB")
    except Exception as exc:
        print(f"  WOULD FAIL: {exc}")

    print("\n-- window resources " + "-" * 48)
    for name in ("progress.html", "settings.html"):
        path = UI_DIR / name
        print(f"  {name:<18} {'ok' if path.is_file() else 'MISSING'}  {path}")
    icon = _icon_path()
    print(f"  {'icon':<18} {icon or 'none found'}")

    print("\nnothing was installed or started.")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="launcher", description=__doc__)
    parser.add_argument("--diagnose", action="store_true",
                        help="report what startup would do, without a window and without installing")
    args = parser.parse_args(argv)

    if args.diagnose:
        return diagnose()

    applog.banner()

    import webview

    launcher = Launcher()
    launcher.window = webview.create_window(
        config.APP_TITLE,
        str(UI_DIR / "progress.html"),
        width=1280,
        height=860,
        min_size=(940, 620),
    )
    launcher.window.events.closing += launcher._on_closing

    launcher.tray = launcher.build_tray()
    # pywebview needs the main thread on Windows, so the tray goes to a worker. It is
    # a daemon: if the window loop ends unexpectedly the process must still exit.
    threading.Thread(target=launcher.tray.run, name="tray", daemon=True).start()

    log.info("tray started")

    if state.get("autostart") and not autostart.is_enabled():
        # Default-on has to be applied once, on the first run that owns the setting.
        with contextlib.suppress(Exception):
            autostart.enable()

    log.info("entering the window loop")
    webview.start(launcher.bootstrap, debug=False)
    log.info("window loop ended; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
