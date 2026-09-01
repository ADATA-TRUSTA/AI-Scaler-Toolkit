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
import time
import webbrowser
from pathlib import Path
from typing import Any

from . import (
    applog,
    autostart,
    component_manager,
    config,
    env_file,
    preflight,
    progress,
    provision,
    service_manager,
    state,
    updater,
)

UI_DIR = config.resource_root() / "launcher" / "ui"

log = applog.get("main")


def _icon_data_uri() -> str:
    """The window is HTML, so the icon has to travel as data -- a file:// reference
    from a page loaded over http would be blocked."""
    path = _icon_path()
    if path is None:
        return ""
    try:
        import base64

        return "data:image/x-icon;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


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


class WindowApi:
    """Exposed to the setup page. It only needs one thing: the button at the end."""

    def __init__(self, launcher: Launcher) -> None:
        self._launcher = launcher

    def open_app(self) -> None:
        self._launcher.reload_window()


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
        """The full interface, in the system browser.

        The ?ui=full marker matters. config.json asks for the collapsed view because
        that is right for the desktop window, and the page has no other way to tell
        the two apart -- without it, "open the full interface" would open the narrow
        one and offer a button to expand it, which is absurd in a browser tab.
        """
        webbrowser.open(config.service_url(self.service.port) + "/?ui=full")

    def open_settings(self) -> None:
        """Open the settings window, or raise the one already open.

        The reference has to be cleared when the window closes. Calling show() on a
        closed pywebview window does not raise -- it quietly does nothing -- so
        relying on an exception to detect it meant settings could be opened exactly
        once per session.
        """
        import webview

        if self.settings_window is not None:
            log.info("settings window already open; raising it")
            with contextlib.suppress(Exception):
                self.settings_window.show()
                self.settings_window.restore()
            return

        log.info("opening the settings window")
        window = webview.create_window(
            f"{config.APP_TITLE} — Settings",
            str(UI_DIR / "settings.html"),
            js_api=SettingsApi(self),
            width=720,
            height=820,
            min_size=(600, 560),
        )
        self.settings_window = window

        def _forget() -> None:
            log.info("settings window closed")
            self.settings_window = None

        window.events.closed += _forget

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

    def _set_step(self, name: str, fraction: float | None, detail: str, elapsed: int) -> None:
        value = "null" if fraction is None else f"{fraction:.4f}"
        self._js(
            f"window.setProgress({_js_string(name)}, {value}, {_js_string(detail)}, {elapsed})"
        )

    def _report_progress(self, update: progress.Progress) -> None:
        self._set_step(update.label, update.fraction, update.detail, update.elapsed)

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

            self._js(f"window.setIcon({_js_string(_icon_data_uri())})")

            provisioned = provision.is_provisioned()
            log.info("bootstrap: already provisioned = %s", provisioned)
            started_at = time.monotonic()

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

                log.info("bootstrap: handing over to provisioning")
                provision.run(
                    on_phase=lambda key, _label: self._js(f"window.setPhase({_js_string(key)})"),
                    on_progress=self._report_progress,
                )

            log.info("bootstrap: starting the service")
            self._js(f"window.setPhase({_js_string('done')})")
            self._set_step("Starting the service", None, "loading the GPU runtime", 0)
            self.service.start(
                on_wait=lambda seconds: self._set_step(
                    "Starting the service", None, "loading the GPU runtime", seconds
                )
            )

            if provisioned:
                # Not a first run: there is nothing to celebrate and nobody waiting,
                # so go straight to the interface.
                log.info("bootstrap: loading the interface")
                self.reload_window()
            else:
                # After a wait this long, finishing silently by swapping the page reads
                # as one more thing happening rather than as "done".
                total = int(time.monotonic() - started_at)
                minutes, seconds = divmod(total, 60)
                spent = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
                accelerator = state.get("accelerator") or "your CPU"
                log.info("bootstrap: setup finished in %s, waiting for the user", spent)
                self._js(
                    "window.showDone("
                    + _js_string(f"Ready in {spent}, using {accelerator}.")
                    + ")"
                )
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
        js_api=WindowApi(launcher),
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
