"""Tray application: first-run setup, then a window onto the existing WebUI.

Division of labour, matching what a desktop AI tool is expected to do: the WebUI owns
the work (training and inference settings), and the tray owns the environment (the
service, its port, optional components, updates).

Threading is not incidental here. pywebview must own the main thread on Windows, so
the tray icon runs on a worker thread and all of the first-run sequencing happens in
the callable passed to webview.start().
"""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from . import (
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
        try:
            self.window.evaluate_js(script)
        except Exception:
            pass

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
            report = preflight.run()
            if not report.ok:
                self._report_blocking(report)
                return

            if not provision.is_provisioned():
                phases = [{"key": p.key, "label": p.label} for p in provision.iter_phases()]
                self._js(f"window.setPhases({_js_json(phases)})")

                port = service_manager.resolve_first_run_port(self._choose_port)
                state.set("service_port", port)
                env_file.ensure(port)
                if port != config.DEFAULT_SERVICE_PORT:
                    # The frontend's address is baked in at packaging time, so a
                    # non-default port has to be written there too or the UI loads and
                    # then cannot reach the backend.
                    service_manager.set_port(port)

                provision.run(
                    on_line=lambda line: self._js(f"window.appendLog({_js_string(line)})"),
                    on_phase=lambda key, _label: self._js(f"window.setPhase({_js_string(key)})"),
                )

            self.service.start()
            self.reload_window()

        except Exception as exc:
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
                try:
                    self.window.destroy()
                except Exception:
                    pass


def _js_string(value: str) -> str:
    import json

    return json.dumps(value)


def _js_json(value: Any) -> str:
    import json

    return json.dumps(value)


def main() -> int:
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

    if state.get("autostart") and not autostart.is_enabled():
        # Default-on has to be applied once, on the first run that owns the setting.
        try:
            autostart.enable()
        except Exception:
            pass

    webview.start(launcher.bootstrap, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
