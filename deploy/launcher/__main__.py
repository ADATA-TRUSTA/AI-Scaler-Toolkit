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
    brand,
    components,
    config,
    env_file,
    maintenance,
    preflight,
    progress,
    provision,
    remedy,
    service_manager,
    single_instance,
    state,
    updater,
)

UI_DIR = config.resource_root() / "launcher" / "ui"

# Setup needs a small window: it shows one heading, one bar and a row of pips. The
# interface that follows needs a large one, and the shape depends on whether the chat
# panel is showing -- a column laid out for 480px in a 1320px frame is mostly empty.
SETUP_SIZE = (660, 580)
COMPACT_SIZE = (640, 880)
EXPANDED_SIZE = (1320, 880)

# Long enough to read "Setup complete" and the time it took, short enough not to be a
# step. The button was removed: after a nine-minute wait, asking for a click before
# showing the thing that was being installed is friction, not ceremony.
DONE_PAUSE_SECONDS = 2.2

# Injected into every page the window loads, so the interface can tell the window to
# change shape without knowing anything about how the window is implemented.
HOST_HOOK = """
window.__hostViewChange = function (compact) {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.set_compact) {
    window.pywebview.api.set_compact(!!compact);
  }
};
"""

log = applog.get("main")


def _mark_data_uri() -> str:
    """The application mark for the setup window.

    Sent as data because the page is loaded from a file: a reference to the payload
    directory would not resolve, and would break again once the window navigates to the
    interface over http.
    """
    path = config.payload_dir() / layout_mark_name()
    if not path.is_file():
        log.info("no application mark at %s; the lettered tile stands in", path)
        return ""
    try:
        import base64

        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        log.warning("could not read the application mark at %s", path)
        return ""
    log.info("application mark: %s (%d bytes)", path.name, path.stat().st_size)
    return "data:image/png;base64," + data


def layout_mark_name() -> str:
    """Kept in step with what packaging writes, without importing the packaging code
    into the shipped application."""
    return "app-mark.png"


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
    """Exposed to every page the main window loads."""

    def __init__(self, launcher: Launcher) -> None:
        self._launcher = launcher

    def open_app(self) -> None:
        self._launcher.reload_window()

    def set_compact(self, compact: bool) -> None:
        """The interface hid or showed its chat panel; match the window to it."""
        self._launcher.set_window_size(COMPACT_SIZE if compact else EXPANDED_SIZE)


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
            "last_setup": stored["last_setup"],
            "package_count": stored["package_count"],
            # Shown read-only. The launcher owns the service port and nothing else --
            # these two belong to an inference configuration, and the load-model
            # request carries its own, so anything written here would be overwritten.
            "llama_port": config.LLAMA_SERVER_PORT,
            "vllm_port": config.VLLM_PORT,
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

    def repair_environment(self) -> dict[str, Any]:
        """Throw the installed environment away and build it again.

        The counterpart to this -- removing the application itself -- belongs to the
        uninstaller, not to a settings page. This one exists because the environment
        can be broken without the application being broken: a half-finished install, a
        virus scanner that ate a file, a torch build that no longer imports. Repairing
        it should not mean reinstalling everything.

        The model cache is kept. It is not part of what breaks and re-downloading tens
        of gigabytes to fix a virtual environment would be its own kind of failure.
        """
        try:
            self._launcher.begin_repair()
            return {"ok": True}
        except Exception as exc:
            log.exception("could not start the repair")
            return {"ok": False, "error": str(exc)}

    def check_update(self) -> dict[str, Any]:
        self._update = updater.check()
        log.info("update check: latest=%s available=%s reason=%s",
                 self._update.latest, self._update.available, self._update.reason)
        return {
            "current": self._update.current,
            "latest": self._update.latest,
            "available": self._update.available,
            "reason": self._update.reason,
        }

    def open_release_page(self) -> None:
        if self._update:
            updater.open_release_page(self._update)

    def open_url(self, url: str) -> None:
        if str(url).startswith(("http://", "https://")):
            webbrowser.open(url)


    # ------------------------------------------------------------- components

    def list_components(self) -> list[dict[str, Any]]:
        return components.listing()

    def install_component(self, component_id: str) -> dict[str, Any]:
        started, error = components.start(str(component_id))
        return {"ok": started, "error": error}

    def component_progress(self) -> dict[str, Any] | None:
        """Polled by the page while an install runs.

        Returning the whole log every second is wasteful in principle and free in
        practice -- it is a few hundred lines over a local bridge -- and it means a
        page reopened mid-install shows the history rather than starting blank.
        """
        return components.current_job()


class Launcher:
    def __init__(self) -> None:
        self.service = service_manager.ServiceManager()
        self.window: Any = None
        self.settings_window: Any = None
        self.tray: Any = None
        self._quitting = threading.Event()
        # What the window is currently sized to, so a page asking for the size it is
        # already at costs nothing.
        self._window_size: tuple[int, int] | None = None

    # ---------------------------------------------------------------- window

    def _js(self, script: str) -> None:
        """Push a UI update, tolerating a window that is mid-navigation or gone."""
        if self.window is None:
            return
        with contextlib.suppress(Exception):
            self.window.evaluate_js(script)

    def set_window_size(self, size: tuple[int, int]) -> None:
        if self.window is None:
            return
        width, height = size
        # Every page reports the width it needs when it mounts, so the size it is
        # already at is reported constantly. Acting on those made the log read like a
        # window resizing itself in a loop -- which is what it looked like to me when
        # someone was simply switching between two pages that want different widths.
        if self._window_size == (width, height):
            return
        self._window_size = (width, height)
        log.info("resizing the window to %dx%d", width, height)
        with contextlib.suppress(Exception):
            self.window.resize(width, height)

    def reload_window(self) -> None:
        if self.window is None:
            return
        compact = self._starts_compact()
        # Asked for in the URL, because the page cannot tell a window from a browser
        # tab any other way -- and config.json, being per-deployment, describes both.
        url = config.interface_url(self.service.port, compact=compact)
        # Sized before the page loads rather than after, so the interface does not
        # appear at the setup window's size and then jump.
        self.set_window_size(COMPACT_SIZE if compact else EXPANDED_SIZE)
        log.info("loading %s", url)
        self.window.load_url(url)

    def _starts_compact(self) -> bool:
        """What the packaged config.json asks for. The interface may be toggled away
        from it later, which is what set_compact is for."""
        try:
            import json

            data = json.loads(config.frontend_config().read_text(encoding="utf-8"))
            return data.get("hideChatPanel") is True
        except (OSError, ValueError):
            return False

    def _inject_host_hook(self) -> None:
        """Re-injected on every page load, because navigation clears it."""
        self._js(HOST_HOOK)

    def show_window(self) -> None:
        if self.window is None:
            return
        with contextlib.suppress(Exception):
            self.window.show()
        with contextlib.suppress(Exception):
            # Closing the window hides it and minimising leaves it minimised; a
            # request to come forward has to undo either.
            self.window.restore()

    def open_in_browser(self) -> None:
        """The full interface, in the system browser.

        No marker needed: only ?ui=compact collapses the view, and only the desktop
        window ever asks for that. A browser gets the full interface by default, which
        is the only thing that makes sense in a browser tab.
        """
        webbrowser.open(config.service_url(self.service.port))

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

        def _paint() -> None:
            with contextlib.suppress(Exception):
                window.evaluate_js(
                    f"window.setBrandBackground({_js_string(brand.page_background())})"
                )

        window.events.loaded += _paint

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

    def _show_progress_page(self, heading: str, detail: str = "") -> None:
        """Put the window in front showing the setup page, ready to be driven.

        The same surface first-run uses. A service action that takes a minute or two
        with nothing on screen is indistinguishable from one that did nothing, which is
        exactly how "Start backend" and "Restart backend" behaved: the work happened
        and no one could tell.
        """
        self.set_window_size(SETUP_SIZE)
        with contextlib.suppress(Exception):
            self.window.load_url(str(UI_DIR / "progress.html"))
        self.show_window()
        time.sleep(1.0)  # the page has to load before anything can be pushed into it
        self._js(f"window.setBrandBackground({_js_string(brand.page_background())})")
        self._js(f"window.setMark({_js_string(_mark_data_uri())})")
        self._set_step(heading, None, detail, 0)

    def open_window(self) -> None:
        """Show the window, and make sure there is something behind it.

        Opening used to only raise the window, which is fine while the backend happens
        to be up and shows an unreachable interface when it is not. Since nothing in
        the tray starts the backend any more, opening has to: all or nothing, and this
        is the "all".
        """
        self.show_window()
        if self.service.status() == "ours":
            return
        log.info("opening the window with no backend running; starting it")
        self.begin_service_action("start")

    def begin_service_action(self, kind: str) -> None:
        """Start, stop or restart the backend, with the window showing what happens."""
        threading.Thread(target=self._service_action, args=(kind,),
                         name=f"service-{kind}", daemon=True).start()

    def _service_action(self, kind: str) -> None:
        headings = {
            "start": ("Starting the service", "loading the GPU runtime"),
            "restart": ("Restarting the service", "loading the GPU runtime"),
            "stop": ("Stopping the service", ""),
        }
        heading, detail = headings[kind]
        log.info("tray: %s backend", kind)
        try:
            self._show_progress_page(heading, detail)
            if kind == "stop":
                self.service.unwatch()
                self.service.stop()
                self._set_step("The service is stopped", 1.0,
                               "Start it again from the tray when you need it", 0)
                return

            action = self.service.start if kind == "start" else self.service.restart
            action(on_wait=lambda seconds: self._set_step(heading, None, detail, seconds))
            self.service.watch(self._service_lost)
            self.reload_window()
        except Exception as exc:
            log.exception("tray: %s failed", kind)
            self._js(f"window.showError({_js_string(str(exc))})")

    def begin_repair(self) -> None:
        """Stop the service, clear the environment, and set up again in the window."""
        log.info("repair requested")
        if self.settings_window is not None:
            with contextlib.suppress(Exception):
                self.settings_window.destroy()
            self.settings_window = None

        threading.Thread(target=self._repair, name="repair", daemon=True).start()

    def _repair(self) -> None:
        try:
            self.service.unwatch()
            self.service.stop()

            self.set_window_size(SETUP_SIZE)
            self.window.load_url(str(UI_DIR / "progress.html"))
            self.show_window()
            # The page has to finish loading before anything can be pushed into it.
            time.sleep(1.0)
            self._js(f"window.setBrandBackground({_js_string(brand.page_background())})")
            self._js(f"window.setMark({_js_string(_mark_data_uri())})")
            self._set_step("Removing the current environment", None, "", 0)

            result = maintenance.provisioning_output()
            log.info("repair: %s", result.summary())
            if not result.ok:
                raise RuntimeError(
                    "Could not remove " + ", ".join(result.failed) +
                    ". Something is still using it -- close the application and try again."
                )

            provision.run(
                on_phase=lambda key, _label: self._js(f"window.setPhase({_js_string(key)})"),
                on_progress=self._report_progress,
                on_phases=self._publish_phases,
            )
            self.service.start(on_wait=lambda seconds: self._set_step(
                "Starting the service", None, "loading the GPU runtime", seconds))
            self.service.watch(self._service_lost)
            self.reload_window()
            log.info("repair: finished")
        except Exception as exc:
            log.exception("repair failed")
            self._js(f"window.showError({_js_string(str(exc))})")

    def _service_lost(self, detail: str) -> None:
        """The backend went away after it had started.

        Said plainly rather than restarted automatically: something that just crashed
        usually crashes again, and a silent restart loop hides the reason. The log
        keeps whatever it printed on the way out.
        """
        log.warning("service lost: %s", detail)
        message = (
            f"The backend service stopped ({detail}).\n\n"
            "Use “Restart backend” in the tray menu to bring it back. "
            f"What it printed before stopping is in {config.log_dir() / 'service.log'}."
        )
        self._js(f"window.showError({_js_string(message)})")

    def _publish_phases(self, phases) -> None:
        """Send the step list once, with the steps this machine will skip left out.

        A phase that never runs would otherwise sit uncoloured in the middle of the
        row for the whole install, which reads as something broken rather than as a
        check that does not apply to this hardware.
        """
        items = [{"key": phase.key, "label": phase.label} for phase in phases]
        log.info("bootstrap: %d steps apply here (%s)", len(items),
                 ", ".join(item["key"] for item in items))
        self._js(f"window.setPhases({_js_json(items)})")

    def _report_progress(self, update: progress.Progress) -> None:
        self._set_step(update.label, update.fraction, update.detail, update.elapsed)

    # ------------------------------------------------------------- first run

    def _install_missing(self, report: preflight.Report) -> preflight.Report:
        """Install the prerequisites setup itself needs, then check again.

        These are ours to fix: without them setup cannot run at all, and telling
        someone to go and install a C++ runtime before they can use the thing they
        just downloaded is not an installation experience. Optional third-party tools
        are the opposite case and are only ever shown, never installed.
        """
        fixable = [c for c in report.blocking if remedy.can_fix(c.fixer)]
        unfixable = [c for c in report.blocking if not remedy.can_fix(c.fixer)]
        if not fixable:
            return report

        if unfixable:
            # No point spending minutes on downloads when something else will stop us
            # anyway.
            log.info("not attempting repairs: %s cannot be fixed from here",
                     ", ".join(c.label for c in unfixable))
            return report

        for check in fixable:
            log.info("installing missing prerequisite: %s", check.label)
            self._set_step(f"Installing {check.label}", None, "one-time, needed by setup", 0)
            if not remedy.apply(check.fixer):
                log.warning("could not install %s", check.label)

        log.info("re-running preflight after the repairs")
        return preflight.run()

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

        The dialog is the platform's own, and on Windows that means OK and Cancel, not
        Yes and No. So the text names what OK does instead of asking a question the
        buttons cannot answer -- "Use 8572 instead?" above a Cancel button reads as a
        question with the wrong answers.
        """
        import webview

        accepted = webview.windows[0].create_confirmation_dialog(
            f"Port {requested} is in use",
            f"Another program is already using port {requested}.\n\n"
            f"OK: start on port {suggested} instead.\n"
            f"Cancel: stop, so you can free port {requested} first.",
        )
        if accepted:
            return suggested
        # Declining is a decision, not a failure to resolve. Returning the busy port
        # would surface as "port N is also in use", which reads as a contradiction of
        # what the user just chose.
        raise RuntimeError(
            f"Setup stopped because port {requested} is in use. Close whatever is "
            f"using it and start this application again, or accept port {suggested} "
            "when asked."
        )

    def bootstrap(self) -> None:
        """Everything between launch and a usable window. Runs off the main thread."""
        try:
            self._js(f"window.setBrandBackground({_js_string(brand.page_background())})")
            self._js(f"window.setMark({_js_string(_mark_data_uri())})")

            log.info("bootstrap: preflight")
            report = preflight.run()
            if not report.ok:
                report = self._install_missing(report)
            if not report.ok:
                self._report_blocking(report)
                return

            provisioned = provision.is_provisioned()
            log.info("bootstrap: already provisioned = %s", provisioned)
            started_at = time.monotonic()

            if not provisioned:
                # A previous attempt that started and never finished. Nothing is lost
                # -- setup_env installs only what is missing, so the finished parts go
                # past in seconds -- but silence about it is what makes a resumed run
                # look like the whole thing starting over on an application the user
                # thought was installed.
                resuming = state.get("setup_started")
                state.set("setup_started", True)
                if resuming:
                    log.info("bootstrap: the previous setup did not finish; resuming")
                    self._set_step("Resuming setup", None,
                                   "picking up where the last attempt stopped", 0)

                # The step list is published by provisioning itself, once it knows the
                # accelerator -- see _publish_phases.
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
                    on_phases=self._publish_phases,
                )

            if self._quitting.is_set():
                # Quit landed while setup was running. It has been cancelled; starting
                # a service now would leave one behind with nothing to manage it.
                log.info("bootstrap: quitting, so not starting the service")
                return

            log.info("bootstrap: starting the service")
            self._js(f"window.setPhase({_js_string('done')})")
            self._set_step("Starting the service", None, "loading the GPU runtime", 0)
            self.service.start(
                on_wait=lambda seconds: self._set_step(
                    "Starting the service", None, "loading the GPU runtime", seconds
                )
            )

            self.service.watch(self._service_lost)

            if provisioned:
                # Not a first run: there is nothing to celebrate and nobody waiting,
                # so go straight to the interface.
                log.info("bootstrap: loading the interface")
                self.reload_window()
            else:
                # Say it finished, then go. Swapping the page silently after a
                # nine-minute wait reads as one more thing happening rather than as
                # done; making the user click reads as a chore.
                total = int(time.monotonic() - started_at)
                minutes, seconds = divmod(total, 60)
                spent = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
                accelerator = state.get("accelerator") or "your CPU"
                state.set("last_setup", f"{spent} ({accelerator})")
                state.set("setup_started", False)
                log.info("bootstrap: setup finished in %s", spent)
                self._js("window.showDone(" + _js_string(f"Ready in {spent}, using {accelerator}.") + ")")
                time.sleep(DONE_PAUSE_SECONDS)
                self.reload_window()
            log.info("bootstrap: done")

        except provision.Cancelled:
            # Quit landed mid-setup. Nothing is wrong and there is nothing on screen
            # left to tell.
            log.info("bootstrap: setup was cancelled by quitting")
        except Exception as exc:
            log.exception("bootstrap failed")
            self._js(f"window.showError({_js_string(str(exc))})")

    # ------------------------------------------------------------------ tray

    def build_tray(self) -> Any:
        import pystray
        from PIL import Image

        icon_file = _icon_path()
        image = Image.open(icon_file) if icon_file else Image.new("RGB", (64, 64), (47, 111, 237))

        # Four items, and none of them mention the backend.
        #
        # This menu used to be a service console: a status line plus Start, Stop and
        # Restart. All of it exposed an implementation detail as something to manage,
        # and every one of those items cost a bug -- a status line that froze, a Start
        # offered while it was already running, a Stop that silently failed on a
        # service this launcher had adopted rather than spawned.
        #
        # The rule is now all or nothing: opening the window makes sure the backend is
        # up, and Quit takes everything down. There is no in-between state for anyone
        # to manage, so there is nothing to put in a menu.
        #
        # Checking for updates went to the settings page. It belongs where someone
        # goes to look for it, not behind a right-click.
        menu = pystray.Menu(
            pystray.MenuItem("Open window", lambda: self.open_window(), default=True),
            pystray.MenuItem("Open full interface in browser", lambda: self.open_in_browser()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", lambda: self.open_settings()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self.quit()),
        )
        return pystray.Icon(config.APP_ID, image, config.APP_TITLE, menu)

    def _guard(self, action: Any) -> None:
        try:
            action()
        except Exception as exc:
            self._js(f"window.showError({_js_string(str(exc))})")

    def _notify(self, title: str, message: str) -> None:
        """Say it in a dialog the reader has to dismiss.

        This was a tray balloon, and a balloon is not an answer: Windows suppresses
        them under focus assist, per-application notification settings, and during
        full-screen use, so "Check for updates" was reported as doing nothing at all.
        A message box cannot be silently swallowed, and an update check is a thing
        someone asked for and is waiting on -- interrupting them is the point.
        """
        log.info("notification: %s -- %s", title, message)
        if sys.platform != "win32":
            with contextlib.suppress(Exception):
                if self.tray is not None:
                    self.tray.notify(message, title)
            return

        import ctypes

        MB_OK, MB_ICONINFORMATION, MB_SETFOREGROUND, MB_TOPMOST = 0x0, 0x40, 0x10000, 0x40000

        def show() -> None:
            with contextlib.suppress(Exception):
                ctypes.windll.user32.MessageBoxW(
                    None, message, f"{config.APP_TITLE} — {title}",
                    MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST,
                )

        # On its own thread. A message box blocks the thread that raised it, and these
        # are raised from menu callbacks, which pystray runs on the thread that also
        # serves the tray icon -- so a dialog left on screen made the whole tray
        # unresponsive, with no menu until someone found the dialog and dismissed it.
        threading.Thread(target=show, name="notify", daemon=True).start()

    # ------------------------------------------------------------------ exit

    def quit(self) -> None:
        """The only path that stops the service.

        A component install is asked about first, and the reason is a real casualty:
        an install interrupted partway left a third-party tool with a virtual
        environment and no packages in it -- broken, and reporting itself as present.
        Its installer is a child of this process reading from a pipe we own, so this
        process going away is enough to abort it mid-write.

        Closing the settings window is safe; the install runs here, not there. Quitting
        is what takes it down, so quitting is what asks.

        Work in the backend is asked about for a different reason. The service holds
        loaded models and serves them to other applications, not only to this window,
        so stopping it can break something the user is doing somewhere else entirely --
        which they will not connect to having quit a tray icon. Only when something is
        actually loaded or training, because a confirmation that appears every time is
        one nobody reads.
        """
        if not self._confirm_quit_while_busy():
            log.info("quit cancelled: the backend is in use")
            return

        job = components.current_job()
        if job and job["running"] and not self._confirm_quit_during_install(job["id"]):
            log.info("quit cancelled: %s is still installing", job["id"])
            return
        self._quitting.set()
        # From here on a launch must not be told "already running, I will come
        # forward": this copy is on its way out and will not come forward, so the
        # click did nothing and the user had to click a second time.
        with contextlib.suppress(Exception):
            single_instance.begin_quitting()

        # Everything visible goes now, and the service is stopped afterwards in
        # shutdown(), once the window loop has ended. Stopping first meant up to
        # fifteen seconds of a window that would not close while the backend was
        # terminated and its port waited on -- the application looked hung at exactly
        # the moment the user had finished with it.
        if self.tray is not None:
            with contextlib.suppress(Exception):
                self.tray.stop()
        # Every window, not just the main one. pywebview's loop runs until the last
        # window closes, so leaving the settings window open meant Quit removed the
        # tray icon and the main window while the process kept running -- an
        # application with no way left to reach it and no way to quit it.
        for window in (self.settings_window, self.window):
            if window is not None:
                with contextlib.suppress(Exception):
                    window.destroy()
        self.settings_window = None

    def shutdown(self) -> None:
        """Stop the service. Runs after the window loop ends, on the way out.

        Deliberately not part of quit(): the user should not watch a dead window while
        a subprocess is terminated and a port is waited on. By the time this runs there
        is nothing on screen to keep waiting.
        """
        log.info("shutting down the service")
        # Provisioning first, because it is what starts the service. Closing the
        # window ends the window loop but leaves bootstrap running in its thread:
        # setup ran to completion in a process with no interface and then started a
        # backend, while the next launch -- seeing the guard gone -- provisioned and
        # started its own on the same port. Setup is resumable, so stopping it costs
        # the user nothing.
        with contextlib.suppress(Exception):
            if provision.cancel():
                log.info("cancelled the setup that was still running")
        with contextlib.suppress(Exception):
            self.service.unwatch()
        with contextlib.suppress(Exception):
            self.service.stop()

    def _confirm_quit_while_busy(self) -> bool:
        """True to go ahead. Silent unless the backend is holding something."""
        body = service_manager.health(self.service.port, timeout=2.0)
        if not body:
            return True
        doing = []
        if body.get("training_active"):
            doing.append("a training run is in progress")
        if body.get("model_loaded"):
            doing.append("a model is loaded")
        if not doing:
            return True

        log.info("quit while busy: %s", "; ".join(doing))
        return self._ask(
            "The backend is in use",
            f"{' and '.join(doing).capitalize()}.\n\n"
            "Other applications can be using the loaded model through this service, "
            "and quitting stops it for them too.\n\n"
            "OK: quit anyway.\nCancel: leave it running.",
        )

    def _ask(self, title: str, message: str) -> bool:
        """A yes/no the user cannot miss, without needing a window to hang it on."""
        if sys.platform != "win32":
            target = self.settings_window or self.window
            if target is None:
                return True
            with contextlib.suppress(Exception):
                return bool(target.create_confirmation_dialog(title, message))
            return True

        import ctypes

        MB_OKCANCEL, MB_ICONWARNING = 0x1, 0x30
        MB_SETFOREGROUND, MB_TOPMOST = 0x10000, 0x40000
        IDOK = 1
        try:
            answer = ctypes.windll.user32.MessageBoxW(
                None, message, f"{config.APP_TITLE} — {title}",
                MB_OKCANCEL | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST,
            )
        except Exception:
            log.exception("could not ask")
            return True
        return answer == IDOK

    def _confirm_quit_during_install(self, component_id: str) -> bool:
        """Ask before abandoning an install. Defaults to quitting if we cannot ask."""
        import webview

        name = components.BY_ID[component_id].name if component_id in components.BY_ID \
            else component_id
        target = self.settings_window or self.window
        if target is None:
            return True
        try:
            return bool(target.create_confirmation_dialog(
                f"{name} is still installing",
                f"Quitting now interrupts the {name} installer, which can leave it "
                "half-installed.\n\nOK: quit anyway.\nCancel: leave it running.",
            ))
        except Exception:
            log.exception("could not ask about quitting mid-install")
            return True


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
    parser.add_argument(autostart.LOGIN_FLAG, dest="autostart", action="store_true",
                        help="this launch came from the autostart entry, not from a person")
    args = parser.parse_args(argv)

    if args.diagnose:
        return diagnose()

    applog.banner()

    # Logging in is not a request to install anything. Setup takes eight minutes, opens
    # a window and downloads gigabytes, and doing that because someone turned their
    # computer on is a surprise -- particularly since autostart is on by default, so
    # nobody chose it. Whoever launches it deliberately gets the first-run flow; a
    # login gets nothing until then.
    if args.autostart and not provision.is_provisioned():
        log.info("started by the autostart entry with setup not yet done; "
                 "leaving it for a deliberate launch")
        return 0

    import webview

    launcher = Launcher()

    # Before anything else is created. A second launch should raise the window that is
    # already open, not add a tray icon and a second manager for one service -- where
    # "Quit" on either would stop the service under the other.
    if not single_instance.acquire(lambda: launcher.show_window()):
        log.info("another copy is already running; exiting")
        return 0

    # Sized before it is created, not after. Every launch used to open at the setup
    # size and then jump to the interface size once the service answered -- which on
    # an already-installed machine is a small window sitting there for a minute or
    # more and then resizing, for no reason the user can see.
    if provision.is_provisioned():
        width, height = COMPACT_SIZE if launcher._starts_compact() else EXPANDED_SIZE
        log.info("already provisioned; opening at the interface size %dx%d", width, height)
    else:
        width, height = SETUP_SIZE
        log.info("setup is needed; opening at the setup size %dx%d", width, height)

    launcher.window = webview.create_window(
        config.APP_TITLE,
        str(UI_DIR / "progress.html"),
        js_api=WindowApi(launcher),
        width=width,
        height=height,
        min_size=(520, 480),
    )
    launcher.window.events.closing += launcher._on_closing
    # Navigation clears injected globals, so the hook is put back after every load.
    launcher.window.events.loaded += launcher._inject_host_hook

    launcher.tray = launcher.build_tray()
    # pywebview needs the main thread on Windows, so the tray goes to a worker. It is
    # a daemon: if the window loop ends unexpectedly the process must still exit.
    threading.Thread(target=launcher.tray.run, name="tray", daemon=True).start()

    log.info("tray started")

    if state.get("autostart") and not autostart.is_enabled():
        # Default-on has to be applied once, on the first run that owns the setting.
        with contextlib.suppress(Exception):
            autostart.enable()
    else:
        # An existing entry records an absolute path, and this executable moves --
        # a reinstall to a different directory, an upgrade, a developer build
        # replaced by a real one. Checked on every start because the entry going
        # stale produces no error at all: nothing launches at login, silently.
        with contextlib.suppress(Exception):
            if autostart.refresh():
                log.info("autostart pointed at another copy; repointed it here")

    log.info("entering the window loop")
    try:
        webview.start(launcher.bootstrap, debug=False)
    finally:
        # After the windows are gone, so nobody is looking at a frozen one while the
        # backend is terminated. The loop only ends on Quit, and if it ever ends for
        # another reason the service should stop then too.
        launcher.shutdown()
        single_instance.release()
    log.info("window loop ended; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
