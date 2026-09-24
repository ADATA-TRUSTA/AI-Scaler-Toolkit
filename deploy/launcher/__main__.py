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
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import (
    applog,
    autostart,
    brand,
    browser,
    components,
    config,
    env_file,
    firewall,
    hardening,
    maintenance,
    preflight,
    progress,
    provision,
    remedy,
    service_manager,
    shortcuts,
    single_instance,
    starter_model,
    state,
    updater,
)

UI_DIR = config.resource_root() / "launcher" / "ui"

# Setup needs a small window: it shows one heading, one bar and a row of pips. The
# interface that follows needs a large one, and the shape depends on whether the chat
# panel is showing -- a column laid out for 480px in a 1320px frame is mostly empty.
SETUP_SIZE = (660, 580)

# Used only when the screen cannot be established; otherwise the interface is sized to
# it by interface_size below.
COMPACT_SIZE = (640, 880)
EXPANDED_SIZE = (1320, 880)

# The settings column is laid out at a fixed 480px and does not reflow, so the compact
# window has one correct width whatever the screen is -- widening it adds empty space,
# not content. Only the expanded shape, which is that column plus a chat panel that
# takes what is left, has anything to do with a bigger screen.
COMPACT_WIDTH = 640

# How much of the screen the window asks for, and the most it is allowed to take. The
# fraction is what leaves room for the desktop's own furniture, and that slack is doing
# real work: nothing here can measure a dock. Wayland does not tell a client where the
# struts are -- Gdk's get_workarea returns the whole screen, and so does the webview's
# screen.availWidth -- and it does not let a client place its own window either, so a
# window that comes out too wide cannot be measured or moved back afterwards. Leaving
# a fifth of the width unasked-for is what keeps it on screen wherever it is put.
#
# Caps only, deliberately no lower bound. A first version had one, and on a 1024x768
# display it pushed the width up to a figure the screen could not give, which then got
# clamped to 97% of it -- and the dock took the rest. A minimum wider than the screen
# is not a minimum, it is an overflow. MIN_WINDOW_SIZE below is the real floor.
MAX_WINDOW_SIZE = (1600, 1000)
SCREEN_FRACTION = (0.80, 0.85)

# Nothing smaller is usable, so the sizes above are never trimmed below this even on a
# screen that cannot fit it -- a window running off the edge is better than one with no
# room for its own controls.
MIN_WINDOW_SIZE = (520, 480)

# The floor above is the compact shape's: one 480px column and its own frame. The
# expanded shape is that column plus a chat panel, and the column does not reflow, so
# every pixel below this comes out of the chat panel -- which has no minimum of its
# own and simply gets narrower until its contents no longer fit in it.
#
# Not hypothetical. A window dragged narrow while expanded was remembered at 636, and
# every later "Show chat" restored it: 480 for the column, and a 150px strip for the
# chat panel, in which the message area had vanished and only the composer's
# background was still visible. 480 for the column plus 360 for a panel that can hold
# a line of conversation, plus the frame.
EXPANDED_MIN_WIDTH = 880

# Taken off the screen before a window is sized to it. A desktop keeps some of the
# screen for itself -- a top bar, a dock, the window's own title bar -- and a window
# sized to the full height opens with its lower edge under whatever is there. Width is
# less contended than height, hence the difference.
SCREEN_MARGIN = (32, 96)

# How long after asking for a size a resize still counts as ours rather than the user's.
# Not a size tolerance: the window manager does not grant what was asked for -- asking
# for 640x653 produced an event reporting 692x742, its own decorations -- and any
# tolerance wide enough to cover that is wide enough to swallow a real drag.
_OUR_RESIZE_WINDOW = 2.0

# How long the window must sit still before a drag is treated as finished. A drag
# delivers an event per frame: one measured drag produced thirty in two seconds, and
# each would otherwise rewrite the state file for a decision the user had not made yet.
_DRAG_SETTLE_SECONDS = 0.4


def claim_desktop_identity() -> None:
    """Say which application this window belongs to, before one is created.

    A Linux desktop matches a window to its .desktop file -- and so to its icon and its
    place in the dock -- by the window's app_id, which GTK takes from the program name.
    pywebview's GTK backend sets no program name, no WM_CLASS and no application id:
    checked, there is not a line in it that does. So the window arrived anonymous, the
    shell could not pair it with ai-scaler-toolkit.desktop, and the dock showed a
    generic icon beside an entry whose Icon= was correct all along.

    Set here rather than anywhere nearer the window, because GTK reads the program name
    when the window is realised and ignores it afterwards.

    Windows does not work this way -- the taskbar follows the executable, which the
    installer already owns -- and there is nothing to do there.
    """
    if config.IS_WINDOWS:
        return
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib

        GLib.set_prgname(config.APP_ID)
        GLib.set_application_name(config.APP_TITLE)
    except Exception as exc:  # noqa: BLE001 - an unnamed window is a cosmetic loss
        log.info("could not claim the desktop identity: %s", exc)
        return
    log.info("desktop identity: %s (%s)", config.APP_ID, config.APP_TITLE)


def screen_size() -> tuple[int, int] | None:
    """The screen a window will open on, or None when that cannot be established.

    pywebview answers this from its platform layer, which is up before the loop starts,
    so it can be asked while deciding the size to create a window at. A session that
    reports no screens at all -- headless, or an unusual compositor -- gets None, and
    the caller then uses the size it asked for, which is what this did before.

    The first screen, not the largest: that is the primary on every backend that orders
    them, and it is where a window opens unless something else places it.
    """
    try:
        import webview

        screens = webview.screens
    except Exception:  # noqa: BLE001 - any failure here means "cannot tell"
        return None
    for screen in screens or ():
        width, height = getattr(screen, "width", 0), getattr(screen, "height", 0)
        if width and height:
            return int(width), int(height)
    return None


def fit_to_screen(size: tuple[int, int]) -> tuple[int, int]:
    """Trim a wanted size to what the screen can actually show.

    The sizes above are written for a desktop monitor and were applied whatever the
    screen was. On a 1024x768 display -- a VM, a projector, an older laptop -- asking
    for the interface put a 1320x880 window past two edges at once, with the controls
    down the right-hand side of the page unreachable. Showing the chat panel is where
    that surfaced, because it is the only action that makes the window wider.
    """
    screen = screen_size()
    if screen is None:
        return size
    return (
        min(size[0], max(MIN_WINDOW_SIZE[0], screen[0] - SCREEN_MARGIN[0])),
        min(size[1], max(MIN_WINDOW_SIZE[1], screen[1] - SCREEN_MARGIN[1])),
    )


def size_key(compact: bool) -> str:
    return "compact_size" if compact else "expanded_size"


def minimum_size(compact: bool) -> tuple[int, int]:
    """The smallest this shape can be and still show what it is for.

    Per shape, like the remembered size: the compact window is one column, and 520 is
    enough for it. The expanded one is that column plus a chat panel, so its floor is
    wider -- see EXPANDED_MIN_WIDTH.
    """
    if compact:
        return MIN_WINDOW_SIZE
    return (max(MIN_WINDOW_SIZE[0], EXPANDED_MIN_WIDTH), MIN_WINDOW_SIZE[1])


def remembered_size(compact: bool) -> tuple[int, int] | None:
    """The size the user last dragged this shape to, if they ever did.

    Per shape, because the window has two and they are not interchangeable: the
    settings column alone, and that column with the chat panel beside it. Trimmed to
    the current screen on the way out, so a size chosen on a desktop monitor does not
    hang off the edge of the laptop it is next opened on.
    """
    stored = state.get(size_key(compact))
    try:
        width, height = int(stored[0]), int(stored[1])
    except (TypeError, ValueError, IndexError):
        return None
    floor = minimum_size(compact)
    if width < MIN_WINDOW_SIZE[0] or height < MIN_WINDOW_SIZE[1]:
        return None
    # Raised rather than rejected: a width this shape cannot survive is the one part
    # of their choice to overrule, and discarding the whole size would throw away a
    # height they meant. The stored value is left alone -- the next drag replaces it,
    # and a size we corrected is not a size they chose.
    return fit_to_screen((max(width, floor[0]), max(height, floor[1])))


def interface_size(compact: bool) -> tuple[int, int]:
    """How big to make the window for the interface, on this screen.

    Sized to the screen rather than fixed. The constants above were written on one
    monitor and applied everywhere: on a 1024x768 display the expanded window ran past
    two edges at once, and on a large one it stayed a small window in the middle of an
    empty desktop.

    Width only adapts when expanded -- see COMPACT_WIDTH. Height always does, since
    every part of the interface is a list that can usefully be longer.

    A size the user dragged this shape to wins, because they decided it. Toggling the
    chat panel still moves the window -- to the other shape's size -- so the control
    always does something visible, which is what one size for both shapes took away.
    """
    chosen = remembered_size(compact)
    if chosen is not None:
        return chosen

    screen = screen_size()
    if screen is None:
        return COMPACT_SIZE if compact else EXPANDED_SIZE

    width = COMPACT_WIDTH
    if not compact:
        width = min(MAX_WINDOW_SIZE[0], round(screen[0] * SCREEN_FRACTION[0]))
    height = min(MAX_WINDOW_SIZE[1], round(screen[1] * SCREEN_FRACTION[1]))
    # Through fit_to_screen, not instead of it: the bounds above are what the interface
    # wants, and the screen still has the last word on a display smaller than the lower
    # bound.
    return fit_to_screen((width, height))

# Long enough to read "Setup complete" and the time it took, short enough not to be a
# step. The button was removed: after a nine-minute wait, asking for a click before
# showing the thing that was being installed is friction, not ceremony.
DONE_PAUSE_SECONDS = 2.2

# Injected into every page the window loads, so the interface can tell the window to
# change shape, and open this application's settings, without knowing anything about
# how the window is implemented.
#
# The event goes last. These are installed on the window's loaded event, which is after
# the interface has rendered: a hook the page only ever calls can be looked up when it
# is called, but the settings button decides whether to be drawn at all, so it has to be
# told when the hook arrives. A browser never sends it, and never shows the button.
HOST_HOOK = """
window.__hostViewChange = function (compact) {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.set_compact) {
    window.pywebview.api.set_compact(!!compact);
  }
};
window.__hostOpenSettings = function () {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.open_settings) {
    window.pywebview.api.open_settings();
  }
};
window.dispatchEvent(new Event("hostready"));
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


def _has_appindicator_typelib() -> bool:
    """Whether the desktop can show a tray icon through AppIndicator.

    The same directories AppRun checks before offering to install the package, kept in
    step with it: this is the fallback for a machine where that install did not happen.
    """
    for directory in ("/usr/lib/x86_64-linux-gnu/girepository-1.0",
                      "/usr/lib/aarch64-linux-gnu/girepository-1.0",
                      "/usr/lib64/girepository-1.0",
                      "/usr/lib/girepository-1.0"):
        path = Path(directory)
        if not path.is_dir():
            continue
        for pattern in ("AyatanaAppIndicator3-*.typelib", "AppIndicator3-*.typelib"):
            if any(path.glob(pattern)):
                return True
    return False


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

    def close_unsupported(self) -> None:
        """Exit, for the screen that says this machine cannot run it."""
        self._launcher.quit()

    def set_compact(self, compact: bool) -> None:
        """The interface hid or showed its chat panel; match the window to it."""
        self._launcher.set_compact(compact)

    def open_settings(self) -> None:
        """The interface's settings button: the same window as the tray's Settings.

        One window, not a second copy of it: open_settings raises the one already
        open rather than stacking another, whichever of the two asked.
        """
        self._launcher.open_settings()


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
            "installed_version": config.app_version(),
            "accelerator": stored["accelerator"],
            "last_setup": stored["last_setup"],
            "package_count": stored["package_count"],
            # The address, not just the number. "Backend service: 8571" reads as one
            # of several services with several ports; the interface the reader is
            # looking at is served on this one, and showing the URL says so without
            # anyone having to read a paragraph.
            "service_url": config.service_url(stored["service_port"]),
            # Shown read-only. The launcher owns the application port and nothing else
            # -- this belongs to an inference configuration, and the load-model request
            # carries its own, so anything written here would be overwritten.
            "llama_port": config.LLAMA_SERVER_PORT,
            # Omitted where vLLM cannot run at all: on Windows the backend refuses the
            # engine up front, so a port for it is a number for something that will
            # never start.
            "vllm_port": None if sys.platform == "win32" else config.VLLM_PORT,
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
        # Through the launcher, which is where _open_url_async lives. Calling it on
        # self raised AttributeError, and pywebview swallows what a bridge method
        # raises -- so every "Vendor page" button did nothing at all, on both
        # platforms, with nothing in the log to say why.
        if str(url).startswith(("http://", "https://")):
            self._launcher._open_url_async(url)


    # ------------------------------------------------------------- components

    def list_components(self) -> list[dict[str, Any]]:
        return components.listing()

    def set_lan_access(self, enabled: bool) -> dict[str, Any]:
        """Let other machines on the network reach the service, or stop letting them.

        The service binds where .env says, so this rewrites that one key and restarts
        it -- a listening socket's address cannot be changed under it.

        There is no authentication on this API. Anyone who can reach the port can use
        the loaded model, read the settings and load or unload. That is the user's call
        to make, which is why this exists, and why the page states it plainly rather
        than describing the switch as "network mode".
        """
        enabled = bool(enabled)
        previous = bool(state.get("lan_access"))
        if enabled == previous:
            return {"ok": True, **self.network_details()}

        # Written first because bind_host reads them, and put back if the restart does
        # not happen. A failure that leaves them written is not a failure: the page
        # reverts its switch and says so, while the state file still says the port is
        # open -- so the next repaint shows the switch back on, and the next start
        # binds to the network with nobody having agreed to it. Nothing is "on" until
        # a service is listening that way.
        state.set("lan_access", enabled)
        env_file.set_host(config.bind_host())
        try:
            self._launcher.service.restart()
        except Exception as exc:  # noqa: BLE001 - reported, not raised at the page
            log.exception("could not restart the service after changing the bind address")
            state.set("lan_access", previous)
            env_file.set_host(config.bind_host())
            with contextlib.suppress(Exception):
                # The old address is the one the user still believes is in force, so
                # the service is put back on it rather than left wherever it stopped.
                self._launcher.service.restart()
            return {"ok": False, "error": f"Could not restart the service: {exc}",
                    **self.network_details()}
        log.info("lan access %s; service bound to %s",
                 "enabled" if enabled else "disabled", config.bind_host())
        return {"ok": True, **self.network_details()}

    def network_details(self) -> dict[str, Any]:
        """What another machine would need, and whether it can be used yet.

        The commands are given as text to copy rather than a file to download: the
        other machine's Hermes owns its own configuration, and `hermes config set` is
        its supported way to change it. Handing over a fragment of YAML instead would
        mean someone pasting it into a two-thousand-line file that is not ours.
        """
        port = self._launcher.service.port
        address = config.lan_address()
        enabled = bool(state.get("lan_access"))
        base = f"http://{address}:{port}/v1" if address else None
        return {
            "lan_access": enabled,
            "address": address or "",
            "base_url": base or "",
            # What a component pointed here would be given. Reported so the card can
            # say it before the button is pressed rather than after: "it will use
            # qwen2.5-7b" and "there is no model loaded yet" are different enough that
            # finding out afterwards is a surprise, and disabling the button instead
            # would block the one moment someone most wants to connect a tool -- while
            # setting the machine up, before any model is chosen.
            "model": self._loaded_model_id(port) or "",
            # Built from the same definition the button writes, so the two cannot
            # describe different endpoints again -- see components.hermes_provider.
            "commands": components.hermes_commands(base) if base and enabled else [],
            # Only ever about the rule this application adds, never about whether the
            # port can be reached -- nothing on this side can establish that. See
            # launcher/firewall.py for the measurements behind that.
            #
            # Whether the rule exists is deliberately absent: asking costs three and a
            # half seconds, measured, and this runs every time the Components tab is
            # drawn. It made the tab arrive empty and fill in late. The page asks for
            # it through firewall_state, once, when somebody opens the section that
            # shows it.
            #
            # The commands themselves are not sent either. They were, back when the
            # card printed them beside the buttons for the reader to check before
            # agreeing to an administrator prompt; the section no longer does, and
            # what Windows names in the prompt is the command itself. Three strings
            # built on every draw that nothing read.
            "firewall": {
                "supported": firewall.supported(),
                "linux_hint": "" if firewall.supported() else firewall.linux_hint(port),
            },
        }

    def firewall_state(self) -> dict[str, Any]:
        """Whether the rule this application adds is in place. Slow, so asked on demand."""
        return {"has_rule": firewall.has_rule(self._launcher.service.port)}

    def firewall_allow(self, temporary: bool = False) -> dict[str, Any]:
        """Add the rule, through the system's own administrator prompt.

        The only place this application asks for an administrator. Opening a port with
        no password on it to the network is a privileged act the user asked for, and
        the system prompt is the right thing to ask it with; nothing else here has
        changed.
        """
        port = self._launcher.service.port
        temporary = bool(temporary)
        ok, error = firewall.run_elevated(
            firewall.add_command(port, temporary=temporary), keep_open=temporary)
        # The temporary one is not looked for: it is deliberately under another name
        # and the window it opened is what says it is there.
        if temporary:
            return {"ok": ok, "error": error, "has_rule": False}
        # Checked, not assumed -- and checked whether the launch called itself a
        # success or not, because it is wrong in both directions. Start-Process
        # reports whether it started a process, not whether that process did
        # anything, which is how a command that was being mangled on its way in
        # reported success for a rule it never created. And a prompt this side
        # stopped waiting on may still have been answered afterwards. The firewall is
        # the only thing that knows, so it is the only thing asked.
        if firewall.has_rule(port):
            # Returned so the page can repaint from it. Asking again on the other
            # side would be the same three and a half seconds for an answer in hand.
            return {"ok": True, "error": "", "has_rule": True}
        if not ok:
            return {"ok": False, "error": error, "has_rule": False}
        return {"ok": False, "has_rule": False,
                "error": "Windows ran the command without reporting a problem, but "
                         "the rule is not there."}

    def firewall_remove(self) -> dict[str, Any]:
        port = self._launcher.service.port
        ok, error = firewall.run_elevated(firewall.remove_command(port), keep_open=False)
        if not ok:
            # Removing something that was not there is not a failure, and is the one
            # way the record above corrects itself: somebody deleted the rule by hand,
            # and this is where we find out.
            if not firewall.has_rule(port):
                return {"ok": True, "error": "", "has_rule": False}
            return {"ok": False, "error": error, "has_rule": True}
        if firewall.has_rule(port):
            return {"ok": False, "error": "The rule is still there.", "has_rule": True}
        return {"ok": True, "error": "", "has_rule": False}

    def open_component(self, component_id: str) -> dict[str, Any]:
        started, error, kind = components.open_component(str(component_id))
        return {"ok": started, "error": error, "kind": kind}

    def point_component_here(self, component_id: str) -> dict[str, Any]:
        """Point a component at this machine's model service.

        Only Hermes for now, and only because it takes an OpenAI-compatible endpoint --
        which this backend exposes whatever engine is loaded.

        Refused until a model is loaded. Writing the address alone looked harmless and
        was not: Hermes keeps a default model beside the endpoint, so an endpoint
        pointed here while its default still read upstage/solar-pro4:free -- Hermes's
        own, which this service does not serve -- described a working setup that could
        not answer a single message. There is also nothing to be gained by allowing it,
        since the button's offer is this machine's model and there is not one yet.
        """
        if str(component_id) != "hermes":
            return {"ok": False, "error": "That component has nothing to point here.",
                    "kind": components.FAILED}
        port = self._launcher.service.port
        model = self._loaded_model_id(port)
        if not model:
            return {"ok": False, "error": "Load a model first; there is nothing to "
                                          "point Hermes at yet.",
                    "kind": components.BLOCKED}
        refusal = self._tool_calling_refusal(port)
        if refusal:
            return {"ok": False, "error": f"Hermes sends tools with every request, and "
                                          f"this service will not take them. {refusal}",
                    "kind": components.BLOCKED}
        ok, message, kind = components.point_hermes_here(config.openai_base_url(port), model)
        return {"ok": ok, "error": "" if ok else message,
                "detail": message if ok else "", "kind": kind}

    def _tool_calling_refusal(self, port: int) -> str:
        """The service's own reason a request carrying tools would be refused, or "".

        Asked rather than worked out. Whether tool calling works depends on the loaded
        engine, and for vLLM on a parser having been configured as well -- conditions
        the service already evaluates, and that this side would otherwise keep a second
        copy of, to drift from the first. Hermes sends tools on every request, so a
        Hermes pointed at a service that refuses them fails on its first message with
        an error the user reads there rather than here.

        Cheap because the check runs before any generation: measured at two
        milliseconds when the answer is no. When it is yes the reply is one token.
        """
        import json
        import urllib.error
        import urllib.request

        probe = {
            "model": "probe",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1,
            "tools": [{"type": "function",
                       "function": {"name": "probe",
                                    "parameters": {"type": "object", "properties": {}}}}],
        }
        request = urllib.request.Request(
            f"{config.openai_base_url(port)}/chat/completions",
            data=json.dumps(probe).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20):
                return ""
        except urllib.error.HTTPError as exc:
            try:
                detail = str(json.loads(exc.read().decode("utf-8")).get("detail") or "")
            except Exception:  # noqa: BLE001 - the status is the answer; the text is a bonus
                detail = ""
            # Only the refusal this asked about. Any other error means the request did
            # not get far enough to answer the question: a dead inference server
            # answers 500, and reporting that as "this service will not take tools"
            # sends someone to change engines over an engine that has crashed.
            if exc.code == 400 and "tool" in detail.lower():
                return detail
            log.info("tool-calling probe got HTTP %d: %s", exc.code, detail[:200])
            return ""
        except Exception:  # noqa: BLE001 - unreachable is not a refusal; let the write proceed
            return ""

    def _loaded_model_id(self, port: int) -> str | None:
        """What /v1/models reports, or None. Never raises: this is a convenience."""
        import json
        import urllib.request
        try:
            with urllib.request.urlopen(
                    f"{config.openai_base_url(port)}/models", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - no model name is a fine answer
            return None
        entries = data.get("data") or []
        return str(entries[0].get("id")) if entries and entries[0].get("id") else None

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
        # Set when preflight found something this machine cannot be talked out
        # of -- no supported GPU, most of all. Closing the window then means
        # exiting rather than hiding to the tray: there is nothing left for a
        # tray icon to manage, and one that does nothing is worse than none.
        self._unsupported = False
        # What the window is currently sized to, so a page asking for the size it is
        # already at costs nothing.
        self._window_size: tuple[int, int] | None = None
        # Which shape the interface is in, so a drag is remembered against the right
        # one. The packaged config.json decides how it starts; the page changes it.
        self._compact = self._starts_compact()
        # When we last asked for a size, and how much bigger the frame came back than
        # the content area. The window manager does not grant what is asked for -- it
        # adds its decorations and reports the result -- so the offset is learnt from
        # our own resizes and every later event is read through it. Without that, one
        # toggle's reported size fed into the next toggle's request and the window grew
        # by the decorations each time: measured 744 -> 796 -> 848.
        self._resize_asked_at = time.monotonic()
        self._decoration = (0, 0)
        self._maximized = False
        # A drag delivers an event per frame -- around thirty in two seconds, measured.
        # Writing state on each one rewrites the whole file thirty times for one
        # decision, so the write waits until the drag stops.
        self._resize_timer: threading.Timer | None = None

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
        width, height = fit_to_screen(size)
        # Every page reports the width it needs when it mounts, so the size it is
        # already at is reported constantly. Acting on those made the log read like a
        # window resizing itself in a loop -- which is what it looked like to me when
        # someone was simply switching between two pages that want different widths.
        if self._window_size == (width, height):
            return
        self._window_size = (width, height)
        # Stamped before the call, so the event this is about to cause is recognised as
        # ours rather than mistaken for the user dragging the frame.
        self._resize_asked_at = time.monotonic()
        log.info("resizing the window to %dx%d", width, height)
        with contextlib.suppress(Exception):
            self.window.resize(width, height)

    def set_compact(self, compact: bool) -> None:
        """The interface hid or showed its chat panel; match the window to it."""
        self._compact = compact
        self.set_window_size(interface_size(compact))

    def _on_maximized(self) -> None:
        self._maximized = True

    def _on_restored(self) -> None:
        self._maximized = False

    def _on_resized(self, width: int, height: int) -> None:
        """Remember a size the user chose for the shape the window is in.

        Every resize arrives here, ours included, and the two numbers are not in the
        same units: resize() sets the content area and the event reports the frame
        around it -- 52 by 89 pixels of difference on GNOME. So what arrives just after
        we asked is the answer to our own request and gives us that offset; anything
        else is the user, read back through it.

        Not written here. A drag delivers an event per frame, so the write is deferred
        until they stop, and only the size they stopped at is kept.
        """
        if time.monotonic() - self._resize_asked_at < _OUR_RESIZE_WINDOW:
            if self._window_size is not None:
                self._decoration = (width - self._window_size[0], height - self._window_size[1])
            return
        if self._maximized:
            # A maximised window is the desktop's size, not a size anyone chose, and
            # restoring would otherwise leave that recorded as a preference.
            return

        content = (width - self._decoration[0], height - self._decoration[1])
        if content[0] < MIN_WINDOW_SIZE[0] or content[1] < MIN_WINDOW_SIZE[1]:
            return
        self._window_size = content
        if self._resize_timer is not None:
            self._resize_timer.cancel()
        self._resize_timer = threading.Timer(_DRAG_SETTLE_SECONDS, self._remember_size, (content,))
        self._resize_timer.daemon = True
        self._resize_timer.start()

    def _remember_size(self, content: tuple[int, int]) -> None:
        key = size_key(self._compact)
        if state.get(key) == list(content):
            return
        log.info("remembering the %s window at %dx%d",
                 "compact" if self._compact else "expanded", content[0], content[1])
        state.set(key, list(content))

    def reload_window(self) -> None:
        if self.window is None:
            return
        compact = self._starts_compact()
        # Asked for in the URL, because the page cannot tell a window from a browser
        # tab any other way -- and config.json, being per-deployment, describes both.
        url = config.interface_url(self.service.port, compact=compact)
        # Sized before the page loads rather than after, so the interface does not
        # appear at the setup window's size and then jump.
        self.set_window_size(interface_size(compact))
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
        self._open_url_async(config.service_url(self.service.port))

    def _open_url_async(self, url: str) -> None:
        """Opening a URL can block for as long as the desktop's opener feels like
        taking, and every caller here is a thread that must not be held up: a tray
        menu item, or the page's own JavaScript bridge."""
        def work() -> None:
            if browser.open_url(url):
                return
            # Silently doing nothing is the failure this replaces. Even when no
            # browser can be reached the address is still useful: the service is
            # running and anything on the machine can talk to it.
            self._notify(
                "No browser could be opened",
                f"The interface is running at\n\n  {url}\n\n"
                "Open that address in a browser to use it.",
            )

        threading.Thread(target=work, name="open-url", daemon=True).start()

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
        if self._quitting.is_set() or self._unsupported:
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

        # Nothing we can install our way out of means the window is the end of the
        # road, so it offers a way to close and closing exits. The alternative -- the
        # window vanishing by itself once the message is shown -- is indistinguishable
        # from a crash, which is the failure this whole screen exists to avoid.
        self._unsupported = not any(remedy.can_fix(c.fixer) for c in report.blocking)
        self._js(
            f"window.showError({_js_string(chr(10).join(lines))}, "
            f"{'true' if self._unsupported else 'false'})"
        )

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

            # A service that outlived the launcher is normally adopted. Not after an
            # upgrade: the payload was replaced under it, which is the directory it
            # was started in and the code it is running. Its working directory no
            # longer exists, so every relative path it touches fails with
            #
            #   FileNotFoundError: [Errno 2] No such file or directory
            #
            # while /health goes on answering "healthy" -- a backend that reports
            # itself fine and cannot load a model. Replacing it is also simply
            # correct: an upgrade should leave the new backend running, not the old.
            # By version as well: the payload counts as refreshed only in the first
            # run after an upgrade, and a run that ends before getting here -- closed
            # during startup, or closed by the next installer -- left the following
            # one adopting the old backend. The version it was started by does not
            # forget.
            started_by = state.get("service_version")
            if ((config.payload_was_refreshed() or started_by != config.app_version())
                    and self.service.status() == "ours"):
                log.info("the running service was started by %s (this is %s)%s; "
                         "restarting it on the new one", started_by or "an older launcher",
                         config.app_version(),
                         ", and the payload was replaced under it"
                         if config.payload_was_refreshed() else "")
                self._set_step("Restarting the service", None,
                               "the update replaced it", 0)
                with contextlib.suppress(Exception):
                    self.service.stop()

            log.info("bootstrap: starting the service")
            self._js(f"window.setPhase({_js_string('done')})")
            self._set_step("Starting the service", None, "loading the GPU runtime", 0)
            self.service.start(
                on_wait=lambda seconds: self._set_step(
                    "Starting the service", None, "loading the GPU runtime", seconds
                )
            )

            self.service.watch(self._service_lost)

            # The starter model, before the interface, so the first time the window
            # opens there is something in the list to load. Once per machine, and it
            # never keeps the window closed -- see starter_model.py. Guarded anyway:
            # this is a convenience, and nothing about it is worth failing startup for.
            try:
                note = starter_model.fetch(
                    self.service.port,
                    lambda detail, fraction, elapsed: self._set_step(
                        "Downloading a starter model", fraction, detail, elapsed),
                )
            except Exception:
                log.exception("the starter model step failed; carrying on without it")
                note = None
            if note:
                self._set_step("Starter model", None, note, 0)
                time.sleep(DONE_PAUSE_SECONDS * 2)

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
        # pystray tries appindicator, then gtk, then xorg, and takes the first that
        # imports. The last one is the problem: on a Wayland session -- the default on
        # Ubuntu since 22.04 -- it imports fine and then draws a window nobody sees, so
        # a missing typelib turns into a tray icon that is simply absent with nothing
        # in the log. Naming the backend when the typelib is there makes that case
        # raise instead, and the caller logs it. AppRun installs the typelib when it
        # can; this covers the machines where it could not.
        if not config.IS_WINDOWS and _has_appindicator_typelib():
            os.environ.setdefault("PYSTRAY_BACKEND", "appindicator")

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
            pystray.MenuItem("Open window", self._menu_action(self.open_window, "open"),
                             default=True),
            pystray.MenuItem("Open full interface in browser",
                             self._menu_action(self.open_in_browser, "browser")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self._menu_action(self.open_settings, "settings")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._menu_action(self.quit, "quit")),
        )
        return pystray.Icon(config.APP_ID, image, config.APP_TITLE, menu)

    def _menu_action(self, action: Any, name: str) -> Any:
        """Run a menu item on a worker, never on the thread that serves the UI.

        On Linux the tray is attached to the window's GTK main loop (see run_detached
        below), so a menu callback runs on the very thread pywebview needs in order to
        do anything -- and two of these items are nothing but pywebview calls.

        Both failures are silent. create_window does its work only when it is called
        from somewhere other than the main thread:

            if threading.current_thread().name != 'MainThread' and guilib:
                ...
                guilib.create_window(window)
            return window

        so Settings appended a window to a list, returned it, logged "opening the
        settings window", and showed nothing, forever. create_confirmation_dialog is
        the other shape of the same problem: it queues work on the main loop and waits
        for the answer, which on the main loop is a deadlock -- reached only by
        quitting with a model loaded, which is why it went unnoticed.

        Windows already ran these on pystray's own thread; a worker changes nothing
        there, and one rule for both platforms is easier to keep right than two.
        """
        def start() -> None:
            threading.Thread(target=self._run_menu_action, args=(action, name),
                             name=f"tray-{name}", daemon=True).start()
        return start

    def _run_menu_action(self, action: Any, name: str) -> None:
        try:
            action()
        except Exception:
            log.exception("tray: %s failed", name)

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
            # Not tray.notify(). pystray's AppIndicator backend -- the one build_tray
            # now asks for by name -- implements no _notify, so the base class raises
            # NotImplementedError and the suppress around it threw the message away.
            # Every call here was silently doing nothing, which is the fault the
            # caller of this exists to report.
            #
            # The same cascade AppRun uses before Python is available, for the same
            # reason: what a desktop has to show a message with varies, and stderr
            # reaches nobody when the application was started from an icon.
            body = f"{title}\n\n{message}"
            for command in (["zenity", "--info", "--no-wrap", f"--title={config.APP_TITLE}",
                             f"--text={body}"],
                            ["kdialog", "--title", config.APP_TITLE, "--msgbox", body],
                            ["notify-send", title, message]):
                if shutil.which(command[0]) is None:
                    continue
                with contextlib.suppress(OSError):
                    # Detached: zenity and kdialog stay up until dismissed, and this is
                    # called from a worker that has other things to do.
                    subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
                    return
            log.warning("no way to show a notification on this desktop")
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
        """The only path that stops the service. Quit means quit.

        There used to be a second question here, asked whenever a model was loaded or
        a training run was going, on the grounds that the service serves other
        applications too. It is gone. A model is loaded almost all of the time this is
        running, so the dialog was not the rare warning it was designed as -- it stood
        between the user and the one button whose meaning is not in doubt.

        The install below still asks, and it is a different thing: not "are you sure",
        but a warning that stopping now leaves something broken on disk. An install
        interrupted partway left a third-party tool with a virtual environment and no
        packages in it, reporting itself as present. Its installer is a child of this
        process reading from a pipe we own, so this process going away is enough to
        abort it mid-write.

        Closing the settings window is safe; the install runs here, not there. Quitting
        is what takes it down, so quitting is what asks.
        """
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

    def _confirm_quit_during_install(self, component_id: str) -> bool:
        """Ask before abandoning an install. Defaults to quitting if we cannot ask."""
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

    # Before anything reads a path. An installer's child inherits a mitigation that
    # makes uv's managed Python unreachable, and a first run under it installs nothing
    # at all -- see hardening.py.
    if hardening.step_out_of_the_way():
        log.info("handed over to a copy without the mitigation; exiting")
        return 0

    # Also before anything reads a path. Inside an AppImage the payload is mounted
    # read-only, and both the backend's .env and the llama build land inside it.
    try:
        working = config.ensure_writable_payload()
        log.info("payload: %s", working)
    except OSError:
        # Not "continuing with the shipped one": where this can fail at all the shipped
        # payload is a read-only mount, so what follows runs against whatever state the
        # working copy was left in. Anything preserved from it is in payload.keep and
        # comes back on the next attempt, so the honest thing is to say so and let the
        # next step fail where the user can see it.
        log.exception("could not prepare a writable payload at %s; what follows may "
                      "not work, and a restart will try again", config.working_payload_dir())

    # Logging in is not a request to install anything. Setup takes eight minutes, opens
    # a window and downloads gigabytes, and doing that because someone turned their
    # computer on is a surprise -- particularly since autostart is on by default, so
    # nobody chose it. Whoever launches it deliberately gets the first-run flow; a
    # login gets nothing until then.
    if args.autostart and not provision.is_provisioned():
        log.info("started by the autostart entry with setup not yet done; "
                 "leaving it for a deliberate launch")
        return 0

    claim_desktop_identity()

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
    # Trimmed to the screen here as well as on every later resize: a window created
    # larger than the display opens past its edges, and the first thing the user sees
    # is the one thing they cannot drag back.
    if provision.is_provisioned():
        width, height = interface_size(launcher._starts_compact())
        log.info("already provisioned; opening at the interface size %dx%d", width, height)
    else:
        width, height = fit_to_screen(SETUP_SIZE)
        log.info("setup is needed; opening at the setup size %dx%d", width, height)
    log.info("screen: %s", screen_size() or "unknown")

    launcher.window = webview.create_window(
        config.APP_TITLE,
        str(UI_DIR / "progress.html"),
        js_api=WindowApi(launcher),
        width=width,
        height=height,
        min_size=MIN_WINDOW_SIZE,
    )
    # What it was created at, before any event can arrive. Without this the first
    # resized event -- the window simply being drawn -- is measured against nothing.
    launcher._window_size = (width, height)
    launcher.window.events.closing += launcher._on_closing
    launcher.window.events.resized += launcher._on_resized
    launcher.window.events.maximized += launcher._on_maximized
    launcher.window.events.restored += launcher._on_restored
    # Navigation clears injected globals, so the hook is put back after every load.
    launcher.window.events.loaded += launcher._inject_host_hook

    # The tray is a convenience; the window is the product. Building it used to be
    # unguarded, so a desktop where the tray backend cannot start took the whole
    # application with it -- observed on Linux, where pystray picks its Xorg backend
    # at import time and raises DisplayNameError if it cannot reach an X server. The
    # user's machine is then one that shows nothing at all rather than one without a
    # tray icon.
    try:
        launcher.tray = launcher.build_tray()
    except Exception:
        launcher.tray = None
        log.exception("could not create the tray icon; carrying on without one")
    else:
        if config.IS_WINDOWS:
            # pywebview needs the main thread on Windows, so the tray goes to a
            # worker. It is a daemon: if the window loop ends unexpectedly the
            # process must still exit.
            threading.Thread(target=launcher.tray.run, name="tray", daemon=True).start()
            log.info("tray started")
        else:
            # Not a thread on Linux. Both halves are GTK here -- pystray's
            # AppIndicator backend and pywebview's GTK backend -- and GTK has one
            # default main context, which whichever starts first takes. Running the
            # tray in a worker took it, and pywebview then said
            #
            #   GLib-GIO-CRITICAL: g_application_run() cannot acquire the default
            #   main context because it is already acquired by another thread!
            #
            # and the window loop ended immediately, leaving a process with a tray
            # icon and no window.
            #
            # run_detached is pystray's answer for exactly this: it attaches the icon
            # to the main loop the host is about to run instead of starting one.
            launcher.tray.run_detached()
            log.info("tray attached to the main loop")

    # The menu entry and the desktop icon, which Linux had neither of: the AppImage
    # was the only way in, so between logins the user had to go and find the file.
    # Rewritten whenever it disagrees with this copy, for the same reason autostart is
    # refreshed below -- the recorded path goes stale when the AppImage moves.
    with contextlib.suppress(Exception):
        shortcuts.ensure()

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

    # Ended deliberately rather than by returning, because returning was not enough.
    # WebKitGTK and the GTK tray leave native threads behind -- llvmpipe, VBlankMonitor,
    # PressureMonitor, several unnamed ones -- and with them running the interpreter's
    # own shutdown does not finish. Observed on Linux: this line was logged, main
    # returned, and the process was still there afterwards with its tray icon still in
    # the system bar. Someone who pressed Close then had an application that had said
    # goodbye and not gone.
    #
    # Everything that must happen has happened by now: the service is stopped, the
    # single-instance socket is released, and the log is flushed below. os._exit is the
    # right tool once there is nothing left to unwind.
    applog.flush()
    if not config.IS_WINDOWS:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
