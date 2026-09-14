"""Open a URL in the user's browser, and know whether it worked.

`webbrowser.open` is not enough on Linux. It picks xdg-open, starts it with
`start_new_session=True`, and returns True when the process has not exited yet --
which is the same answer for "the browser is starting" and "xdg-open is stuck". On a
desktop with no registered http handler both xdg-open and `gio open` block
indefinitely with no output, so the tray's "Open full interface in browser" reported
success, produced no browser, and left nothing in the log. Observed on a fresh Ubuntu
24.04 install where Firefox ships as a snap and nothing had ever claimed
x-scheme-handler/http.

So the rule here is the opposite one: a handoff tool has *succeeded* only when it has
exited, and cleanly. Both of them hand the URL to something else and return
immediately, so a process still alive after a few seconds is stuck, not working. It
is killed and the next candidate is tried.

Nothing in here may run on the UI thread: the whole point is that these commands can
block forever, and on Linux the tray shares the window's GTK main loop.
"""

from __future__ import annotations

import shutil
import subprocess
import webbrowser

from . import applog, config

log = applog.get("browser")

# How long a handoff tool gets to exit before it is treated as stuck. A working one
# returns in well under a second, so this is already generous; it used to be 8, which
# with two tools to try meant sixteen seconds of nothing happening after a click.
_HANDOFF_TIMEOUT = 4.0

# Tried in turn when the desktop's own opener does not work, which is the case this
# exists for: the desktop has no http handler, but a browser is installed.
_BROWSERS = ("firefox", "google-chrome", "chromium", "chromium-browser",
             "microsoft-edge", "brave-browser", "epiphany")


def _desktop_has_http_handler() -> bool:
    """Whether anything is registered to open an http URL.

    Asked before xdg-open and gio are tried, because when nothing is registered they do
    not fail -- they hang, and each one then costs its full timeout. That is the case
    this module exists for, so it was also the slow path: the tray's "open in browser"
    took sixteen seconds to do anything on a desktop where no browser had ever claimed
    x-scheme-handler/http. Measured at 0.03s to ask, against 8s to find out the hard
    way, twice.

    A "yes" here is not a promise that the opener works, so the timeouts still apply;
    it only skips a tool that is certain to be useless.

    xdg-mime and nothing else. `gio mime x-scheme-handler/http` was tried as a second
    opinion and is worse than useless for this: with no handler registered it prints
    "No default applications for ..." on stdout and exits 0, so "did it answer with
    something" reads that as a yes. Returning True when the tool is missing is the
    honest default -- unknown means try it and let the timeout decide.
    """
    if shutil.which("xdg-mime") is None:
        return True
    try:
        result = subprocess.run(["xdg-mime", "query", "default", "x-scheme-handler/http"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(result.stdout.strip())


def _handoff(command: list[str]) -> bool:
    """Run a tool that hands the URL to a browser and exits. True if it did."""
    try:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        log.info("%s could not start: %s", command[0], exc)
        return False
    try:
        return proc.wait(timeout=_HANDOFF_TIMEOUT) == 0
    except subprocess.TimeoutExpired:
        log.info("%s did not return within %.0fs; treating it as stuck",
                 command[0], _HANDOFF_TIMEOUT)
        proc.kill()
        return False


def _spawn(command: list[str]) -> bool:
    """Start a browser directly.

    Success has two shapes here, and counting only one of them was a bug. A browser
    with no instance running becomes the window and stays alive, so still running after
    a moment means it worked. A browser that already has an instance forwards the URL
    to it and exits 0 immediately -- the common case, and it was being read as a
    failure: every remaining candidate was tried and the user was told no browser could
    be opened, in front of the tab that had just opened.
    """
    try:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        log.info("%s could not start: %s", command[0], exc)
        return False
    try:
        code = proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return True  # still running, which for a browser is what success looks like
    if code != 0:
        log.info("%s exited immediately with %s", command[0], code)
    return code == 0


def open_url(url: str) -> bool:
    """Show `url` to the user. Returns whether a browser was actually reached.

    Call this off the UI thread.
    """
    log.info("opening %s", url)
    if config.IS_WINDOWS:
        # ShellExecute, which reports its own failures. None of the above applies.
        return webbrowser.open(url)

    if _desktop_has_http_handler():
        # One handoff tool, not both. On a desktop that has xdg-open, gio is what
        # xdg-open delegates to -- so when xdg-open cannot open the URL, gio fails the
        # same way, and on the desktop this was measured on it fails by hanging until
        # its timeout. Trying both doubled the wait for nothing. gio is still the
        # fallback where xdg-open is absent, which is what which() decides.
        if shutil.which("xdg-open"):
            command = ["xdg-open", url]
        elif shutil.which("gio"):
            command = ["gio", "open", url]
        else:
            command = []
        if command and _handoff(command):
            log.info("%s opened it", command[0])
            return True
    else:
        log.info("no application claims http; going straight to the browsers")

    for name in _BROWSERS:
        path = shutil.which(name)
        if path and _spawn([path, url]):
            log.info("started %s directly", name)
            return True

    log.warning("nothing could open %s", url)
    return False
