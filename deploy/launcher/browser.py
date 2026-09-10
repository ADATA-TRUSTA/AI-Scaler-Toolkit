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

# How long a handoff tool gets to exit before it is treated as stuck. They normally
# return in well under a second; this is generous so that a slow-but-working desktop
# is not cut off.
_HANDOFF_TIMEOUT = 8.0

# Tried in turn when the desktop's own opener does not work, which is the case this
# exists for: the desktop has no http handler, but a browser is installed.
_BROWSERS = ("firefox", "google-chrome", "chromium", "chromium-browser",
             "microsoft-edge", "brave-browser", "epiphany")


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
    """Start a browser directly. It keeps running, so the test is the opposite one:
    it must still be alive once it has had a moment to fail."""
    try:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        log.info("%s could not start: %s", command[0], exc)
        return False
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return True  # still running, which for a browser is what success looks like
    log.info("%s exited immediately with %s", command[0], proc.returncode)
    return False


def open_url(url: str) -> bool:
    """Show `url` to the user. Returns whether a browser was actually reached.

    Call this off the UI thread.
    """
    log.info("opening %s", url)
    if config.IS_WINDOWS:
        # ShellExecute, which reports its own failures. None of the above applies.
        return webbrowser.open(url)

    for tool in ("xdg-open", "gio"):
        if not shutil.which(tool):
            continue
        command = [tool, url] if tool == "xdg-open" else [tool, "open", url]
        if _handoff(command):
            log.info("%s opened it", tool)
            return True

    for name in _BROWSERS:
        path = shutil.which(name)
        if path and _spawn([path, url]):
            log.info("started %s directly", name)
            return True

    log.warning("nothing could open %s", url)
    return False
