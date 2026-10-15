"""Let one copy of the application run, and let later launches raise it.

A tray application that also starts at login is launched twice as a matter of course:
once by Windows, once by whoever clicks the desktop shortcut. Without this, that gives
two tray icons, two windows, and two managers of one service -- and "Quit" on either
stops the service under the other, leaving a window that can no longer reach a backend.

The mechanism is a loopback socket, which is what most desktop toolkits use for this:
binding it is the lock, and connecting to it is how a second launch says "you are
already running, show yourself". A mutex would only do the first half.

The same socket also takes "quit". The build host is someone's desktop, and the copy
open there holds this port and the service's: the release pipeline's test could not
start its own. The pipeline runs as another account, which cannot stop this one's
processes, so it asks -- and the copy quits the way the tray's Quit does.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Callable

from . import applog

log = applog.get("instance")

# Loopback only -- this listens for nothing but our own launches.
HOST = "127.0.0.1"
PORT = 48571

# Exchanged so that a port held by something else is not mistaken for another copy of
# this application. Being wrong in that direction would mean refusing to start at all.
HELLO = b"ai-scaler-toolkit/1 activate\n"
QUIT = b"ai-scaler-toolkit/1 quit\n"
ACK = b"ai-scaler-toolkit/1 ok\n"
# Answered while this copy is on its way out. A launch that arrives then must not be
# told "already running, I will come forward" -- the copy answering is closing and will
# not come forward, so the click did nothing and the user had to click again.
BYE = b"ai-scaler-toolkit/1 closing\n"

_TIMEOUT = 2.0
# How long a launch waits for a closing copy to let go of the port before giving up on
# it. Shutting down stops a service and waits for its port, which is seconds, not
# minutes.
_HANDOVER = 20.0


class _Listener(threading.Thread):
    def __init__(self, server: socket.socket, on_activate: Callable[[], None],
                 on_quit: Callable[[], None] | None) -> None:
        super().__init__(name="single-instance", daemon=True)
        self._server = server
        self._on_activate = on_activate
        self._on_quit = on_quit
        self._stop = threading.Event()
        self._quitting = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(_TIMEOUT)
                    message = conn.recv(len(HELLO))
                    if message != HELLO and not (message == QUIT and self._on_quit):
                        continue
                    if self._quitting.is_set():
                        conn.sendall(BYE)
                        log.info("a launch arrived while closing; told it to wait")
                        continue
                    conn.sendall(ACK)
                except OSError:
                    continue
            if message == QUIT and self._on_quit:
                log.info("asked to quit over the single-instance port")
                try:
                    self._on_quit()
                except Exception:
                    log.exception("could not quit when asked")
                continue
            log.info("another launch asked us to come forward")
            try:
                self._on_activate()
            except Exception:
                log.exception("could not bring the window forward")

    def begin_quitting(self) -> None:
        self._quitting.set()

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._server.close()


_listener: _Listener | None = None


def begin_quitting() -> None:
    """Stop claiming to be a copy that can be raised.

    Called when shutdown starts, which takes a few seconds: it stops a service and
    waits for its port. A launch during that window used to be answered with "already
    running", exit quietly, and appear to do nothing at all -- the user pressed Quit,
    clicked the shortcut, and had to click a second time.
    """
    if _listener is not None:
        _listener.begin_quitting()


def _ask_running_instance() -> str:
    """"raised" if a copy took over, "closing" if one is on its way out, else "none"."""
    try:
        with socket.create_connection((HOST, PORT), timeout=_TIMEOUT) as client:
            client.sendall(HELLO)
            answer = client.recv(max(len(ACK), len(BYE)))
    except OSError:
        return "none"
    if answer == ACK:
        return "raised"
    if answer == BYE:
        return "closing"
    return "none"


def acquire(on_activate: Callable[[], None],
            on_quit: Callable[[], None] | None = None) -> bool:
    """Claim the single-instance slot.

    True means this process owns it and should carry on starting. False means another
    copy is already running, has been asked to show its window, and this process should
    exit quietly. `on_quit` is what a "quit" request runs; without one it is ignored.
    """
    global _listener

    deadline = time.monotonic() + _HANDOVER
    while True:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Deliberately not SO_REUSEADDR: the point here is that the second bind fails.
            server.bind((HOST, PORT))
            break
        except OSError:
            server.close()

        answer = _ask_running_instance()
        if answer == "raised":
            log.info("already running; asked the existing window to come forward")
            return False
        if answer == "closing" and time.monotonic() < deadline:
            # It is going away. Waiting for it is what the person who just clicked
            # expects; exiting quietly would look like the click did nothing.
            time.sleep(0.5)
            continue
        if answer == "closing":
            log.warning("the closing copy still holds port %d after %.0fs; starting "
                        "without the single-instance guard", PORT, _HANDOVER)
            return True
        # The port is held by something that is not us. Refusing to start over a port
        # clash would be a worse outcome than running twice, so carry on and say so.
        log.warning("port %d is in use by something else; starting without the "
                    "single-instance guard", PORT)
        return True

    server.listen(4)
    _listener = _Listener(server, on_activate, on_quit)
    _listener.start()
    log.info("single-instance guard held on %s:%d", HOST, PORT)
    return True


def release() -> None:
    global _listener
    if _listener is not None:
        _listener.close()
        _listener = None
