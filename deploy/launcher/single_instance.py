"""Let one copy of the application run, and let later launches raise it.

A tray application that also starts at login is launched twice as a matter of course:
once by Windows, once by whoever clicks the desktop shortcut. Without this, that gives
two tray icons, two windows, and two managers of one service -- and "Quit" on either
stops the service under the other, leaving a window that can no longer reach a backend.

The mechanism is a loopback socket, which is what most desktop toolkits use for this:
binding it is the lock, and connecting to it is how a second launch says "you are
already running, show yourself". A mutex would only do the first half.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Callable

from . import applog

log = applog.get("instance")

# Loopback only -- this listens for nothing but our own launches.
HOST = "127.0.0.1"
PORT = 48571

# Exchanged so that a port held by something else is not mistaken for another copy of
# this application. Being wrong in that direction would mean refusing to start at all.
HELLO = b"ai-scaler-toolkit/1 activate\n"
ACK = b"ai-scaler-toolkit/1 ok\n"

_TIMEOUT = 2.0


class _Listener(threading.Thread):
    def __init__(self, server: socket.socket, on_activate: Callable[[], None]) -> None:
        super().__init__(name="single-instance", daemon=True)
        self._server = server
        self._on_activate = on_activate
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(_TIMEOUT)
                    if conn.recv(len(HELLO)) != HELLO:
                        continue
                    conn.sendall(ACK)
                except OSError:
                    continue
            log.info("another launch asked us to come forward")
            try:
                self._on_activate()
            except Exception:
                log.exception("could not bring the window forward")

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._server.close()


_listener: _Listener | None = None


def _tell_running_instance() -> bool:
    """True when an existing copy answered and was asked to show itself."""
    try:
        with socket.create_connection((HOST, PORT), timeout=_TIMEOUT) as client:
            client.sendall(HELLO)
            return client.recv(len(ACK)) == ACK
    except OSError:
        return False


def acquire(on_activate: Callable[[], None]) -> bool:
    """Claim the single-instance slot.

    True means this process owns it and should carry on starting. False means another
    copy is already running, has been asked to show its window, and this process should
    exit quietly.
    """
    global _listener

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Deliberately not SO_REUSEADDR: the point here is that the second bind fails.
        server.bind((HOST, PORT))
    except OSError:
        server.close()
        if _tell_running_instance():
            log.info("already running; asked the existing window to come forward")
            return False
        # The port is held by something that is not us. Refusing to start over a port
        # clash would be a worse outcome than running twice, so carry on and say so.
        log.warning("port %d is in use by something else; starting without the "
                    "single-instance guard", PORT)
        return True

    server.listen(4)
    _listener = _Listener(server, on_activate)
    _listener.start()
    log.info("single-instance guard held on %s:%d", HOST, PORT)
    return True


def release() -> None:
    global _listener
    if _listener is not None:
        _listener.close()
        _listener = None
