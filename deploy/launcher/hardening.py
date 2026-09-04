"""Refuse to run under a mitigation that makes the Python environment unreachable.

Windows lets a process enforce "redirection trust", which refuses any path that
crosses a junction the process does not trust. Installers turn it on -- sensibly: an
installer must not be tricked into following a junction someone planted in %TEMP% --
and a child process inherits it.

We cannot live with it. uv keeps its managed Python at cpython-3.12.13-... and refers
to it everywhere through the minor-version alias cpython-3.12-..., which on Windows is
a junction. Under the mitigation every path through that alias is refused with
ERROR_UNTRUSTED_MOUNT_POINT, so `uv sync` cannot inspect the interpreter and the
service cannot be spawned. What that looked like in practice: ticking "Start AI Scaler
Toolkit" on the installer's last page gave a first run that installed no Python
packages at all and reported success.

The mitigation cannot be turned off once set -- SetProcessMitigationPolicy only ever
tightens. The only way out is a different parent, so this starts one copy through
Explorer, which runs at the ordinary desktop context, and lets this one go.

The installer no longer launches us directly, so this should never fire. It is here
because "launched by something hardened" is not only the installer's to do: deployment
tools, remote management agents and scheduled tasks all start processes, and a first
run that silently installs nothing is far too quiet a way to fail.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import os
import subprocess
import sys

from . import applog, config

log = applog.get("hardening")

# Set on the copy we start, so a relaunch that is still hardened runs anyway rather
# than handing off forever.
RELAUNCH_FLAG = "AI_SCALER_TOOLKIT_RELAUNCHED"

# ProcessRedirectionTrustPolicy. Bit 0 is EnforceRedirectionTrust, bit 1 is
# AuditRedirectionTrust, which only logs and is not our problem.
_POLICY_REDIRECTION_TRUST = 16
_ENFORCE = 0x1


def enforces_redirection_trust() -> bool:
    """Whether this process is refused paths that cross an untrusted junction."""
    if not config.IS_WINDOWS:
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declared rather than left to default. GetCurrentProcess returns the pseudo
        # handle (HANDLE)-1, and with the default restype of int that arrives as
        # 0x00000000FFFFFFFF instead of all ones: the call then fails with
        # ERROR_INVALID_HANDLE and this reports a hardened process as an ordinary one,
        # which is exactly the case it exists to catch.
        kernel32.GetCurrentProcess.restype = w.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetProcessMitigationPolicy.restype = w.BOOL
        kernel32.GetProcessMitigationPolicy.argtypes = [
            w.HANDLE, ctypes.c_int, ctypes.POINTER(w.DWORD), ctypes.c_size_t]

        value = w.DWORD(0)
        ok = kernel32.GetProcessMitigationPolicy(
            kernel32.GetCurrentProcess(), _POLICY_REDIRECTION_TRUST,
            ctypes.byref(value), ctypes.sizeof(value))
    except (OSError, AttributeError, ValueError):
        # Older Windows has neither the policy nor the API. Nothing to work around.
        return False
    if not ok:
        log.warning("could not read the redirection-trust policy (error %s)",
                    ctypes.get_last_error())
        return False
    return bool(value.value & _ENFORCE)


def _relaunch_through_explorer() -> bool:
    """Start one copy from the desktop shell. True if it was started.

    Explorer is used rather than ShellExecute because ShellExecute from this process
    would keep this process as the parent, and the parent is the whole problem.
    """
    target = sys.executable if getattr(sys, "frozen", False) else None
    if not target or not os.path.isfile(target):
        log.warning("cannot tell what to relaunch (frozen=%s, executable=%s)",
                    getattr(sys, "frozen", False), sys.executable)
        return False

    explorer = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "explorer.exe")
    environment = os.environ.copy()
    environment[RELAUNCH_FLAG] = "1"
    try:
        # Explorer returns immediately; the copy it starts is its child, not ours.
        subprocess.Popen([explorer, target], env=environment,
                         creationflags=config.subprocess_flags())
    except OSError:
        log.exception("could not relaunch through Explorer")
        return False
    log.info("relaunched through Explorer: %s", target)
    return True


def step_out_of_the_way() -> bool:
    """True when this process should exit because a better-placed copy was started."""
    if not enforces_redirection_trust():
        return False

    if os.environ.get(RELAUNCH_FLAG):
        # We already handed off once and the new copy is hardened too. Carrying on is
        # better than a chain of processes, and setup will now fail loudly rather than
        # silently -- setup_env stops on a failed sync.
        log.warning("still running with redirection trust enforced after a relaunch; "
                    "continuing, but the Python environment may be unreachable")
        return False

    log.warning("started with redirection trust enforced, which makes uv's managed "
                "Python unreachable; handing over to a copy started from the desktop")
    return _relaunch_through_explorer()
