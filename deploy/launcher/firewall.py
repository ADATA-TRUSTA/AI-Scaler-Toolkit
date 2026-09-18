"""The firewall rule this application adds, and nothing else about the firewall.

Two questions look alike and only one of them can be answered from here.

"Can another computer reach this port" cannot. A connection made from this machine to
its own address succeeds whatever the firewall says -- local traffic does not traverse
the inbound rules -- so a test button here could only ever report success, including
on the machine where nobody else can get in. Reading the rules instead does not settle
it either: measured on a real desktop, enumerating every port filter took ten seconds,
needed a pipeline to avoid an access-denied, and still answered "blocked" on a machine
another computer was reaching perfectly well -- because Windows had allowed the
listening program rather than the port. An answer that is slow and sometimes wrong is
worse than no answer.

"Is our rule present" can be, in 260 milliseconds, because the rule has a name we gave
it. That is the question the buttons are about, and the only thing claimed anywhere
here. Adding and removing need an administrator; this application does not ask for one
anywhere else and does not silently start now -- each button raises the standard
prompt, on a click, for one command the user can read first.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess

from . import applog, config

log = applog.get("firewall")

# Long enough for a cold query against the firewall service, short enough that a
# wedged one cannot hold the settings page open.
_TIMEOUT = 25

# The other side of the administrator prompt is a person, not a process. The permanent
# rule is waited on, so this covers however long they take to read the prompt and
# answer it -- and at the 25 seconds above it expired on anyone who paused, reported
# Python's raw TimeoutExpired text as the explanation, and left them with a rule that
# appeared a moment later anyway. Killing the wait does not take the prompt off the
# screen, so a short timeout here buys nothing and costs the truth.
_ELEVATION_TIMEOUT = 300

# Distinct names, because they are removed on different terms: one when the window
# that holds it is closed with Enter, one only when somebody asks.
PERMANENT_RULE = "AI Scaler Toolkit"
TEMPORARY_RULE = "AI Scaler Toolkit (temporary)"


def rule_name(port: int, *, temporary: bool = False) -> str:
    return f"{TEMPORARY_RULE if temporary else PERMANENT_RULE} {port}"


def supported() -> bool:
    """Whether this platform has a firewall worth offering to change.

    Windows only, for now. Linux has whatever the distribution chose -- Ubuntu ships
    ufw and leaves it inactive, Fedora runs firewalld, others have neither -- so the
    common case there is that nothing needs doing, and a button to fix what is not
    broken is worse than no button. See linux_hint for what is offered instead.
    """
    return config.IS_WINDOWS


def has_rule(port: int) -> bool:
    """Whether a rule this application added is in place for this port.

    Not whether the port is reachable. Nothing here knows that -- see the module
    docstring for why it is not worth pretending otherwise.
    """
    if not supported():
        return False
    script = (
        f'$r = Get-NetFirewallRule -DisplayName "{PERMANENT_RULE} {port}" '
        "-ErrorAction SilentlyContinue; if ($r) { 'yes' } else { 'no' }"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    return out is not None and out.strip() == "yes"


def add_command(port: int, *, temporary: bool = False) -> str:
    """The command a button runs.

    Legible on purpose, because the administrator prompt Windows raises names this
    command and the reader is entitled to understand what they are agreeing to.
    """
    name = rule_name(port, temporary=temporary)
    add = (f'New-NetFirewallRule -DisplayName "{name}" -Direction Inbound '
           f"-LocalPort {port} -Protocol TCP -Action Allow")
    if not temporary:
        return add
    # The rule lives as long as the window does, which is the only honest way to make
    # a firewall rule temporary: Windows has no expiry for them, and an application
    # that promised to remove one later would be promising across its own crash.
    # try/finally covers Enter and Ctrl+C. Closing the window with the X kills the
    # shell, and the rule stays -- said plainly on the button rather than hidden.
    return (f"{add} | Out-Null; "
            "try { Read-Host 'Allowed. Press Enter to remove the rule' | Out-Null } "
            f"finally {{ Remove-NetFirewallRule -DisplayName \"{name}\" }}")


def remove_command(port: int) -> str:
    return f'Remove-NetFirewallRule -DisplayName "{rule_name(port)}"'


def _encoded(command: str) -> str:
    """The command as -EncodedCommand takes it: base64 of UTF-16LE.

    Not as text, and not as a file either. Both were tried and both were eaten in the
    same place. The command reaches the elevated shell through two parsers: this
    process builds a -Command string for one PowerShell, which calls Start-Process,
    whose -ArgumentList is re-split by Windows for another. The rule's name has spaces
    in it, so it is in double quotes, and the second split took those quotes -- the
    elevated shell received -DisplayName AI Scaler Toolkit 8571 and answered "a
    positional parameter cannot be found that accepts argument 'Scaler'".

    Writing the command to a .ps1 and passing -File moved the spaces rather than
    removing them. Start-Process does not quote the elements of -ArgumentList, so a
    temporary directory under a profile named "Ada Lovelace" splits at exactly the
    same place. Measured: with a space anywhere in the path the script never runs,
    and Start-Process still exits 0, which is the original fault back again on the
    machines least likely to be the ones it was tested on.

    Base64 is letters, digits and +/=. There is nothing in it for a split to take, no
    file to leave behind in a temporary directory, and nothing on disk that could be
    swapped between writing it and Windows running it as an administrator.
    """
    return base64.b64encode(command.encode("utf-16-le")).decode("ascii")


def run_elevated(command: str, *, keep_open: bool) -> tuple[bool, str]:
    """Run one command in an elevated shell, through the system's own prompt.

    Waited on, except for the one that must not be. The permanent rule takes
    milliseconds, and the caller redraws from what the firewall says afterwards -- so
    returning before the elevated shell had even started meant the redraw asked too
    early, saw no rule, and left the buttons offering to add the one that had just
    been added. The temporary rule's shell sits waiting for Enter, and waiting on that
    would hang the settings window for as long as the window stays open, which is the
    whole point of it.

    A refused prompt is not a failure of this application and does not read as one.
    """
    if not supported():
        return False, "Only Windows has a firewall this can change."

    # -NoExit for the temporary rule, whose window is the rule's lifetime. The
    # permanent one is waited on and hidden: it takes milliseconds, and the caller
    # redraws from what the firewall says afterwards, which it cannot do before the
    # rule exists.
    parts = ["'-NoProfile'"]
    if keep_open:
        parts.append("'-NoExit'")
    parts += ["'-EncodedCommand'", _quote(_encoded(command))]
    launcher = ("Start-Process powershell -Verb RunAs "
                + ("" if keep_open else "-WindowStyle Hidden -Wait ")
                + "-ArgumentList " + ",".join(parts))
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", launcher],
            capture_output=True, text=True, timeout=_ELEVATION_TIMEOUT,
            creationflags=config.subprocess_flags())
    except subprocess.TimeoutExpired:
        # Said without claiming what became of the rule, because this side no longer
        # knows: the prompt is still on screen and may yet be answered. The caller
        # asks the firewall itself rather than believing this.
        return False, "The administrator prompt went unanswered."
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "canceled by the user" in detail.lower() or "已取消" in detail:
            return False, "The administrator prompt was declined."
        log.info("elevation failed: %s", detail[:300])
        return False, "Windows would not start the elevated command."
    return True, ""


def linux_hint(port: int) -> str:
    """What to do on a Linux desktop that does have a firewall running, or "".

    Read as an ordinary user, and silent when the answer needs root: a machine whose
    firewall cannot be read is not a machine to give instructions about.
    """
    if shutil.which("firewall-cmd"):
        state = _run(["firewall-cmd", "--state"])
        if state and "running" in state:
            ports = _run(["firewall-cmd", "--list-ports"]) or ""
            if not _mentions_port(ports, port):
                return (f"sudo firewall-cmd --add-port={port}/tcp --permanent && "
                        "sudo firewall-cmd --reload")
    if shutil.which("ufw"):
        status = _run(["ufw", "status"])
        if status and "inactive" not in status and not _mentions_port(status, port):
            return f"sudo ufw allow {port}/tcp"
    return ""


def _mentions_port(text: str, port: int) -> bool:
    """Whether this listing already has a rule for this TCP port.

    On its own boundaries, because a substring is not a port. ufw's listing was
    searched for the bare digits, which an existing rule for 18571 or 85710 contains,
    and firewalld's for "8571/tcp", which "18571/tcp" contains -- either one would
    have withheld the hint from a machine that needed it. A bare number in ufw means
    any protocol, so that counts; 8571/udp does not.
    """
    return re.search(rf"(?<!\d){port}(?:/tcp)?(?![\w/])", text) is not None


def _quote(value: str) -> str:
    """For PowerShell, where a single-quoted string escapes its quote by doubling."""
    return "'" + value.replace("'", "''") + "'"


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=_TIMEOUT,
                                creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("%s: %s", command[0], exc)
        return None
    if result.returncode != 0:
        log.info("%s exited %d", command[0], result.returncode)
        return None
    return result.stdout.strip().lower()
