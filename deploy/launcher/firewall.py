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
import os
import re
import shutil
import subprocess
from pathlib import Path

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


# The rules that stand in for a block rule Windows made on the service's interpreter --
# see add_command. Grouped, so they are found again by what they are rather than by a
# name a person might also have used.
GROUP = "AI Scaler Toolkit"
_NARROWED = "AI Scaler Toolkit: keeps blocking"


def service_programs() -> list[str]:
    """The executables Windows sees listening when the service runs, lower case.

    The service runs from a virtual environment, whose python.exe only starts the
    interpreter it was made from -- that one is the process holding the port, and the
    one a firewall rule names. Both spellings are returned, the path pyvenv.cfg records
    and the path it resolves to: uv records a version directory and also keeps a
    shorter name linked to it, and the rule Windows made named the resolved one.
    """
    found: list[str] = []
    try:
        cfg = (config.venv_dir() / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return found
    match = re.search(r"^home\s*=\s*(.+?)\s*$", cfg, re.MULTILINE)
    if not match:
        return found
    exe = Path(match.group(1)) / "python.exe"
    for candidate in (str(exe), os.path.realpath(exe)):
        lowered = candidate.lower()
        if lowered not in found:
            found.append(lowered)
    return found


def _undo_narrowing() -> str:
    """Put back the block rules add_command stood in for, and remove its stand-ins."""
    return (f'Get-NetFirewallRule -Group "{GROUP}" -ErrorAction SilentlyContinue | '
            f'Where-Object {{ $_.DisplayName -like "{_NARROWED}*" }} | ForEach-Object {{ '
            "if ($_.Description) { Enable-NetFirewallRule -Name $_.Description "
            "-ErrorAction SilentlyContinue }; Remove-NetFirewallRule -Name $_.Name }")


def _narrow_blocks(port: int, programs: list[str]) -> str:
    """Make room for this port in any block rule on the service's interpreter.

    Cancelling the prompt Windows raises the first time a program listens is not a
    neutral answer: Windows records it as an inbound block rule on that program, and a
    block rule beats every allow rule -- so the port rule this application adds did
    nothing, and the button said "allowed" on a machine nobody could reach. Measured on
    the Windows test VM: two such rules on uv's python.exe, TCP and UDP, on the Public
    profile, and with them in place another computer got no answer at all.

    Removing the rule would be simplest, and wrong: the interpreter is uv's, shared with
    anything else run on that Python -- Hermes among them -- and the person said no to
    all of it listening. So the TCP block is replaced with one that blocks every port
    but this one, same program, same profiles, and the original is disabled rather than
    deleted, its name kept on the stand-in so removing our rule can put it back. Only
    rules on any port are touched; one somebody wrote for particular ports was a
    decision, not an answer to a prompt.
    """
    if not programs:
        return ""
    targets = ",".join(_quote(p) for p in programs)
    ranges = []
    if port > 1:
        ranges.append(f"'1-{port - 1}'")
    if port < 65535:
        ranges.append(f"'{port + 1}-65535'")
    return (
        f"$targets = @({targets}); "
        "Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True "
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "$rule = $_; "
        "$program = ($rule | Get-NetFirewallApplicationFilter).Program; "
        "if (-not $program) { return }; "
        "$program = [Environment]::ExpandEnvironmentVariables($program).ToLowerInvariant(); "
        "if ($targets -notcontains $program) { return }; "
        "$ports = $rule | Get-NetFirewallPortFilter; "
        "if ($ports.LocalPort -ne 'Any') { return }; "
        "if ($ports.Protocol -ne 'TCP' -and $ports.Protocol -ne 'Any') { return }; "
        f'New-NetFirewallRule -DisplayName "{_NARROWED} $($rule.DisplayName) except {port}" '
        f'-Group "{GROUP}" -Description $rule.Name -Direction Inbound -Action Block '
        f"-Program $program -Protocol TCP -LocalPort {','.join(ranges)} "
        "-Profile $rule.Profile | Out-Null; "
        "if ($ports.Protocol -eq 'Any') { "
        f'New-NetFirewallRule -DisplayName "{_NARROWED} $($rule.DisplayName) (UDP)" '
        f'-Group "{GROUP}" -Description $rule.Name -Direction Inbound -Action Block '
        "-Program $program -Protocol UDP -Profile $rule.Profile | Out-Null }; "
        "Disable-NetFirewallRule -Name $rule.Name }"
    )


def add_command(port: int, *, temporary: bool = False,
                programs: list[str] | None = None) -> str:
    """The command a button runs.

    Legible on purpose, because the administrator prompt Windows raises names this
    command and the reader is entitled to understand what they are agreeing to.

    Three steps: undo any earlier stand-ins (the port may have changed since), add the
    allow rule, and make room for the port in a block rule Windows made on the
    service's interpreter -- without that last one the allow rule does nothing on a
    machine where the first prompt was cancelled. See _narrow_blocks.
    """
    if programs is None:
        programs = service_programs()
    name = rule_name(port, temporary=temporary)
    # Only when missing: pressed twice, it used to leave two identical rules.
    add = (f'if (-not (Get-NetFirewallRule -DisplayName "{name}" -ErrorAction SilentlyContinue)) '
           f'{{ New-NetFirewallRule -DisplayName "{name}" -Direction Inbound '
           f"-LocalPort {port} -Protocol TCP -Action Allow | Out-Null }}")
    steps = [_undo_narrowing(), add, _narrow_blocks(port, programs)]
    if not temporary:
        # A permanent rule of ours for a port the service has since moved off. Left
        # behind, it kept that port open to the network with nothing of ours on it,
        # and nothing on the page mentions it once the port has changed.
        steps.insert(1, (
            f'Get-NetFirewallRule -DisplayName "{PERMANENT_RULE} *" -ErrorAction SilentlyContinue | '
            f"Where-Object {{ $_.DisplayName -match '^{PERMANENT_RULE} \\d+$' "
            f'-and $_.DisplayName -ne "{name}" }} | Remove-NetFirewallRule'))
    command = "; ".join(step for step in steps if step)
    if not temporary:
        return command
    # The rule lives as long as the window does, which is the only honest way to make
    # a firewall rule temporary: Windows has no expiry for them, and an application
    # that promised to remove one later would be promising across its own crash.
    # try/finally covers Enter and Ctrl+C. Closing the window with the X kills the
    # shell, and the rule stays -- said plainly on the button rather than hidden. The
    # stand-ins go with it, and the block rules they replaced come back.
    return (f"{command}; "
            "try { Read-Host 'Allowed. Press Enter to remove the rule' | Out-Null } "
            f"finally {{ Remove-NetFirewallRule -DisplayName \"{name}\"; "
            f"{_undo_narrowing()} }}")


def remove_command(port: int) -> str:
    """Remove our allow rules, and give back any block rule add_command stood in for.

    Every port's, not only this one's: the port can change after a rule was added, and
    a remove that looked for the current port alone left the old one open with nothing
    of ours listening on it. There is one service, so every such rule is its.
    """
    del port  # kept for the callers; the rules are found by name, whatever their port
    return (f'Get-NetFirewallRule -DisplayName "{PERMANENT_RULE} *" -ErrorAction SilentlyContinue | '
            f"Where-Object {{ $_.DisplayName -match '^{PERMANENT_RULE} \\d+$' }} | "
            f"Remove-NetFirewallRule; {_undo_narrowing()}")


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

    Read as an ordinary user. Whether a firewall is on can always be found out that
    way; whether it already allows the port sometimes cannot, and then the command is
    offered regardless, since running it again changes nothing.
    """
    if shutil.which("firewall-cmd") and _firewalld_running():
        # Same as ufw below: the port list may be closed to us, and then the command
        # is offered regardless -- firewalld answers an existing port with a warning
        # and changes nothing.
        ports = _run(["firewall-cmd", "--list-ports"])
        if ports is None or not _mentions_port(ports, port):
            return (f"sudo firewall-cmd --add-port={port}/tcp --permanent && "
                    "sudo firewall-cmd --reload")
    if shutil.which("ufw") and _ufw_enabled():
        # Its rules are root's to read, so whether this port is already allowed is
        # usually unknown here -- and then the command is offered anyway: running it
        # twice is harmless, ufw skips a rule it already has. When they can be read,
        # an allowed port gets no hint.
        status = _run(["ufw", "status"])
        if status is None or not _mentions_port(status, port):
            return f"sudo ufw allow {port}/tcp"
    return ""


def _firewalld_running() -> bool:
    """Whether firewalld is running, asked of systemd rather than of firewalld.

    `firewall-cmd --state` goes through polkit, and for an ordinary user it answered
    "Authorization failed" -- over SSH and from the desktop session alike, checked on
    the Linux test VM -- so a running firewalld was never noticed. Whether the service
    is up is something systemd tells anyone.
    """
    state = _run(["systemctl", "is-active", "firewalld"])
    if state is not None:
        return state == "active"
    state = _run(["firewall-cmd", "--state"])
    return state is not None and "running" in state


def _ufw_enabled(conf: Path = Path("/etc/ufw/ufw.conf")) -> bool:
    """Whether ufw is switched on, read where any user can.

    `ufw status` was asked, and it answers only root: as anyone else it prints "You need
    to be root" and nothing more -- so on Ubuntu, where ufw is the firewall, an enabled
    one was never noticed and the page offered no help to a machine nobody could reach.
    Checked on the Linux test VM. The switch itself is ENABLED= in ufw.conf, which is
    readable by everyone. (systemd's view is no use: the unit is "active" with the
    firewall off, because it runs either way.)
    """
    try:
        text = conf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return re.search(r"^\s*ENABLED\s*=\s*\"?yes\"?\s*$", text, re.MULTILINE | re.IGNORECASE) is not None


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
