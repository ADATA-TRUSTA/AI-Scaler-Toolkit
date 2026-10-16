"""Install the optional tools this application is expected to sit alongside.

This reverses an earlier decision, and the reversal is worth recording. The original
manifest showed an install command and never ran it, which bought three things: no
output panel to build, no elevation to handle, and nothing to break when a vendor
changed their installer. What it cost was the only thing a user actually wanted --
the tool installed. "Here is a command, go and paste it somewhere" is a to-do item
dressed as a feature.

So we run them, and we pay those three costs deliberately:

- Output is streamed to the page, because an install that takes minutes and says
  nothing is indistinguishable from one that has hung.
- Both installs are per-user by design (Hermes into %LOCALAPPDATA%\\hermes, ComfyUI
  into this application's own data directory), so neither needs an administrator.
  That is not a detail: this application installs without elevation and would look
  broken if a component demanded it.
- We do not control Hermes's installer. When it fails we say what failed and leave the
  vendor's own page one click away, rather than pretending the failure is ours to fix.
  ComfyUI is installed by this application itself -- see comfyui.py for why.

Detection is deliberately looser than the install path: it looks for whatever is
actually on the machine, including copies this application never installed, because
telling someone to install something they already have is its own kind of wrong.
"""

from __future__ import annotations

import contextlib
import locale
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import applog, comfyui, config, shortcuts

log = applog.get("components")

HERMES_PS1 = "https://hermes-agent.nousresearch.com/install.ps1"
HERMES_SH = "https://hermes-agent.nousresearch.com/install.sh"
HERMES_PAGE = "https://github.com/nousresearch/hermes-agent"

_MAX_LINES = 400
_INSTALL_TIMEOUT = 3600

# How long a component gets to fail before it is called started. An application that is
# going to refuse -- a missing library, a broken install -- says so and exits at once;
# one that is going to run is still running. Long enough to catch the first, short
# enough not to be a pause the user notices.
_START_GRACE_SECONDS = 2.5

# How an outcome reads, so the page can colour it without knowing what it says. The
# distinction that matters is the middle one: "load a model first" and "Hermes is not
# installed" are the state of things, not faults, and painting them the same red as
# "it would not start" tells the reader something broke when nothing did.
DONE = "done"
BLOCKED = "blocked"
FAILED = "failed"


# --------------------------------------------------------------------------- detect

def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")


def _user_environment(name: str) -> str | None:
    """Read a persistent user environment variable, not this process's copy of it.

    Windows does not push environment changes into running processes, so an installer
    that sets HERMES_HOME cannot be seen by an application that was already open --
    which is exactly the application offering the Install button. Without this, a
    successful install reports itself as missing until the app is restarted.
    """
    if not config.IS_WINDOWS:
        return None
    try:
        result = subprocess.run(
            ["reg", "query", r"HKCU\Environment", "/v", name],
            capture_output=True, text=True, timeout=30,
            creationflags=config.subprocess_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if name in line and "REG_" in line:
            # NAME    REG_SZ    value
            parts = line.split(None, 2)
            if len(parts) == 3:
                return parts[2].strip()
    return None


def _as_spelled(path: str) -> str:
    """The path as the filesystem spells it.

    shutil.which on Windows does not read the directory. It appends each PATHEXT entry
    to the stem and asks whether that file exists, and PATHEXT is written in capitals,
    so an ordinary hermes.exe came back as "hermes.EXE" -- which works everywhere and
    is wrong on the card, where it sits beside a path found by listing a directory.

    Only the spelling is corrected. realpath would also have fixed it and would have
    followed links on the way, turning the command on PATH into wherever it points,
    which is not the same answer to "where is Hermes".
    """
    entry = Path(path)
    try:
        for sibling in entry.parent.iterdir():
            if sibling.name.lower() == entry.name.lower():
                return str(sibling)
    except OSError:
        pass
    return path


def detect_hermes() -> str | None:
    found = shutil.which("hermes")
    if found:
        return _as_spelled(found)

    roots: list[Path] = []
    for candidate in (os.environ.get("HERMES_HOME"), _user_environment("HERMES_HOME")):
        if candidate:
            root = Path(candidate)
            if root not in roots:
                roots.append(root)
    # The two places it installs itself on Windows: its historical default, and outside
    # AppData, where it has to go on machines that veto directory renames under AppData.
    roots += [_local_appdata() / "hermes", Path.home() / "hermes"]

    for root in roots:
        hit = _first_existing(root / "bin" / "hermes.exe", root / "bin" / "hermes")
        if hit:
            return str(hit)

    # Linux, when ~/.local/bin is not on this process's PATH -- a launcher started by
    # the desktop inherits whatever the session had, which is not always the shell's.
    # The installer puts the command there as a shim, and the program itself inside
    # ~/.hermes, which is its default home on Linux and appears in none of the roots
    # above.
    if not config.IS_WINDOWS:
        hit = _first_existing(Path.home() / ".local" / "bin" / "hermes",
                              Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes")
        if hit:
            return str(hit)
    return None


# -------------------------------------------------------------------------- install

class Progress:
    """What an install can say, kept separate by who is saying it.

    Not a cosmetic distinction. _explain_failure reads the log back to recognise known
    failures, and our own reassurance -- "it installs for you alone, so it will not ask
    for an administrator" -- contains the exact word that check looks for. Scanning our
    own narration made the page blame elevation for a failure that had nothing to do
    with it, and hid the real reason underneath a confident wrong answer.
    """

    def __init__(self, job: Job) -> None:
        self._job = job

    def line(self, text: str) -> None:
        """The installer said this."""
        self._job.record(text, ours=False)

    def note(self, text: str) -> None:
        """We are saying this."""
        self._job.record(text, ours=True)

    def tail(self, lines: int) -> list[str]:
        """The installer's recent output, for deciding what to do about a failure.

        Its own output only -- see the class docstring for what reading our narration
        back cost the last time.
        """
        return self._job.installer_lines()[-lines:]


def _decode(raw: bytes) -> str:
    """Decode a line of installer output without guessing wrong silently.

    Console tools on Windows write in the OEM code page, not UTF-8, so reading their
    output as UTF-8 turns every non-ASCII character into a replacement mark -- which is
    exactly the part of an error message that matters when the system language is not
    English. UTF-8 is tried strictly first precisely so that it can fail and let the
    machine's own encoding have a turn.
    """
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp1252"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _stream(command: list[str], out: Progress, log_name: str) -> int:
    """Run a command, forwarding each line as it appears.

    Line by line rather than at the end: these take minutes, and a panel that stays
    empty until the process exits tells the user nothing about whether it is working.

    The output goes to a file that the child owns, and we tail the file. A pipe would
    be the obvious choice and is the wrong one: the pipe belongs to this process, so
    this process going away breaks the installer's next write and aborts it partway.
    That is not hypothetical -- it left a third-party tool with an empty virtual
    environment, broken and still reporting itself as installed. Writing to a file
    means an install survives us, and leaves its log behind either way.
    """
    log.info("running: %s", " ".join(command))
    config.log_dir().mkdir(parents=True, exist_ok=True)
    path = config.log_dir() / log_name

    with path.open("wb") as sink:
        process = subprocess.Popen(
            command,
            stdout=sink,
            stderr=subprocess.STDOUT,
            creationflags=config.subprocess_flags(),
        )
        deadline = time.monotonic() + _INSTALL_TIMEOUT
        with path.open("rb") as reader:
            while True:
                raw = reader.readline()
                if raw:
                    text = _decode(raw).rstrip()
                    if text:
                        out.line(text)
                    continue
                if process.poll() is not None:
                    for rest in reader:  # whatever landed after the last read
                        text = _decode(rest).rstrip()
                        if text:
                            out.line(text)
                    break
                if time.monotonic() > deadline:
                    out.note(f"Giving up after {_INSTALL_TIMEOUT // 60} minutes.")
                    process.kill()
                    break
                time.sleep(0.3)

    out.note(f"Full output: {path}")
    return process.returncode if process.returncode is not None else 1


def _download(url: str, target: Path, out: Progress, what: str = "the installer") -> bool:
    out.note(f"Downloading {what}.")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AI-Scaler-Toolkit"})
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            step = 0
            while chunk := response.read(1024 * 512):
                handle.write(chunk)
                done += len(chunk)
                # Every 10%, not every chunk: 180 MB of progress lines is not progress.
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    out.note(f"  {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    except (OSError, ValueError) as exc:
        out.note(f"Download failed: {exc}")
        return False
    out.note(f"Downloaded {target.stat().st_size / 1e6:.0f} MB.")
    return True


def _hermes_desktop_exe(root: Path) -> Path | None:
    """The windowed application, which is a separate build from the command line tool.

    electron-builder writes it to release/<platform>-unpacked/, named after the product.
    The known names are tried first; after them, any *-unpacked directory, so that a
    target this list does not name -- another architecture, a renamed platform folder
    -- is still found rather than reported as never built.
    """
    release = root / "hermes-agent" / "apps" / "desktop" / "release"
    known = _first_existing(release / "win-unpacked" / "Hermes.exe",
                            release / "win-arm64-unpacked" / "Hermes.exe",
                            release / "linux-unpacked" / "Hermes",
                            release / "linux-arm64-unpacked" / "Hermes")
    if known:
        return known
    name = "Hermes.exe" if config.IS_WINDOWS else "Hermes"
    try:
        folders = sorted(release.glob("*-unpacked"))
    except OSError:
        return None
    for folder in folders:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _hermes_home() -> Path:
    """Where Hermes actually is, preferring the disk over any environment variable.

    The process copy of HERMES_HOME is the least reliable answer available: it was
    captured when this application started, and an install that moved Hermes since
    then updated the persistent variable, which running processes never see. Taking it
    at face value pointed at the previous location, which is how this was found.

    A home is recognised by hermes-agent/, the checkout itself. Recognising it by a
    bin/ directory as well was wrong on Linux: there the command on PATH is a shim in
    ~/.local/bin, so "two levels up from the command" is ~/.local -- which has a bin/,
    passed, and sent every later question to a directory with no Hermes in it.
    """
    located = detect_hermes()
    candidates: list[Path] = []
    if located:
        candidates.append(Path(located).parent.parent)
        # The program the Linux shim runs: ~/.hermes/hermes-agent/venv/bin/hermes.
        candidates.append(Path(located).parent.parent.parent.parent)
    for value in (_user_environment("HERMES_HOME"), os.environ.get("HERMES_HOME")):
        if value:
            candidates.append(Path(value))
    candidates += [Path.home() / ".hermes", _local_appdata() / "hermes", Path.home() / "hermes"]
    for home in candidates:
        if (home / "hermes-agent").is_dir():
            return home
    for value in (_user_environment("HERMES_HOME"), os.environ.get("HERMES_HOME")):
        if value:
            return Path(value)
    # Nothing installed: each platform's own default, not Windows' on both.
    return _local_appdata() / "hermes" if config.IS_WINDOWS else Path.home() / ".hermes"


def _default_install_home() -> Path:
    """Where the vendor's script will put Hermes if we do not tell it otherwise.

    Deliberately not _hermes_home(): that answers "where is it", by looking at the
    disk, and the two are different questions. Deciding whether an install is heading
    somewhere that cannot work has to use the destination the script will pick, which
    is its own default -- HERMES_HOME as the child process will inherit it, else
    %LOCALAPPDATA%\\hermes. Using the disk answer here meant the retry never fired
    when an existing install already sat somewhere that works.
    """
    inherited = os.environ.get("HERMES_HOME")
    return Path(inherited) if inherited else _local_appdata() / "hermes"


def _under_appdata(path: Path) -> bool:
    appdata = Path.home() / "AppData"
    try:
        path.resolve().relative_to(appdata.resolve())
    except (ValueError, OSError):
        return False
    return True


# The symptom of a machine that vetoes directory renames under AppData. uv extracts a
# managed Python to a temporary directory and renames it into place; where the rename
# is blocked, uv reports the *target* as missing, which is true and useless.
_APPDATA_RENAME_VETO = (
    "missing expected target directory for python minor version link",
    "python 3.11 not available",
)


def _run_hermes_installer(script: Path, out: Progress, log_name: str,
                          home: Path | None) -> int:
    """Run the vendor's script from a file so its own switches can be passed.

    -IncludeDesktop is the reason. Without it the script installs the command line
    tool and stops: the windowed application is opt-in upstream, and its build step is
    also what creates the Start Menu and desktop shortcuts. Piping the script into
    Invoke-Expression, as the documented one-liner does, cannot pass a switch at all.
    """
    prelude = "$ProgressPreference = 'SilentlyContinue'; "
    invocation = f"& {_ps_string(str(script))} -IncludeDesktop"
    if home is not None:
        # The environment variable, and the parameter as well.
        #
        # -HermesHome alone does not work: the script's Python stage recomputes its
        # install directory from %LOCALAPPDATA% and ignores the parameter, so a retry
        # that passed only -HermesHome went straight back to the location that cannot
        # work. The environment variable is what was observed to work by hand, and
        # observation beats the tidier-looking option.
        prelude += f"$env:HERMES_HOME = {_ps_string(str(home))}; "
        invocation += f" -HermesHome {_ps_string(str(home))}"
    command = [
        "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-Command", prelude + invocation,
    ]
    return _stream(command, out, log_name)


def _ps_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _install_hermes_linux(out: Progress) -> bool:
    """The vendor's script with --include-desktop, as Windows passes -IncludeDesktop.

    The desktop application is opt-in upstream on both platforms, and this used to ask
    for it on Windows only -- so Linux got the command line tool and nothing to open,
    while the source tree for the windowed build sat in apps/desktop unbuilt. The
    switch is the vendor's own, and its build stage carries its own stall timeout and
    mirror fallback for the Electron download, which is more than a second build step
    of ours would.

    From a file rather than piped into bash, for the same reason as on Windows: a pipe
    cannot pass the switch. `bash -s -- --include-desktop` could, but a file also gives
    a real path to report errors against.

    And --non-interactive, which skips the vendor's setup wizard and its gateway
    install. The script runs them on /dev/tty whenever it can open one, so started
    from a terminal it stopped at "How would you like to set up Hermes?" with nobody
    reading that terminal, and the card spun forever. They are not needed: this
    application configures Hermes itself, through `hermes config set` (see
    hermes_commands). Started with no terminal they were skipped anyway, which is why
    it went unnoticed.
    """
    script = Path(tempfile.gettempdir()) / "hermes-install.sh"
    if not _download(HERMES_SH, script, out):
        return False
    out.note("Installing the command line tool and building the desktop application. "
             "The desktop build is a separate step and takes several minutes.")
    code = _stream(["bash", str(script), "--include-desktop", "--non-interactive"], out,
                   "install-hermes.log")
    if code != 0:
        out.note(f"The installer exited with code {code}.")

    # Judged by what is on disk, as on Windows, and for the same reason: which half
    # was built is what decides what the card and the Open button can offer.
    located = detect_hermes()
    desktop = _hermes_desktop_exe(_hermes_home())
    if desktop is not None:
        # Hermes writes a hermes.desktop of its own the first time it is opened, so an
        # entry of ours would be a second "Hermes" in the menu opening a terminal. One
        # from an earlier version of this application, which could only offer the
        # terminal, goes too.
        shortcuts.remove_component("hermes")
        out.note(f"Desktop application built: {desktop}")
        # The installer's last step on Linux is sudo, to make Chromium's sandbox helper
        # setuid root, and it fails the whole desktop stage when sudo cannot ask for a
        # password -- which it cannot here, with no terminal. The build is already on
        # disk by then and runs without the helper, so the failure it prints above
        # reads as far worse than it is. Said plainly, beside it.
        if "Cannot configure Electron sandbox helper" in "\n".join(out.tail(200)):
            out.note("The installer could not set up Chromium's sandbox helper, which "
                     "needs your password, and there is no terminal here to ask for it. "
                     "That is the error above, and it does not stop the application: "
                     "Hermes opens without the sandbox, and its card says how to turn "
                     "the sandbox on.")
        return True
    if located is None:
        out.note("Hermes did not install.")
        return False
    # The command line tool is there and the window is not. Still worth a menu entry,
    # since it is the one thing that runs: its own installer says to reload the shell,
    # which is a per-terminal fix and no help to anyone looking for it in a menu, and
    # the entry names the absolute path because a desktop does not read .bashrc.
    if shortcuts.ensure_component("hermes", Path(located), "Hermes Agent", terminal=True,
                                  comment="Self-improving AI agent by Nous Research"):
        out.note("Added the command line tool to the applications menu; it opens in a "
                 "terminal.")
    out.note("The command line tool installed, but the desktop application was not "
             "built. Its output is above; installing again retries the build.")
    return False


def install_hermes(out: Progress) -> bool:
    if sys.platform != "win32":
        return _install_hermes_linux(out)

    script = Path(tempfile.gettempdir()) / "hermes-install.ps1"
    if not _download(HERMES_PS1, script, out):
        return False

    out.note("Installing the command line tool and building the desktop application. "
             "The desktop build is a separate step and takes several minutes.")
    code = _run_hermes_installer(script, out, "install-hermes.log", home=None)
    if code != 0:
        out.note(f"The installer exited with code {code}.")

    # Judged by what is on disk, never by the exit code. This installer prints
    # "[X] Installation failed" and then exits 0, so a retry gated on the exit code
    # never fires and a success gated on it is a coin toss. The exit code is reported
    # above, for the reader, and decides nothing.
    #
    # One retry, and only for the failure with a known cause and a known cure: the
    # default location is inside AppData, and on a machine that vetoes directory
    # renames there the install cannot complete wherever else it is tried. Retrying
    # blindly would fail twice and take twice as long.
    if _hermes_desktop_exe(_hermes_home()) is None and _under_appdata(_default_install_home()):
        tail = "\n".join(out.tail(80)).lower()
        if any(marker in tail for marker in _APPDATA_RENAME_VETO):
            fallback = Path.home() / "hermes"
            out.note(
                "That failed while installing its own Python, in a way that means this "
                f"machine will not allow it under {Path.home() / 'AppData'}. Retrying "
                f"in {fallback}, which is outside it."
            )
            retry = _run_hermes_installer(script, out, "install-hermes-retry.log",
                                          home=fallback)
            if retry != 0:
                out.note(f"The retry exited with code {retry}.")

    # Both halves have to be there. The command line tool with no window is not what
    # was asked for, and the script reports success having built only the first.
    desktop = _hermes_desktop_exe(_hermes_home())
    if desktop is None:
        if detect_hermes() is None:
            out.note("Hermes did not install.")
        else:
            out.note("The command line tool installed, but the desktop application was "
                     "not built. Run `hermes desktop` to build it.")
        return False
    out.note(f"Desktop application built: {desktop}")
    return True


def install_comfyui(out: Progress) -> bool:
    """Install for the card that would be picked by default -- see comfyui.py."""
    return comfyui.install(comfyui.default_device(comfyui.devices()), out)


def select_comfyui(device_id: str) -> tuple[bool, str, str]:
    """Make this the device ComfyUI uses, installing what it needs first.

    Recorded, not started: the image and video skills start ComfyUI on it when they
    need it -- see comfyui.py for why it is theirs to start. Returns (ok, message,
    kind). When something has to be installed this starts a job and returns at once;
    the page follows it the way it follows any install.
    """
    device = comfyui.find_device(device_id)
    if device is None:
        return False, "That device is no longer on this computer.", FAILED
    if not device.usable:
        return False, device.note or f"{device.name} cannot run ComfyUI.", BLOCKED

    if comfyui.source_installed() and comfyui.env_ready(device.backend):
        comfyui.write_selection(device)
        return True, "", DONE

    def work(out: Progress) -> bool:
        if not comfyui.install(device, out):
            return False
        # The menu entry an earlier version wrote for Comfy Desktop on Linux names a
        # program this application no longer installs.
        with contextlib.suppress(Exception):
            shortcuts.remove_component("comfyui")
        comfyui.write_selection(device)
        return True

    started, error = start("comfyui", work=work,
                           done=f"ComfyUI is installed and set to use {device.name}.")
    return (True, "installing", DONE) if started else (False, error, BLOCKED)


# ------------------------------------------------------------------------ catalogue

def explain_hermes(log_tail: str) -> str | None:
    """Hermes cannot always move an existing environment aside.

    Known, reproducible on a machine with a filesystem filter driver, and completely
    opaque otherwise: the installer prints a path and no instruction. Diagnosing it
    once took a day; recognising it costs one string match.
    """
    if any(marker in log_tail for marker in _APPDATA_RENAME_VETO):
        return ("Hermes could not install its own Python. On this machine a directory "
                f"cannot be moved into place anywhere under {Path.home() / 'AppData'}, "
                "which is where it installs by default, and its installer reports that "
                "as a missing directory. Installing it outside AppData works: "
                f"`$env:HERMES_HOME='{Path.home() / 'hermes'}'` before running the "
                "vendor's script.")
    if "venv" in log_tail and ("could not move" in log_tail or "access is denied" in log_tail
                               or "拒絕存取" in log_tail):
        return ("Hermes could not replace its existing Python environment. Delete "
                "%LOCALAPPDATA%\\hermes\\hermes-agent\\venv and run this again: with no "
                "environment present the installer creates a fresh one instead of "
                "trying to rename the old one.")
    return None


def open_hermes() -> str | None:
    """The windowed application, when it was built. Windows only.

    Detection finds the command line tool, because that is what proves Hermes is
    installed and what a PATH lookup returns. Opening it is a different question: on
    Windows the installer also builds a windowed application, and starting the command
    line tool there gives a console window with a shell prompt in it -- which is what
    the Open button did.

    None when it is not there, and open_component then falls back to what detection
    found. On Linux the built application is never opened by its path -- see
    hermes_open_command for why that is not safe there.
    """
    if not config.IS_WINDOWS:
        return None
    desktop = _hermes_desktop_exe(_hermes_home())
    return str(desktop) if desktop else None


# Where the last launch of the desktop application wrote its output. Read back by the
# card, because it is the only place that says whether the window came up sandboxed.
HERMES_DESKTOP_LOG = "hermes-desktop.log"

# What Hermes's launcher prints when it has to start Electron unsandboxed. Its own
# words, matched as they are: rewording them upstream would lose the note on the card,
# which is the right way for that to fail -- quietly, and only the note.
_HERMES_NO_SANDBOX = "Falling back to --no-sandbox"


def hermes_open_command() -> list[str] | None:
    """Linux: `hermes desktop --skip-build`, not the built binary.

    Starting the binary directly skips the part of Hermes's launcher that decides how
    Chromium is sandboxed, and on Ubuntu 23.10 and later that decision is not optional:
    unprivileged user namespaces are restricted, so Electron needs either a setuid-root
    chrome-sandbox or --no-sandbox, and without either it aborts on start. The launcher
    uses the sandbox when it can, sets up the setuid helper with sudo when it can ask,
    and falls back to --no-sandbox when it cannot -- which is the case from here, with
    no terminal for sudo to ask in. That policy is the vendor's to make; running the
    binary ourselves would take it over and get it wrong on exactly those machines.

    --skip-build because Open should open. Its menu entry runs `hermes desktop` and
    rebuilds when the sources have changed; Hermes's own update runs --build-only for
    that, so the release directory is kept current without a click that sits for three
    minutes before a window appears.
    """
    if config.IS_WINDOWS or _hermes_desktop_exe(_hermes_home()) is None:
        return None
    cli = detect_hermes()
    return [cli, "desktop", "--skip-build"] if cli else None


def _hermes_ran_unsandboxed() -> bool:
    try:
        text = (config.log_dir() / HERMES_DESKTOP_LOG).read_text(encoding="utf-8",
                                                                errors="replace")
    except OSError:
        return False
    return _HERMES_NO_SANDBOX in text


def describe_hermes() -> tuple[str, bool]:
    """Say whether the windowed application is there, not just the command line tool.

    Returns (text, settled) -- settled False marks something the reader should act on,
    so the page can colour it without knowing what any of it means.

    Running without the sandbox is settled: the window works, and turning the sandbox
    on is the user's call, made with their own password in their own terminal. This
    application does not ask for it on their behalf -- nothing it installs does.
    """
    if _hermes_desktop_exe(_hermes_home()) is None:
        return ("Command line tool only — the desktop application is not built. "
                "Reinstall to build it.", False)
    if not config.IS_WINDOWS and _hermes_ran_unsandboxed():
        return ("Desktop application built. On this system it runs without Chromium's "
                "sandbox; to turn the sandbox on, run `hermes desktop` once in a "
                "terminal — it asks for your password.", True)
    return "Desktop application built.", True


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    description: str
    note: str
    page: str
    detect: Callable[[], str | None]
    install: Callable[[Progress], bool]
    windows_only: bool = False
    # Whether this is a command-line tool, which decides how opening it is arranged: a
    # terminal has to be found for it, and its menu entry asks the desktop for one.
    # Hermes ships a windowed build on Windows and a command on Linux, so this is
    # per-platform rather than a property of the component.
    terminal: bool = False
    # Turns this component's own known failures into an instruction. Keeping it here
    # rather than in a shared if-chain keyed on id means the knowledge sits with the
    # thing it is about, and adding a component cannot break another one's advice.
    explain: Callable[[str], str | None] | None = None
    # Anything about an installed copy worth saying on the card. "Installed" is one
    # bit, and one bit was not enough for Hermes: the command line tool and the
    # windowed application are separate builds, and a card reading "Installed" beside
    # a missing desktop application is the kind of half-truth this file keeps fixing.
    describe: Callable[[], tuple[str, bool]] | None = None
    # What Open should start, when that is not the file detection found. Returning
    # None means "the same thing", which is the case for every component whose one
    # build is both the proof it is installed and the thing you open.
    open_target: Callable[[], str | None] | None = None
    # A command to run for Open, taking precedence over the file and its menu entry,
    # for a component whose own launcher has to be the one to start it. Returning None
    # means "no such command here", and Open carries on as it would without one.
    open_command: Callable[[], list[str] | None] | None = None
    # Where that command's output goes, relative to the log directory.
    open_log: str = ""


CATALOGUE: tuple[Component, ...] = (
    Component(
        id="hermes",
        name="Hermes Agent",
        description="Self-improving AI agent by Nous Research.",
        note="Installs into your user profile. Takes several minutes: it provisions a "
             "Python environment of its own, then builds the desktop application.",
        page=HERMES_PAGE,
        detect=detect_hermes,
        install=install_hermes,
        explain=explain_hermes,
        describe=describe_hermes,
        open_target=open_hermes,
        open_command=hermes_open_command,
        open_log=HERMES_DESKTOP_LOG,
        # Only for the fallback: on Linux, when the desktop application did not build,
        # what is left to open is the command line tool, and that needs a terminal.
        # With the desktop build present, open_command starts it and this is not used.
        terminal=not config.IS_WINDOWS,
    ),
    Component(
        id="comfyui",
        name="ComfyUI",
        description="Node-based generative AI workflows, run here without a window of "
                    "its own and opened in the browser.",
        note="Pick the card ComfyUI uses; the image and video skills start it there "
             "when they need it. Each kind of card gets its own PyTorch, downloaded the "
             "first time it is chosen: about 2.3 GB for NVIDIA, 1.8 GB for Intel.",
        page=comfyui.PAGE,
        detect=comfyui.detect,
        install=install_comfyui,
    ),
)

BY_ID = {component.id: component for component in CATALOGUE}


# ------------------------------------------------------------------------------ job

@dataclass
class Job:
    """One install in flight. Only one runs at a time.

    The page polls this rather than being pushed to. Polling is the unglamorous choice
    and the right one here: pywebview's bridge is request/response, and calling into a
    page that may have been closed mid-install is a crash waiting to be written.
    """

    component_id: str
    lines: list[str] = field(default_factory=list)
    # Parallel to lines: True where we wrote it, False where the installer did.
    _ours: list[bool] = field(default_factory=list, repr=False)
    running: bool = True
    ok: bool = False
    message: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, line: str, ours: bool) -> None:
        with self._lock:
            self.lines.append(line)
            self._ours.append(ours)
            if len(self.lines) > _MAX_LINES:
                cut = len(self.lines) - _MAX_LINES
                del self.lines[:cut]
                del self._ours[:cut]

    def installer_lines(self) -> list[str]:
        """Only what the installer itself produced -- see Progress."""
        with self._lock:
            # strict: the two lists are appended together under this lock, so a length
            # difference means that stopped being true and should be loud.
            return [line for line, ours in zip(self.lines, self._ours, strict=True)
                    if not ours]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.component_id,
                "running": self.running,
                "ok": self.ok,
                "message": self.message,
                "lines": list(self.lines),
            }


_job: Job | None = None
_job_lock = threading.Lock()


def current_job() -> dict | None:
    with _job_lock:
        job = _job
    return job.snapshot() if job else None


def _settle(component: Component, seconds: int = 25) -> str | None:
    """Give the installer a moment to finish before asking whether it worked.

    An installer can return before the program is on disk -- it hands off to a child
    process, or the shell writes the last files after the parent exits. Asking the
    instant it returns reported a successful ComfyUI install as missing.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            found = component.detect()
        except Exception:
            log.exception("detecting %s failed", component.id)
            found = None
        if found or time.monotonic() >= deadline:
            return found
        time.sleep(1.0)


def _explain_failure(component: Component, lines: list[str]) -> str:
    """Turn the tail of an install log into something worth reading.

    `lines` is the installer's own output. Passing everything the page shows would
    include our narration, and matching on that produces confident wrong answers.
    """
    blob = "\n".join(lines[-40:]).lower()

    if component.explain is not None:
        try:
            specific = component.explain(blob)
        except Exception:
            log.exception("explaining %s's failure raised", component.id)
            specific = None
        if specific:
            return specific

    # Applies to any of them: an installer that wants elevation cannot be satisfied
    # here, and saying so is more use than repeating that it failed.
    if "elevation" in blob or "administrator" in blob:
        return (f"{component.name}'s installer asked for administrator rights, which "
                "this application does not have. Install it from the vendor's page "
                "instead.")
    return (f"{component.name} did not install. The log above is the installer's own "
            "output; the vendor's page has the manual instructions.")


def start(component_id: str, work: Callable[[Progress], bool] | None = None,
          done: str = "") -> tuple[bool, str]:
    """Begin an install. Returns (started, error).

    `work` replaces the component's own install for this run, and `done` what success
    says -- ComfyUI's Start installs whatever its card is missing and then starts it,
    which is one job to watch rather than two.
    """
    global _job
    component = BY_ID.get(component_id)
    if component is None:
        return False, "Unknown component."
    if component.windows_only and sys.platform != "win32":
        return False, f"{component.name} is only published for Windows."

    with _job_lock:
        if _job is not None and _job.running:
            return False, "Another component is still installing."
        job = Job(component_id=component_id)
        _job = job

    out = Progress(job)
    out.note(f"Installing {component.name} on {platform.system()}.")

    def worker() -> None:
        try:
            ok = (work or component.install)(out)
        except Exception as exc:
            log.exception("installing %s failed", component_id)
            out.note(f"Unexpected error: {exc}")
            ok = False

        located = _settle(component) if ok else component.detect()
        if ok and located is None:
            # The installer said it worked and the program is not there. Trust the disk.
            out.note("The installer reported success but the program cannot be found.")
            ok = False

        job.ok = ok
        job.message = ((done or f"{component.name} is installed.") if ok
                       else _explain_failure(component, job.installer_lines()))
        job.running = False
        log.info("component %s finished ok=%s at %s", component_id, ok, located or "-")

    threading.Thread(target=worker, name=f"install-{component_id}", daemon=True).start()
    return True, ""


def _entry_program(entry: Path) -> str:
    """The program a desktop entry runs, without its arguments or quoting."""
    try:
        text = entry.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if not line.startswith("Exec="):
            continue
        rest = line[len("Exec="):].strip()
        # We quote the path because it may contain spaces; a vendor's may not.
        return rest[1:].split('"', 1)[0] if rest.startswith('"') else rest.split(" ", 1)[0]
    return ""


def _write_entry(component: Component, program: Path) -> bool:
    """(Re)write a component's menu entry for whatever is runnable now."""
    return shortcuts.ensure_component(component.id, program, component.name,
                                      terminal=component.terminal,
                                      comment=component.description)


def open_component(component_id: str) -> tuple[bool, str, str]:
    """Start an installed component. Returns (started, message).

    Through the desktop entry where there is one, rather than running the program
    ourselves: the entry already says how this component wants to be started -- with a
    terminal or without, with the vendor's own arguments -- and the desktop is the
    thing that knows which terminal to use. Running it directly would have to guess
    both, and guessing wrong on a command-line tool means a click that does nothing.
    """
    component = BY_ID.get(component_id)
    if component is None:
        return False, "Unknown component.", FAILED
    located = component.detect()
    if not located:
        return False, f"{component.name} does not look installed.", BLOCKED
    # Detection answers "is it installed"; this answers "what does Open start". They
    # are the same file for most components and not for Hermes -- see open_hermes.
    program = Path((component.open_target and component.open_target()) or located)

    if config.IS_WINDOWS:
        try:
            os.startfile(program)  # noqa: S606 - the point is to open it as the shell would
        except OSError as exc:
            return False, f"Could not open {component.name}: {exc}", FAILED
        return True, "", DONE

    command = component.open_command() if component.open_command else None
    if command:
        log_file = config.log_dir() / (component.open_log or f"{component_id}-open.log")
        log.info("opening %s with %s (output to %s)", component_id, " ".join(command), log_file)
        # Unbuffered, because the log is read while the program is still running. A
        # Python launcher writing to a file buffers its output in blocks and, when it
        # waits on the application it started -- as `hermes desktop` waits on Electron
        # -- nothing it printed reaches the file until the window is closed. Measured:
        # the line saying Hermes had fallen back to --no-sandbox was missing from the
        # log of a window that was running with --no-sandbox.
        started, detail = _spawn(command, log_to=log_file,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if started:
            return True, "", DONE
        log.info("%s would not start: %s", component.name, detail)
        return False, f"{component.name} would not start. {detail}".strip(), FAILED

    entry = config.component_entry_target(component_id)
    # An entry outlives the thing it names: one written by an earlier version of this
    # application can name a program that is no longer what starts the component.
    # Rewriting it before use fixes the click and the menu at once; deferring to an
    # entry we can see is wrong would be deferring for its own sake.
    if entry.is_file() and _entry_program(entry) != str(program):
        log.info("entry for %s names %r, not %s -- rewriting it",
                 component_id, _entry_program(entry), program)
        _write_entry(component, program)

    launcher_cmd = shutil.which("gio")
    if entry.is_file() and launcher_cmd:
        try:
            subprocess.Popen([launcher_cmd, "launch", str(entry)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        except OSError as exc:
            log.info("gio launch failed for %s: %s", component_id, exc)
        else:
            return True, "", DONE

    if component.terminal:
        # No desktop entry to defer to, so the terminal has to be found. Ordered by how
        # likely it is to be the one this desktop actually uses: the Debian alternative
        # first, because it is whatever the user or the distribution chose.
        for name, flag in (("x-terminal-emulator", "-e"), ("gnome-terminal", "--"),
                           ("konsole", "-e"), ("xfce4-terminal", "-x"),
                           ("kitty", ""), ("alacritty", "-e"), ("xterm", "-e")):
            found = shutil.which(name)
            if not found:
                continue
            command = [found, flag, str(program)] if flag else [found, str(program)]
            started, detail = _spawn(command)
            if started:
                return True, "", DONE
            log.info("%s could not start %s: %s", name, component.name, detail)
        return False, (f"No terminal found to run {component.name} in. "
                       f"Run it yourself with: {program}"), BLOCKED

    started, detail = _spawn([str(program)])
    if not started:
        return False, f"{component.name} would not start. {detail}".strip(), FAILED
    return True, "", DONE


def _spawn(command: list[str], log_to: Path | None = None,
           env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Start something and find out whether it stayed started.

    Popen not raising means the file could be executed, which is a much weaker claim
    than the program running: a ComfyUI AppImage on a machine without libfuse2 starts,
    prints that it needs FUSE and exits, and reporting that as success gives the user a
    button that does nothing and says nothing. Measured on the VM, where exactly that
    happened.

    So it is given a moment to fail. Still running afterwards is success. So is
    exiting cleanly in that time, which is what a launcher that hands off does: GNOME
    Terminal asks its server for a window and exits 0 at once. Counting that as a
    failure is how opening Hermes on the VM went -- each terminal in turn opened a
    window, was reported as "could not start" with nothing after the colon, because it
    had printed nothing, and the next one was tried, until the list ran out and the
    page said no terminal could be found beside the windows it had opened.

    log_to sends the program's output to a file instead of a pipe. A pipe is only safe
    for as long as someone reads it, and nobody does once this returns: a chatty
    program -- Electron logs plenty -- fills it and blocks on its next write.
    """
    sink = None
    try:
        if log_to is not None:
            log_to.parent.mkdir(parents=True, exist_ok=True)
            sink = log_to.open("w", encoding="utf-8", errors="replace")
        process = subprocess.Popen(
            command,
            stdout=sink if sink is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace",
            start_new_session=True, env=env, creationflags=config.subprocess_flags())
    except OSError as exc:
        if sink is not None:
            sink.close()
        return False, str(exc)
    try:
        code = process.wait(timeout=_START_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return True, ""
    finally:
        # The child has its own copy of the descriptor; ours is not needed either way.
        if sink is not None:
            sink.close()
    if code == 0:
        return True, ""
    output = ""
    with contextlib.suppress(Exception):
        if log_to is not None:
            output = log_to.read_text(encoding="utf-8", errors="replace").strip()
        elif process.stdout:
            output = (process.stdout.read() or "").strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False, f"It exited with code {code}."
    # The first line that reads as a failure, else the first line. Neither end alone
    # works: an AppImage without FUSE says what is wrong first and "for more
    # information" last, while Hermes's launcher announces what it is about to do
    # before the line marked ✗ that says why it could not.
    failure = next((line for line in lines
                    if line.startswith("✗") or "error" in line.lower() or "fail" in line.lower()),
                   lines[0])
    return False, failure[:200]


# An endpoint of our own in Hermes's `providers:` map, rather than its built-in
# "custom" slot. Writing to "custom" was wrong twice over: it is one shared slot, so
# pointing Hermes here quietly overwrote whatever else somebody had put in it, and it
# carries no name -- Hermes's endpoint editor showed "Custom / custom" with a default
# model of upstage/solar-pro4:free, which is Hermes's own default and not something
# this service serves. A named provider shows what it is and leaves "custom" alone.
HERMES_PROVIDER_ID = "trusta-ast"
HERMES_PROVIDER_NAME = "TRUSTA AST"

# A stable name for "whatever is loaded here", rather than the id of whatever happens
# to be loaded at the moment the button is pressed. This service answers on the model
# it has loaded and ignores the name asked for -- checked against it: a request naming
# trusta-ast-default came back answered by Qwen/Qwen2.5-0.5B-Instruct -- so writing the
# real id would buy nothing and cost a Hermes endpoint that silently names yesterday's
# model every time a different one is loaded here.
HERMES_MODEL_ALIAS = "trusta-ast-default"

# Providers Hermes ships selected. Finding one of these means nobody has chosen yet;
# anything else is a decision, and is not ours to overwrite. Newer Hermes installs
# ship "auto" (pick by whichever API key is present); without it here, a fresh
# install was taken for a deliberate choice and the button refused to touch it.
HERMES_STOCK_PROVIDERS = ("openrouter", "custom", "auto")


def hermes_provider(service_url: str, api_key: str = "") -> dict[str, object]:
    """The endpoint this application asks Hermes to keep, as Hermes stores it.

    One definition, written twice over: once by the button, a key at a time through
    `hermes config set`, and once into the commands the page hands out for a Hermes on
    another computer. They were two hand-written lists and they drifted -- the button
    moved to a named provider and the commands were left setting the old shared
    "custom" slot, so the same card offered two different results depending on which
    machine you were setting up.

    discover_models is what keeps the endpoint honest: Hermes asks the service what it
    is serving on every request, so the model below is a stable name for "whatever is
    loaded there" rather than the id of whatever happened to be loaded when this was
    written.

    api_key is for a Hermes on another computer, which the service asks for one. The
    button on this machine passes none: loopback needs no key, and a key written into
    this machine's Hermes would be one more copy to go stale when it is changed.
    """
    provider: dict[str, object] = {
        "name": HERMES_PROVIDER_NAME,
        "base_url": service_url,
        "discover_models": True,
        "model": HERMES_MODEL_ALIAS,
    }
    if api_key:
        # Hermes sends it as "Authorization: Bearer", on chat and on the /v1/models
        # probe discover_models makes -- read from hermes_cli/config_providers.py.
        provider["api_key"] = api_key
    return provider


def hermes_selection() -> list[tuple[str, str]]:
    """The two keys that make the endpoint above the one Hermes actually uses.

    model.default is not redundant with the provider's own model. Unset it and Hermes
    sends an empty model field -- checked against this service, which logged
    `"model": ""` -- so the provider's copy is what its endpoint editor shows, and this
    is what goes out on the wire.
    """
    return [("model.provider", HERMES_PROVIDER_ID),
            ("model.default", HERMES_MODEL_ALIAS)]


def _as_config_value(value: object) -> str:
    """`hermes config set` takes text. Its YAML reads true, not True."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def hermes_commands(service_url: str, api_key: str = "") -> list[str]:
    """The same endpoint, as commands to run on a Hermes somewhere else.

    Three, not six: `hermes config set` takes one key and one value, but the value may
    be a JSON object -- verified against it -- so the whole provider goes in one. The
    two selection keys cannot join it: providers and model are separate branches of the
    configuration, and model.default is not redundant with the provider's own model --
    unset it and Hermes sends an empty model field.

    Single-quoted because that is a literal string in both cmd-era PowerShell and in
    bash, and the other computer may be either.
    """
    import json

    provider = json.dumps(hermes_provider(service_url, api_key), separators=(",", ":"))
    return [f"hermes config set providers.{HERMES_PROVIDER_ID} '{provider}'",
            *(f"hermes config set {key} {value}" for key, value in hermes_selection())]


def _hermes_manual_steps(service_url: str) -> str:
    """The commands to run when this declines to run them, as lines to read.

    Built from hermes_commands, so what is offered by hand is what the button would
    have done. Two hand-written copies of this drifted once already.
    """
    return "".join("\n  " + line for line in hermes_commands(service_url))


def point_hermes_here(service_url: str, model: str | None) -> tuple[bool, str, str]:
    """Point an installed Hermes at this machine's model service.

    Through `hermes config set`, which is the vendor's own supported way to change it.
    Editing ~/.hermes/config.yaml directly would mean parsing and rewriting two
    thousand lines of someone else's commented configuration, and getting the merge
    wrong loses settings that are not ours.

    It refuses when base_url is already something other than the shipped default: that
    is a decision somebody made, and silently replacing it is how an application earns
    a reputation for meddling. The values are reported instead, to be applied by hand.
    """
    program = detect_hermes()
    if not program:
        return False, "Hermes does not look installed.", BLOCKED

    active = _hermes_config_get(program, "model.provider")
    if active is UNKNOWN:
        # Not read, so not overwritten. Treating this as "nobody has chosen" is how
        # a deliberate choice gets replaced by an application that could not even
        # see it.
        return False, ("Hermes is installed but its configuration could not be read, "
                       "so it has been left alone. To set it by hand:" +
                       _hermes_manual_steps(service_url)), BLOCKED
    if active not in (None, "", HERMES_PROVIDER_ID) and active not in HERMES_STOCK_PROVIDERS:
        return False, (f"Hermes is set to use the \"{active}\" provider, which somebody "
                       f"chose. Leaving it alone. To point it here by hand:"
                       + _hermes_manual_steps(service_url)), BLOCKED

    settings = [(f"providers.{HERMES_PROVIDER_ID}.{key}", _as_config_value(value))
                for key, value in hermes_provider(service_url).items()]
    settings += hermes_selection()
    for key, value in settings:
        ok, detail = _hermes_config_set(program, key, value)
        if not ok:
            return False, f"Could not set {key}: {detail}", FAILED

    log.info("pointed Hermes at %s (model %s)", service_url, model or "unset")
    if model:
        return True, f"Hermes now uses {model} on this machine.", DONE
    return True, ("Hermes now uses this machine. Load a model here and it will be "
                  "offered; until then Hermes has nothing to talk to."), DONE


def _hermes_command(program: str) -> list[str]:
    return [program]


def _run_hermes(program: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a Hermes command, its output decoded by _decode.

    Not text=True: that reads in the Windows codepage, and Hermes is a Python program
    run with -I, which ignores PYTHONIOENCODING -- so whether it writes UTF-8 or the
    codepage is up to its own runtime. Decoded strictly the wrong way, a model name
    outside the codepage raised past the callers' except clauses.
    """
    result = subprocess.run(_hermes_command(program) + args, capture_output=True,
                            timeout=120, creationflags=config.subprocess_flags())
    return subprocess.CompletedProcess(result.args, result.returncode,
                                       _decode(result.stdout or b""), _decode(result.stderr or b""))


# What _hermes_config_get returns when the question could not be put to Hermes at all,
# as distinct from a key that is simply not set. They are not the same answer, and the
# guard in point_hermes_here acts on the difference: not set means nobody has chosen,
# unreadable means we do not know, and overwriting on "we do not know" is the one thing
# that guard exists to prevent.
UNKNOWN = object()


def _hermes_config_get(program: str, key: str) -> str | None | object:
    """The value, None when the key is not set, UNKNOWN when it could not be read."""
    try:
        result = _run_hermes(program, ["config", "get", key])
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if result.returncode != 0:
        # Hermes exits non-zero for an unset key and says so in as many words; any
        # other non-zero exit happened before it could look.
        said = (result.stdout + result.stderr).lower()
        return None if "not set" in said else UNKNOWN
    return result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else None


def _hermes_config_set(program: str, key: str, value: str) -> tuple[bool, str]:
    try:
        result = _run_hermes(program, ["config", "set", key, value])
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[:200]
    return True, ""


def listing() -> list[dict]:
    job = current_job()
    busy = job["id"] if job and job["running"] else ""
    out = []
    for component in CATALOGUE:
        try:
            located = component.detect()
        except Exception:
            log.exception("detecting %s failed", component.id)
            located = None
        detail, detail_settled = "", True
        if located is not None and component.describe is not None:
            try:
                detail, detail_settled = component.describe()
            except Exception:
                log.exception("describing %s failed", component.id)
        out.append({
            "id": component.id,
            "name": component.name,
            "description": component.description,
            "note": component.note,
            "page": component.page,
            "installed": located is not None,
            "detail": detail,
            "detail_settled": detail_settled,
            "location": located or "",
            "available": not (component.windows_only and sys.platform != "win32"),
            "busy": busy == component.id,
            "blocked": bool(busy) and busy != component.id,
            # Said here rather than worked out in the page, so the card does not have
            # to know which components are command-line tools or which platform it is
            # on -- both are this file's business.
            "terminal": component.terminal,
            "can_point_here": component.id == "hermes" and located is not None,
            # ComfyUI's card is a device picker with Start and Stop rather than an
            # Install button; its contents come from comfyui_state.
            "picker": component.id == "comfyui",
        })
    return out
