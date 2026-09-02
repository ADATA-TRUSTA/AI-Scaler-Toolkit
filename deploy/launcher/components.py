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
  through electron-builder's per-user NSIS target), so neither needs an administrator.
  That is not a detail: this application installs without elevation and would look
  broken if a component demanded it.
- We do not control either installer. When one fails we say what failed and leave the
  vendor's own page one click away, rather than pretending the failure is ours to fix.

Detection is deliberately looser than the install path: it looks for whatever is
actually on the machine, including copies this application never installed, because
telling someone to install something they already have is its own kind of wrong.
"""

from __future__ import annotations

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

from . import applog, config

log = applog.get("components")

HERMES_PS1 = "https://hermes-agent.nousresearch.com/install.ps1"
HERMES_SH = "https://hermes-agent.nousresearch.com/install.sh"
HERMES_PAGE = "https://github.com/nousresearch/hermes-agent"

# electron-builder's conventional download path. It redirects to the current build, so
# it does not go stale the way a pinned release asset would.
COMFY_WINDOWS = "https://download.comfy.org/windows/nsis/x64"
COMFY_PAGE = "https://comfy.org/download/"

_MAX_LINES = 400
_INSTALL_TIMEOUT = 3600


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


def detect_hermes() -> str | None:
    found = shutil.which("hermes")
    if found:
        return found

    roots: list[Path] = []
    for candidate in (os.environ.get("HERMES_HOME"), _user_environment("HERMES_HOME")):
        if candidate:
            root = Path(candidate)
            if root not in roots:
                roots.append(root)
    # The two places it installs itself: its historical default, and outside AppData,
    # where it has to go on machines that veto directory renames under AppData.
    roots += [_local_appdata() / "hermes", Path.home() / "hermes"]

    for root in roots:
        hit = _first_existing(root / "bin" / "hermes.exe", root / "bin" / "hermes")
        if hit:
            return str(hit)
    return None


def _program_in(folder: Path) -> Path | None:
    """The application executable in an install folder, if there is one.

    Not a fixed filename. A real install put its program at "Comfy Desktop.exe" while
    every guess in circulation -- including the one this file shipped with -- said
    "ComfyUI.exe", so the install worked and the page reported it missing. What is
    reliable is the shape of the folder: one program, plus an uninstaller named after
    it.
    """
    try:
        executables = [p for p in folder.glob("*.exe")
                       if not p.name.lower().startswith(("uninstall", "unins"))]
    except OSError:
        return None
    if not executables:
        return None
    # Deterministic when a folder holds more than one: the shortest name is the
    # application, the longer ones are its helpers.
    return min(executables, key=lambda p: (len(p.name), p.name))


def detect_comfyui() -> str | None:
    """Find the install folder by name, then ask the folder what it contains."""
    roots = [_local_appdata() / "Programs"]
    if os.environ.get("PROGRAMFILES"):
        roots.append(Path(os.environ["PROGRAMFILES"]))
    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if "comfy" not in entry.name.lower() or not entry.is_dir():
                continue
            program = _program_in(entry)
            if program:
                return str(program)

    found = shutil.which("comfyui") or shutil.which("ComfyUI")
    if found:
        return found

    hit = _first_existing(Path.home() / ".local/bin/ComfyUI", Path("/opt/ComfyUI/ComfyUI"))
    return str(hit) if hit else None


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


def _download(url: str, target: Path, out: Progress) -> bool:
    out.note("Downloading the installer.")
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


def install_hermes(out: Progress) -> bool:
    if sys.platform == "win32":
        command = [
            "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-Command",
            "$ProgressPreference = 'SilentlyContinue'; "
            f"Invoke-RestMethod {HERMES_PS1} | Invoke-Expression",
        ]
    else:
        command = ["bash", "-c", f"curl -fsSL {HERMES_SH} | bash"]
    code = _stream(command, out, "install-hermes.log")
    if code != 0:
        out.note(f"The installer exited with code {code}.")
    return code == 0


def install_comfyui(out: Progress) -> bool:
    if sys.platform != "win32":
        out.note("ComfyUI Desktop only publishes a Windows build. The download page has "
                 "the instructions for other systems.")
        return False
    target = Path(tempfile.gettempdir()) / "ComfyUI-Setup.exe"
    if not _download(COMFY_WINDOWS, target, out):
        return False
    out.note("Running ComfyUI's own installer. It installs for you alone, so it will "
             "not ask for an administrator.")
    code = _stream([str(target), "/S"], out, "install-comfyui.log")
    if code != 0:
        out.note(f"The installer exited with code {code}.")
        return False
    out.note("Installed. The first time you open ComfyUI it downloads its own copy of "
             "PyTorch, around 2.5 GB, separate from this application's.")
    return True


# ------------------------------------------------------------------------ catalogue

def explain_hermes(log_tail: str) -> str | None:
    """Hermes cannot always move an existing environment aside.

    Known, reproducible on a machine with a filesystem filter driver, and completely
    opaque otherwise: the installer prints a path and no instruction. Diagnosing it
    once took a day; recognising it costs one string match.
    """
    if "venv" in log_tail and ("could not move" in log_tail or "access is denied" in log_tail
                               or "拒絕存取" in log_tail):
        return ("Hermes could not replace its existing Python environment. Delete "
                "%LOCALAPPDATA%\\hermes\\hermes-agent\\venv and run this again: with no "
                "environment present the installer creates a fresh one instead of "
                "trying to rename the old one.")
    return None


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
    # Turns this component's own known failures into an instruction. Keeping it here
    # rather than in a shared if-chain keyed on id means the knowledge sits with the
    # thing it is about, and adding a component cannot break another one's advice.
    explain: Callable[[str], str | None] | None = None


CATALOGUE: tuple[Component, ...] = (
    Component(
        id="hermes",
        name="Hermes Agent",
        description="Self-improving AI agent by Nous Research.",
        note="Installs into your user profile. Takes a few minutes: it provisions a "
             "Python environment of its own.",
        page=HERMES_PAGE,
        detect=detect_hermes,
        install=install_hermes,
        explain=explain_hermes,
    ),
    Component(
        id="comfyui",
        name="ComfyUI",
        description="Node-based generative AI workflows, the official desktop build.",
        note="A 180 MB download. On first launch it fetches its own PyTorch, about "
             "2.5 GB and separate from this application's, so budget the disk twice.",
        page=COMFY_PAGE,
        detect=detect_comfyui,
        install=install_comfyui,
        windows_only=True,
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
            return [line for line, ours in zip(self.lines, self._ours) if not ours]

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


def start(component_id: str) -> tuple[bool, str]:
    """Begin an install. Returns (started, error)."""
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
            ok = component.install(out)
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
        job.message = (f"{component.name} is installed." if ok
                       else _explain_failure(component, job.installer_lines()))
        job.running = False
        log.info("component %s finished ok=%s at %s", component_id, ok, located or "-")

    threading.Thread(target=worker, name=f"install-{component_id}", daemon=True).start()
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
        out.append({
            "id": component.id,
            "name": component.name,
            "description": component.description,
            "note": component.note,
            "page": component.page,
            "installed": located is not None,
            "location": located or "",
            "available": not (component.windows_only and sys.platform != "win32"),
            "busy": busy == component.id,
            "blocked": bool(busy) and busy != component.id,
        })
    return out
