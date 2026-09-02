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

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
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


def detect_hermes() -> str | None:
    found = shutil.which("hermes")
    if found:
        return found
    roots = []
    if os.environ.get("HERMES_HOME"):
        roots.append(Path(os.environ["HERMES_HOME"]))
    roots.append(_local_appdata() / "hermes")
    for root in roots:
        hit = _first_existing(root / "bin" / "hermes.exe", root / "bin" / "hermes")
        if hit:
            return str(hit)
    return None


def detect_comfyui() -> str | None:
    """Scan rather than hardcode.

    ComfyUI Desktop installs under a directory named after its npm scope, which is the
    sort of name that changes without anyone considering it a breaking change. Matching
    on "comfy" survives that; a literal path does not.
    """
    try:
        for entry in (_local_appdata() / "Programs").iterdir():
            if "comfy" not in entry.name.lower() or not entry.is_dir():
                continue
            hit = _first_existing(entry / "ComfyUI.exe", entry / "comfyui.exe")
            if hit:
                return str(hit)
    except OSError:
        pass

    found = shutil.which("comfyui") or shutil.which("ComfyUI")
    if found:
        return found

    candidates = [Path.home() / ".local/bin/ComfyUI", Path("/opt/ComfyUI/ComfyUI")]
    if os.environ.get("PROGRAMFILES"):
        candidates.append(Path(os.environ["PROGRAMFILES"]) / "ComfyUI" / "ComfyUI.exe")
    hit = _first_existing(*candidates)
    return str(hit) if hit else None


# -------------------------------------------------------------------------- install

def _stream(command: list[str], on_line: Callable[[str], None]) -> int:
    """Run a command, forwarding each line as it appears.

    Line by line rather than at the end: these take minutes, and a panel that stays
    empty until the process exits tells the user nothing about whether it is working.
    """
    log.info("running: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=config.subprocess_flags(),
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if line:
            on_line(line)
    return process.wait(timeout=_INSTALL_TIMEOUT)


def _download(url: str, target: Path, on_line: Callable[[str], None]) -> bool:
    on_line("Downloading the installer.")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AI-Scaler-Toolkit"})
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            step = 0
            while chunk := response.read(1024 * 512):
                out.write(chunk)
                done += len(chunk)
                # Every 10%, not every chunk: 180 MB of progress lines is not progress.
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    on_line(f"  {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    except (OSError, ValueError) as exc:
        on_line(f"Download failed: {exc}")
        return False
    on_line(f"Downloaded {target.stat().st_size / 1e6:.0f} MB.")
    return True


def install_hermes(on_line: Callable[[str], None]) -> bool:
    if sys.platform == "win32":
        command = [
            "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-Command",
            "$ProgressPreference = 'SilentlyContinue'; "
            f"Invoke-RestMethod {HERMES_PS1} | Invoke-Expression",
        ]
    else:
        command = ["bash", "-c", f"curl -fsSL {HERMES_SH} | bash"]
    code = _stream(command, on_line)
    if code != 0:
        on_line(f"The installer exited with code {code}.")
    return code == 0


def install_comfyui(on_line: Callable[[str], None]) -> bool:
    if sys.platform != "win32":
        on_line("ComfyUI Desktop only publishes a Windows build. The download page has "
                "the instructions for other systems.")
        return False
    target = Path(tempfile.gettempdir()) / "ComfyUI-Setup.exe"
    if not _download(COMFY_WINDOWS, target, on_line):
        return False
    on_line("Running ComfyUI's own installer. It installs for you alone, so it will "
            "not ask for an administrator.")
    code = _stream([str(target), "/S"], on_line)
    if code != 0:
        on_line(f"The installer exited with code {code}.")
        return False
    on_line("Installed. The first time you open ComfyUI it downloads its own copy of "
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
    install: Callable[[Callable[[str], None]], bool]
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
    running: bool = True
    ok: bool = False
    message: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            if len(self.lines) > _MAX_LINES:
                del self.lines[: len(self.lines) - _MAX_LINES]

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


def _explain_failure(component: Component, lines: list[str]) -> str:
    """Turn the tail of an install log into something worth reading."""
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

    job.append(f"Installing {component.name} on {platform.system()}.")

    def worker() -> None:
        try:
            ok = component.install(job.append)
        except Exception as exc:
            log.exception("installing %s failed", component_id)
            job.append(f"Unexpected error: {exc}")
            ok = False

        located = component.detect()
        if ok and located is None:
            # The installer said it worked and the program is not there. Trust the disk.
            job.append("The installer reported success but the program cannot be found.")
            ok = False

        job.ok = ok
        job.message = (f"{component.name} is installed." if ok
                       else _explain_failure(component, job.lines))
        job.running = False
        log.info("component %s finished ok=%s", component_id, ok)

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
