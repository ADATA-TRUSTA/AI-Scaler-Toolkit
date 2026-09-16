"""Where everything lives, per platform.

Every path is per-user. Nothing here needs administrator rights, which is what keeps
the installer per-user too -- the same bargain Ollama makes.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_NAME = "AIScalerToolkit"
APP_ID = "ai-scaler-toolkit"
APP_TITLE = "AI Scaler Toolkit"

IS_WINDOWS = sys.platform == "win32"

# Service port. The backend's own default is 8000; the desktop build moves off it so a
# hand-run backend and the packaged one can coexist on one machine.
DEFAULT_SERVICE_PORT = 8571
SERVICE_HOST = "127.0.0.1"

# Left alone on purpose. Both are part of an inference configuration that the
# load-model payload carries, so anything written here is overwritten in normal use.
LLAMA_SERVER_PORT = 5001
VLLM_PORT = 5000

# Provisioning downloads a venv (~3.7 GB), the GPU build of torch (~2.7 GB) and the
# llama binaries. Checked before starting rather than discovered halfway through.
MIN_FREE_DISK_GB = 10


# Windows only. Without it every helper this launcher runs -- reg query, the
# provisioning script, the service -- flashes or holds open a console window, which
# in a windowed application looks like something has gone wrong.
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


def subprocess_flags() -> int:
    return NO_WINDOW


def _win_local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))


def app_dir() -> Path:
    """Installed program directory.

    Frozen: the directory the executable sits in. From source: the repo's deploy/.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """Root for files bundled with the launcher (UI pages, the component list).

    PyInstaller unpacks bundled data under sys._MEIPASS, which is not where the
    modules appear to live, so resolving these from __file__ works from a checkout and
    breaks in the shipped build -- the failure being a missing UI page at first launch.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def shipped_payload_dir() -> Path:
    """The payload as delivered (backend/, frontend/).

    Installed builds put it beside the launcher; running from a source checkout it is
    the repo root one level up.
    """
    here = app_dir()
    if (here / "backend").is_dir():
        return here
    return here.parent


def working_payload_dir() -> Path:
    """Where the payload is copied to when what shipped cannot be written to."""
    return data_dir() / "payload"


def _is_writable(path: Path) -> bool:
    """Tried, not asked.

    os.access answers from the permission bits, and a read-only mount still has
    permissive ones: an AppImage's squashfs is owned by whoever ran it and marked
    drwxr-xr-x, so os.access says yes and the first real write says EROFS. Measured
    here -- the copy below did not happen, and nothing said why.
    """
    probe = path / f".write-probe-{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


@lru_cache(maxsize=1)
def payload_dir() -> Path:
    """The payload the launcher actually works from.

    Normally the shipped one. On Linux the shipped one is inside an AppImage, which is
    a read-only squashfs mount -- and the payload is not read-only in practice: the
    launcher writes backend/.env into it, and setup_env.sh builds llama.cpp into
    backend/service/utils/llama-bin. Both would fail, the first quietly (the backend
    falls back to .env.example and every setting is wrong) and the second loudly, in
    the middle of a first run.

    Rather than move each write somewhere else -- which fixes the two that exist today
    and not the third someone adds later -- the payload is copied somewhere writable
    and everything carries on as it does on Windows. See ensure_writable_payload.

    Cached because it is asked for constantly and the answer cannot change within a
    run.
    """
    shipped = shipped_payload_dir()
    if _is_writable(shipped):
        return shipped
    return working_payload_dir()


def _payload_stamp(shipped: Path) -> str:
    """Identifies the delivered payload, so an upgrade replaces the working copy.

    The AppImage's own size and modification time when there is one -- a new release
    is a new file, and that is the cheapest thing that changes. Falling back to a walk
    of the payload when it is not an AppImage, which is only development.
    """
    appimage = os.environ.get("APPIMAGE")
    if appimage:
        try:
            st = Path(appimage).stat()
            return f"appimage:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            pass
    count = 0
    total = 0
    newest = 0.0
    for path in shipped.rglob("*"):
        if path.is_file():
            try:
                st = path.stat()
            except OSError:
                continue
            count += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return f"walk:{count}:{total}:{int(newest)}"


# What a payload refresh must carry across, relative to the payload root.
#
# These are made on this machine rather than shipped, and they live inside the backend
# project because that is where the backend looks for them. Replacing the payload
# wholesale therefore deleted, on every upgrade: the user's registered models, the
# settings the launcher and setup_env recorded, and a llama build that takes ten
# minutes of CPU to make again. Nothing said so -- the next start simply had no models
# in it and no llama binary, while state.json still reported the machine as
# provisioned.
#
# Windows never had this: its installer overwrites rather than replaces, so all of it
# survived there. This brings Linux to the same place.
#
# An entry may name one path or end in a * to mean every file matching it, which is
# how service/configs is handled: the backend writes models_registry.json,
# current_inference_config.json, current_training_config.json and
# last_finetuned_model.json into it, and naming them one by one is how three of the
# four came to be missing here -- only the registry was listed, and an upgrade went on
# quietly deleting the rest. The directory cannot be kept whole either, because it also
# holds deepspeed configurations that ship with the release and must come from the new
# one. Matching the files the backend writes says both things at once.
PRESERVED_PAYLOAD_PATHS = (
    "backend/.env",
    "backend/service/configs/*.json",
    "backend/service/utils/llama-bin",
    "backend/service/utils/llama-build",
    "backend/service/utils/llama.cpp",
)


_payload_refreshed = False


def payload_was_refreshed() -> bool:
    """Whether this run replaced the working payload, i.e. this is the first start
    after an upgrade. False until ensure_writable_payload has run."""
    return _payload_refreshed


def _move_preserved(source: Path, target: Path) -> None:
    """Move whatever of PRESERVED_PAYLOAD_PATHS exists from one tree to another.

    A move, not a copy: on one filesystem it is a rename, and llama-build alone is
    several gigabytes.

    Never over something already there, which is what makes an interrupted refresh
    safe to repeat. A set-aside copy exists only because a previous attempt did not
    finish putting it back, so it is the one to keep; and nothing in the list is ever
    shipped, so on the way back there is nothing of the new release to overwrite.
    """
    import shutil

    for pattern in PRESERVED_PAYLOAD_PATHS:
        if "*" in pattern:
            found = sorted(source.glob(pattern))
        else:
            candidate = source / pattern
            found = [candidate] if candidate.exists() else []
        for origin in found:
            destination = target / origin.relative_to(source)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(origin), str(destination))


def ensure_writable_payload() -> Path:
    """Make sure the payload the launcher works from can be written to.

    A no-op wherever the shipped payload is writable, which is every Windows install
    and every source checkout. Inside an AppImage it is not: the mount is read-only,
    so the payload is copied into the data directory once and used from there.

    The copy is stamped with what it came from and remade when that changes, because
    an AppImage is upgraded by replacing the file -- with no installer to notice, a
    stale copy would go on being used and the upgrade would appear to do nothing.
    """
    import shutil

    global _payload_refreshed

    shipped = shipped_payload_dir()
    if _is_writable(shipped):
        return shipped

    working = working_payload_dir()
    stamp_file = working.parent / "payload.stamp"
    stamp = _payload_stamp(shipped)
    if (working.is_dir() and stamp_file.is_file()
            and stamp_file.read_text(encoding="utf-8").strip() == stamp):
        return working

    # Replaced rather than merged. A copy over the top of an older one leaves whatever
    # the new release removed, which is the same class of fault the Windows installer
    # needed [InstallDelete] for -- but everything in PRESERVED_PAYLOAD_PATHS has to
    # come through it, so it is set aside first and put back afterwards.
    keep = working.parent / "payload.keep"
    if working.exists():
        _move_preserved(working, keep)
        # Not ignore_errors: whatever it could not remove is still there, copytree
        # then fails with FileExistsError, and the caller -- which catches OSError and
        # logs "continuing with the shipped one" -- goes on using a payload that has
        # been half deleted and not replaced. On a read-only AppImage there is no
        # shipped one to continue with, so that message is wrong as well as the state.
        shutil.rmtree(working)
        if working.exists():
            raise OSError(f"{working} could not be emptied before replacing it")
    working.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(shipped, working)
    if keep.is_dir():
        _move_preserved(keep, working)
        shutil.rmtree(keep, ignore_errors=True)
    stamp_file.write_text(stamp + "\n", encoding="utf-8")
    _payload_refreshed = True
    return working


def backend_dir() -> Path:
    return payload_dir() / "backend"


def vcruntime_dir() -> Path:
    """The Visual C++ runtime the payload ships, which setup_env places beside the
    binaries that import it. See deploy/package/vcruntime.py for why it is shipped.

    Absent from a source checkout, where nothing has assembled a payload -- and not
    needed there, since a machine that builds this has the runtime installed."""
    return payload_dir() / "vcruntime"


def uv_binary() -> Path | None:
    """The uv the payload ships, or None when there is none to ship.

    setup_env installs uv from astral.sh when it cannot find one, which means running a
    downloaded script; an installed build carries a pinned, hashed copy instead, so that
    never happens on a user's machine. A source checkout has no payload and falls back,
    which is the right answer for a developer's machine.
    """
    name = "uv.exe" if IS_WINDOWS else "uv"
    candidate = payload_dir() / "uv" / name
    return candidate if candidate.is_file() else None


def frontend_config() -> Path:
    """The runtime config the WebUI fetches. Holds apiBaseUrl and hideChatPanel."""
    return payload_dir() / "frontend" / "dist" / "config.json"


def data_dir() -> Path:
    if IS_WINDOWS:
        return _win_local_appdata() / APP_NAME
    return Path.home() / ".local" / "share" / APP_ID


def config_dir() -> Path:
    if IS_WINDOWS:
        return _win_local_appdata() / APP_NAME
    return Path.home() / ".config" / APP_ID


def state_file() -> Path:
    return config_dir() / "state.json"


def log_dir() -> Path:
    return data_dir() / "logs"


def venv_dir() -> Path:
    """Where the backend's virtual environment goes.

    Deliberately outside the program directory: an uninstall-reinstall cycle should not
    have to re-download 6+ GB, and a program directory that grows by gigabytes after
    install surprises people.

    uv would otherwise create .venv inside the backend project, so provision.py passes
    this through UV_PROJECT_ENVIRONMENT.
    """
    return data_dir() / "service" / ".venv"


def hf_home() -> Path:
    """Model cache. Left in the conventional location so it is shared with any other
    Hugging Face tooling the user already has, and so uninstalling can offer to keep
    it -- it reaches tens of gigabytes."""
    return Path.home() / ".cache" / "huggingface"


def llama_dir() -> Path:
    """Where setup_env installs the llama runtime: inside the backend project.

    It used to land in a per-user directory on PATH, which is why the legacy location
    below is still cleaned up -- a purge that misses the real one leaves the next run
    reporting "already present" and skipping the step it was supposed to exercise.
    """
    return backend_dir() / "service" / "utils" / "llama-bin"


def llama_build_dir() -> Path:
    """Where setup_env extracts the llama.cpp sources and compiles them on Linux.

    The largest thing provisioning leaves behind -- 400 MB of sources and object files
    on a measured install -- and it was missing from the reset, which claims to remove
    everything setup_env put on the machine. Kept across upgrades on purpose (see
    PRESERVED_PAYLOAD_PATHS), which is exactly why a reset has to be the thing that
    removes it.
    """
    return backend_dir() / "service" / "utils" / "llama-build"


def llama_binary() -> Path:
    """The executable itself. setup_env records the resolved path in .env; this is
    only used for detection and cleanup."""
    return llama_dir() / ("llama-server.exe" if IS_WINDOWS else "llama-server")


def legacy_llama_binary() -> Path:
    """Earlier installs put it on PATH. Still removed on purge so an old machine ends
    up in the same state as a new one."""
    if IS_WINDOWS:
        return _win_local_appdata() / "Microsoft" / "WindowsApps" / "llama.exe"
    return Path.home() / ".local" / "bin" / "llama"


def llama_staging_dir() -> Path:
    """Scratch directory the llama installer stages downloads in. Uninstall clears it."""
    if IS_WINDOWS:
        return _win_local_appdata() / "llama-app"
    return Path.home() / ".llama-app"


def gguf_tools_dir() -> Path:
    """Sparse checkout of the llama.cpp conversion scripts that setup_env creates."""
    return backend_dir() / "service" / "utils" / "llama.cpp"


def setup_env_script() -> Path:
    """The backend's own provisioning script.

    Not reimplemented here. It already detects the accelerator, runs uv sync with the
    right extras, installs llama, fetches the GGUF conversion tooling and writes its
    part of .env. A second copy of that logic would drift from this one within a
    release or two.
    """
    scripts = backend_dir() / "scripts"
    return scripts / "windows" / "setup_env.ps1" if IS_WINDOWS else scripts / "linux" / "setup_env.sh"


def autostart_target() -> Path:
    """Per-user autostart entry. Windows uses a registry value instead of a file, so
    this is only meaningful on Linux."""
    return Path.home() / ".config" / "autostart" / f"{APP_ID}.desktop"


def menu_entry_target() -> Path:
    """The applications-menu entry. Linux only; on Windows the installer writes a
    Start Menu shortcut."""
    return Path.home() / ".local" / "share" / "applications" / f"{APP_ID}.desktop"


def desktop_shortcut_target() -> Path:
    """The icon on the desktop, matching the one the Windows installer offers.

    XDG_DESKTOP_DIR when the user has one -- a localised desktop directory is normal
    outside English installs, and writing to ~/Desktop there leaves a file nobody
    sees.
    """
    return _xdg_desktop_dir() / f"{APP_ID}.desktop"


def _xdg_desktop_dir() -> Path:
    config_file = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ) / "user-dirs.dirs"
    try:
        for line in config_file.read_text(encoding="utf-8").splitlines():
            if not line.startswith("XDG_DESKTOP_DIR="):
                continue
            value = line.split("=", 1)[1].strip().strip('"')
            expanded = os.path.expandvars(value.replace("$HOME", str(Path.home())))
            if expanded:
                return Path(expanded)
    except OSError:
        pass
    return Path.home() / "Desktop"


def component_entry_target(component_id: str) -> Path:
    """The menu entry for a component this application installed.

    Namespaced under APP_ID so it cannot collide with an entry the component's own
    packaging writes, and so uninstalling can find every entry we are responsible for
    without guessing at names.
    """
    return (Path.home() / ".local" / "share" / "applications"
            / f"{APP_ID}-{component_id}.desktop")


def component_icon_target(component_id: str) -> Path:
    return (Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256"
            / "apps" / f"{APP_ID}-{component_id}.png")


def menu_icon_target() -> Path:
    """Where the menu entry's Icon= name resolves from. hicolor is the theme every
    desktop falls back to, so an icon there is found without one being configured."""
    return (Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256"
            / "apps" / f"{APP_ID}.png")


def service_url(port: int) -> str:
    return f"http://{SERVICE_HOST}:{port}"


def openai_base_url(port: int) -> str:
    """Where an OpenAI-compatible client should point.

    The backend exposes /v1/chat/completions and /v1/models whatever engine is loaded,
    so this is a stable address to hand to a tool that speaks that protocol -- which is
    how a third-party agent uses a model served here instead of a cloud API.
    """
    return f"{service_url(port)}/v1"


def interface_url(port: int, *, compact: bool = False) -> str:
    """The interface itself, not the root that redirects to it.

    The backend answers / with a redirect to /frontend/, and that redirect drops the
    query string -- so a request for /?ui=compact arrives as /frontend/ with nothing
    attached, and the page renders as though it had been asked for the full view.
    Addressing the real path avoids depending on a redirect to carry anything.
    """
    url = f"{service_url(port)}/frontend/"
    return url + "?ui=compact" if compact else url
