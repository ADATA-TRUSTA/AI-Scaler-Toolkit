"""ComfyUI without its desktop shell, run by this application on the device you pick.

Why not Comfy Desktop, which this used to install: Desktop owns the ComfyUI process.
The user starts it from Desktop's window, with Desktop's arguments, so nothing else can
decide how it is launched -- and the image and video skills that sit beside this
application have to, because the one guarantee they rely on (--reserve-vram, sized at
launch to whatever the hosted LLM is holding) only exists if they are the ones starting
it. Desktop also picks the PyTorch build from a fixed vendor order, NVIDIA before Intel,
so a machine with both had no say in which card ran it.

So this keeps ComfyUI itself and nothing around it:

- One copy of the ComfyUI source, downloaded as a release archive. Not a git clone:
  nothing else this application does needs git, and a clone would make it a
  prerequisite for one optional component.
- One virtual environment per PyTorch build -- cuda, xpu, cpu -- next to it. The build is
  what decides which vendor's cards ComfyUI can see at all, so switching from the NVIDIA
  card to the Intel one is switching environments, while switching between two NVIDIA
  cards is only an argument. Environments are created the first time their device is
  started, not up front: each is gigabytes, and most machines will only ever use one.
- The same torch versions the backend pins. uv keeps one cache, so on a machine whose
  backend already downloaded torch for this card the environment is mostly hard links.

The models, custom nodes and outputs live in the one source tree, so every environment
sees the same ones.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from . import applog, config, state

log = applog.get("comfyui")

REPO = "Comfy-Org/ComfyUI"
PAGE = "https://github.com/Comfy-Org/ComfyUI"
# Used only when the releases feed cannot be read. A known release beats failing an
# install over a metadata request.
FALLBACK_TAG = "v0.38.0"
ARCHIVE = "https://github.com/{repo}/archive/refs/tags/{tag}.zip"

HOST = "127.0.0.1"
PORT = 8188
PYTHON = "3.12"

# Kept in step with the backend's own pins (backend/pyproject.toml), on purpose: uv has
# one cache, and matching versions mean the backend's download is reused rather than
# repeated. torchaudio stops at 2.11.0 because that is the newest that exists.
TORCH_VERSIONS = {"torch": "2.13.0", "torchvision": "0.28.0", "torchaudio": "2.11.0"}


@dataclass(frozen=True)
class Backend:
    id: str
    label: str
    index: str
    local: str          # the wheel's local version tag
    # Roughly what a first install downloads, for the card to say before the click.
    # The figures are what Comfy Desktop reports for the same builds.
    download_gb: float


BACKENDS = {
    "cuda": Backend("cuda", "CUDA", "https://download.pytorch.org/whl/cu130", "cu130", 2.3),
    "xpu": Backend("xpu", "XPU", "https://download.pytorch.org/whl/xpu", "xpu", 1.8),
    "cpu": Backend("cpu", "CPU", "https://download.pytorch.org/whl/cpu", "cpu", 0.7),
}

# CUDA 13 wheels need an R580 driver or newer.
NVIDIA_MIN_DRIVER = 580
# torch 2.13.0+xpu bundles the oneAPI 2026.0 runtime, which needs this Intel driver or
# newer on Windows -- the same floor backend/pyproject.toml records for its xpu extra.
INTEL_MIN_DRIVER = (32, 0, 101, 8801)

_VENDORS = {"10DE": "nvidia", "8086": "intel", "1002": "amd"}


# --------------------------------------------------------------------------- layout

def home() -> Path:
    return config.data_dir() / "components" / "comfyui"


def source_dir() -> Path:
    return home() / "ComfyUI"


def env_dir(backend: str) -> Path:
    return home() / "envs" / backend


def env_python(backend: str) -> Path:
    sub = "Scripts/python.exe" if config.IS_WINDOWS else "bin/python"
    return env_dir(backend) / sub


def _ready_marker(backend: str) -> Path:
    """Written only once an environment has computed something on its device.

    An environment that exists is not one that works: torch can install, import, report
    the device as available and then fail on the first kernel when its bundled runtime
    does not match the driver. The marker is the difference between those.
    """
    return env_dir(backend) / ".trusta-ready.json"


def source_installed() -> bool:
    return (source_dir() / "main.py").is_file()


def source_version() -> str:
    with contextlib.suppress(OSError):
        return (source_dir() / ".trusta-version").read_text(encoding="utf-8").strip()
    return ""


def env_ready(backend: str) -> bool:
    return _ready_marker(backend).is_file() and env_python(backend).is_file()


def installed_backends() -> list[str]:
    return [b for b in BACKENDS if env_ready(b)]


def detect() -> str | None:
    """Installed means there is something that can be started: source and one env."""
    if source_installed() and installed_backends():
        return str(source_dir())
    return None


# -------------------------------------------------------------------------- devices

@dataclass(frozen=True)
class Device:
    id: str             # "cuda:0", "xpu:0", "cpu", "amd:0"
    backend: str        # a BACKENDS key, or "" when nothing here can run on it
    vendor: str
    name: str
    index: int          # within its backend, as ComfyUI will be told
    memory_mb: int | None
    driver: str
    usable: bool
    note: str           # why it cannot be used, or what to know before using it


def _run(command: list[str], timeout: float = 15) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                errors="replace", creationflags=config.subprocess_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _nvidia_devices() -> list[Device]:
    """nvidia-smi, because its numbering is the one ComfyUI will be given.

    CUDA numbers devices fastest-first by default and nvidia-smi numbers them by PCI
    bus. Launches set CUDA_DEVICE_ORDER=PCI_BUS_ID so the two agree, which makes the
    index here the right --cuda-device.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    out = _run([smi, "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    devices = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        index, name, memory, driver = int(parts[0]), parts[1], parts[2], parts[3]
        try:
            major = int(driver.split(".")[0])
        except ValueError:
            major = 0
        usable = major >= NVIDIA_MIN_DRIVER
        devices.append(Device(
            id=f"cuda:{index}", backend="cuda", vendor="nvidia", name=name, index=index,
            memory_mb=int(memory) if memory.isdigit() else None, driver=driver,
            usable=usable,
            note="" if usable else f"Driver {driver} is too old; this needs "
                                   f"{NVIDIA_MIN_DRIVER} or newer.",
        ))
    return devices


def _windows_adapters() -> list[dict]:
    script = ("[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
              "Get-CimInstance Win32_VideoController | "
              "Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress")
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    if not out or not out.strip():
        return []
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _linux_adapters() -> list[dict]:
    out = _run(["lspci", "-nn"]) or ""
    adapters = []
    for line in out.splitlines():
        if not re.search(r"vga|3d|display", line, re.IGNORECASE):
            continue
        match = re.search(r"\[([0-9a-f]{4}):[0-9a-f]{4}\]", line, re.IGNORECASE)
        if not match:
            continue
        name = line.split(": ", 1)[-1]
        name = re.sub(r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\].*$", "", name, flags=re.IGNORECASE)
        adapters.append({"Name": name, "PNPDeviceID": f"VEN_{match.group(1)}",
                         "DriverVersion": ""})
    return adapters


def _driver_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in text.split("."))
    except ValueError:
        return ()


def _intel_note(driver: str) -> tuple[bool, str]:
    if not config.IS_WINDOWS or not driver:
        return True, ""
    found = _driver_tuple(driver)
    if found and found < INTEL_MIN_DRIVER:
        floor = ".".join(str(p) for p in INTEL_MIN_DRIVER)
        return False, f"Driver {driver} is too old; this needs {floor} or newer."
    return True, ""


def detect_devices() -> list[Device]:
    """Every card something here could run on, then the CPU.

    NVIDIA comes from nvidia-smi, everything else from the system's adapter list. An
    NVIDIA card the adapter list has and nvidia-smi does not is still listed -- its
    driver is what is missing, and saying so beats leaving the card out.
    """
    devices = _nvidia_devices()
    adapters = _windows_adapters() if config.IS_WINDOWS else _linux_adapters()
    counts = {"intel": 0, "amd": 0, "nvidia": 0}
    for adapter in adapters:
        match = re.search(r"VEN_([0-9A-F]{4})", str(adapter.get("PNPDeviceID") or ""),
                          re.IGNORECASE)
        vendor = _VENDORS.get(match.group(1).upper()) if match else None
        if vendor is None:
            continue  # Microsoft Basic Display, remote-desktop adapters, the rest
        name = str(adapter.get("Name") or vendor).strip()
        driver = str(adapter.get("DriverVersion") or "").strip()
        index = counts[vendor]
        counts[vendor] += 1
        if vendor == "nvidia":
            if devices and any(d.vendor == "nvidia" for d in devices):
                continue
            devices.append(Device(id=f"nvidia:{index}", backend="", vendor="nvidia",
                                  name=name, index=index, memory_mb=None, driver=driver,
                                  usable=False,
                                  note="nvidia-smi was not found, so the NVIDIA driver is "
                                       "missing or broken. Install the driver from NVIDIA."))
        elif vendor == "intel":
            usable, note = _intel_note(driver)
            devices.append(Device(id=f"xpu:{index}", backend="xpu", vendor="intel",
                                  name=name, index=index, memory_mb=None, driver=driver,
                                  usable=usable, note=note))
        else:
            devices.append(Device(id=f"amd:{index}", backend="", vendor="amd", name=name,
                                  index=index, memory_mb=None, driver=driver, usable=False,
                                  note="AMD cards are not supported by this build yet."))
    devices.append(Device(id="cpu", backend="cpu", vendor="cpu", name="CPU", index=0,
                          memory_mb=None, driver="", usable=True,
                          note="Very slow — minutes per image, and video is out of reach. "
                               "For testing only."))
    return devices


_cache: tuple[float, list[Device]] | None = None
_cache_lock = threading.Lock()
_CACHE_SECONDS = 60.0


def devices(refresh: bool = False) -> list[Device]:
    """Cached for a minute. The tab redraws often and PowerShell takes a second."""
    global _cache
    with _cache_lock:
        if not refresh and _cache and time.monotonic() - _cache[0] < _CACHE_SECONDS:
            return _cache[1]
    found = detect_devices()
    with _cache_lock:
        _cache = (time.monotonic(), found)
    return found


def recommended_device(found: list[Device]) -> Device:
    """NVIDIA, then Intel, then the CPU: the order the card labels as recommended."""
    for backend in ("cuda", "xpu", "cpu"):
        for device in found:
            if device.backend == backend and device.usable:
                return device
    return found[-1]


def default_device(found: list[Device]) -> Device:
    """The remembered choice if it is still here, else the recommended one."""
    remembered = state.get("comfyui_device")
    for device in found:
        if device.id == remembered and device.usable:
            return device
    return recommended_device(found)


def find_device(device_id: str) -> Device | None:
    for device in devices():
        if device.id == device_id:
            return device
    return None


# -------------------------------------------------------------------------- install

def _uv() -> str | None:
    shipped = config.uv_binary()
    return str(shipped) if shipped else shutil.which("uv")


def _latest_tag(out) -> str:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/latest",
        headers={"User-Agent": "AI-Scaler-Toolkit", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            tag = json.loads(response.read().decode("utf-8")).get("tag_name")
    except (OSError, ValueError) as exc:
        out.note(f"Could not read ComfyUI's latest release ({exc}); using {FALLBACK_TAG}.")
        return FALLBACK_TAG
    return str(tag) if tag else FALLBACK_TAG


def ensure_source(out) -> bool:
    """Download and unpack the ComfyUI source once. Every environment shares it."""
    from .components import _download

    if source_installed():
        version = source_version()
        out.note(f"ComfyUI {version} is already here." if version
                 else "ComfyUI is already here.")
        return True
    tag = _latest_tag(out)
    archive = Path(tempfile.gettempdir()) / f"ComfyUI-{tag}.zip"
    if not _download(ARCHIVE.format(repo=REPO, tag=tag), archive, out,
                     what=f"ComfyUI {tag}"):
        return False

    # Unpacked beside its destination and moved into place whole, so an interrupted
    # unpack cannot leave a tree that detection reads as installed.
    work = home() / "ComfyUI.partial"
    shutil.rmtree(work, ignore_errors=True)
    try:
        work.mkdir(parents=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(work)
        roots = [p for p in work.iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "main.py").is_file():
            out.note("The archive did not contain ComfyUI.")
            return False
        (roots[0] / ".trusta-version").write_text(tag + "\n", encoding="utf-8")
        shutil.rmtree(source_dir(), ignore_errors=True)
        roots[0].replace(source_dir())
    except (OSError, zipfile.BadZipFile) as exc:
        out.note(f"Could not unpack ComfyUI: {exc}")
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
        archive.unlink(missing_ok=True)
    out.note(f"ComfyUI {tag} unpacked to {source_dir()}.")
    return True


_VERIFY = r"""
import sys, torch
kind = sys.argv[1]
print("torch", torch.__version__)
if kind == "cuda":
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to this torch build.")
    device = torch.device("cuda", 0)
    print("device", torch.cuda.get_device_name(0))
elif kind == "xpu":
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        sys.exit("No Intel GPU is available to this torch build.")
    device = torch.device("xpu", 0)
    print("device", torch.xpu.get_device_name(0))
else:
    device = torch.device("cpu")
x = torch.randn(1024, 1024, device=device)
value = float((x @ x).sum().item())
print("ok", value == value)
"""


def _verify(backend: str, out) -> bool:
    """Compute on the device, not just ask whether it is there.

    Under a timeout, because an Intel runtime that does not match its driver has been
    seen to hang compiling the first kernel rather than fail.
    """
    from .components import _decode

    out.note(f"Checking that {BACKENDS[backend].label} can compute on this machine.")
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    try:
        result = subprocess.run([str(env_python(backend)), "-c", _VERIFY, backend],
                                capture_output=True, timeout=300, env=env,
                                creationflags=config.subprocess_flags())
    except subprocess.TimeoutExpired:
        out.note("The check did not finish in five minutes. The driver and this torch "
                 "build probably do not match; updating the graphics driver usually "
                 "fixes it.")
        return False
    except OSError as exc:
        out.note(f"Could not run the check: {exc}")
        return False
    for line in _decode(result.stdout + result.stderr).splitlines():
        if line.strip():
            out.line(line.rstrip())
    if result.returncode != 0:
        out.note("The check failed, so this environment is not marked usable.")
        return False
    _ready_marker(backend).write_text(json.dumps({
        "torch": TORCH_VERSIONS["torch"] + "+" + BACKENDS[backend].local,
        "verified": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }) + "\n", encoding="utf-8")
    return True


def ensure_env(backend: str, out) -> bool:
    """Create and verify one backend's environment, unless it already works."""
    from .components import _stream

    if env_ready(backend):
        out.note(f"The {BACKENDS[backend].label} environment is already set up.")
        return True
    uv = _uv()
    if not uv:
        out.note("uv was not found, so no Python environment can be created.")
        return False
    spec = BACKENDS[backend]
    target = env_dir(backend)
    python = env_python(backend)

    out.note(f"Creating the {spec.label} environment (Python {PYTHON}).")
    if not python.is_file():
        code = _stream([uv, "venv", "--python", PYTHON, "--seed", str(target)], out,
                       f"comfyui-{backend}-venv.log")
        if code != 0 or not python.is_file():
            out.note("Could not create the environment.")
            return False

    pins = [f"{name}=={version}+{spec.local}" for name, version in TORCH_VERSIONS.items()]
    out.note(f"Installing PyTorch {TORCH_VERSIONS['torch']} for {spec.label}. About "
             f"{spec.download_gb:g} GB the first time; less if this machine already "
             "has it cached.")
    # --extra-index-url, not --index-url: uv looks at extra indexes first, so torch and
    # friends come from PyTorch's index and everything else from PyPI.
    code = _stream([uv, "pip", "install", "--python", str(python),
                    "--extra-index-url", spec.index, *pins], out,
                   f"comfyui-{backend}-torch.log")
    if code != 0:
        out.note("PyTorch did not install.")
        return False

    out.note("Installing ComfyUI's own requirements.")
    # torch is already satisfied, and uv pip install does not upgrade what is
    # satisfied, so the unpinned "torch" in requirements.txt leaves this build alone.
    code = _stream([uv, "pip", "install", "--python", str(python),
                    "--extra-index-url", spec.index,
                    "-r", str(source_dir() / "requirements.txt")], out,
                   f"comfyui-{backend}-requirements.log")
    if code != 0:
        out.note("ComfyUI's requirements did not install.")
        return False
    return _verify(backend, out)


def install(device: Device, out) -> bool:
    return ensure_source(out) and ensure_env(device.backend, out)


# --------------------------------------------------------------------------- runtime

_process: subprocess.Popen | None = None
_running_device: Device | None = None
_proc_lock = threading.Lock()


def url() -> str:
    return f"http://{HOST}:{PORT}"


def _get(path: str, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(url() + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _ours_alive() -> bool:
    return _process is not None and _process.poll() is None


def log_path() -> Path:
    return config.log_dir() / "comfyui.log"


def _log_tail(lines: int = 12) -> str:
    with contextlib.suppress(OSError):
        text = log_path().read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.strip().splitlines()[-lines:])
    return ""


def status() -> dict:
    """What is on the port, and whether it is the one this application started."""
    ours = _ours_alive()
    stats = _get("/system_stats")
    device = _running_device if ours else None
    if ours and stats is None:
        phase = "starting"
    elif ours:
        phase = "running"
    elif stats is not None:
        phase = "foreign"   # something else answers as ComfyUI: Desktop, a portable copy
    else:
        phase = "stopped"
    return {
        "phase": phase,
        "url": url(),
        "device_id": device.id if device else "",
        "device_name": device.name if device else "",
        "log": str(log_path()),
    }


def busy() -> bool:
    """True while a prompt is running or queued."""
    queue = _get("/queue")
    if not queue:
        return False
    return bool(queue.get("queue_running") or queue.get("queue_pending"))


def _device_args(device: Device) -> tuple[list[str], dict[str, str]]:
    """What puts ComfyUI on this device: its arguments, and the environment for them.

    Separate from the host and port because it is also what selected.json hands the
    skills, which launch ComfyUI themselves on a port and with a reserve of their own.
    """
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if device.backend == "cuda":
        return ["--cuda-device", str(device.index)], env
    if device.backend == "xpu":
        # ComfyUI's --cuda-device does not reach Level Zero; this does. Only set when
        # it narrows something, so a single-GPU machine runs with stock settings.
        if sum(1 for d in devices() if d.backend == "xpu") > 1:
            env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{device.index}"
        return [], env
    return ["--cpu"], env


def _launch_command(device: Device) -> tuple[list[str], dict[str, str]]:
    args, device_env = _device_args(device)
    command = [str(env_python(device.backend)), "main.py",
               "--listen", HOST, "--port", str(PORT), "--disable-auto-launch", *args]
    return command, {**os.environ, "PYTHONUNBUFFERED": "1", **device_env}


def selection_path() -> Path:
    return home() / "selected.json"


def write_selection(device: Device) -> None:
    """Tell the image and video skills which device was picked, and how to use it.

    They launch ComfyUI themselves -- the --reserve-vram that protects the hosted LLM
    only exists if they do -- so they need what this card knows: which environment's
    Python, which arguments put it on the chosen card, and which card to measure.
    Without it they guess, and guessed wrong: they started ComfyUI on the NVIDIA card
    after Intel had been picked here, and measured the NVIDIA card while ComfyUI ran
    on Intel.

    Written beside the install, replaced whole so a reader never sees half of it.
    """
    args, env = _device_args(device)
    payload = {
        "schema": 1,
        "comfy_home": str(home()),
        "comfy_root": str(source_dir()),
        "python": str(env_python(device.backend)),
        "device": device.id,
        "backend": device.backend,
        "vendor": device.vendor,
        "index": device.index,
        "name": device.name,
        "args": args,
        "env": env,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    target = selection_path()
    partial = target.with_suffix(".json.partial")
    try:
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        partial.replace(target)
    except OSError:
        log.exception("could not write %s", target)


def stop(timeout: float = 15) -> None:
    global _process, _running_device
    with _proc_lock:
        process, _process, _running_device = _process, None, None
    if process is None or process.poll() is not None:
        return
    log.info("stopping ComfyUI (pid %s)", process.pid)
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def launch(device: Device) -> tuple[bool, str]:
    """Start ComfyUI on this device, replacing a copy of ours on another one.

    Returns (started, error). Started means the process stayed up for its first
    seconds; the page then watches status() until the port answers.
    """
    global _process, _running_device
    if not env_ready(device.backend) or not source_installed():
        return False, "That device's environment is not installed yet."
    if not _ours_alive() and _get("/system_stats") is not None:
        return False, (f"Another ComfyUI is already using {url()}. "
                       "Close it to start this one.")
    stop()

    command, env = _launch_command(device)
    config.log_dir().mkdir(parents=True, exist_ok=True)
    log.info("starting ComfyUI on %s: %s", device.id, " ".join(command))
    sink = log_path().open("w", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(command, cwd=source_dir(), env=env, stdout=sink,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=not config.IS_WINDOWS,
                                   creationflags=config.subprocess_flags())
    except OSError as exc:
        return False, f"Could not start ComfyUI: {exc}"
    finally:
        sink.close()
    try:
        code = process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with _proc_lock:
            _process, _running_device = process, device
        state.set("comfyui_device", device.id)
        write_selection(device)
        return True, ""
    tail = _log_tail()
    return False, f"ComfyUI exited at once (code {code})." + (f"\n{tail}" if tail else "")


def device_rows(found: list[Device]) -> list[dict]:
    ready = set(installed_backends())
    rows = []
    for device in found:
        row = asdict(device)
        row["installed"] = device.backend in ready
        row["download_gb"] = BACKENDS[device.backend].download_gb if device.backend else 0
        row["backend_label"] = BACKENDS[device.backend].label if device.backend else ""
        rows.append(row)
    return rows


def describe() -> dict:
    """Everything the card shows, in one call."""
    found = devices()
    return {
        "devices": device_rows(found),
        # What the card starts out selecting (the last choice), and what it labels as
        # recommended (the vendor order). Different questions: someone who picked the
        # Intel card still has an NVIDIA card that is the better one.
        "default": default_device(found).id,
        "recommended": recommended_device(found).id,
        "status": status(),
        "busy": busy() if _ours_alive() else False,
        "version": source_version(),
    }


if __name__ == "__main__":  # a quick look from a terminal: python -m launcher.comfyui
    json.dump(describe(), sys.stdout, indent=2, ensure_ascii=False)
    print()
