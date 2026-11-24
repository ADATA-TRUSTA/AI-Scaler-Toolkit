"""
Strata engine: a managed Strata server, reached over its OpenAI-compatible API.

Strata (github.com/Niko1221/Strata) serves a MoE model with the experts in DRAM and the
hot ones cached on the GPU. Its own ``serve/server.py`` wraps the engine binary in an
OpenAI-compatible HTTP server. The worker starts that server and waits for it here;
generations then go from the main process straight to it (async_server_client), as for
llama-server and vLLM.

The model is described by the run config Strata's setup writes (``strata-<model>.json``:
the engine executable, the pack, the GGUF shards, the vision encoder). The engine copies
it into STRATA_RUNTIME_DIR with the load request's overrides applied and starts the
server with that copy, so the user's file is never changed.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import IO, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import dotenv_values

from ...config_models import InferenceConfig
from ...server_args import flag_groups
from ...settings import (
    PROJECT_ROOT,
    STRATA_DATA_DIR,
    STRATA_ENGINE_BINARY,
    STRATA_HEALTH_TIMEOUT,
    STRATA_HOME,
    STRATA_INSTALL_DIR,
    STRATA_PORT,
    STRATA_PYTHON,
    STRATA_RUNTIME_DIR,
    STRATA_VISION_BINARY,
    configure_logging,
)
from ...utils.process_identity import process_identity
from ..strata_models import STRATA_GGUF_MODELS, strata_model_choice
from .base_engine import BaseEngine

logger = configure_logging(__name__)

STRATA_HOST = "127.0.0.1"
# Loading a 50 GiB expert arena through the page cache pinned it while the kernel kept
# reclaiming, and the session's memory pressure stayed above systemd-oomd's 50% long
# enough for it to kill the desktop session (measured 69.8%). Read unbuffered and without
# transparent huge pages (whose fault-time compaction alone drove 44.8%), the same load
# peaked at 0.2% and read faster (9.1-10.0 GiB/s against 6.4). Set only when the caller
# did not choose: either variable in the environment wins.
_LOAD_ENV_DEFAULTS = {"STRATA_UNBUFFERED_LOAD": "1", "STRATA_NO_ARENA_THP": "1"}
# Windows counts page-locked memory against the commit limit (RAM + page file). Strata 0.2.7
# may lock the PLE table in RAM by itself (26.8 GiB for Qwen3.8-Flash-Next), and the sleep
# backup (about 4.6 GB, also page-locked) then no longer fits: every sleep answered 409
# ("backup needs 4596 MiB, only 512 MiB available"). With the table left where the run config
# puts it, sleep and wake worked. Like the two above, the environment wins.
_WINDOWS_LOAD_ENV_DEFAULTS = {"STRATA_PLE_AUTO_RAM": "0"}
# The Intel engine's CPU expert pool can read a layer's routing before the GPU has finished
# publishing it. Since strata-sleep 0.2.11-sleep.2 (patches 0014, 0015) that cannot crash the
# engine any more, but on Windows (no sysfs driver name for the long spin bound) the CPU can
# still compute from a stale payload: on an Arc B390 the replies drift unless the engine
# checks the doorbell. Same rule as the others: the environment wins.
_WINDOWS_INTEL_LOAD_ENV_DEFAULTS = {"STRATA_DOORBELL_CHECK": "1"}
_LOG_TAIL_LINES = 40
# Where Linux lists the graphics devices; a module constant so a test can point it elsewhere.
_DRM_DEVICES = Path("/sys/class/drm")


def strata_settings_file() -> Path:
    """
    The settings file Strata's setup remembers the data folder in.

    Named rather than left to default, because the default is the one a command-line
    Strata the user installed themselves uses (%APPDATA%/Strata/settings.json). Sharing
    it meant a managed install adopted their data folder and added itself to their list
    of installs -- both observed on a machine that had one. Strata reads this from
    STRATA_SETTINGS_FILE and then neither touches the other.
    """
    return Path(STRATA_RUNTIME_DIR).expanduser() / "strata-settings.json"


def _with_strata_settings(env: dict[str, str]) -> dict[str, str]:
    """``env`` pointed at this backend's own Strata settings, the caller's choice winning."""
    env.setdefault("STRATA_SETTINGS_FILE", str(strata_settings_file()))
    return env


def strata_setup_problem() -> str | None:
    """Why engine=strata cannot start on this host, or None when it can."""
    if not STRATA_HOME.strip():
        return (
            "Strata is not set up: STRATA_HOME is empty. Point it at a Strata install "
            "prepared by Strata's own setup (the directory holding serve/server.py), and "
            "STRATA_PYTHON at its interpreter."
        )
    server = Path(STRATA_HOME).expanduser() / "serve" / "server.py"
    if not server.is_file():
        return f"STRATA_HOME has no serve/server.py: {server}"
    python = strata_python()
    if not Path(python).is_file():
        return f"The Strata interpreter does not exist: {python} (STRATA_PYTHON)"
    # The run config names the engine Strata's setup installed; this one is started instead,
    # and engine_sleeps() decides whether it is asked to sleep.
    if not STRATA_ENGINE_BINARY.strip():
        return (
            "STRATA_ENGINE_BINARY is empty: point it at the Strata engine of the installed "
            "package (engine/strata[.exe])."
        )
    if not Path(STRATA_ENGINE_BINARY).expanduser().is_file():
        return f"STRATA_ENGINE_BINARY does not exist: {STRATA_ENGINE_BINARY}"
    return None


def strata_installable() -> bool:
    """Whether a load can install Strata itself: a GPU it has a package for, and the script."""
    return (
        gpu_backend() is not None and _install_script().is_file() and strata_package() is not None
    )


def _install_script() -> Path:
    name = "windows/install_strata.ps1" if os.name == "nt" else "linux/install_strata.sh"
    return PROJECT_ROOT / "scripts" / name


def _intel_gpu_windows() -> bool:
    """
    An Intel display adapter on Windows, from the display class registry.

    Windows has no /sys/class/drm. Strata's own setup reads Win32_VideoController and this
    same registry key (its patch 0009, intel_gpus_windows); this reads only what the question
    needs, through winreg, so it costs no subprocess on a path the service calls often.
    """
    # The platform check, not the ImportError, is what lets pyright on Linux type-check winreg.
    if sys.platform != "win32":
        return False
    import winreg

    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as cls:
            for i in range(winreg.QueryInfoKey(cls)[0]):
                try:
                    with winreg.OpenKey(cls, winreg.EnumKey(cls, i)) as sub:
                        value = str(winreg.QueryValueEx(sub, "MatchingDeviceId")[0])
                except OSError:
                    continue
                if "ven_8086" in value.lower():
                    return True
    except OSError:
        return False
    return False


def gpu_backend() -> str | None:
    """
    Which engine this host needs: "cuda" for an NVIDIA GPU, "sycl" for an Intel one, None
    for neither.

    NVIDIA first: on a laptop with both, the discrete card is the one Strata is wanted for.
    Intel is read from the DRM devices (vendor 0x8086) rather than a tool, because nothing
    like nvidia-smi is installed with the driver; on Windows, which has no DRM devices, from
    the display class registry. Upstream has no Intel build for Windows, but strata-sleep
    makes one (its patches 0007, 0009 and 0010), so both platforms can be Intel.
    """
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    if os.name == "nt":
        return "sycl" if _intel_gpu_windows() else None
    try:
        for vendor in sorted(_DRM_DEVICES.glob("card*/device/vendor")):
            if vendor.read_text().strip().lower() == "0x8086":
                return "sycl"
    except OSError:
        pass
    return None


def strata_package() -> Path | None:
    """
    The strata-sleep package in vendor/ for this host's GPU, or None.

    The CUDA archives are named strata-sleep-<version>-<platform>.<ext>; another backend's
    carries its name last (strata-<version>-linux-x86_64-sycl.tar.gz: no -sleep, it has no
    deep sleep; before 0.2.11-sleep.1 it was strata-sleep-<version>-...-sycl, which the
    strata-* pattern still finds), so the two never match each other's glob and a payload
    may hold both.
    """
    backend = gpu_backend()
    if backend is None:
        return None
    platform = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    ext = "zip" if os.name == "nt" else "tar.gz"
    pattern = (
        f"strata-sleep-*-{platform}.{ext}"
        if backend == "cuda"
        else f"strata-*-{platform}-{backend}.{ext}"
    )
    found = list((PROJECT_ROOT / "vendor").glob(pattern))
    # The newest version, not the last name: strata-sleep-0.2.7-... sorts after
    # strata-sleep-0.2.11-..., and an upgrade used to leave the older package beside the new one.
    return max(found, key=_package_sort_key) if found else None


_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-sleep\.(\d+))?")
# strata-sleep-<version>-<platform>.<ext> (CUDA) and strata-<version>-<platform>-sycl.<ext>
# (Intel; before 0.2.11-sleep.1 it was strata-sleep-<version>-...-sycl).
_PACKAGE_VERSION_RE = re.compile(r"^strata(?:-sleep)?-(\d+\.\d+\.\d+(?:-sleep\.\d+)?)-")


def _version_key(text: str) -> tuple[int, int, int, int] | None:
    """(major, minor, patch, sleep) of '0.2.11-sleep.2', compared as numbers, or None."""
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    major, minor, patch, sleep = match.groups()
    return int(major), int(minor), int(patch), int(sleep or 0)


def strata_package_version(package: Path | None = None) -> str | None:
    """The version in the name of the package vendor/ carries for this host, or None."""
    package = package if package is not None else strata_package()
    if package is None:
        return None
    match = _PACKAGE_VERSION_RE.match(package.name)
    return match.group(1) if match else None


def _package_sort_key(package: Path) -> tuple[tuple[int, int, int, int], str]:
    version = strata_package_version(package)
    return (_version_key(version) if version else None) or (0, 0, 0, 0), package.name


def strata_installed_version() -> str | None:
    """The first line of VERSION in STRATA_HOME, or None when there is none to read."""
    if not STRATA_HOME.strip():
        return None
    try:
        text = (Path(STRATA_HOME).expanduser() / "VERSION").read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    return (lines[0].strip() or None) if lines else None


def _install_dir() -> Path:
    """Where install_strata.* puts Strata when this backend asks it to."""
    if STRATA_INSTALL_DIR.strip():
        return Path(STRATA_INSTALL_DIR).expanduser()
    return PROJECT_ROOT.parent / "strata"


def _resolved(path: str | Path) -> str:
    # Lexical, not realpath: a venv's python is a link to the interpreter uv manages
    # (~/.local/share/uv/python/... on Linux), which realpath would put outside the install.
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _is_inside(path: str, root: Path) -> bool:
    inner, outer = _resolved(path), _resolved(root)
    return inner == outer or inner.startswith(outer + os.sep)


def strata_update_needed() -> str | None:
    """
    Why the installed Strata should be replaced by the package vendor/ carries, or None.

    An upgrade replaces the backend and the package beside it but not a Strata that an
    earlier build already installed: STRATA_HOME names a working install, so a load only ran
    that install's own setup and the engine fixes the new build ships never reached the
    machine. install_strata.* copies over an earlier install (its venv and run configs stay),
    so the update is that script run again.

    Only an install this backend made is touched: STRATA_HOME is the directory it installs
    into and the interpreter and engine lie inside it. A hand-made install, or an engine
    pinned somewhere else, is the user's own and is left alone, as is one that is newer, the
    same or whose version cannot be read.
    """
    if not strata_installable():
        return None
    install = _install_dir()
    if not STRATA_HOME.strip() or _resolved(STRATA_HOME) != _resolved(install):
        return None
    for pinned in (STRATA_PYTHON, STRATA_ENGINE_BINARY):
        if not pinned.strip() or not _is_inside(pinned, install):
            return None
    installed, shipped = strata_installed_version(), strata_package_version()
    if not installed or not shipped:
        return None
    have, want = _version_key(installed), _version_key(shipped)
    if have is None or want is None or have >= want:
        return None
    return f"the installed Strata {installed} is older than the {shipped} this build ships"


def engine_build_info() -> dict[str, Any]:
    """
    BUILD.json beside STRATA_ENGINE_BINARY: what the engine was built with, or {}.

    scripts/build-*.sh in strata-sleep write it, and strata_sleep.py copies it into the
    package. A hand-made install may have none, so every reader of it has a fallback.

    The load runs in a worker process, and the install a first load makes is recorded in .env
    after the API process started, so this process may still hold the empty value it read at
    startup. Asking engine_sleeps() then said "no deep sleep" for an engine that has it, until
    the service was restarted (F-0096). So an empty value is looked up in .env first, which is
    where install_strata records it.
    """
    if not STRATA_ENGINE_BINARY.strip():
        _adopt_recorded_install()
    if not STRATA_ENGINE_BINARY.strip():
        return {}
    path = Path(STRATA_ENGINE_BINARY).expanduser().parent / "BUILD.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def engine_backend() -> str:
    """The engine's backend: BUILD.json's, else "cuda" (every package before Intel was one)."""
    backend = engine_build_info().get("backend")
    return backend if isinstance(backend, str) and backend else "cuda"


def engine_sleeps() -> bool:
    """
    Whether this engine has deep sleep (--sleep-mode).

    BUILD.json says so outright. Without it, the flag's name is in the binary of every build
    that parses it and in no other, which is how an engine from before BUILD.json carried the
    field is recognised. The Intel (SYCL) engine has none: deep sleep is CUDA driver virtual
    memory, and Level Zero has no equivalent, so it is started without the flag and never
    asked to sleep.
    """
    sleeps = engine_build_info().get("sleep_mode")
    if isinstance(sleeps, bool):
        return sleeps
    try:
        return b"--sleep-mode" in Path(STRATA_ENGINE_BINARY).expanduser().read_bytes()
    except OSError:
        return False


def strata_python() -> str:
    """STRATA_PYTHON, else the checkout's own .venv interpreter."""
    if STRATA_PYTHON.strip():
        return str(Path(STRATA_PYTHON).expanduser())
    venv = Path(STRATA_HOME).expanduser() / ".venv"
    return str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def _set_flag(args: list[str], flag: str, value: str) -> list[str]:
    """``args`` with ``flag value`` replacing every earlier occurrence (or appended)."""
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        if args[i].startswith(flag + "="):
            i += 1
            continue
        out.append(args[i])
        i += 1
    return [*out, flag, value]


# Flags that choose between the same thing two ways: setting one drops the other, since
# the engine would otherwise apply both (a RAM budget and the resident low-RAM mode).
_EXCLUSIVE_FLAGS: tuple[frozenset[str], ...] = (
    frozenset({"--resident-budget-gib", "--resident-experts"}),
)


def _apply_extra_args(args: list[str], extra: list[str]) -> list[str]:
    """
    ``args`` with each flag of ``extra`` replacing the same flag and its values, or added.

    A run config's args are a flag followed by its values, some flags repeated on purpose
    (--native-dense-gguf once per shard): a flag given here replaces all of its
    occurrences, so a repeated one is given here in full.
    """
    groups = flag_groups(extra)
    replaced = {flag for flag, _ in groups}
    for flag in list(replaced):
        for pair in _EXCLUSIVE_FLAGS:
            if flag in pair:
                replaced |= pair
    kept = [tokens for flag, tokens in flag_groups(args) if flag not in replaced]
    return [token for tokens in kept + [tokens for _, tokens in groups] for token in tokens]


def vision_binary() -> Path | None:
    """
    Strata's image encoder: STRATA_VISION_BINARY, else the reader beside the engine.

    An Intel package carries two, as Strata's own strata_control/vision_reader.py expects:
    strata-vision-sycl runs the encoder on the card, and strata-vision is the CPU one that
    starts anywhere. The GPU one is taken when it is there. A CUDA package has one file for
    both (strata-vision, with --gpu), so nothing changes there.
    """
    if STRATA_VISION_BINARY.strip():
        return Path(STRATA_VISION_BINARY).expanduser()
    if not STRATA_ENGINE_BINARY.strip():
        return None
    folder = Path(STRATA_ENGINE_BINARY).expanduser().parent
    suffix = ".exe" if os.name == "nt" else ""
    names = [f"strata-vision-{engine_backend()}{suffix}", f"strata-vision{suffix}"]
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def find_mmproj(gguf_path: str) -> Path | None:
    """An mmproj GGUF beside the model or one folder up (BF16, then F16, first), as llama-server finds it."""
    model = Path(gguf_path).expanduser()
    for folder in dict.fromkeys([model.parent, model.parent.parent]):
        try:
            found = [
                f
                for f in folder.iterdir()
                if "mmproj" in f.name.lower() and f.name.lower().endswith(".gguf")
            ]
        except OSError:
            continue
        if found:
            return min(
                found,
                key=lambda f: ("bf16" not in f.name.lower(), "f16" not in f.name.lower(), f.name),
            )
    return None


def _apply_vision(run: dict[str, Any], config: InferenceConfig, args: list[str]) -> None:
    """
    Image input as the load asks: strata_vision false drops the run config's encoder,
    true keeps it or adds one; omitted leaves the run config as it is.

    Strata's setup is run with --vision no (it would otherwise compile an encoder, and could
    replace the engine), so a model it prepared has none. The package carries the encoder
    ready-made (vision_binary); it is model-agnostic and needs only its exe, the mmproj and
    the model's GGUF.
    """
    vision = run.get("vision")
    wanted = config.strata_vision if config.strata_vision is not None else isinstance(vision, dict)
    if not wanted:
        run.pop("vision", None)
        run["args"] = [a for a in args if a != "--vision"]  # the engine would wait for images
        return
    vision = dict(vision) if isinstance(vision, dict) else {}
    if config.strata_mmproj:
        vision["mmproj"] = str(Path(config.strata_mmproj).expanduser())
    if not config.strata_vision:  # the run config's own encoder, as it wrote it
        run["vision"] = vision
        return
    if not vision.get("model"):
        native = args[args.index("--native") + 1] if "--native" in args[:-1] else None
        vision["model"] = native or config.model_path
    if not vision.get("mmproj"):
        found = find_mmproj(str(vision["model"])) if vision.get("model") else None
        if found is None:
            raise RuntimeError(
                "strata_vision needs the model's image projector: no mmproj .gguf beside "
                f"{vision.get('model')} or one folder up; set strata_mmproj"
            )
        vision["mmproj"] = str(found)
    if not vision.get("exe"):
        exe = vision_binary()
        if exe is None:
            raise RuntimeError(
                "strata_vision needs Strata's image encoder (strata-vision): set "
                "STRATA_VISION_BINARY, or put it beside STRATA_ENGINE_BINARY"
            )
        vision["exe"] = str(exe)
    run["vision"] = vision


def _same_file(a: str, b: str) -> bool:
    try:
        # resolve(): a Hugging Face snapshot path and the blob it links to are the same file
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return False


_SETUP_TAIL_LINES = 40


def _adopt_recorded_install() -> None:
    """
    Take the paths install_strata just recorded in .env into this process.

    settings.py read .env when the process started, before the install wrote to it.
    """
    recorded = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("STRATA_HOME", "STRATA_PYTHON", "STRATA_ENGINE_BINARY"):
        value = (recorded.get(key) or "").strip()
        if value:
            os.environ[key] = value
            globals()[key] = value


def _run_setup_step(cmd: list[str], cwd: Path | None, env: dict[str, str]) -> None:
    """Run an install or setup command, its output to the log; raise with its tail on failure."""
    logger.info("[Strata] %s", " ".join(cmd))
    tail: deque[str] = deque(maxlen=_SETUP_TAIL_LINES)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for raw in process.stdout or ():
        line = raw.rstrip()
        if line:
            tail.append(line)
            logger.info("[Strata setup] %s", line)
    if process.wait() != 0:
        raise RuntimeError(
            f"Strata's setup failed (exit {process.returncode}):\n" + "\n".join(tail)
        )


def prepare_model(gguf_path: str) -> Path:
    """
    Install Strata if this host has none, run its setup for ``gguf_path``, and return the
    run config it wrote.

    The model is a GGUF the model registry already has (downloaded like any llama.cpp
    model); setup takes it in place with --gguf-dir and only adds what Strata derives from
    it (the expert pack, the draft layer) and the run config, sized for this machine. A
    first load pays for this once: minutes, not the download.
    """
    choice = strata_model_choice(gguf_path)
    if choice is None:
        raise RuntimeError(
            f"Strata cannot run {Path(gguf_path).name}: its setup takes only these GGUF files "
            f"(the first of each): {', '.join(sorted(STRATA_GGUF_MODELS))}"
        )
    family, model = choice
    gguf_dir = str(Path(gguf_path).expanduser().parent)
    env = _with_strata_settings(os.environ.copy())
    env["PYTHONIOENCODING"] = "utf-8"
    # --set-default-data-dir records the folder without moving anything into it: the
    # expert pack and the draft layer are made where they are wanted, and a data folder
    # someone already has is left alone. Omitted when unset, so Strata's own default
    # (Strata-data beside the install) still applies to a hand-made install.
    data_dir_args = (
        ["--data-dir", str(Path(STRATA_DATA_DIR).expanduser()), "--set-default-data-dir"]
        if STRATA_DATA_DIR.strip()
        else []
    )
    update = strata_update_needed()
    if update:
        logger.info("[Strata] %s: updating it in place from the package in vendor/", update)
    if strata_setup_problem() is None and update is None:
        _run_setup_step(
            [
                strata_python(),
                "setup.py",
                "--yes",
                "--no-start",
                "--vision",
                "no",
                "--family",
                family,
                "--model",
                model,
                "--gguf-dir",
                gguf_dir,
                *data_dir_args,
            ],
            Path(STRATA_HOME).expanduser(),
            env,
        )
    else:
        if not strata_installable():
            raise RuntimeError(strata_setup_problem() or "Strata cannot be installed here")
        # install_strata.* unpacks the package, makes its venv, runs the same setup and
        # records STRATA_HOME / STRATA_PYTHON / STRATA_ENGINE_BINARY in .env.
        env["TRUSTA_STRATA_SETUP_ARGS"] = f"--family {family} --model {model}"
        env["TRUSTA_STRATA_GGUF_DIR"] = gguf_dir  # its own variable: a path may hold spaces
        if STRATA_DATA_DIR.strip():
            env["TRUSTA_STRATA_DATA_DIR"] = str(Path(STRATA_DATA_DIR).expanduser())  # likewise
        if STRATA_INSTALL_DIR.strip():
            env["TRUSTA_STRATA_DIR"] = STRATA_INSTALL_DIR
        # The script picks a package by itself with a glob that matches only a CUDA archive,
        # so on an Intel host it would find none (vendor/ has only the -sycl one) or take the
        # CUDA one beside it. Hand it the package strata_installable() just checked for. A
        # package someone named in the environment is theirs to choose, and is left alone.
        package = strata_package()
        if package is not None:
            env.setdefault("TRUSTA_STRATA_PACKAGE", str(package))
        script = str(_install_script())
        cmd = (
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script]
            if os.name == "nt"
            else ["bash", script]
        )
        _run_setup_step(cmd, None, env)
        _adopt_recorded_install()
        problem = strata_setup_problem()
        if problem:
            raise RuntimeError(f"Strata was installed but cannot start: {problem}")
    found = find_run_config(gguf_path)
    if found is None:
        raise RuntimeError(f"Strata's setup finished but wrote no run config for {gguf_path}")
    return found


def find_run_config(gguf_path: str) -> Path | None:
    """
    The run config in STRATA_HOME whose model is ``gguf_path``, or None.

    Strata's setup writes one strata-<model>.json per installed model beside serve/, naming
    the model's first GGUF shard in --native. Matching on it lets a load name the model the
    way the model registry lists it, without knowing the run config exists.
    """
    home = Path(STRATA_HOME).expanduser()
    for candidate in sorted(home.glob("strata-*.json")):
        try:
            run = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        args = run.get("args") if isinstance(run, dict) else None
        if not isinstance(args, list) or "--native" not in args[:-1]:
            continue
        if _same_file(str(args[args.index("--native") + 1]), gguf_path):
            return candidate
    return None


def free_vram_mib(gpu: int) -> int:
    """
    Free VRAM on ``gpu`` in MiB, read through NVML or nvidia-smi.

    Not through torch: that would open a CUDA context in this worker and hold a few
    hundred MiB of the very VRAM being measured for as long as the worker lives.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu))
            return int(info.free) // 2**20
        finally:
            pynvml.nvmlShutdown()
    except Exception as nvml_error:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            return int(out.strip().splitlines()[0])
        except Exception as e:
            raise RuntimeError(
                f"strata_vram_limit_gb needs the free VRAM of GPU {gpu}, and neither NVML "
                f"({nvml_error}) nor nvidia-smi ({e}) reported it; leave it unset to use the "
                "run config's --vram-reserve-mib"
            ) from e


def vram_reserve_for_limit(limit_gb: float, free_mib: int) -> int:
    """--vram-reserve-mib that holds the engine to ``limit_gb`` of what is free now."""
    # Strata fills what is free at start-up less the reserve, its vision encoder included:
    # an 8 GB limit measured 8.3 GB in use (engine 6.5 + vision 1.6), so the cap lands
    # within a few hundred MiB of what was asked.
    reserve = free_mib - int(limit_gb * 1024)
    if reserve < 0:
        logger.warning(
            "[Strata] strata_vram_limit_gb=%s but only %.1f GB of VRAM is free; using all of it",
            limit_gb,
            free_mib / 1024,
        )
    return max(reserve, 0)


def build_run_config(
    config: InferenceConfig,
    source: dict[str, Any],
    log_path: str,
    vram_reserve_mib: int | None = None,
) -> dict[str, Any]:
    """
    The run config the server is started with: the model's own, with this load's choices.

    Only the keys the load request owns change; everything else (the pack, the shards,
    the expert profile, the MTP head, KV format, sampling defaults) stays as Strata's setup
    wrote it, unless strata_extra_args or strata_vision sets it. ``vram_reserve_mib`` is
    strata_vram_limit_gb already turned into a reserve.
    """
    run = dict(source)
    args = [str(a) for a in run.get("args") or []]
    if config.strata_extra_args:
        args = _apply_extra_args(args, config.strata_extra_args)
    if config.strata_max_context is not None:
        args = _set_flag(args, "--max-context", str(config.strata_max_context))
    if vram_reserve_mib is not None:
        args = _set_flag(args, "--vram-reserve-mib", str(vram_reserve_mib))
    args = [a for a in args if a != "--sleep-mode"]  # the run config's sleep_mode key decides
    run["args"] = args
    _apply_vision(run, config, args)
    # On where the engine has it: it costs nothing until the first sleep, and freeing the GPU
    # for another program is what this engine is chosen for. The Intel engine has no deep
    # sleep (engine_sleeps), and asking an engine that cannot sleep to sleep stops its start.
    run["sleep_mode"] = engine_sleeps()
    run["exe"] = str(Path(STRATA_ENGINE_BINARY).expanduser())
    run["log"] = log_path
    run.pop("port", None)  # the server is told its port on the command line
    return run


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "model"


def _load_env_defaults(windows: bool, backend: str) -> dict[str, str]:
    """What the server's environment gets when the caller did not set it (see the constants)."""
    defaults = dict(_LOAD_ENV_DEFAULTS)
    if windows:
        defaults.update(_WINDOWS_LOAD_ENV_DEFAULTS)
        if backend == "sycl":
            defaults.update(_WINDOWS_INTEL_LOAD_ENV_DEFAULTS)
    return defaults


class StrataEngine(BaseEngine):
    """Starts Strata's server for the model and stops it again (worker side)."""

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ) -> None:
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.server_process: subprocess.Popen | None = None
        self.base_url = f"http://{STRATA_HOST}:{STRATA_PORT}"
        self.runtime_dir = Path(STRATA_RUNTIME_DIR)
        self.slot_dir = self.runtime_dir / "slots"
        self.server_log_path = self.runtime_dir / "strata-server.log"
        self._log_handle: IO[str] | None = None
        self._model_key = "model"

    # ---------------------- HTTP ----------------------
    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> tuple[int, dict[str, Any]]:
        """One call to the managed server -> (HTTP status, JSON body)."""
        data = json.dumps(payload or {}).encode("utf-8") if method.upper() == "POST" else None
        req = urllib_request.Request(  # noqa: S310 - the loopback server this engine started
            f"{self.base_url}{path}", data=data, method=method.upper()
        )
        req.add_header("Content-Type", "application/json")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - as above
                body = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            status = e.code
        try:
            return status, (json.loads(body) if body.strip() else {})
        except json.JSONDecodeError:
            return status, {"raw": body}

    def _health(self) -> dict[str, Any] | None:
        try:
            status, body = self._request("GET", "/health", timeout=3.0)
        except (urllib_error.URLError, OSError):
            return None
        return body if status == 200 else None

    # ---------------------- process ----------------------
    def _log_tail(self) -> str:
        try:
            with open(self.server_log_path, encoding="utf-8", errors="replace") as f:
                return "".join(deque(f, maxlen=_LOG_TAIL_LINES)).strip()
        except OSError:
            return ""

    def _start_server(self, run_config_path: Path) -> None:
        home = Path(STRATA_HOME).expanduser()
        # Ask whether anything is listening rather than trying to bind: a reload right after
        # an unload finds the last server's connections in TIME_WAIT, which fail a plain
        # bind() although the new server (bound with SO_REUSEADDR) starts fine on the port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((STRATA_HOST, STRATA_PORT)) == 0:
                raise RuntimeError(
                    f"STRATA_PORT {STRATA_PORT} is already in use on {STRATA_HOST}; stop what "
                    "holds it or set another STRATA_PORT"
                )
        cmd = [
            strata_python(),
            str(home / "serve" / "server.py"),
            "--engine",
            "strata",
            "--config",
            str(run_config_path),
            "--host",
            STRATA_HOST,
            "--port",
            str(STRATA_PORT),
            # an OpenAI client's max_tokens beyond the room left is shortened, as
            # llama-server does, instead of refused with a 400
            "--fit-max-tokens",
            "--slot-save-path",
            str(self.slot_dir),
        ]
        env = _with_strata_settings(os.environ.copy())
        defaults = _load_env_defaults(os.name == "nt", engine_backend())
        for key, value in defaults.items():
            env.setdefault(key, value)
        logger.info("[Strata] starting the server: %s", " ".join(cmd))
        self._log_handle = open(self.server_log_path, "a", encoding="utf-8")  # noqa: SIM115 - held for the process's life
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # its own process group: the server, the engine and the vision encoder end together
            kwargs["start_new_session"] = True
        self.server_process = subprocess.Popen(
            cmd,
            cwd=str(home),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
        # Its own session also means it outlives this worker when the worker is killed. The
        # API process stops whatever is left once the worker is gone (F-0086), and needs
        # the PID and start time to know it is still this server.
        create_time = process_identity(self.server_process.pid)
        if create_time is None:
            return  # already gone: nothing to leave behind
        self.status_queue.put(
            {
                "status": "server_started",
                "pid": self.server_process.pid,
                "create_time": create_time,
                "own_session": True,
            }
        )

    def _wait_ready(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            proc = self.server_process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"the Strata server exited with code {proc.returncode} while loading. "
                    f"Its log ({self.server_log_path}) ends with:\n{self._log_tail()}"
                )
            health = self._health()
            if health and health.get("loaded"):
                return health
            time.sleep(1.0)
        raise RuntimeError(
            f"the Strata server did not report the model loaded within {timeout_sec:.0f}s "
            f"(STRATA_HEALTH_TIMEOUT). Its log ends with:\n{self._log_tail()}"
        )

    def _stop_server(self) -> None:
        proc, self.server_process = self.server_process, None
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError) as e:
                    logger.warning("[Strata] the server did not stop: %s", e)
            logger.info("[Strata] server stopped")
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    # ---------------------- slots ----------------------
    def _slot_file(self) -> str:
        return f"{self._model_key}.slot-0.bin"

    def _restore_slot(self) -> dict[str, Any]:
        if not (self.slot_dir / self._slot_file()).is_file():
            return {"restored": False, "reason": "no saved conversation"}
        status, body = self._request(
            "POST", "/slots/0?action=restore", {"filename": self._slot_file()}, timeout=600
        )
        if status != 200:
            logger.warning("[Strata] the saved conversation was not restored: %s", body)
            return {"restored": False, "reason": body.get("error", body)}
        logger.info("[Strata] restored the saved conversation: %s", body)
        return {"restored": True, "tokens": body.get("n_restored")}

    def _save_slot(self) -> None:
        try:
            status, state = self._request("GET", "/is_sleeping", timeout=5)
            if status == 200 and state.get("is_sleeping"):
                # A save would have to wake it, taking back VRAM another program holds
                logger.warning("[Strata] asleep at unload: the conversation is not saved")
                return
            status, body = self._request(
                "POST", "/slots/0?action=save", {"filename": self._slot_file()}, timeout=600
            )
        except (urllib_error.URLError, OSError) as e:
            logger.warning("[Strata] the conversation was not saved: %s", e)
            return
        if status == 200:
            logger.info("[Strata] saved the conversation: %s", body)
        else:
            logger.warning("[Strata] the conversation was not saved: %s", body)

    # ---------------------- lifecycle ----------------------
    def load_model(self, config: InferenceConfig) -> None:
        """Start Strata's server with the model's run config and wait until it is loaded."""
        self.config = config
        problem = strata_setup_problem()
        if config.strata_config_path:
            if problem:
                raise RuntimeError(problem)
            source_path = Path(config.strata_config_path).expanduser()
        elif config.model_path:
            # An older install of this backend is brought up to date even when its run config
            # for the model exists: that config only starts the engine the install has.
            stale = None if problem else strata_update_needed()
            found = None if problem or stale else find_run_config(config.model_path)
            if found is None:
                # The first load of this model (or of Strata at all), or one that finds an
                # older install: install and prepare it here, so a GGUF from the model
                # downloads is all a user has to provide.
                self.status_queue.put({"status": "loading", "stage": "strata_setup"})
                found = prepare_model(config.model_path)
            source_path = found
        else:
            raise RuntimeError(
                "engine=strata needs model_path (the model's first GGUF shard, as the model "
                "registry lists it) or strata_config_path (the strata-<model>.json Strata's "
                "setup wrote for it)"
            )
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RuntimeError(f"cannot read the Strata run config {source_path}: {e}") from e
        if not isinstance(source, dict) or not source.get("exe") or not source.get("args"):
            raise RuntimeError(f"{source_path} is not a Strata run config (no exe / args)")

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self._model_key = _safe_name(source_path.stem)
        self._stop_server()  # before the VRAM is measured: a stale server would hold it
        reserve = None
        if config.strata_vram_limit_gb is not None:
            gpu_key = source.get("gpu")
            gpu = gpu_key if isinstance(gpu_key, int) and not isinstance(gpu_key, bool) else 0
            reserve = vram_reserve_for_limit(config.strata_vram_limit_gb, free_vram_mib(gpu))
        run = build_run_config(
            config, source, str(self.runtime_dir / "strata-engine.log"), vram_reserve_mib=reserve
        )
        run_path = self.runtime_dir / f"{self._model_key}.run.json"
        run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")

        self._start_server(run_path)
        try:
            health = self._wait_ready(STRATA_HEALTH_TIMEOUT)
        except Exception:
            self._stop_server()
            raise
        slot = self._restore_slot()

        capabilities = ["chat"]
        if health.get("images"):
            capabilities += ["vision", "multimodal"]
        self.status_queue.put(
            {
                "status": "ready",
                "config": config.model_dump(),
                # the engine's own backend: "cuda", or "sycl" on an Intel Arc package
                "device": engine_backend(),
                # read by the model card and /inference/status like llama-server's tags
                "llama_capabilities": capabilities,
                "served_context_length": health.get("max_context"),
                "slot_restore_summary": slot,
            }
        )
        logger.info(
            "[Strata] %s loaded (context %s, images %s)",
            health.get("model"),
            health.get("max_context"),
            bool(health.get("images")),
        )

    def unload(self) -> None:
        """Save the conversation, stop the server, and reset state."""
        if self.config is not None and self.server_process:
            self._save_slot()
        self._stop_server()
        self.config = None
        self.stop_generation_flag.clear()
        self.status_queue.put({"status": "unloaded", "message": "Strata engine unloaded"})
