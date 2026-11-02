"""Single source of truth for every path this stack uses (Windows + Linux).

Nothing else in the engine or the skills may hardcode an absolute path. They
import this module, and this module resolves everything from, in order:

  1. an already-set environment variable  (highest priority, per-invocation)
  2. `hermes-comfy.conf`                  (the one file you edit on a new box)
  3. a platform-native default            (works if you installed in the usual place)

Layout differences this module absorbs
--------------------------------------
ComfyUI has three shapes this pack meets, and they nest differently:

  Windows portable   COMFY_HOME/ComfyUI/main.py     + COMFY_HOME/python_embeded/python.exe
  Linux git clone    COMFY_HOME/main.py             + COMFY_HOME/venv/bin/python
  AI Scaler Toolkit  COMFY_HOME/ComfyUI/main.py     + COMFY_HOME/envs/<cuda|xpu|cpu>/...

The third also writes COMFY_HOME/selected.json, naming the card picked in its
settings; see _load_selection.

so `main.py` is what we look for, not a fixed depth. Everything else
(input/, output/, models/) is resolved relative to whichever one we found,
which is also what ComfyUI itself does.

Editing the config
------------------
`hermes-comfy.conf` is KEY=VALUE, `#` comments, `~` and $VARS expanded.
Only two keys normally matter on a fresh machine:

    COMFY_HOME=/opt/ComfyUI
    HERMES_COMFY_ENGINE=/home/you/.hermes-comfy/engine

Run `python tools/doctor.py` after editing; it prints every resolved path and
tells you which ones do not exist yet.
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"
ENGINE_DIR = str(Path(__file__).resolve().parent)


def _detect_wsl() -> bool:
    """True inside a WSL distro.

    Matters because WSL is Linux for every syscall we make, yet ComfyUI, the
    models and the co-resident LLM may all live on the Windows side of the
    boundary -- a split this module has to describe accurately or the planner
    budgets against the wrong machine.
    """
    if WINDOWS:
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


WSL = _detect_wsl()


# --------------------------------------------------------------------------
# config file
# --------------------------------------------------------------------------
def _hermes_home() -> Path:
    """Mirror of hermes_constants.get_hermes_home() — kept in sync deliberately.

    We do not import Hermes here: the engine must stay runnable from a bare
    `python engine/start_comfy.py` with no Hermes on sys.path at all.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    if WINDOWS:
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


HERMES_HOME = str(_hermes_home())
SKILLS_DIR = str(_hermes_home() / "skills")


def find_uv(home: Path | None = None) -> str | None:
    """uv, which installs every Python package this pack needs. None without one.

    In order: UV (AI Scaler Toolkit passes the one its payload carries, so the
    version is the one that release was tested with), Hermes's own copy in
    <hermes home>/bin -- its installer puts one there and builds its venv with
    it -- then whatever is on PATH.

    uv rather than pip because the environments it builds have no pip: Hermes's
    venv is one of them, and `python -m pip` is a ModuleNotFoundError there.
    """
    explicit = os.environ.get("UV", "").strip()
    own = (home or _hermes_home()) / "bin" / ("uv.exe" if WINDOWS else "uv")
    for candidate in (explicit, str(own)):
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("uv")


def pip_install_command(python: str, packages: str) -> str:
    """The command that installs `packages` into `python`, for a reader to run."""
    uv = find_uv()
    if uv:
        return f'"{uv}" pip install --python "{python}" {packages}'
    return f'"{python}" -m pip install {packages}'

CONF_NAME = "hermes-comfy.conf"


def _conf_candidates():
    explicit = os.environ.get("HERMES_COMFY_CONF", "").strip()
    if explicit:
        yield Path(explicit)
    yield Path(ENGINE_DIR) / CONF_NAME
    yield Path(ENGINE_DIR).parent / CONF_NAME
    yield _hermes_home() / CONF_NAME


def _load_conf() -> str | None:
    """Fold the first config file found into os.environ. Env vars already set win."""
    for p in _conf_candidates():
        try:
            if not p.is_file():
                continue
            for raw in p.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if not k or os.environ.get(k):
                    continue          # a real env var always beats the file
                os.environ[k] = os.path.expanduser(os.path.expandvars(v))
            return str(p)
        except OSError:
            continue
    return None


CONF_FILE = _load_conf()


def _env_path(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return str(Path(os.path.expanduser(os.path.expandvars(v or default))))


# --------------------------------------------------------------------------
# ComfyUI
# --------------------------------------------------------------------------
def ast_comfy_home() -> Path:
    """Where AI Scaler Toolkit installs ComfyUI, whether or not it has.

    Its layout is neither of the two above: the source at <home>/ComfyUI, and one
    virtual environment per PyTorch build at <home>/envs/<cuda|xpu|cpu>, because the
    build decides which vendor's cards ComfyUI can see and the user picks the card.
    """
    if WINDOWS:
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "AIScalerToolkit" / "components" / "comfyui"
    return Path.home() / ".local" / "share" / "ai-scaler-toolkit" / "components" / "comfyui"


def _load_selection() -> dict:
    """The device picked on AI Scaler Toolkit's ComfyUI card, if one was.

    selected.json is written there each time ComfyUI is started from that card, and
    names the Python, the arguments and the card that choice means. Reading it is
    what makes the skills launch and measure on the card the user picked, rather
    than on whichever one a probe finds first -- they used to start ComfyUI on the
    NVIDIA card after Intel had been chosen, and size the reserve from the NVIDIA
    card while ComfyUI ran on Intel.

    COMFY_SELECTION overrides the location; an unreadable file is simply absent.
    """
    import json
    explicit = os.environ.get("COMFY_SELECTION", "").strip()
    path = (Path(os.path.expanduser(explicit)) if explicit
            else ast_comfy_home() / "selected.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != 1:
        return {}
    data["_path"] = str(path)
    return data


SELECTION = _load_selection()


def _default_comfy_home() -> str:
    """The selection's install, then AI Scaler Toolkit's, then the platform default."""
    if SELECTION.get("comfy_home"):
        return str(SELECTION["comfy_home"])
    ast_home = ast_comfy_home()
    if (ast_home / "ComfyUI" / "main.py").is_file():
        return str(ast_home)
    if WINDOWS:
        return str(Path.home() / "ComfyUI_windows_portable")
    return str(Path.home() / "ComfyUI")


COMFY_HOME = _env_path("COMFY_HOME", _default_comfy_home())
# The selection only describes the install it came from. A COMFY_HOME pointing
# somewhere else -- a portable build named in the conf -- must not inherit its
# Python, arguments or card.
if SELECTION and Path(SELECTION.get("comfy_home", "")) != Path(COMFY_HOME):
    SELECTION = {}


def wsl_host_ip(route_file: str = "/proc/net/route") -> str | None:
    """The Windows host's address as seen from inside WSL.

    Only needed in WSL's default NAT mode, where the distro gets its own
    network namespace and 127.0.0.1 is the distro, not Windows. The default
    route's gateway is the host. Under WSL2 mirrored networking there is no
    separate namespace and 127.0.0.1 already reaches Windows, so this returns
    None and the plain loopback URL is correct.

    `route_file` is injectable so the hex/endianness parsing can be tested off
    a real WSL box -- that parsing is the part most likely to be wrong.
    """
    if not WSL:
        return None
    try:
        with open(route_file, encoding="ascii") as f:
            next(f)                                   # header
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":   # destination 0.0.0.0
                    gw = int(parts[2], 16)            # little-endian hex
                    return "%d.%d.%d.%d" % (gw & 0xFF, (gw >> 8) & 0xFF,
                                            (gw >> 16) & 0xFF, (gw >> 24) & 0xFF)
    except (OSError, StopIteration, ValueError):
        pass
    return None


COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def comfy_url_candidates() -> list:
    """URLs to try, in order, when reaching ComfyUI.

    Configured value first -- an explicit COMFY_URL is always honoured. The WSL
    gateway is appended only as a fallback, because a NAT-mode distro cannot
    reach a Windows ComfyUI on loopback and the host IP changes across WSL
    restarts, so pinning it in the config would go stale.
    """
    out = [COMFY_URL]
    if WSL:
        host = wsl_host_ip()
        if host:
            from urllib.parse import urlparse, urlunparse
            u = urlparse(COMFY_URL)
            alt = urlunparse(u._replace(netloc="%s:%d" % (host, u.port or 8188)))
            if alt not in out:
                out.append(alt)
    return out


def _find_comfy_root(home: str) -> str:
    """The directory that actually contains main.py.

    Portable builds nest it one level down; git clones do not. Checking for the
    file is the only way that survives both without a flag the user must know.
    """
    home_p = Path(home)
    for cand in (home_p / "ComfyUI", home_p):
        if (cand / "main.py").is_file():
            return str(cand)
    # Not installed yet. Return the conventional slot so error messages point
    # somewhere useful instead of at a half-resolved path.
    return str(home_p / "ComfyUI" if WINDOWS else home_p)


COMFY_ROOT = _env_path("COMFY_ROOT", _find_comfy_root(COMFY_HOME))
COMFY_MAIN = str(Path(COMFY_ROOT) / "main.py")
COMFY_INPUT = _env_path("COMFY_INPUT", str(Path(COMFY_ROOT) / "input"))
COMFY_OUTPUT = _env_path("COMFY_OUTPUT", str(Path(COMFY_ROOT) / "output"))
COMFY_MODELS = _env_path("COMFY_MODELS", str(Path(COMFY_ROOT) / "models"))
COMFY_NODES = _env_path("COMFY_NODES", str(Path(COMFY_ROOT) / "custom_nodes"))


def _find_comfy_python(home: str, root: str) -> str:
    """The interpreter that runs h3_media.py (MiniMax-H3's audio mux).

    It needs PyAV and Pillow — notably NOT torch — so any interpreter with
    those will do; ComfyUI's own is simply the one most likely to have them.

    Windows executables are never offered on a non-Windows host. That is not
    hypothetical tidiness: under WSL a Windows portable ComfyUI is visible
    through /mnt/c, so `python_embeded/python.exe` passes an exists() check and
    would be selected — then invoked from Linux, with Linux paths, and fail.
    On WSL the right answer is the distro's own python plus `pip install av`.
    """
    home_p, root_p = Path(home), Path(root)
    cands = []
    if SELECTION.get("python"):
        cands.append(Path(SELECTION["python"]))                  # the card picked
    if WINDOWS:
        cands.append(home_p / "python_embeded" / "python.exe")   # portable build
    rel = "Scripts/python.exe" if WINDOWS else "bin/python"
    for base in (home_p, root_p):
        cands += [base / "venv" / rel, base / ".venv" / rel]
    # AI Scaler Toolkit's per-build environments, in its own recommended order.
    cands += [home_p / "envs" / b / rel for b in ("cuda", "xpu", "cpu")]
    for c in cands:
        if c.is_file():
            return str(c)
    return sys.executable


COMFY_PYTHON = _env_path("COMFY_PYTHON", _find_comfy_python(COMFY_HOME, COMFY_ROOT))

# Extra flags for the ComfyUI process. The portable build wants
# --windows-standalone-build; a Linux clone rejects nothing but gains nothing.
COMFY_EXTRA_ARGS = [a for a in os.environ.get("COMFY_EXTRA_ARGS", "").split() if a]
if not COMFY_EXTRA_ARGS and WINDOWS and (Path(COMFY_HOME) / "python_embeded").is_dir():
    COMFY_EXTRA_ARGS = ["--windows-standalone-build"]
# What puts ComfyUI on the selected card (--cuda-device N, --cpu). Added unless the
# operator set COMFY_EXTRA_ARGS, which is a decision about the whole command line.
if not os.environ.get("COMFY_EXTRA_ARGS", "").strip():
    COMFY_EXTRA_ARGS += [str(a) for a in SELECTION.get("args") or []]

# Environment the ComfyUI process needs for that card: CUDA_DEVICE_ORDER, so the
# index above means what nvidia-smi says, and ONEAPI_DEVICE_SELECTOR for Intel.
COMFY_ENV = {str(k): str(v) for k, v in (SELECTION.get("env") or {}).items()}

# Which card the planner measures: the selection's vendor and index, unless the
# conf or the environment says otherwise. osmem reads these from os.environ, so
# they are put there for it.
_VENDOR_BY_BACKEND = {"cuda": "nvidia", "xpu": "intel"}
if SELECTION.get("backend") in _VENDOR_BY_BACKEND:
    os.environ.setdefault("VIDGEN_GPU_VENDOR", _VENDOR_BY_BACKEND[SELECTION["backend"]])
    os.environ.setdefault("VIDGEN_GPU_INDEX", str(SELECTION.get("index", 0)))
GPU_VENDOR = os.environ.get("VIDGEN_GPU_VENDOR", "").strip().lower()
GPU_INDEX = os.environ.get("VIDGEN_GPU_INDEX", "").strip()

# Run silently. --windows-standalone-build sets args.auto_launch = True (see
# comfy/cli_args.py), which pops a browser tab open every time the launcher
# brings ComfyUI up -- and this launcher brings it up on demand, so a long
# session sprouts a tab per restart. --disable-auto-launch is evaluated after
# it in the same block, so it wins. Appended rather than substituted so an
# operator who really wants the tab can ask for it with COMFY_EXTRA_ARGS
# containing --auto-launch.
if not {"--auto-launch", "--disable-auto-launch"} & set(COMFY_EXTRA_ARGS):
    COMFY_EXTRA_ARGS.append("--disable-auto-launch")

# No paid cloud "partner" nodes by default. An agent that bypassed gen.py once
# hand-built a workflow around RecraftImageInpaintingNode; only a missing
# login stopped it. With the flag the node does not exist at all. Set
# COMFY_ALLOW_API_NODES=1 to load them anyway.
if (os.environ.get("COMFY_ALLOW_API_NODES", "").strip() != "1"
        and "--disable-api-nodes" not in COMFY_EXTRA_ARGS):
    COMFY_EXTRA_ARGS.append("--disable-api-nodes")


def on_foreign_mount(path: str, mounts_file: str = "/proc/mounts") -> bool:
    """True if `path` lives on a Windows drive mounted into this distro.

    Asks /proc/mounts for the filesystem type rather than matching a "/mnt/"
    prefix: the automount root is configurable in /etc/wsl.conf, and a manual
    `mount -t drvfs` can put a Windows drive anywhere. The longest matching
    mount point wins, since mounts nest.
    """
    if not WSL:
        return False
    # Plain string work on purpose: these are POSIX paths, and Path() would
    # rewrite the separators when this logic is exercised from a Windows host
    # (which is exactly how it gets tested).
    p = str(path).replace("\\", "/").rstrip("/") or "/"
    try:
        with open(mounts_file, encoding="utf-8", errors="replace") as f:
            rows = [ln.split() for ln in f]
    except OSError:
        return p.startswith("/mnt/")                      # last-resort heuristic
    best = ""
    for r in rows:
        if len(r) < 3:
            continue
        mp, fstype = r[1].replace("\\040", " ").rstrip("/") or "/", r[2]
        if fstype not in ("9p", "drvfs", "virtiofs", "drvfs.exe"):
            continue
        # Compare on path boundaries, so /mnt/cdrom never matches /mnt/c.
        if (p == mp or p.startswith(mp + "/")) and len(mp) > len(best):
            best = mp
    return bool(best)


def _detect_remote() -> bool:
    """True when this process cannot start ComfyUI itself, only talk to it.

    The WSL case is the reason this exists: from inside the distro a Windows
    ComfyUI is perfectly *visible* through /mnt/c, so an existence check on
    main.py says "local" and the launcher then tries to run a Windows build
    under a Linux interpreter. Ownership, not visibility, is the real test --
    a ComfyUI sitting on a Windows drive mount belongs to Windows.
    """
    raw = os.environ.get("COMFY_REMOTE", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    if on_foreign_mount(COMFY_HOME):
        return True
    return not Path(COMFY_MAIN).is_file()


COMFY_REMOTE = _detect_remote()


# --------------------------------------------------------------------------
# engine-owned directories (created on demand, never shipped)
# --------------------------------------------------------------------------
WF_DIR = _env_path("HERMES_COMFY_WORKFLOWS", str(Path(ENGINE_DIR) / "workflows"))
STATE_DIR = _env_path("HERMES_COMFY_STATE", ENGINE_DIR)
OUT_DIR = _env_path("HERMES_COMFY_OUT", str(Path(STATE_DIR) / "out"))
JOBS_DIR = _env_path("HERMES_COMFY_JOBS", str(Path(STATE_DIR) / "jobs"))
LOG_DIR = _env_path("HERMES_COMFY_LOGS", str(Path(STATE_DIR) / "logs"))
# start_comfy.py writes the path of the log its server is writing here.
CURRENT_LOG_POINTER = str(Path(LOG_DIR) / "comfy_current.txt")

# The TRUSTA-AST backend. Everything this pack needs to know about the
# co-resident LLM is asked here -- whether a model is loaded, which engine is
# hosting it, how much of the card it holds -- so no inference runtime is ever
# named or probed directly. See astapi.py.
# 8571 is AI Scaler Toolkit's own default port. The backend's run_service scripts
# use 8000 when run by hand from a checkout, which sets VIDGEN_AST_URL instead.
AST_DEFAULT_URL = "http://127.0.0.1:8571"
AST_URL = (os.environ.get("VIDGEN_AST_URL") or AST_DEFAULT_URL).rstrip("/")
# osmem's last-resort source asks the backend too, and reads only os.environ.
os.environ.setdefault("VIDGEN_AST_URL", AST_URL)

# Optional escape hatch: the process whose clean, file-backed weight pages may
# be handed back to the OS once a render is confirmed running. There is
# deliberately NO default -- naming an inference binary here is what used to
# tie the pack to one runtime. Left unset, astapi derives it from the engine
# the backend reports, and if that yields nothing the planner simply does not
# count reclaimable DRAM (the conservative direction).
LLM_PROCESS = os.environ.get("VIDGEN_LLM_PROCESS", "").strip()


def ensure_dirs() -> None:
    for d in (OUT_DIR, JOBS_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def describe() -> dict:
    """Everything doctor.py and the installers report. Keep it JSON-safe."""
    return {
        "platform": sys.platform + ("  (WSL)" if WSL else ""),
        "wsl": WSL,
        "wsl_host_ip": wsl_host_ip(),
        "comfy_remote": COMFY_REMOTE,
        "comfy_url_candidates": comfy_url_candidates(),
        "conf_file": CONF_FILE,
        "hermes_home": HERMES_HOME,
        "skills_dir": SKILLS_DIR,
        "engine_dir": ENGINE_DIR,
        "comfy_home": COMFY_HOME,
        "comfy_root": COMFY_ROOT,
        "comfy_main": COMFY_MAIN,
        "comfy_python": COMFY_PYTHON,
        "comfy_models": COMFY_MODELS,
        "comfy_nodes": COMFY_NODES,
        "comfy_input": COMFY_INPUT,
        "comfy_output": COMFY_OUTPUT,
        "comfy_url": COMFY_URL,
        "comfy_extra_args": COMFY_EXTRA_ARGS,
        "comfy_env": COMFY_ENV,
        "selection": ("%s (%s)" % (SELECTION.get("device"), SELECTION.get("_path"))
                      if SELECTION else None),
        "gpu_measured": (GPU_VENDOR + (":" + GPU_INDEX if GPU_INDEX else "")) or "first found",
        "workflows": WF_DIR,
        "out_dir": OUT_DIR,
        "jobs_dir": JOBS_DIR,
        "log_dir": LOG_DIR,
        "ast_url": AST_URL,
        "llm_process_override": LLM_PROCESS or None,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
