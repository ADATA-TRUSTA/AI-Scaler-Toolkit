"""Protected launcher for the local (portable) ComfyUI.

Starts ComfyUI with a `--reserve-vram` sized to whatever ELSE is on the GPU
(above all the LLM the TRUSTA-AST backend is hosting), so ComfyUI can never
evict it — it offloads its
own weights to system RAM instead (slower, but stable). Leave this server
running; Hermes's built-in `creative/comfyui` skill just connects to :8188
and therefore inherits the protection automatically.

Usage:
  python start_comfy.py                 # auto: reserve = external VRAM + margin
  python start_comfy.py --margin 2      # bigger safety margin (GB)
  python start_comfy.py --reserve 9     # force an explicit reserve (GB)
  python start_comfy.py --force         # launch even if something answers :8188
"""
import os, sys, time, json, socket, argparse, subprocess, urllib.request
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # embeddable python (._pth) does not auto-add script dir

import paths          # resolves hermes-comfy.conf into os.environ -- import first
import astapi         # the only thing that talks to the TRUSTA-AST backend
import osmem
import sysmon

COMFY_URL = paths.COMFY_URL
COMFY_HOME = paths.COMFY_HOME
LOG_DIR = paths.LOG_DIR


def _host_port():
    u = urlparse(COMFY_URL)
    return (u.hostname or "127.0.0.1"), (u.port or 8188)


def is_up():
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _port_bound():
    host, port = _host_port()
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _wake_llm():
    """Make the LLM promote its VRAM before we measure.

    Windows/WDDM demotes an idle process's GPU allocations to system memory:
    the hosted 35B measured 6.6 GB while serving and only 2.5 GB after sitting
    idle. Sizing --reserve-vram against the idle number would under-reserve and
    let ComfyUI take space the LLM needs the moment someone talks to it. One
    throwaway token brings it back, so we always measure the hot footprint.

    The token goes to AST's own OpenAI-compatible endpoint. The old route --
    name an inference binary, grep `--port` out of its command line, POST to
    that port -- worked for exactly one engine on one deployment shape, and
    silently did nothing on any other.
    """
    st = astapi.status()
    if not st or not st.get("loaded"):
        return
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1,
                       "model": st.get("model_name") or "default"}).encode()
    req = urllib.request.Request(astapi.base_url() + "/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    print(f"waking the hosted LLM via {astapi.base_url()} so its VRAM is measured hot...")
    try:
        urllib.request.urlopen(req, timeout=180).read()
        time.sleep(3)
    except Exception as e:
        print(f"WARN: could not wake the LLM ({e}); measuring as-is.", file=sys.stderr)


# ComfyUI has no authentication at all. Bound beyond loopback, anyone who can reach the
# port can queue work on the GPU, upload files, read every image and video it made, and
# -- with ComfyUI-Manager installed, which is common -- install custom nodes, which is
# running code on this machine. So it stays on 127.0.0.1, and this launcher no longer
# offers --listen. The one case that needed it, Hermes in a NAT-mode WSL distro with
# ComfyUI on Windows, is served by WSL's mirrored networking, where loopback is shared.
# Another computer gets the LLM through AI Scaler Toolkit's /v1 with its API key; image
# and video generation run where Hermes runs.
NO_LISTEN = (
    "ComfyUI stays on 127.0.0.1: it has no authentication, so bound to the network "
    "anyone who can reach it can run work, read every output and, with "
    "ComfyUI-Manager, install code on this machine.\n"
    "For Hermes in WSL, turn on mirrored networking instead -- in "
    "%USERPROFILE%\\.wslconfig:\n"
    "    [wsl2]\n"
    "    networkingMode=mirrored\n"
    "then run `wsl --shutdown`; 127.0.0.1 then reaches ComfyUI from WSL as well."
)


def _refuse_listen(argv: list) -> None:
    """Stop before anything starts when asked to bind ComfyUI beyond loopback."""
    asked = [a for a in argv if a == "--listen" or a.startswith("--listen=")]
    if asked:
        print("ERROR: --listen is not supported.\n" + NO_LISTEN, file=sys.stderr)
        sys.exit(2)


def main():
    _refuse_listen(sys.argv[1:])
    # The conf can carry arguments too; a --listen there would bind it all the same.
    _refuse_listen(paths.COMFY_EXTRA_ARGS)
    ap = argparse.ArgumentParser(description="Launch portable ComfyUI with VRAM protection.")
    ap.add_argument("--margin", type=float,
                    default=float(os.environ.get("COMFY_RESERVE_MARGIN", "2.0")),
                    help="GB headroom on top of measured external VRAM use (default 2.0)")
    ap.add_argument("--reserve", type=float, help="Force reserve in GB (skip auto-measure)")
    ap.add_argument("--force", action="store_true", help="Launch even if :8188 is bound/answering")
    ap.add_argument("--no-wake", dest="wake", action="store_false", default=True,
                    help="Do not ping the hosted LLM before measuring (measures whatever is resident now)")
    a = ap.parse_args()

    if is_up() and not a.force:
        print(f"ComfyUI already running at {COMFY_URL}. Nothing to do (use --force to override).")
        return
    if _port_bound() and not a.force:
        print(f"ERROR: {COMFY_URL} is bound but not answering as ComfyUI "
              "(hung/foreign process). Stop it first, or pass --force.", file=sys.stderr)
        sys.exit(2)

    # An LLM that is lent for every render (llmsleep: Strata fills the card on
    # purpose) must not be counted: --reserve-vram is fixed for ComfyUI's
    # lifetime, so reserving its 14 GB would leave every later render ~1 GB
    # even while the LLM sleeps. Nor is it pinged awake to be measured.
    import llmsleep
    lent_llm_gb = 0.0
    if llmsleep.lendable() and not llmsleep.is_lent():
        lent_llm_gb = llmsleep.llm_vram_gb() or 0.0
        print(f"the hosted LLM is lent to each render (put to sleep or unloaded), so its "
              f"{lent_llm_gb:.1f} GB is not reserved")
    elif a.wake and a.reserve is None and not llmsleep.is_lent():
        _wake_llm()          # size the reserve against the LLM's HOT footprint
    g = sysmon.gpu_vram()
    if "error" not in g and lent_llm_gb:
        g = dict(g, used_mb=max(0, g["used_mb"] - int(lent_llm_gb * 1024)))
    if a.reserve is not None:
        reserve = a.reserve
        print(f"Using forced reserve: --reserve-vram {reserve}")
    elif "error" in g:
        reserve = None
        print(f"WARN: no GPU measurement source answered ({g['error']}); "
              f"launching WITHOUT reserve.", file=sys.stderr)
    else:
        reserve = round(g["used_mb"] / 1024.0 + a.margin, 1)
        print(f"GPU already used by others: {g['used_mb']/1024:.1f} GB "
              f"(+{a.margin} margin)  ->  --reserve-vram {reserve}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"comfy_{time.strftime('%Y%m%d_%H%M%S')}.log")
    py = paths.COMFY_PYTHON
    main_py = paths.COMFY_MAIN
    if not os.path.isfile(main_py):
        print("ERROR: ComfyUI not found at %s.\n"
              "       Set COMFY_HOME in %s to the directory that contains ComfyUI "
              "(portable: the folder holding ComfyUI\\ and python_embeded\\; "
              "Linux: the cloned ComfyUI checkout itself)."
              % (main_py, paths.CONF_FILE or paths.CONF_NAME), file=sys.stderr)
        sys.exit(2)
    # -s keeps the embedded interpreter from picking up a user site-packages;
    # harmless on a venv, so it stays on both platforms.
    args = [py, "-s", main_py] + paths.COMFY_EXTRA_ARGS
    if reserve:
        args += ["--reserve-vram", str(reserve)]

    # The selected card's environment (CUDA_DEVICE_ORDER, ONEAPI_DEVICE_SELECTOR)
    # on top of ours; empty unless AI Scaler Toolkit's selected.json supplied one.
    env = {**os.environ, **paths.COMFY_ENV} if paths.COMFY_ENV else None
    with open(log_path, "w") as log:
        subprocess.Popen(args, cwd=paths.COMFY_ROOT, stdout=log, stderr=log, env=env,
                         **osmem.detach_kwargs())
    # Where this server is logging, for imggen to watch: its sampler writes a
    # progress line per step, and a log that keeps growing is the only sign over
    # HTTP that a slow render is working rather than stuck.
    try:
        with open(paths.CURRENT_LOG_POINTER, "w", encoding="utf-8") as f:
            f.write(log_path + "\n")
    except OSError:
        pass
    print(f"launching ComfyUI... (log: {log_path})")
    for _ in range(150):  # up to 5 min for cold model/driver init
        if is_up():
            print(f"OK - ComfyUI ready at {COMFY_URL}  (reserve_vram = {reserve} GB). "
                  "Generate through the imggen / vidgen skills (gen.py), not the built-in comfyui skill.")
            return
        time.sleep(2)
    print("ERROR: ComfyUI did not become ready in time; check the log above.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
