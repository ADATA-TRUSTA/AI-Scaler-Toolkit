#!/usr/bin/env bash
# Set up the WSL (Ubuntu) side of a split install:
#
#   WSL Ubuntu   Hermes + the skills + this engine        <- what this script builds
#   Windows      ComfyUI, the model files, TRUSTA-AST    <- stays where it is
#
# The two halves talk over HTTP (ComfyUI's API) and over /mnt/c (staging input
# images, reading rendered output). Nothing is copied or duplicated.
#
#   ./wsl_bootstrap.sh                     # detect everything, then install
#   ./wsl_bootstrap.sh --dry-run           # show what it found, change nothing
#   ./wsl_bootstrap.sh --comfy-home /mnt/c/Users/you/ComfyUI_windows_portable
#   ./wsl_bootstrap.sh --skip-hermes       # Hermes already installed here
set -euo pipefail

PACK="$(cd "$(dirname "$0")/.." && pwd)"
DRY=0
SKIP_HERMES=0
COMFY_HOME_ARG=""
COMFY_PORT="${COMFY_PORT:-8188}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)     DRY=1 ;;
        --skip-hermes) SKIP_HERMES=1 ;;
        --comfy-home)  COMFY_HOME_ARG="$2"; shift ;;
        --port)        COMFY_PORT="$2"; shift ;;
        -h|--help)     sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   \033[33mWARN\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "environment"
# ---------------------------------------------------------------------------
grep -qi microsoft /proc/version || die "this is not WSL. Run install.sh instead."
info "WSL distro : ${WSL_DISTRO_NAME:-unknown}"
if [ -r /etc/os-release ]; then
    . /etc/os-release
    info "distro     : ${PRETTY_NAME:-unknown}"
fi
PYV="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
info "python3    : $PYV"
case "$PYV" in
    3.11|3.12|3.13) ;;
    none) die "python3 not found. sudo apt install -y python3 python3-venv python3-pip" ;;
    *) warn "Hermes requires python 3.11-3.13; found $PYV. Its installer may pick another." ;;
esac

# ---------------------------------------------------------------------------
say "locating the Windows side"
# ---------------------------------------------------------------------------
# The Windows user profile, asked of Windows itself rather than guessed: the
# drive letter and user name both vary, and /mnt/c is only the default mount.
WINHOME=""
if command -v wslpath >/dev/null 2>&1 && command -v cmd.exe >/dev/null 2>&1; then
    WINHOME_RAW="$(cd /mnt/c 2>/dev/null && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' || true)"
    [ -n "$WINHOME_RAW" ] && WINHOME="$(wslpath -u "$WINHOME_RAW" 2>/dev/null || true)"
fi
[ -n "$WINHOME" ] && info "windows home : $WINHOME" || warn "could not ask Windows for %USERPROFILE% (interop off?)"

COMFY_HOME=""
if [ -n "$COMFY_HOME_ARG" ]; then
    COMFY_HOME="$COMFY_HOME_ARG"
else
    for c in "$WINHOME/ComfyUI_windows_portable" "$WINHOME/ComfyUI" \
             /mnt/c/ComfyUI_windows_portable /mnt/c/ComfyUI; do
        [ -n "$c" ] || continue
        if [ -f "$c/ComfyUI/main.py" ] || [ -f "$c/main.py" ]; then COMFY_HOME="$c"; break; fi
    done
fi
[ -n "$COMFY_HOME" ] || die "ComfyUI not found under /mnt/c. Pass --comfy-home <path>."
info "comfy home   : $COMFY_HOME"

if [ -f "$COMFY_HOME/ComfyUI/main.py" ]; then COMFY_ROOT="$COMFY_HOME/ComfyUI"; else COMFY_ROOT="$COMFY_HOME"; fi
info "comfy root   : $COMFY_ROOT"
[ -d "$COMFY_ROOT/models" ] || warn "no models/ under $COMFY_ROOT"
[ -d "$COMFY_ROOT/custom_nodes/ComfyUI-GGUF" ] || warn "ComfyUI-GGUF custom node missing on the Windows side"

# ---------------------------------------------------------------------------
say "reaching ComfyUI across the WSL boundary"
# ---------------------------------------------------------------------------
# In WSL's default NAT mode the distro has its own network namespace, so the
# host's 127.0.0.1 is NOT ours -- the default gateway is Windows. Under WSL2
# mirrored networking there is no separate namespace and loopback just works.
GW="$(ip route show default 2>/dev/null | awk '{print $3; exit}')"
info "default gw   : ${GW:-none}"
COMFY_URL=""
for u in "http://127.0.0.1:$COMFY_PORT" ${GW:+"http://$GW:$COMFY_PORT"}; do
    if curl -fsS --max-time 4 "$u/system_stats" >/dev/null 2>&1; then
        COMFY_URL="$u"; info "reachable at : $u"; break
    fi
    info "not reachable: $u"
done
if [ -z "$COMFY_URL" ]; then
    COMFY_URL="http://127.0.0.1:$COMFY_PORT"
    warn "ComfyUI is not answering yet. That is fine for install, but before"
    warn "generating you must start it ON WINDOWS:"
    warn "    python <engine>\\start_comfy.py"
    warn "It listens on 127.0.0.1 only, which a NAT-mode distro cannot reach:"
    warn "turn on WSL mirrored networking (%USERPROFILE%\\.wslconfig:"
    warn "[wsl2] networkingMode=mirrored, then wsl --shutdown)."
fi

# ---------------------------------------------------------------------------
say "GPU"
# ---------------------------------------------------------------------------
# Vendor-agnostic, same order the engine uses. Any one of these answering is
# enough; ComfyUI's /system_stats is the no-tooling fallback checked by doctor.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.total --format=csv,noheader >/dev/null 2>&1; then
    info "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1)"
elif command -v xpu-smi >/dev/null 2>&1 && xpu-smi discovery >/dev/null 2>&1; then
    info "xpu-smi:    $(xpu-smi discovery 2>/dev/null | head -3 | tr '\n' ' ')"
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showmeminfo vram >/dev/null 2>&1; then
    info "rocm-smi:   $(rocm-smi --showmeminfo vram 2>/dev/null | head -3 | tr '\n' ' ')"
else
    warn "no GPU vendor CLI on PATH (nvidia-smi / xpu-smi / rocm-smi)."
    warn "Not necessarily fatal: the engine falls back to ComfyUI's own"
    warn "/system_stats, which reports whichever backend torch is on. Run"
    warn "doctor.py once ComfyUI is up to see which source answers, and set"
    warn "VIDGEN_GPU_QUERY_CMD if you would rather supply the numbers yourself."
fi

if [ "$DRY" = "1" ]; then
    say "dry run -- stopping before any change"
    exit 0
fi

# ---------------------------------------------------------------------------
say "system packages"
# ---------------------------------------------------------------------------
NEED=""
for p in python3-venv python3-pip curl git; do
    dpkg -s "$p" >/dev/null 2>&1 || NEED="$NEED $p"
done
if [ -n "$NEED" ]; then
    info "installing:$NEED (sudo)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq $NEED
else
    info "all present"
fi

# ---------------------------------------------------------------------------
say "Hermes"
# ---------------------------------------------------------------------------
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
if [ "$SKIP_HERMES" = "1" ]; then
    info "skipped by request"
elif [ -x "$HERMES_HOME/hermes-agent/venv/bin/python" ]; then
    info "already installed at $HERMES_HOME/hermes-agent"
else
    info "installing from hermes-agent.nousresearch.com (--skip-setup: configure it later)"
    # Deliberately not fatal. The official installer exits non-zero when an
    # OPTIONAL stage fails -- the Node/browser-tool `npm install` has a 600 s
    # cap and times out on a cold WSL distro -- even though the Python agent
    # itself installed fine. Aborting there would leave a perfectly usable
    # Hermes with the skills never installed, so judge by the artifact instead.
    curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup \
        || warn "the Hermes installer reported a failure; checking whether the core survived"
fi

HERMES_PY="$HERMES_HOME/hermes-agent/venv/bin/python"
if [ -x "$HERMES_PY" ]; then
    info "python for skills: $HERMES_PY ($("$HERMES_PY" -V 2>&1))"
    if ! "$HERMES_PY" -c 'import hermes_constants' >/dev/null 2>&1; then
        warn "Hermes venv exists but the agent package does not import; the skills"
        warn "will still work (they only need a python), but 'hermes' may not run."
    fi
    [ -x "$HERMES_HOME/hermes-agent/venv/bin/hermes" ] \
        || warn "no 'hermes' entrypoint -- browser/node tools likely skipped. Harmless here."
else
    HERMES_PY="$(command -v python3)"
    warn "Hermes venv not found; falling back to $HERMES_PY"
    warn "The skills only need a python with PyAV, so this still works."
fi

# ---------------------------------------------------------------------------
say "media dependencies"
# ---------------------------------------------------------------------------
# h3_media.py muxes MiniMax-H3's native audio into the mp4. It needs PyAV and
# Pillow -- NOT torch -- so the local interpreter can host it and we never have
# to call the Windows python across the boundary.
#
# The Hermes venv is built by uv, which does NOT install pip into it, so
# `python -m pip` is a ModuleNotFoundError there. Drive uv directly when it is
# available (the Hermes installer drops it in $HERMES_HOME/bin) and keep pip as
# the fallback for a plain python3.
PKGS="av pillow pillow-heif"
UV="$HERMES_HOME/bin/uv"
[ -x "$UV" ] || UV="$(command -v uv || true)"
installed=0
if [ -n "$UV" ] && [ -x "$UV" ]; then
    info "using uv: $UV"
    "$UV" pip install --quiet --python "$HERMES_PY" $PKGS && installed=1 \
        || warn "uv pip install failed"
fi
if [ "$installed" = "0" ]; then
    "$HERMES_PY" -m pip install --quiet --upgrade $PKGS && installed=1 \
        || warn "pip install failed too"
fi
if "$HERMES_PY" -c 'import av, PIL; print("   av", av.__version__, "| pillow", PIL.__version__)'; then
    :
else
    warn "PyAV/Pillow still not importable under $HERMES_PY."
    warn "Images will work; only MiniMax-H3 audio muxing (vidgen r2v) needs them."
    warn "Fix later with:  $UV pip install --python $HERMES_PY $PKGS"
fi

# ---------------------------------------------------------------------------
say "installing the pack"
# ---------------------------------------------------------------------------
ENGINE_DIR="$HERMES_HOME/comfy-engine"
"$HERMES_PY" "$PACK/install.py" --comfy-home "$COMFY_HOME" --python "$HERMES_PY" || true

CONF="$ENGINE_DIR/hermes-comfy.conf"
[ -f "$CONF" ] || die "installer did not produce $CONF"

# The installer writes a local-machine config; correct it for the split setup.
say "writing split-topology settings into $CONF"
python3 - "$CONF" "$COMFY_URL" "$COMFY_ROOT" "$HERMES_PY" <<'PY'
import sys, io, re
conf, url, root, py = sys.argv[1:5]
# The backend sits on the same Windows host as ComfyUI, on its own port.
from urllib.parse import urlsplit
# 8571, the desktop application's port; 8000 was a checkout's default.
ast_url = "http://%s:8571" % (urlsplit(url).hostname or "127.0.0.1")
t = io.open(conf, encoding="utf-8").read()

def setkey(text, key, val, note):
    """Force KEY=val, replacing a commented or previous definition."""
    line = "%s=%s" % (key, val)
    pat = re.compile(r"^#?\s*%s=.*$" % re.escape(key), re.M)
    if pat.search(text):
        return pat.sub(line, text, count=1)
    return text + "\n# %s\n%s\n" % (note, line)

t = setkey(t, "COMFY_URL", url,
           "ComfyUI runs on the Windows host; this is how WSL reaches it.")
t = setkey(t, "COMFY_REMOTE", "1",
           "ComfyUI belongs to Windows: never try to launch it from here.")
t = setkey(t, "COMFY_ROOT", root, "Windows-side ComfyUI, seen through /mnt/c.")
t = setkey(t, "COMFY_PYTHON", py,
           "h3_media.py needs only PyAV+Pillow, so the local venv can host it.")
t = setkey(t, "VIDGEN_AST_URL", ast_url,
           "The TRUSTA-AST backend is on the Windows side too.")
io.open(conf, "w", encoding="utf-8", newline="\n").write(t)
print("   updated COMFY_URL / COMFY_REMOTE / COMFY_ROOT / COMFY_PYTHON / VIDGEN_AST_URL")
PY

# ---------------------------------------------------------------------------
say "verifying"
# ---------------------------------------------------------------------------
HERMES_COMFY_ENGINE="$ENGINE_DIR" "$HERMES_PY" "$PACK/tools/doctor.py" || true

cat <<EOF

$(printf '\033[1m== next\033[0m')
   1. Turn on WSL mirrored networking, so WSL reaches Windows' 127.0.0.1:
          %USERPROFILE%\\.wslconfig  ->  [wsl2] networkingMode=mirrored
          then, in Windows:  wsl --shutdown
      and on WINDOWS start ComfyUI (it listens on 127.0.0.1 only):
          python <windows-engine>\\start_comfy.py
   2. Configure Hermes:  hermes setup
      Point its LLM endpoint at the AI Scaler Toolkit backend on Windows:
          http://127.0.0.1:8571/v1
      (mirrored networking makes that Windows; no API key is needed for a
       local connection. Go through the backend, not the engine's private
       port -- that one listens on loopback with a key only the backend has)
   3. Set terminal.timeout: 600 in $HERMES_HOME/config.yaml
   4. Try it:
          $HERMES_PY $HERMES_HOME/skills/imggen/gen.py zimage "a red fox"

   Read INSTALL.md section "WSL split install" for what the memory guard can
   and cannot do across the OS boundary.
EOF
