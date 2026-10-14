#!/usr/bin/env bash
# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (built from source + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   ./setup_env.sh                          # auto-detect; install llama only if missing
#                                           # (llama-server, llama-quantize and llama-fit-params)
#   TRUSTA_ACCEL=xpu ./setup_env.sh         # force cuda | xpu
#   TRUSTA_SETUP_VLLM=0 ./setup_env.sh      # skip vLLM (installed by default on CUDA hosts)
#   TRUSTA_SETUP_VLLM=1 ./setup_env.sh      # also on XPU hosts, where it is opt-in
#   TRUSTA_INSTALL_LLAMA=0 ./setup_env.sh   # skip llama entirely
#   TRUSTA_INSTALL_LLAMA=1 ./setup_env.sh   # force reinstall even if already present
#   TRUSTA_LLAMA_BACKEND=vulkan ./setup_env.sh  # force the generic Vulkan build (sees every Intel/AMD/NVIDIA card)
#   TRUSTA_LLAMA_BACKEND=cpu ./setup_env.sh     # CPU-only build
#   TRUSTA_APT_INSTALL=1 ./setup_env.sh         # let setup_env apt-install missing build deps
#   TRUSTA_APT_INSTALL=0 ./setup_env.sh         # never apt-install, even as root
#   TRUSTA_LLAMA_ALLOW_FALLBACK=1 ./setup_env.sh  # unattended: accept a degraded llama
#   TRUSTA_SKIP_XPU_CHECK=1 ./setup_env.sh      # skip the post-sync XPU smoke check
#   TRUSTA_XPU_CHECK_TIMEOUT=600 ./setup_env.sh # raise the XPU check kill timeout (seconds)
set -euo pipefail

die() {
    echo "[setup_env] $*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/service"

# GGUF convert tooling (convert_hf_to_gguf.py / convert_lora_to_gguf.py + conversion/ + gguf-py):
# only shipped with the llama.cpp sources, not built by the CMake targets, so they are copied
# out of the same extracted archive the binary is built from — no C++, nothing to compile.
LLAMA_CONVERT_DIR="$SERVICE_DIR/utils/llama.cpp"
# Built binaries live under the project, not in ~/.local/bin: they are ours to place, pin
# and delete, and a per-project directory keeps two checkouts apart.
LLAMA_BIN_DIR="$SERVICE_DIR/utils/llama-bin"
LLAMA_BUILD_DIR="$SERVICE_DIR/utils/llama-build"
# The standalone tools the backend shells out to besides the server: llama-quantize
# (service/utils/conversion_manager.py) and llama-fit-params (gguf_estimator.py).
LLAMA_TOOLS=(llama-quantize llama-fit-params)
# Where those tools go when the server is not ours (LLAMA_SERVER_BINARY or PATH): setup_env
# does not write into someone else's directory, so they get their own and are recorded in .env.
LLAMA_TOOLS_DIR="$SERVICE_DIR/utils/llama-tools"
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp"
# The one place the pinned build is named. TRUSTA_LLAMA_VERSION overrides it.
LLAMA_VERSION_DEFAULT="b10107"
LLAMA_VERSION="${TRUSTA_LLAMA_VERSION:-$LLAMA_VERSION_DEFAULT}"
# The sources are a tarball of the tag, not a clone.
#
# git used to be a hard prerequisite of this script and of the desktop launcher's preflight
# check, purely to fetch these. On a clean machine that cost an installation of git -- and on
# Windows that meant a UAC prompt, in an installer whose whole design is to need none, because
# winget's Git.Git has no per-user installer and elevates whatever --scope it is given.
#
# GitHub serves every tag as an archive, so the source can be had with the same curl/wget/python
# fallback that fetches uv. Nothing is lost by it: the tag is resolved by the server, so the
# convert scripts and the compiled binary come from one download and cannot be different
# revisions -- which is the drift the old two-step (resolve a sha, then fetch it) existed to
# prevent, and which needed a hardcoded offline fallback sha that had to be kept in step by a
# test.
LLAMA_SRC_DIR="$LLAMA_BUILD_DIR/src"
llama_tarball_url() { echo "$LLAMA_CPP_URL/archive/refs/tags/$1.tar.gz"; }
# Written into the extracted tree, because an archive carries no revision of its own and
# "which version is this directory" has to be answerable without the network.
LLAMA_SRC_STAMP=".trusta-llama-version"

CONVERT_PATHS=(convert_hf_to_gguf.py convert_lora_to_gguf.py conversion gguf-py)

# Extract the tag's sources into LLAMA_SRC_DIR, unless they are already there.
ensure_llama_source() {
    local ver="$1" stamp="$LLAMA_SRC_DIR/$LLAMA_SRC_STAMP" archive status
    if [[ -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" == "$ver" && -f "$LLAMA_SRC_DIR/CMakeLists.txt" ]]; then
        echo "[setup_env] reusing the llama.cpp $ver sources at $LLAMA_SRC_DIR"
        return 0
    fi

    echo "[setup_env] downloading llama.cpp $ver"
    archive="$(mktemp --suffix=.tar.gz)"
    fetch_to "$(llama_tarball_url "$ver")" "$archive" && status=0 || status=$?
    if ((status == 2)); then
        rm -f "$archive"
        echo "[setup_env] WARNING: no curl, wget or python3 to download llama.cpp with." >&2
        return 1
    fi
    if ((status != 0)); then
        rm -f "$archive"
        echo "[setup_env] WARNING: could not download $(llama_tarball_url "$ver")" >&2
        echo "[setup_env]   Check the network, or pin a version that exists: TRUSTA_LLAMA_VERSION=bNNNNN" >&2
        return 1
    fi

    # Replaced, not merged. A half-extracted tree from an interrupted run would otherwise be
    # built and reported as the pinned version.
    rm -rf "$LLAMA_SRC_DIR"
    mkdir -p "$LLAMA_SRC_DIR"
    # --strip-components=1 drops the llama.cpp-<tag>/ wrapper the archive adds.
    if ! tar -xzf "$archive" -C "$LLAMA_SRC_DIR" --strip-components=1; then
        rm -f "$archive"
        rm -rf "$LLAMA_SRC_DIR"
        echo "[setup_env] WARNING: the llama.cpp archive could not be extracted (truncated download?)" >&2
        return 1
    fi
    rm -f "$archive"
    printf '%s\n' "$ver" > "$stamp"
    echo "[setup_env] llama.cpp $ver sources ready at $LLAMA_SRC_DIR"
}

detect_accel() {
    if nvidia-smi &>/dev/null 2>&1; then
        echo "cuda"
    elif command -v clinfo &>/dev/null 2>&1 && clinfo 2>/dev/null | grep -qi "Intel"; then
        echo "xpu"
    elif [[ -d /dev/dri ]] && ls /dev/dri/renderD* &>/dev/null 2>&1; then
        # Intel Arc / iGPU usually exposes a renderD device under /dev/dri
        echo "xpu"
    elif [[ -c /dev/dxg ]] && [[ -e /usr/lib/x86_64-linux-gnu/libze_loader.so.1 ]]; then
        # WSL2 has no /dev/dri: the GPU is paravirtualised through /dev/dxg, and Level Zero
        # reaches it via the loader shipped with the distro. Without this branch an Intel-only
        # WSL2 box falls through and installs the wrong torch extra.
        echo "xpu"
    else
        # Not cuda. Falling back to it meant that a machine with no accelerator at all
        # -- an AMD desktop, a plain virtual machine -- installed the CUDA extra and
        # died six minutes later building causal-conv1d, with
        #
        #   NameError: name 'bare_metal_version' is not defined
        #
        # from a setup.py that had looked for a CUDA toolkit and not found one.
        # Nothing in that names the actual problem. Saying "none" turns it into one
        # line, before any download.
        echo "none"
    fi
}

# llama inference backend (decoupled from the torch accel): auto = cuda when NVIDIA is present, else vulkan
resolve_llama_backend() {
    local backend="${TRUSTA_LLAMA_BACKEND:-auto}"
    case "$backend" in
        auto|AUTO|"")
            if nvidia-smi &>/dev/null 2>&1; then echo "cuda"; else echo "vulkan"; fi
            ;;
        cuda|vulkan|cpu)
            echo "$backend"
            ;;
        *)
            die "unsupported TRUSTA_LLAMA_BACKEND: $backend (use auto | cuda | vulkan | cpu)"
            ;;
    esac
}

# Echoes yes/no instead of using the exit status, so the decision (and its validation)
# can be resolved before the long `uv sync` rather than after it.
resolve_setup_vllm() {
    local mode="${TRUSTA_SETUP_VLLM:-auto}"
    case "$mode" in
        1|true|TRUE|yes|YES|on|ON)
            echo "yes"
            ;;
        0|false|FALSE|no|NO|off|OFF)
            echo "no"
            ;;
        auto|AUTO|"")
            # CUDA only by default. The XPU build (vllm-xpu) works on the Panther Lake
            # iGPUs it was tried on, but is opt-in: an iGPU's memory is the system's,
            # the kernel's TTM page pool keeps up to half of RAM after a run and vLLM's
            # startup check counts that as used, so a second load fails; and vLLM
            # documents XPU support for Arc and data-center GPUs, not iGPUs.
            if [[ "$ACCEL" == "cuda" ]]; then echo "yes"; else echo "no"; fi
            ;;
        *)
            die "unsupported TRUSTA_SETUP_VLLM value: $mode (use auto / 1 / 0)"
            ;;
    esac
}

# The XPU check is killed after this long. Validated here rather than handed straight to
# `timeout`, which would otherwise fail with exit 125 and be reported as a failed XPU check.
resolve_xpu_check_timeout() {
    local raw="${TRUSTA_XPU_CHECK_TIMEOUT:-300}"
    if [[ "$raw" =~ ^[0-9]+$ ]] && ((raw > 0)); then
        echo "$raw"
        return 0
    fi
    echo "[setup_env] WARNING: ignoring invalid TRUSTA_XPU_CHECK_TIMEOUT='$raw' (want a positive integer)" >&2
    echo "300"
}

# uv is the one thing this script cannot build for itself, and until now it simply
# assumed it: on a machine without it the first `uv sync` failed with "command not
# found", which is a poor thing to say to someone who installed a one-click package
# precisely so they would not have to install a toolchain.
#
# Two failures, not one. It may be genuinely absent, or it may be installed and
# missing from PATH -- the launcher is started from a desktop icon, and a desktop's
# PATH is not a login shell's, so ~/.local/bin is routinely not on it.
# curl, wget, then python3, into a file. A clean Ubuntu Desktop has no curl -- measured,
# not assumed -- and reaching for it first is how an earlier version of this failed on
# exactly the machines it was written for. python3 is the backstop because GNOME depends
# on it, so a desktop that lacks it is not one this application was going to run on.
#
# Each of the three retries, and the first two give up on a stalled transfer rather than
# on a slow one: a deadline for the whole download is a demand for a minimum speed, and
# these files are tens of megabytes. (The Windows script had exactly that fault -- a
# 100-second default on a 137 MB asset -- so the same shape is avoided here.)
fetch_to() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        # --speed-time/--speed-limit: stop after 30s under 1 KB/s. --retry-all-errors
        # needs curl 7.71 (Ubuntu 22.04 ships 7.81); older ones ignore unknown options
        # with an error, so it is kept last and dropped on failure.
        curl -LsSf --retry 3 --retry-delay 2 --speed-limit 1024 --speed-time 30 \
             --retry-all-errors "$url" -o "$out" \
            || curl -LsSf --retry 3 --retry-delay 2 --speed-limit 1024 --speed-time 30 \
                    "$url" -o "$out"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 --waitretry=2 --read-timeout=30 -O "$out" "$url"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "$url" "$out" <<'FETCH'
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=120) as r:
    open(sys.argv[2], "wb").write(r.read())
FETCH
    else
        return 2
    fi
    [[ -s "$out" ]]
}

ensure_uv() {
    # A packaged installer carries its own uv, pinned and hashed when it was built, and
    # names it here. Taken before anything on the machine: which uv resolved the lock is
    # part of what a release is, and "whatever the user happened to have" is not an
    # answer. Unset in a checkout, where the machine's own uv is the right one.
    if [[ -n "${TRUSTA_UV:-}" && -x "${TRUSTA_UV}" ]]; then
        echo "[setup_env] using the uv the installer carries: $TRUSTA_UV"
        PATH="$(dirname "$TRUSTA_UV"):$PATH"
        export PATH
        return 0
    fi

    command -v uv >/dev/null 2>&1 && return 0

    local user_bin="${XDG_BIN_HOME:-$HOME/.local/bin}"
    if [[ -x "$user_bin/uv" ]]; then
        echo "[setup_env] uv found at $user_bin/uv but not on PATH; adding it"
        PATH="$user_bin:$PATH"
        export PATH
        return 0
    fi

    echo "[setup_env] uv is not installed; fetching it into $user_bin"
    # curl, wget, then python3. A clean Ubuntu Desktop has no curl -- measured, not
    # assumed -- and reaching for it first is how the previous version of this
    # function failed on exactly the machines it was written for. python3 is the
    # backstop because GNOME depends on it, so a desktop that lacks it is not one
    # this application was going to run on anyway.
    local script status
    script="$(mktemp)"
    fetch_to "https://astral.sh/uv/install.sh" "$script" && status=0 || status=$?
    if ((status == 2)); then
        rm -f "$script"
        die "no curl, wget or python3 to fetch uv with; install uv by hand: https://docs.astral.sh/uv/"
    fi
    if ((status != 0)); then
        rm -f "$script"
        die "could not download the uv installer from https://astral.sh/uv - check the network, or install uv by hand"
    fi
    if ! env UV_INSTALL_DIR="$user_bin" UV_UNMANAGED_INSTALL="$user_bin" sh "$script" >/dev/null; then
        rm -f "$script"
        die "the uv installer failed - see the error above, or install uv by hand"
    fi
    rm -f "$script"
    PATH="$user_bin:$PATH"
    export PATH
    command -v uv >/dev/null 2>&1 || die "installed uv into $user_bin but it is still not runnable"
    echo "[setup_env] uv $(uv --version 2>/dev/null | awk '{print $2}') installed"
}

# uv installs into UV_PROJECT_ENVIRONMENT when it is set, so the venv is not always ./.venv
venv_dir() {
    local venv="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"
    [[ "$venv" == /* ]] || venv="$PROJECT_ROOT/$venv"
    echo "$venv"
}

venv_python() {
    echo "$(venv_dir)/bin/python"
}

# auto (default) = install only what is missing; 1 = force reinstall; 0 = skip entirely
llama_mode() {
    local mode="${TRUSTA_INSTALL_LLAMA:-auto}"
    case "$mode" in
        auto|AUTO|"")            echo "auto" ;;
        1|true|TRUE|yes|YES|on|ON)   echo "force" ;;
        0|false|FALSE|no|NO|off|OFF) echo "skip" ;;
        *)
            die "unsupported TRUSTA_INSTALL_LLAMA value: $mode (use auto / 1 / 0)"
            ;;
    esac
}

# The value a setting has for the service: the process environment first, then .env (or
# .env.example when .env is absent, matching service/settings.py). Empty when it is unset.
env_setting() {
    local key="$1"
    if [[ -n "${!key:-}" ]]; then
        echo "${!key}"
        return 0
    fi
    local env_file value=""
    for env_file in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.example"; do
        [[ -f "$env_file" ]] || continue
        value="$(sed -n "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//p" \
            "$env_file" | tail -n 1 | tr -d '"'\''' | sed 's/[[:space:]]*$//')"
        break
    done
    [[ -n "$value" ]] && echo "${value/#\~/$HOME}"
    return 0
}

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise the
# directory setup_env builds into. .env is consulted too, since that is
# where the service itself reads the override from — including service/settings.py's
# fallback to .env.example when .env is absent, so setup and the service agree.
llama_binary_path() {
    local configured
    configured="$(env_setting LLAMA_SERVER_BINARY)"
    if [[ -n "$configured" ]]; then
        echo "$configured"
    else
        # ggml-org/llama-install.sh's ~/.local/bin/llama is deliberately not a fallback:
        # its CUDA builds ship a CPU backend with no vector ISA. Point LLAMA_SERVER_BINARY
        # at one explicitly if you really want it.
        echo "$LLAMA_BIN_DIR/llama-server"
    fi
}

# Echoes the path that actually holds a usable binary — the configured/default one, or
# whatever `llama-server` resolves to on PATH — and returns 1 when there is none. Callers
# report the path that was really found instead of the one that was merely expected.
resolve_llama_binary() {
    local configured on_path default
    configured="$(llama_binary_path)"
    if [[ -x "$configured" ]]; then
        echo "$configured"
        return 0
    fi
    # A configured path that does not exist must not end the search: setup_env may have just
    # built a good binary at the default location, and giving up here skipped the whole
    # verify-and-record step while the run still exited 0, leaving .env pointing at nothing.
    # The mismatch is reported by the caller - this runs in a subshell, so a warning here
    # cannot be de-duplicated and would be printed once per call.
    default="$LLAMA_BIN_DIR/llama-server"
    if [[ "$configured" != "$default" && -x "$default" ]]; then
        echo "$default"
        return 0
    fi
    on_path="$(command -v llama-server 2>/dev/null || true)"
    if [[ -n "$on_path" ]]; then
        echo "$on_path"
        return 0
    fi
    return 1
}

# The .env setting the backend reads first for each tool.
llama_tool_setting() {
    case "$1" in
        llama-quantize) echo "LLAMA_QUANTIZE_BIN" ;;
        llama-fit-params) echo "LLAMA_FIT_PARAMS_BIN" ;;
    esac
}

# Echoes, one per line, the tools the backend would fail to find next to the given server.
# Mirrors the backend's resolvers (ConversionManager._resolve_quantize_binary and
# GGUFEstimator._resolve_fit_binary): an explicit setting, the convert tree, the server's own
# directory, or a unified `llama` that runs both as subcommands.
missing_llama_tools() {
    local server="$1" tool configured dir
    case "$(basename "$server")" in
        llama) return 0 ;;
    esac
    for tool in "${LLAMA_TOOLS[@]}"; do
        configured="$(env_setting "$(llama_tool_setting "$tool")")"
        [[ -n "$configured" && -f "$configured" && -x "$configured" ]] && continue
        for dir in "$LLAMA_CONVERT_DIR/build/bin" "$LLAMA_CONVERT_DIR/bin" "$LLAMA_CONVERT_DIR" \
                   "$(dirname "$server")"; do
            [[ -f "$dir/$tool" && -x "$dir/$tool" ]] && continue 2
        done
        echo "$tool"
    done
}

# Whether a binary lives in the directory setup_env manages, so it may be rebuilt in place.
in_llama_bin_dir() {
    local dir
    dir="$(cd -- "$(dirname -- "$1")" 2>/dev/null && pwd -P)" || return 1
    [[ -d "$LLAMA_BIN_DIR" && "$dir" == "$(cd -- "$LLAMA_BIN_DIR" && pwd -P)" ]]
}

# Whether the binary is one the user chose through LLAMA_SERVER_BINARY (environment or .env)
# and setup_env must not replace. Anything else - our own install, or a prebuilt that merely
# happened to be on PATH - is setup_env's to replace.
is_user_llama_binary() {
    local configured
    configured="$(env_setting LLAMA_SERVER_BINARY)"
    [[ -n "$configured" && -x "$configured" && "$configured" == "$1" ]] &&
        ! in_llama_bin_dir "$1"
}

# The sparse checkout is only useful if the scripts the conversion code calls are actually there.
have_convert_tooling() {
    [[ -f "$LLAMA_CONVERT_DIR/convert_hf_to_gguf.py" ]] &&
        [[ -f "$LLAMA_CONVERT_DIR/convert_lora_to_gguf.py" ]] &&
        [[ -d "$LLAMA_CONVERT_DIR/gguf-py" ]]
}

# Record what setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. The file ends up with
# exactly one line per key: an override that resolves is kept as it is, one that
# points at nothing is replaced by what was actually verified, and duplicates are collapsed.
record_env_setting() {
    local key="$1" resolved="$2" env_file="$PROJECT_ROOT/.env"
    [[ -n "$resolved" ]] || return 0
    if [[ ! -f "$env_file" ]]; then
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        echo "[setup_env] no .env, so $key was not recorded (resolved: $resolved)"
        return 0
    fi
    # Exactly one line for the key survives this. Appending whenever no *live* line
    # matched left .env.example's commented example sitting above a second entry, so the file
    # showed the key twice; and once two live lines exist python-dotenv silently keeps the last,
    # which is not the one someone editing the first would expect.
    local live="^[[:space:]]*$key[[:space:]]*="
    local commented="^[[:space:]]*#[[:space:]]*$key[[:space:]]*="
    local stripped previous dropped total bom
    stripped="$(mktemp)"
    # Drop a leading BOM first: it hides a first-line key from every ^ anchor below, and
    # python-dotenv would read it as part of that key's name.
    bom="$(printf '\357\273\277')"
    sed "1s/^$bom//" "$env_file" > "$stripped"

    total="$(grep -cE "$live" "$stripped" || true)"
    dropped=$(( total > 0 ? total - 1 : 0 ))
    previous="$(grep -m1 -E "$live" "$stripped" 2>/dev/null |
        sed -E "s/$live[[:space:]]*//" | tr -d '"'"'"'' | sed 's/[[:space:]]*$//' || true)"

    local rewritten
    rewritten="$(mktemp)"
    awk -v setting="$key=$resolved" -v live="$live" -v commented="$commented" '
        {
            if ($0 ~ live) {
                if (placed) { next }        # a duplicate: drop it
                buf[++n] = setting
                placed = 1
                next
            }
            buf[++n] = $0
        }
        END {
            if (!placed) {
                # Take over the commented example in place, so the key stays where the template
                # put it instead of the file growing a second block saying the same thing.
                for (i = 1; i <= n; i++) {
                    if (buf[i] ~ commented) { buf[i] = setting; placed = 1; break }
                }
            }
            if (!placed) {
                buf[++n] = ""
                buf[++n] = "# Recorded by setup_env: the llama path it resolved, so the service does not"
                buf[++n] = "# have to re-resolve it from a possibly different PATH."
                buf[++n] = setting
            }
            for (i = 1; i <= n; i++) print buf[i]
        }
    ' "$stripped" > "$rewritten"

    # cat rather than mv: keeps the original inode, owner and mode, which matter for a .env.
    cat "$rewritten" > "$env_file"
    rm -f "$stripped" "$rewritten"

    if [[ "$previous" == "$resolved" ]]; then
        echo "[setup_env] .env already records $key=$resolved"
    elif [[ -n "$previous" ]]; then
        echo "[setup_env] .env $key updated to $resolved (was $previous)"
    else
        echo "[setup_env] recorded $key=$resolved in .env"
    fi
    (( dropped > 0 )) &&
        echo "[setup_env] removed $dropped duplicate $key line(s) from .env"
    return 0
}

record_llama_binary() { record_env_setting LLAMA_SERVER_BINARY "$1"; }

# llama is built from source on Linux, not installed from ggml-org/llama-install.sh.
# Two reasons. First, llama.cpp publishes no Linux CUDA binary at all - every CUDA asset in its
# releases is win-* (checked on b10107, b10644 and b10665) - so there is nothing to download.
# Second, install.sh's CUDA/ROCm presets never set LLAMA_INSTALL_FLAGS, so the binary it gives
# you is a real CUDA build whose *CPU* backend has no vector ISA: 1.69x slower on Linux once any
# weight is computed on the CPU. https://github.com/samhong5668/llama-bench-lab
#
# GGML_NATIVE=ON compiles for this machine's CPU, which is correct here because the build happens
# on the machine that will run it. It also means the ISA is right by construction, so the probe
# reporting "unverified" for a source build is expected rather than a warning sign.
#
# This is the one step that needs sudo (apt) and a few minutes of CPU. Windows does not: there
# the llama.cpp release ships a binary whose CPU backend is already correct.
# Install apt packages, but only when that can be done without stopping to ask.
#
# Default (TRUSTA_APT_INSTALL unset): only as root, or where sudo is already passwordless -
# containers, CI and one-click installers get through untouched, while an ordinary desktop is
# left alone, because setup_env should not rewrite the system packages of a machine that did
# not invite it to. TRUSTA_APT_INSTALL=1 opts in and may prompt for the password, at a terminal
# or in polkit's dialog; TRUSTA_APT_INSTALL=0 forbids installing outright.
#
# The polkit path is what makes the desktop installer work. Double-clicked from a file manager
# there is no terminal, so `sudo` has nowhere to ask and every build tool -- cmake and
# build-essential -- silently stayed missing and llama.cpp was never built. pkexec puts the
# prompt on screen instead, which is how a desktop application is supposed to ask.
#
# Returns 1 without doing anything when it is not allowed to act, so callers fall back to
# printing the command.
apt_install() {
    local opt="${TRUSTA_APT_INSTALL:-auto}"
    local -a pkgs=("$@") sudo_cmd=()
    [[ "$opt" =~ ^(0|false|no|off)$ ]] && return 1
    ((${#pkgs[@]})) || return 0
    command -v apt-get >/dev/null 2>&1 || return 1

    if ((EUID != 0)); then
        if sudo -n true 2>/dev/null; then
            sudo_cmd=(sudo -n)
        elif [[ ! "$opt" =~ ^(1|true|yes|on)$ ]]; then
            return 1                 # not opted in, and the password is not free
        elif [[ -t 0 ]]; then
            sudo_cmd=(sudo)          # a terminal can answer the prompt
        elif command -v pkexec >/dev/null 2>&1 &&
            [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
            # pkexec's own dialog. It runs the command with a minimal environment, so
            # DEBIAN_FRONTEND has to be passed through env below rather than exported.
            sudo_cmd=(pkexec)
        else
            return 1
        fi
    fi

    echo "[setup_env] installing missing packages: ${pkgs[*]}"
    # env, not a bare assignment: sudo drops the caller's environment.
    if ! "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${pkgs[@]}"; then
        echo "[setup_env] apt-get install failed; refreshing the index and retrying once" >&2
        "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update -qq || return 1
        "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${pkgs[@]}" || return 1
    fi
    return 0
}

# The build tools, as apt package names. Echoed one per line so the caller can re-ask after an
# install instead of assuming apt did what it said.
missing_build_tools() {
    command -v cmake >/dev/null 2>&1 || echo cmake
    missing_compilers
}

# A C and a C++ compiler, which build-essential brings together. Both are needed: llama.cpp
# and causal-conv1d are C++, and a machine can have gcc without g++.
missing_compilers() {
    if ! { command -v cc || command -v gcc; } >/dev/null 2>&1 ||
        ! { command -v c++ || command -v g++; } >/dev/null 2>&1; then
        echo build-essential
    fi
}

# What the Vulkan backend needs, as apt package names.
#
# Asked of cmake in a throwaway project that mirrors the two find_package calls in
# ggml/src/ggml-vulkan/CMakeLists.txt, rather than hand-listing binaries to look for. Hand-
# listing is what went wrong first: glslc plus the headers looked sufficient, the check passed,
# and the real configure then failed on SPIRV-Headers - advice that is specific and wrong is
# worse than none. Mirroring the find_package calls cannot drift like that.
missing_vulkan_packages() {
    local probe out
    probe="$(mktemp -d)"
    cat > "$probe/CMakeLists.txt" <<'VKPROBE_EOF'
cmake_minimum_required(VERSION 3.19)
# CXX enabled, not NONE: FindVulkan needs the toolchain's word size to locate the library, and
# with no language it reports libvulkan-dev missing on a host that has it.
project(vkprobe CXX)
if (DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()
find_package(Vulkan COMPONENTS glslc QUIET)
find_package(SPIRV-Headers CONFIG QUIET)
if (NOT Vulkan_FOUND)
    message("TRUSTA_MISSING libvulkan-dev")
endif()
if (NOT Vulkan_glslc_FOUND)
    message("TRUSTA_MISSING glslc")
endif()
if (NOT SPIRV-Headers_FOUND)
    message("TRUSTA_MISSING spirv-headers")
endif()
VKPROBE_EOF
    out="$(cmake -S "$probe" -B "$probe/build" 2>&1 || true)"
    rm -rf "$probe"
    printf '%s\n' "$out" | sed -n 's/^TRUSTA_MISSING //p'
}

# Usage: build_llama_from_source [dest] [target...]
# Defaults to the full install: llama-server and every tool, into LLAMA_BIN_DIR. A tools-only
# build passes its own directory and targets, and then no llama-server is required.
build_llama_from_source() {
    local ver="$LLAMA_VERSION"
    local dest="${1:-$LLAMA_BIN_DIR}"
    local -a targets=("${@:2}")
    ((${#targets[@]})) || targets=(llama-server "${LLAMA_TOOLS[@]}")
    local -a missing=()
    mapfile -t missing < <(missing_build_tools)
    if ((${#missing[@]})); then
        # Re-ask after installing rather than trusting apt's exit code: the packages have to
        # actually put the tools on PATH for the build that follows.
        apt_install "${missing[@]}" && mapfile -t missing < <(missing_build_tools)
    fi
    if ((${#missing[@]})); then
        echo "[setup_env] WARNING: cannot build llama, missing: ${missing[*]}" >&2
        echo "[setup_env]   sudo apt install ${missing[*]}" >&2
        echo "[setup_env]   or let setup_env do it: TRUSTA_APT_INSTALL=1" >&2
        return 1
    fi

    local cmake_args=(
        -S "$LLAMA_BUILD_DIR/src" -B "$LLAMA_BUILD_DIR/build"
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE=ON
        # Look for the shared libraries next to the executable. Without this the binary keeps
        # an RPATH into the build tree, so it only runs while that tree still exists - deleting
        # it, or moving just the binary, breaks the install.
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_APP=OFF
        -DLLAMA_BUILD_UI=OFF
        -DLLAMA_CURL=OFF
    )
    # Every backend flag is passed on every run, ON or OFF. cmake caches what it was given, so
    # only *adding* -DGGML_CUDA=ON for the cuda case would leave it ON in the cache when the
    # backend later changes to cpu - the build would keep producing CUDA while the user asked
    # for something else, silently. Verified: reconfiguring without a flag keeps the cached ON.
    local want_cuda=OFF want_vulkan=OFF
    case "$LLAMA_BACKEND" in
        cuda)
            # nvcc is not on PATH in a default CUDA install.
            if ! command -v nvcc >/dev/null 2>&1; then
                local d
                for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
                    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; break; fi
                done
            fi
            if ! command -v nvcc >/dev/null 2>&1; then
                echo "[setup_env] WARNING: nvcc not found, so the CUDA build cannot be made." >&2
                echo "[setup_env]   install the CUDA Toolkit, or set TRUSTA_LLAMA_BACKEND=cpu" >&2
                return 1
            fi
            want_cuda=ON
            ;;
        vulkan)
            local -a vk_missing=()
            mapfile -t vk_missing < <(missing_vulkan_packages)
            if ((${#vk_missing[@]})); then
                apt_install "${vk_missing[@]}" &&
                    mapfile -t vk_missing < <(missing_vulkan_packages)
            fi
            if ((${#vk_missing[@]})); then
                echo "[setup_env] WARNING: cannot build the Vulkan backend, missing: ${vk_missing[*]}" >&2
                echo "[setup_env]   sudo apt install ${vk_missing[*]}" >&2
                echo "[setup_env]   or let setup_env do it: TRUSTA_APT_INSTALL=1" >&2
                echo "[setup_env]   or build without a GPU: TRUSTA_LLAMA_BACKEND=cpu" >&2
                return 1
            fi
            want_vulkan=ON
            ;;
        cpu) ;;
    esac
    cmake_args+=(-DGGML_CUDA="$want_cuda" -DGGML_VULKAN="$want_vulkan")

    # $ORIGIN, so the binary finds the ggml libraries beside it wherever it is moved.
    #
    # A toolkit assembled from wheels needs a second entry. It sits inside the virtual
    # environment, which is on no default search path, and the first symptom is not a
    # missing library at run time but a failed link: configuring succeeds, every object
    # compiles, and the last step -- where ld resolves libggml-cuda.so's own dependencies
    # to build llama-server -- reports
    #
    #   undefined reference to `cudaStreamCreateWithFlags@libcudart.so.13'
    #
    # -L answers that one, and the RPATH entry answers the run-time half that would
    # otherwise have failed later and less clearly. Both are computed here rather than
    # appended conditionally: cmake caches what it is given, so a value passed on one run
    # and omitted on the next stays in effect, which is the same trap the backend flags
    # above are written to avoid.
    #
    # A system toolkit needs neither, and does not get them: its lib64 is already on the
    # loader's path, and CUDA_HOME/lib exists only in the layout the wheels unpack.
    local rpath='$ORIGIN'
    if [[ "$want_cuda" == "ON" && -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib" ]]; then
        rpath="\$ORIGIN;$CUDA_HOME/lib"
        cmake_args+=(
            -DCMAKE_LIBRARY_PATH="$CUDA_HOME/lib"
            "-DCMAKE_EXE_LINKER_FLAGS=-L$CUDA_HOME/lib"
            "-DCMAKE_SHARED_LINKER_FLAGS=-L$CUDA_HOME/lib"
        )
    fi
    cmake_args+=("-DCMAKE_INSTALL_RPATH=$rpath")

    mkdir -p "$LLAMA_BUILD_DIR"
    ensure_llama_source "$ver" || return 1

    # No cross-check against a resolved revision any more, and none is needed: the archive was
    # requested by tag, so what is in the directory is what the tag points at. The stamp
    # ensure_llama_source writes is what makes a reused directory answerable for its version.

    # llama-server is what the service runs, but the backend also shells out to llama-quantize
    # (service/utils/conversion_manager.py) and llama-fit-params (gguf_estimator.py). Windows
    # gets all three for free from the release zip; here they have to be asked for, and both
    # resolvers look for them beside llama-server.
    echo "[setup_env] building ${targets[*]} $ver ($LLAMA_BACKEND, GGML_NATIVE=ON); this takes a few minutes"
    cmake "${cmake_args[@]}" || return 1
    cmake --build "$LLAMA_BUILD_DIR/build" --target "${targets[@]}" --parallel "$(nproc)" || return 1

    # Collect first, replace second: nothing in dest is touched until the build has produced
    # what it must, so a failed rebuild leaves the previous install working.
    local target produced
    local -a outputs=()
    for target in "${targets[@]}"; do
        produced="$(find "$LLAMA_BUILD_DIR/build" -type f -name "$target" -print -quit)"
        if [[ -z "$produced" ]]; then
            # llama-server missing means the build did not really succeed; the two tools are
            # optional enough that a warning is the right level.
            if [[ "$target" == "llama-server" ]]; then
                echo "[setup_env] WARNING: the build reported success but $target was not found" >&2
                return 1
            fi
            echo "[setup_env] note: $target was not produced, so that feature stays unavailable" >&2
            continue
        fi
        outputs+=("$produced")
    done
    if ((${#outputs[@]} == 0)); then
        echo "[setup_env] WARNING: the build reported success but none of ${targets[*]} was found" >&2
        return 1
    fi

    # Replace the directory rather than copying over it: a cuda build leaves libggml-cuda.so
    # behind, and the next copy would carry it forward into a cpu install, where $ORIGIN would
    # still find it. Same for files dropped between releases.
    rm -rf "$dest"
    mkdir -p "$dest"
    # Copy the binaries and the shared libraries they load at run time, so the build tree can be
    # deleted without breaking the install.
    cp -f "${outputs[@]}" "$dest/"
    # -type l as well as -type f: the loader asks for the SONAME (libggml-base.so.0), which is a
    # symlink to the real libggml-base.so.0.17.0. Copying only regular files leaves the binary
    # unable to find anything. cp -P keeps the links as links instead of duplicating the target.
    find "$LLAMA_BUILD_DIR/build" \( -type f -o -type l \) -name "*.so*" \
        -exec cp -Pf {} "$dest/" \; 2>/dev/null || true
    echo "[setup_env] llama built into $dest: $(basename -a "${outputs[@]}" | tr '\n' ' ')"
    return 0
}

# Verify what was installed instead of trusting it: scripts/llama_backend.py explains why
# "is it CUDA?" is not a sufficient question on its own.
verify_llama_binary() {
    local binary="$1" require="$2"
    local probe="$PROJECT_ROOT/scripts/llama_backend.py"
    local python="$PROJECT_ROOT/.venv/bin/python"
    [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] && python="$UV_PROJECT_ENVIRONMENT/bin/python"
    if [[ ! -f "$probe" ]]; then
        echo "[setup_env] WARNING: $probe not found, cannot verify the llama binary" >&2
        LLAMA_VERIFY="unverified (probe missing)"
        return 0
    fi
    if [[ ! -x "$python" ]]; then
        echo "[setup_env] WARNING: $python not found, cannot verify the llama binary" >&2
        LLAMA_VERIFY="unverified (venv python missing)"
        return 0
    fi
    local args=("-u" "$probe" "$binary")
    [[ -n "$require" ]] && args+=("--require" "$require")
    [[ "$LLAMA_ALLOW_FALLBACK" == "yes" ]] && args+=("--allow-degraded")
    if "$python" "${args[@]}"; then
        if [[ "$LLAMA_ALLOW_FALLBACK" == "yes" ]]; then
            LLAMA_VERIFY="accepted (fallback allowed)"
        else
            LLAMA_VERIFY="ok"
        fi
        return 0
    fi
    LLAMA_VERIFY="FAILED"
    return 1
}

# The same four options the GUI offers, so both paths speak one vocabulary.
show_llama_options() {
    cat >&2 <<'OPTIONS'

  How to proceed (re-run setup_env with one of these):
    1. retry after fixing the toolchain / CUDA Toolkit / network
         TRUSTA_INSTALL_LLAMA=1
    2. use the CPU-only build
         TRUSTA_INSTALL_LLAMA=1 TRUSTA_LLAMA_BACKEND=cpu
    3. use your own binary
         set LLAMA_SERVER_BINARY in .env to its path
    4. skip llama entirely
         TRUSTA_INSTALL_LLAMA=0
  Unattended runs can accept a degraded binary with TRUSTA_LLAMA_ALLOW_FALLBACK=1.

OPTIONS
}

# The Python scripts needed for GGUF conversion: no C++, nothing to compile.
#
# Copied out of the same extracted sources the binary is built from, so the two cannot be
# different revisions. This used to be a sparse, blobless checkout of four paths -- about
# 1.7MB against the archive's ~25MB -- which was the reason for the extra machinery. The
# archive is a few seconds either way on any connection that is about to download several
# gigabytes of CUDA wheels, and it is what removes git from the prerequisites.
get_llama_convert_tooling() {
    ensure_llama_source "$LLAMA_VERSION" || die "could not obtain the llama.cpp sources, so the GGUF convert tooling cannot be installed."

    echo "[setup_env] installing the convert tooling ($LLAMA_VERSION): ${CONVERT_PATHS[*]}"
    mkdir -p "$LLAMA_CONVERT_DIR"
    local path
    for path in "${CONVERT_PATHS[@]}"; do
        if [[ ! -e "$LLAMA_SRC_DIR/$path" ]]; then
            die "the llama.cpp $LLAMA_VERSION sources do not contain $path, which the GGUF conversion code calls. The pinned version may have moved or renamed it."
        fi
        rm -rf "${LLAMA_CONVERT_DIR:?}/$path"
        cp -a "$LLAMA_SRC_DIR/$path" "$LLAMA_CONVERT_DIR/$path"
    done
    printf '%s\n' "$LLAMA_VERSION" > "$LLAMA_CONVERT_DIR/$LLAMA_SRC_STAMP"
    echo "[setup_env] convert tooling ready (pure Python, no build step)"
}

ACCEL="${TRUSTA_ACCEL:-$(detect_accel)}"
echo "[setup_env] accelerator=$ACCEL"

case "$ACCEL" in
    cuda|xpu) ;;
    none)
        die "no supported GPU was found on this machine.

  Looked for: an NVIDIA GPU (nvidia-smi), and an Intel GPU (clinfo, /dev/dri, /dev/dxg).

  This build installs PyTorch for NVIDIA or Intel; there is no CPU-only build, so
  there is nothing to install here. If you do have one of those and it was simply not
  detected, say so and setup will use it:

      TRUSTA_ACCEL=cuda ./setup_env.sh
      TRUSTA_ACCEL=xpu  ./setup_env.sh"
        ;;
    *)
        die "unsupported accelerator: $ACCEL (use cuda or xpu)"
        ;;
esac

# Here rather than at the first `uv` call, for the same reason as everything else in
# this block: what cannot work should be reported before the long part, not during it.
ensure_uv

# Resolve and validate every input up front: `uv sync` below takes minutes and downloads
# gigabytes, so a typo in any of these must be rejected before that, not after it.
SETUP_VLLM="$(resolve_setup_vllm)"
LLAMA_MODE="$(llama_mode)"
LLAMA_BACKEND="$(resolve_llama_backend)"
# Unattended runs may accept a degraded llama; interactive ones should stop instead.
LLAMA_ALLOW_FALLBACK="no"
case "${TRUSTA_LLAMA_ALLOW_FALLBACK:-}" in
    1|true|TRUE|yes|YES|on|ON) LLAMA_ALLOW_FALLBACK="yes" ;;
esac
LLAMA_VERIFY="n/a"
LLAMA_TOOLS_STATUS="n/a"
XPU_CHECK_TIMEOUT="$(resolve_xpu_check_timeout)"

# vLLM is an extra of this project rather than a separate one, so it joins the single
# sync below. There are two builds of the same release: `vllm` (PyPI's, CUDA) beside
# `cuda`, and `vllm-xpu` (vLLM's own index) beside `xpu`; pyproject keeps them apart.
# Linux only -- neither has Windows wheels.
SYNC_EXTRAS=(--extra "$ACCEL")
VLLM_EXTRA="vllm"
[[ "$ACCEL" == "xpu" ]] && VLLM_EXTRA="vllm-xpu"
if [[ "$SETUP_VLLM" == "yes" ]]; then
    SYNC_EXTRAS+=(--extra "$VLLM_EXTRA")
fi

cd "$PROJECT_ROOT"

# uv's own CPython, not the distribution's. causal-conv1d and deepspeed compile C and
# CUDA extensions, which need Python's development headers, and a clean Ubuntu Desktop
# has the interpreter without them:
#
#   fatal error: Python.h: No such file or directory
#
# The headers are in python3-dev, which needs apt and therefore root -- and this
# product installs per-user precisely so that it does not. uv's standalone builds
# ship the headers, so asking for one removes the requirement instead of documenting
# it. Pinned to 3.12 rather than left open: the torch and vllm wheels this project
# resolves are built for it.
UV_PY=(--python 3.12 --python-preference only-managed)


# causal-conv1d compiles a CUDA extension and is declared no-build-isolation, so it
# builds against this project's environment -- which means torch and a CUDA compiler
# have to be installed *before* it is built, not alongside it. Its setup.py finds the
# compiler by looking for nvcc on PATH and under CUDA_HOME, and gives up with
#
#   nvcc was not found. Are you sure your environment has nvcc available?
#   NameError: name 'bare_metal_version' is not defined
#
# The compiler is not missing: nvidia-cuda-nvcc is in the resolution and lands at
# .../site-packages/nvidia/cu13/bin/nvcc. Nothing pointed at it. This worked on the
# machines it was developed on because both had a system CUDA toolkit -- WSL creates
# /usr/local/cuda, and the Windows host has the toolkit installed -- so setup.py found
# one by accident. On an ordinary machine with only a driver, it does not, and the
# failure arrives after several gigabytes have already been downloaded.
#
# The wheels carry a runtime, not a toolkit, and two things a toolkit has are missing
# from what they unpack. Both only matter at link time, so they surface after every
# .cu file has already compiled:
#
#   /usr/bin/ld: cannot find -lcudart: No such file or directory
#
#   1. torch's CUDAExtension passes -L$CUDA_HOME/lib64. The wheels use lib/.
#   2. A toolkit ships libcudart.so beside libcudart.so.13 for the linker to follow.
#      The wheels ship only the versioned file, because nothing links against them --
#      they are loaded by name at run time.
#
# Both are made good inside the virtual environment, which this script owns and
# recreates, so nothing outside it is touched.
# Where this machine's CUDA driver library is. Not in the wheels by design: the driver
# belongs to the kernel module, so it comes from the graphics driver, and its location
# is whatever that driver chose -- /usr/lib/x86_64-linux-gnu normally, /usr/lib/wsl/lib
# under WSL and a GPU-paravirtualised VM. ldconfig knows; the list is the backstop for
# a machine whose cache has not been refreshed since the driver was installed.
find_cuda_driver() {
    local line path
    line="$( { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } \
        | grep -m1 'libcuda\.so\.1 ' || true)"
    path="${line##*=> }"
    if [[ -n "$path" && -e "$path" ]]; then
        echo "$path"
        return 0
    fi
    for path in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 \
                /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/lib/wsl/lib/libcuda.so.1; do
        if [[ -e "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

complete_wheel_toolkit() {
    local root="$1" lib="$1/lib" so base driver
    [[ -d "$lib" ]] || return 0
    [[ -e "$root/lib64" ]] || ln -s lib "$root/lib64" || true
    for so in "$lib"/lib*.so.*; do
        [[ -f "$so" ]] || continue
        base="${so%%.so.*}.so"
        [[ -e "$base" ]] || ln -s "$(basename "$so")" "$base" || true
    done

    # CMake's FindCUDAToolkit builds its CUDA::cuda_driver target from an unversioned
    # libcuda.so, which a real toolkit ships as a stub under lib64/stubs. Without one
    # llama.cpp configures far enough to report "Including CUDA backend" and then stops:
    #
    #   CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:182 (target_link_libraries):
    #     Target "ggml-cuda" links to: CUDA::cuda_driver
    #     but the target was not found.
    #
    # Pointing the stub at the machine's real driver library is what a stub stands in
    # for anyway, and it is the only copy that is certain to match the running kernel
    # module.
    if [[ ! -e "$lib/stubs/libcuda.so" ]] && driver="$(find_cuda_driver)"; then
        mkdir -p "$lib/stubs" && ln -sf "$driver" "$lib/stubs/libcuda.so" || true
    fi
}

# Two passes: everything except causal-conv1d, then point CUDA_HOME at the nvcc that
# has just been installed and build it.
if [[ "$ACCEL" == "cuda" ]]; then
    echo "[setup_env] uv sync ${SYNC_EXTRAS[*]} (without causal-conv1d, so nvcc lands first)"
    uv sync "${UV_PY[@]}" "${SYNC_EXTRAS[@]}" --no-install-package causal-conv1d

    # nvcc from the pip packages, made usable as a toolkit. Needed whenever something is
    # compiled for CUDA: causal-conv1d when there is no prebuilt wheel, and the llama.cpp
    # CUDA build further down, always (llama.cpp publishes no Linux CUDA binary).
    # lib/python3.12/site-packages/nvidia/cu13/bin/nvcc is six levels down, so a
    # shallower -maxdepth silently finds nothing and the message blames the toolkit.
    nvcc_bin="$(find "$(venv_dir)/lib" -maxdepth 7 -type f -name nvcc -path '*/nvidia/*/bin/nvcc' 2>/dev/null | head -n 1)"
    if [[ -n "$nvcc_bin" ]]; then
        CUDA_HOME="$(dirname "$(dirname "$nvcc_bin")")"
        export CUDA_HOME
        export CUDA_PATH="$CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        echo "[setup_env] CUDA_HOME=$CUDA_HOME (nvcc from the pip package, no system toolkit needed)"
        complete_wheel_toolkit "$CUDA_HOME"
    elif command -v nvcc >/dev/null 2>&1; then
        echo "[setup_env] using the nvcc already on PATH: $(command -v nvcc)"
    else
        die "no nvcc after installing the CUDA packages, so neither causal-conv1d nor llama can be built.
  Looked for one under $(venv_dir)/lib/*/site-packages/nvidia/*/bin and on PATH.
  Install the CUDA Toolkit, or set TRUSTA_LLAMA_BACKEND=cpu and use an accelerator that does not need it."
    fi

    # The release pipeline builds causal-conv1d once, against exactly this lock's torch,
    # and ships the wheel in wheels/ beside this script's project. Installing it needs
    # no compiler, no toolkit and no administrator; compiling here is the fallback for a
    # checkout or payload without one (or with one that does not import), and needs all
    # three. --no-deps: everything it depends on is already in the environment.
    cc1d_version="$(sed -n '/^name = "causal-conv1d"$/{n;s/^version = "\(.*\)"$/\1/p}' "$PROJECT_ROOT/uv.lock")"
    cc1d_wheel="$(compgen -G "$PROJECT_ROOT/wheels/causal_conv1d-${cc1d_version}-*.whl" | head -n 1 || true)"
    cc1d_prebuilt=no
    if [[ -n "$cc1d_version" && -n "$cc1d_wheel" ]]; then
        if uv pip install --python "$(venv_dir)/bin/python" --no-deps "$cc1d_wheel" &&
            "$(venv_dir)/bin/python" -c 'import causal_conv1d' 2>/dev/null; then
            cc1d_prebuilt=yes
            echo "[setup_env] causal-conv1d: installed the prebuilt $(basename "$cc1d_wheel"); nothing to compile"
        else
            echo "[setup_env] WARNING: the prebuilt $(basename "$cc1d_wheel") did not install or import; compiling instead" >&2
        fi
    fi

    if [[ "$cc1d_prebuilt" != yes ]]; then
        # causal-conv1d's extension is C++ as well as CUDA, and nvcc hands the host code to the
        # system compiler. A desktop Ubuntu has none until build-essential is installed, and
        # without one the build stops deep inside torch's extension helper with
        #
        #   The current installed version of c++ (0.0.0) is less than the minimum required
        #   version by CUDA 13.0 (6.0.0).
        #
        # after the multi-gigabyte download has already happened. So ask for the compiler the way
        # the llama.cpp build does, and say plainly what is missing when it cannot be installed.
        mapfile -t missing < <(missing_compilers)
        if ((${#missing[@]})); then
            apt_install "${missing[@]}" && mapfile -t missing < <(missing_compilers)
        fi
        if ((${#missing[@]})); then
            die "causal-conv1d has to be compiled, and there is no C++ compiler.
      Install one and run setup again:  sudo apt install ${missing[*]}
      or let setup_env do it:           TRUSTA_APT_INSTALL=1"
        fi

        echo "[setup_env] uv sync ${SYNC_EXTRAS[*]} (building causal-conv1d)"
        uv sync "${UV_PY[@]}" "${SYNC_EXTRAS[@]}"
    fi
else
    echo "[setup_env] uv sync ${SYNC_EXTRAS[*]}"
    uv sync "${UV_PY[@]}" "${SYNC_EXTRAS[@]}"
fi

# XPU only: prove the install can actually compute. torch.xpu.is_available() can report True
# and still fail (or hang) on the first kernel when the wheel's oneAPI runtime does not match
# the installed Intel driver — see scripts/xpu_smoke.py. Run under a timeout to cover the hang.
XPU_CHECKED="n/a"

# What to do about an Intel runtime that cannot drive the GPU, said from here because the
# check can die before it says anything: on a stock 24.04 with a Panther Lake GPU, the
# archive's compute runtime (23.43) made torch segfault while enumerating the device --
# no message, exit 139. Intel's graphics PPA (26.x) fixed it. Same advice as
# xpu_smoke.py gives when it lives long enough to give it.
xpu_runtime_hint() {
    local packages="libze1 libze-intel-gpu1 intel-opencl-icd libze-dev" installed major
    installed="$(dpkg-query -W -f='${Version}' intel-opencl-icd 2>/dev/null || true)"
    if [[ -z "$installed" ]]; then
        echo "[setup_env] The Intel GPU compute runtime (Level Zero) is not installed:" >&2
        echo "[setup_env]   sudo apt install $packages" >&2
        return
    fi
    major="${installed%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]] && ((major >= 25)); then
        echo "[setup_env] The Intel GPU compute runtime ($installed) is current. Check that this" >&2
        echo "[setup_env] user is in the render group (sudo usermod -aG render \$USER, then log in" >&2
        echo "[setup_env] again) and that the xe or i915 kernel driver is loaded (lspci -k)." >&2
        return
    fi
    echo "[setup_env] The Intel GPU compute runtime ($installed) is Ubuntu's own build, which" >&2
    echo "[setup_env] predates recent Intel GPUs (Lunar Lake, Panther Lake, Arc B-series). On one" >&2
    echo "[setup_env] of those, install Intel's build, then open the application again:" >&2
    echo "[setup_env]   sudo add-apt-repository -y ppa:kobuk-team/intel-graphics" >&2
    echo "[setup_env]   sudo apt install $packages" >&2
}

if [[ "$ACCEL" == "xpu" ]]; then
    xpu_python="$(venv_python)"
    if [[ "${TRUSTA_SKIP_XPU_CHECK:-0}" == "1" ]]; then
        echo "[setup_env] skipping the XPU check (TRUSTA_SKIP_XPU_CHECK=1)"
        XPU_CHECKED="skipped"
    elif [[ ! -f "$PROJECT_ROOT/scripts/xpu_smoke.py" ]]; then
        # Keep this non-fatal and identical to setup_env.ps1: a missing prerequisite means an
        # incomplete checkout, not a broken driver, and it must not be reported as a passing check.
        echo "[setup_env] WARNING: scripts/xpu_smoke.py not found, skipping the XPU check" >&2
        XPU_CHECKED="skipped (xpu_smoke.py not found)"
    elif [[ ! -x "$xpu_python" ]]; then
        echo "[setup_env] WARNING: $xpu_python not found, skipping the XPU check" >&2
        XPU_CHECKED="skipped (venv python not found)"
    else
        echo "[setup_env] verifying the XPU install (real GEMM + training step, timeout ${XPU_CHECK_TIMEOUT}s)"
        set +e
        timeout -s KILL "$XPU_CHECK_TIMEOUT" "$xpu_python" -u "$PROJECT_ROOT/scripts/xpu_smoke.py"
        xpu_check_rc=$?
        set -e
        # Killed by a signal (a crash, or the timeout's KILL): the check could not explain
        # itself, so say what it would have. A plain failure has already printed this.
        if [[ $xpu_check_rc -gt 128 ]]; then
            xpu_runtime_hint
        fi
        if [[ $xpu_check_rc -eq 137 ]]; then
            die "the XPU check did not finish within ${XPU_CHECK_TIMEOUT}s: kernel compilation is hanging, which an outdated Intel GPU driver causes. Update the driver, then re-run (or set TRUSTA_SKIP_XPU_CHECK=1 to bypass)."
        elif [[ $xpu_check_rc -ne 0 ]]; then
            die "the XPU check failed (exit $xpu_check_rc); see the message above. Set TRUSTA_SKIP_XPU_CHECK=1 to bypass."
        fi
        XPU_CHECKED="ok (GEMM + training step)"
    fi
fi

# oneCCL, which XPU training initialises even on a single GPU, opens Level Zero by its
# unversioned name, libze_loader.so. Only libze-dev provides that name; libze1 carries
# libze_loader.so.1 alone. So a machine with a working GPU and no -dev package runs
# inference and fails every training job with "ze_data was not initialized" -- seen on
# two Panther Lake laptops. The AppImage asks for libze-dev on first launch; this covers
# a declined dialog and a source install without root: a link inside the environment,
# which whoever starts the service (run_service.sh, the launcher) puts on
# LD_LIBRARY_PATH. Verified there: oneCCL resolves the name through LD_LIBRARY_PATH first.
ZE_LINK_DIR="$(venv_dir)/lib/trusta-ze"
ZE_LOADER_STATUS="n/a"
if [[ "$ACCEL" == "xpu" ]]; then
    # The loader cache lists sonames only, so it names libze_loader.so.1 whether or not the
    # unversioned link exists; the link is looked for beside it, where the dynamic loader's
    # own fallback search would find it.
    ldconfig_bin="$(command -v ldconfig || echo /sbin/ldconfig)"
    ze_so1="$("$ldconfig_bin" -p 2>/dev/null | awk '/libze_loader\.so\.1 .*x86-64/ {print $NF; exit}')"
    if [[ -z "$ze_so1" ]]; then
        ZE_LOADER_STATUS="MISSING: Level Zero is not installed (sudo apt install libze1 libze-dev)"
    elif [[ -e "$(dirname "$ze_so1")/libze_loader.so" ]]; then
        rm -rf "$ZE_LINK_DIR"
        ZE_LOADER_STATUS="system (libze-dev)"
    else
        mkdir -p "$ZE_LINK_DIR"
        ln -sfn "$ze_so1" "$ZE_LINK_DIR/libze_loader.so"
        ZE_LOADER_STATUS="linked in the environment (no libze-dev)"
    fi
    echo "[setup_env] libze_loader.so for training: $ZE_LOADER_STATUS"
fi

if [[ "$SETUP_VLLM" == "yes" ]]; then
    echo "[setup_env] vLLM installed into $(venv_dir) via --extra $VLLM_EXTRA"
else
    echo "[setup_env] skipping vLLM (ACCEL=$ACCEL, TRUSTA_SETUP_VLLM=${TRUSTA_SETUP_VLLM:-auto})"
fi

# DeepSpeed's async NVMe path opens a descriptor per operation and never closes
# it (see scripts/patch_deepspeed_aio.py). This has to run after every install,
# because `uv sync` replaces site-packages wholesale.
DEEPSPEED_AIO_STATUS="not applicable"
if "$PROJECT_ROOT/.venv/bin/python" -c "import deepspeed" >/dev/null 2>&1; then
    if "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/patch_deepspeed_aio.py"; then
        DEEPSPEED_AIO_STATUS="fd close patched"
    else
        # Not fatal: training still runs, it just leaks descriptors on NVMe
        # offload. Loud, though -- a silent skip is how it comes back.
        echo "[setup_env] WARNING: the DeepSpeed aio fd patch did not apply; NVMe offload will leak file descriptors"
        DEEPSPEED_AIO_STATUS="NOT patched (see warning above)"
    fi
fi

if [[ "$LLAMA_MODE" == "skip" ]]; then
    echo "[setup_env] skipping llama (TRUSTA_INSTALL_LLAMA=0)"
    LLAMA_STATUS="skipped"
else
    # Convert tooling first. It is pure Python, independent of the binary, and cheap - doing it
    # before the binary means a binary failure below still leaves GGUF conversion working.
    if [[ "$LLAMA_MODE" != "force" ]] && have_convert_tooling; then
        echo "[setup_env] convert tooling already present at $LLAMA_CONVERT_DIR — leaving it alone"
        llama_convert_status="already present"
    else
        get_llama_convert_tooling
        llama_convert_status="fetched"
    fi

    # Binary
    existing_llama=""
    if [[ "$LLAMA_MODE" != "force" ]] && existing_llama="$(resolve_llama_binary)"; then
        # "Present" means the whole set the backend needs, not only the server: an install
        # from before the tools were built, or a server brought from elsewhere, would otherwise
        # be left without them for good, since every later run stops at this check.
        mapfile -t missing_tools < <(missing_llama_tools "$existing_llama")
        if ((${#missing_tools[@]} == 0)); then
            echo "[setup_env] llama binary already present at $existing_llama — leaving it alone (TRUSTA_INSTALL_LLAMA=1 to reinstall)"
            llama_bin_status="already present"
        elif ! is_user_llama_binary "$existing_llama"; then
            # An older install of ours, or a prebuilt found on PATH: replace it with the full,
            # pinned set in LLAMA_BIN_DIR, so the tools share the server's version and shared
            # libraries. A PATH binary is not deleted (it is not ours), only no longer used.
            # The build leaves the existing install alone until it succeeds.
            echo "[setup_env] llama binary at $existing_llama lacks ${missing_tools[*]}; installing the full set $LLAMA_VERSION into $LLAMA_BIN_DIR"
            if build_llama_from_source; then
                llama_bin_status="built $LLAMA_VERSION ($LLAMA_BACKEND), replacing $existing_llama (lacked ${missing_tools[*]})"
            else
                # Not fatal: the server still runs, only conversion / fit-params are unavailable.
                echo "[setup_env] WARNING: could not build the full set; keeping $existing_llama" >&2
                llama_bin_status="already present (could not build ${missing_tools[*]})"
            fi
        else
            # Set explicitly through LLAMA_SERVER_BINARY: never replaced. Build only the missing
            # tools into their own directory and point the backend at them, rather than writing
            # beside someone else's binary.
            echo "[setup_env] LLAMA_SERVER_BINARY=$existing_llama lacks ${missing_tools[*]}; building them into $LLAMA_TOOLS_DIR"
            if build_llama_from_source "$LLAMA_TOOLS_DIR" "${missing_tools[@]}"; then
                for tool in "${missing_tools[@]}"; do
                    [[ -x "$LLAMA_TOOLS_DIR/$tool" ]] &&
                        record_env_setting "$(llama_tool_setting "$tool")" "$LLAMA_TOOLS_DIR/$tool"
                done
                llama_bin_status="already present; built ${missing_tools[*]} into $LLAMA_TOOLS_DIR"
            else
                echo "[setup_env] WARNING: could not build ${missing_tools[*]}; GGUF quantize / fit-params stay unavailable" >&2
                llama_bin_status="already present (could not build ${missing_tools[*]})"
            fi
        fi
    elif build_llama_from_source; then
        llama_bin_status="built $LLAMA_VERSION ($LLAMA_BACKEND)"
    else
        # Exiting non-zero, not warning: the service has no usable binary either way, and a
        # launcher that only reads the exit code would otherwise report this as a success.
        show_llama_options
        die "could not build llama $LLAMA_VERSION ($LLAMA_BACKEND) - see the reason above. Nothing was recorded in .env."
    fi

    # From here the two paths are the same: whatever binary the service is about to be
    # given has to be the build we asked for. An already-present binary is checked too -
    # it may be a llama.app install from before this change.
    llama_bin="$(resolve_llama_binary || true)"
    if [[ -n "$llama_bin" ]]; then
        # Reported here rather than inside resolve_llama_binary: that runs in a subshell, so a
        # warning there cannot be de-duplicated and prints once per call.
        configured_bin="$(llama_binary_path)"
        if [[ "$llama_bin" != "$configured_bin" && ! -x "$configured_bin" ]]; then
            echo "[setup_env] LLAMA_SERVER_BINARY is set to $configured_bin, which does not exist; using $llama_bin"
        fi
        # A source build reports no ggml CPU variant, so only the backend is required here;
        # its ISA is right by construction. "cpu" is also what the probe reports for a GPU
        # build on a GPU-less host, so it is not required either.
        require=""
        [[ "$LLAMA_BACKEND" != "cpu" ]] && require="$LLAMA_BACKEND"
        if verify_llama_binary "$llama_bin" "$require"; then
            # Only record a binary that passed: writing a rejected one into .env would make
            # the next run treat it as already present and skip the check for good.
            record_llama_binary "$llama_bin"
        else
            show_llama_options
            die "the llama binary at $llama_bin did not pass verification (see above). It was not recorded in .env."
        fi
    else
        # The build either succeeded or already exited above, so getting here means nothing
        # usable could be found. Falling through silently would exit 0 having verified nothing.
        show_llama_options
        die "no llama binary could be resolved: LLAMA_SERVER_BINARY points at '$(llama_binary_path)', which does not exist, and llama-server is not on PATH. Clear that setting to use $LLAMA_BIN_DIR."
    fi

    LLAMA_STATUS="binary: $llama_bin_status / convert tooling: $llama_convert_status"
    # Said in the summary, not only in a note mid-build that scrolls away.
    mapfile -t missing_tools < <(missing_llama_tools "$llama_bin")
    if ((${#missing_tools[@]})); then
        LLAMA_TOOLS_STATUS="MISSING ${missing_tools[*]} (GGUF quantize / fit-params unavailable)"
    else
        LLAMA_TOOLS_STATUS="ok (${LLAMA_TOOLS[*]})"
    fi
fi

echo ""
echo "=========================================="
echo "  Environment setup complete"
echo "  Accelerator : $ACCEL"
echo "  XPU check   : $XPU_CHECKED"
echo "  ZE loader   : $ZE_LOADER_STATUS"
echo "  llama check : $LLAMA_VERIFY"
echo "  Service Dir : $SERVICE_DIR"
echo "  vLLM Setup  : ${TRUSTA_SETUP_VLLM:-auto} (installed: $SETUP_VLLM)"
echo "  DeepSpeed   : $DEEPSPEED_AIO_STATUS"
echo "  llama       : $LLAMA_STATUS"
echo "  llama tools : $LLAMA_TOOLS_STATUS"
echo "=========================================="
