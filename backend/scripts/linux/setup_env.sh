#!/usr/bin/env bash
# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (built from source + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   ./setup_env.sh                          # auto-detect; install llama only if missing
#   TRUSTA_ACCEL=xpu ./setup_env.sh         # force cuda | xpu
#   TRUSTA_SETUP_VLLM=0 ./setup_env.sh      # skip the vllm extra (CUDA hosts install it by default)
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
# only shipped with the llama.cpp sources, not built by the CMake targets. A sparse + blobless
# fetch pulls just these paths (~1.7MB) — no C++, nothing to compile. The pinned revision is
# maintained by hand here.
LLAMA_CONVERT_DIR="$SERVICE_DIR/utils/llama.cpp"
# Built binaries live under the project, not in ~/.local/bin: they are ours to place, pin
# and delete, and a per-project directory keeps two checkouts apart.
LLAMA_BIN_DIR="$SERVICE_DIR/utils/llama-bin"
LLAMA_BUILD_DIR="$SERVICE_DIR/utils/llama-build"
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp"
# The one place the pinned build is named. TRUSTA_LLAMA_VERSION overrides it.
LLAMA_VERSION_DEFAULT="b10107"
LLAMA_VERSION="${TRUSTA_LLAMA_VERSION:-$LLAMA_VERSION_DEFAULT}"
# Offline fallback for the convert tooling's commit. Normally resolved from the tag by
# resolve_llama_cpp_ref below, so the two cannot drift; kept because a machine with no network
# should still be able to finish a setup, and annotated with its tag so
# tests/unit/test_llama_paths_agree.py can check it belongs to the version being pinned.
LLAMA_CPP_REF_FALLBACK="c0bc8591e8815c63cb01dd3f051a8b0df02501c9"  # = tag b10107 HEAD
LLAMA_CPP_REF="$LLAMA_CPP_REF_FALLBACK"
# Whether the sha above came from the remote or is just the fallback. The build refuses to
# compile a checkout that disagrees with a *resolved* sha, but cannot judge one offline.
LLAMA_CPP_REF_RESOLVED="no"

# Ask the remote which commit the tag points at, so the convert scripts and the binary are
# always the same revision. Maintaining both by hand meant a version bump could update one and
# not the other, with no error - just convert scripts from a different build.
resolve_llama_cpp_ref() {
    local line sha
    # git ls-remote rather than the releases API: git is already required, and this needs no token.
    #
    # GIT_TERMINAL_PROMPT=0 and the timeout are both load-bearing: against a URL that answers
    # with an auth challenge, git blocks forever waiting for a username, which would hang the
    # whole setup instead of falling back. Measured - a wrong URL hung until killed.
    line="$(GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true timeout 30 \
        git ls-remote --tags "$LLAMA_CPP_URL" "refs/tags/$LLAMA_VERSION" 2>/dev/null | head -n 1)"
    sha="${line%%[[:space:]]*}"
    if [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
        LLAMA_CPP_REF="$sha"
        LLAMA_CPP_REF_RESOLVED="yes"
        echo "[setup_env] $LLAMA_VERSION resolves to $sha"
    else
        LLAMA_CPP_REF_RESOLVED="no"
        echo "[setup_env] could not resolve $LLAMA_VERSION from the remote; using the pinned fallback $LLAMA_CPP_REF_FALLBACK" >&2
    fi
}

CONVERT_PATHS=(convert_hf_to_gguf.py convert_lora_to_gguf.py conversion gguf-py)

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
        # WSL2 box falls through to the cuda default and installs the wrong torch extra.
        echo "xpu"
    else
        echo "cuda"
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

# uv installs into UV_PROJECT_ENVIRONMENT when it is set, so the venv is not always ./.venv
venv_python() {
    local venv="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"
    [[ "$venv" == /* ]] || venv="$PROJECT_ROOT/$venv"
    echo "$venv/bin/python"
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

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise the
# directory setup_env builds into. .env is consulted too, since that is
# where the service itself reads the override from — including service/settings.py's
# fallback to .env.example when .env is absent, so setup and the service agree.
llama_binary_path() {
    if [[ -n "${LLAMA_SERVER_BINARY:-}" ]]; then
        echo "$LLAMA_SERVER_BINARY"
        return
    fi
    local env_file from_env_file=""
    for env_file in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.example"; do
        [[ -f "$env_file" ]] || continue
        from_env_file="$(sed -n 's/^[[:space:]]*LLAMA_SERVER_BINARY[[:space:]]*=[[:space:]]*//p' \
            "$env_file" | tail -n 1 | tr -d '"'\''' | sed 's/[[:space:]]*$//')"
        break
    done
    if [[ -n "$from_env_file" ]]; then
        echo "${from_env_file/#\~/$HOME}"
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

# The sparse checkout is only useful if the scripts the conversion code calls are actually there.
have_convert_tooling() {
    [[ -f "$LLAMA_CONVERT_DIR/convert_hf_to_gguf.py" ]] &&
        [[ -f "$LLAMA_CONVERT_DIR/convert_lora_to_gguf.py" ]] &&
        [[ -d "$LLAMA_CONVERT_DIR/gguf-py" ]]
}

# Record the binary setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. The file ends up with
# exactly one LLAMA_SERVER_BINARY line: an override that resolves is kept as it is, one that
# points at nothing is replaced by what was actually verified, and duplicates are collapsed.
record_llama_binary() {
    local resolved="$1" env_file="$PROJECT_ROOT/.env"
    [[ -n "$resolved" ]] || return 0
    if [[ ! -f "$env_file" ]]; then
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        echo "[setup_env] no .env, so LLAMA_SERVER_BINARY was not recorded (resolved: $resolved)"
        return 0
    fi
    # Exactly one LLAMA_SERVER_BINARY line survives this. Appending whenever no *live* line
    # matched left .env.example's commented example sitting above a second entry, so the file
    # showed the key twice; and once two live lines exist python-dotenv silently keeps the last,
    # which is not the one someone editing the first would expect.
    local live='^[[:space:]]*LLAMA_SERVER_BINARY[[:space:]]*='
    local commented='^[[:space:]]*#[[:space:]]*LLAMA_SERVER_BINARY[[:space:]]*='
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
    awk -v setting="LLAMA_SERVER_BINARY=$resolved" -v live="$live" -v commented="$commented" '
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
                buf[++n] = "# Recorded by setup_env: the llama binary it resolved, so the service does not"
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
        echo "[setup_env] .env already records LLAMA_SERVER_BINARY=$resolved"
    elif [[ -n "$previous" ]]; then
        echo "[setup_env] .env LLAMA_SERVER_BINARY updated to $resolved (was $previous)"
    else
        echo "[setup_env] recorded LLAMA_SERVER_BINARY=$resolved in .env"
    fi
    (( dropped > 0 )) &&
        echo "[setup_env] removed $dropped duplicate LLAMA_SERVER_BINARY line(s) from .env"
    return 0
}

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
# not invite it to. TRUSTA_APT_INSTALL=1 opts in and may prompt for a sudo password when there
# is a terminal to answer it; TRUSTA_APT_INSTALL=0 forbids installing outright.
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
        elif [[ "$opt" =~ ^(1|true|yes|on)$ && -t 0 ]]; then
            sudo_cmd=(sudo)          # opted in, and there is a terminal for the prompt
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
    command -v git >/dev/null 2>&1 || echo git
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
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

build_llama_from_source() {
    local ver="$LLAMA_VERSION"
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
        '-DCMAKE_INSTALL_RPATH=$ORIGIN'
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

    mkdir -p "$LLAMA_BUILD_DIR"
    # Reuse a checkout that still has working git metadata, and just move it to the pinned ref;
    # the source tree is ~200MB and re-cloning it on every run is wasteful.
    #
    # Failures here are fatal rather than swallowed. A failed fetch can leave FETCH_HEAD from a
    # previous run, so `checkout FETCH_HEAD` would succeed on the *old* revision while the log
    # and .env both claim the new version - a build of the wrong source that reports success.
    if [[ -e "$LLAMA_BUILD_DIR/src/.git" ]] &&
        git -C "$LLAMA_BUILD_DIR/src" rev-parse --git-dir &>/dev/null; then
        echo "[setup_env] reusing the llama.cpp checkout at $LLAMA_BUILD_DIR/src"
        if ! git -C "$LLAMA_BUILD_DIR/src" fetch --depth 1 origin "refs/tags/$ver" &>/dev/null; then
            echo "[setup_env] WARNING: could not fetch $ver into the existing checkout at $LLAMA_BUILD_DIR/src." >&2
            echo "[setup_env]   Delete that directory to force a fresh clone, or check the network." >&2
            return 1
        fi
        if ! git -C "$LLAMA_BUILD_DIR/src" checkout --detach FETCH_HEAD &>/dev/null; then
            echo "[setup_env] WARNING: could not check out $ver in $LLAMA_BUILD_DIR/src (local changes?)." >&2
            echo "[setup_env]   Move or delete that directory, then re-run; nothing was deleted for you." >&2
            return 1
        fi
    else
        rm -rf "$LLAMA_BUILD_DIR/src"
        echo "[setup_env] cloning llama.cpp $ver (shallow)"
        git clone --depth 1 --branch "$ver" "$LLAMA_CPP_URL" "$LLAMA_BUILD_DIR/src" || return 1
    fi

    # Say out loud which revision is about to be compiled, and cross-check it against the tag
    # when the remote could be reached, so "built $ver" is never a claim about something else.
    local head
    head="$(git -C "$LLAMA_BUILD_DIR/src" rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$head" ]]; then
        echo "[setup_env] WARNING: could not read HEAD in $LLAMA_BUILD_DIR/src" >&2
        return 1
    fi
    resolve_llama_cpp_ref
    if [[ "$LLAMA_CPP_REF_RESOLVED" == "yes" && "$head" != "$LLAMA_CPP_REF" ]]; then
        echo "[setup_env] WARNING: $LLAMA_BUILD_DIR/src is at $head but $ver is $LLAMA_CPP_REF" >&2
        echo "[setup_env]   Delete that directory to force a fresh clone." >&2
        return 1
    fi
    echo "[setup_env] building from $head"

    # llama-server is what the service runs, but the backend also shells out to llama-quantize
    # (service/utils/conversion_manager.py) and llama-fit-params (gguf_estimator.py). Windows
    # gets all three for free from the release zip; here they have to be asked for, and both
    # resolvers look for them beside llama-server.
    local targets=(llama-server llama-quantize llama-fit-params)

    echo "[setup_env] building llama $ver ($LLAMA_BACKEND, GGML_NATIVE=ON); this takes a few minutes"
    cmake "${cmake_args[@]}" || return 1
    cmake --build "$LLAMA_BUILD_DIR/build" --target "${targets[@]}" --parallel "$(nproc)" || return 1

    # Replace the directory rather than copying over it: a cuda build leaves libggml-cuda.so
    # behind, and the next copy would carry it forward into a cpu install, where $ORIGIN would
    # still find it. Same for files dropped between releases.
    rm -rf "$LLAMA_BIN_DIR"
    mkdir -p "$LLAMA_BIN_DIR"
    # Copy the binaries and the shared libraries they load at run time, so the build tree can be
    # deleted without breaking the install.
    local target produced
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
        cp -f "$produced" "$LLAMA_BIN_DIR/"
    done
    # -type l as well as -type f: the loader asks for the SONAME (libggml-base.so.0), which is a
    # symlink to the real libggml-base.so.0.17.0. Copying only regular files leaves the binary
    # unable to find anything. cp -P keeps the links as links instead of duplicating the target.
    find "$LLAMA_BUILD_DIR/build" \( -type f -o -type l \) -name "*.so*" \
        -exec cp -Pf {} "$LLAMA_BIN_DIR/" \; 2>/dev/null || true
    echo "[setup_env] llama built: $LLAMA_BIN_DIR/llama-server"
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

# Fetch only the Python scripts needed for GGUF conversion (sparse + blobless shallow): no C++, nothing to compile.
get_llama_convert_tooling() {
    # Reuse any checkout whose git metadata still works, and just move it to the pinned revision.
    # `.git` is a *file*, not a directory, in a full clone left over from the old submodule layout;
    # testing for a directory would classify that as "no checkout" and delete the whole tree.
    # Require .git to exist first, or `git -C` would walk up and resolve the parent project repo.
    if [[ -e "$LLAMA_CONVERT_DIR/.git" ]] && git -C "$LLAMA_CONVERT_DIR" rev-parse --git-dir &>/dev/null; then
        echo "[setup_env] reusing the existing checkout: $LLAMA_CONVERT_DIR"
        git -C "$LLAMA_CONVERT_DIR" remote get-url origin &>/dev/null \
            || git -C "$LLAMA_CONVERT_DIR" remote add origin "$LLAMA_CPP_URL"
    else
        echo "[setup_env] sparse checkout of the convert tooling: $LLAMA_CPP_URL"
        rm -rf "$LLAMA_CONVERT_DIR"
        git init "$LLAMA_CONVERT_DIR"
        git -C "$LLAMA_CONVERT_DIR" remote add origin "$LLAMA_CPP_URL"
        git -C "$LLAMA_CONVERT_DIR" sparse-checkout set --no-cone "${CONVERT_PATHS[@]}"
    fi
    echo "[setup_env] fetching the convert scripts (pinned to $LLAMA_CPP_REF, only ${CONVERT_PATHS[*]})"
    git -C "$LLAMA_CONVERT_DIR" fetch --depth 1 --filter=blob:none origin "$LLAMA_CPP_REF"
    # Reusing a checkout means the checkout can be refused: git will not overwrite untracked
    # files. Say what to do instead of letting the raw git error stand — and do not "fix" it by
    # deleting the directory, which is the data loss this reuse path exists to avoid.
    if ! git -C "$LLAMA_CONVERT_DIR" checkout --detach FETCH_HEAD; then
        die "could not check out $LLAMA_CPP_REF in $LLAMA_CONVERT_DIR (see the git error above). Untracked files in that directory usually cause this. Move or delete the directory, then re-run; nothing was deleted for you."
    fi
    echo "[setup_env] convert tooling ready (pure Python, no build step)"
}

ACCEL="${TRUSTA_ACCEL:-$(detect_accel)}"
echo "[setup_env] accelerator=$ACCEL"

case "$ACCEL" in
    cuda|xpu) ;;
    *)
        die "unsupported accelerator: $ACCEL (use cuda or xpu)"
        ;;
esac

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
XPU_CHECK_TIMEOUT="$(resolve_xpu_check_timeout)"

# vLLM is an extra of this project rather than a separate one, so it joins the single
# sync below. It is Linux + CUDA only (no Windows wheels, and it pins torch), which is
# why resolve_setup_vllm only answers yes for ACCEL=cuda.
SYNC_EXTRAS=(--extra "$ACCEL")
if [[ "$SETUP_VLLM" == "yes" ]]; then
    SYNC_EXTRAS+=(--extra vllm)
fi

cd "$PROJECT_ROOT"
echo "[setup_env] uv sync ${SYNC_EXTRAS[*]}"
uv sync "${SYNC_EXTRAS[@]}"

# XPU only: prove the install can actually compute. torch.xpu.is_available() can report True
# and still fail (or hang) on the first kernel when the wheel's oneAPI runtime does not match
# the installed Intel driver — see scripts/xpu_smoke.py. Run under a timeout to cover the hang.
XPU_CHECKED="n/a"
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
        if [[ $xpu_check_rc -eq 137 ]]; then
            die "the XPU check did not finish within ${XPU_CHECK_TIMEOUT}s: kernel compilation is hanging, which an outdated Intel GPU driver causes. Update the driver, then re-run (or set TRUSTA_SKIP_XPU_CHECK=1 to bypass)."
        elif [[ $xpu_check_rc -ne 0 ]]; then
            die "the XPU check failed (exit $xpu_check_rc); see the message above. Set TRUSTA_SKIP_XPU_CHECK=1 to bypass."
        fi
        XPU_CHECKED="ok (GEMM + training step)"
    fi
fi

if [[ "$SETUP_VLLM" == "yes" ]]; then
    echo "[setup_env] vLLM installed into $PROJECT_ROOT/.venv via --extra vllm"
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
        # Resolve the tag to a commit first: the convert scripts have to be the same revision
        # as the binary.
        resolve_llama_cpp_ref
        get_llama_convert_tooling
        llama_convert_status="fetched"
    fi

    # Binary
    existing_llama=""
    if [[ "$LLAMA_MODE" != "force" ]] && existing_llama="$(resolve_llama_binary)"; then
        echo "[setup_env] llama binary already present at $existing_llama — leaving it alone (TRUSTA_INSTALL_LLAMA=1 to reinstall)"
        llama_bin_status="already present"
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
fi

echo ""
echo "=========================================="
echo "  Environment setup complete"
echo "  Accelerator : $ACCEL"
echo "  XPU check   : $XPU_CHECKED"
echo "  llama check : $LLAMA_VERIFY"
echo "  Service Dir : $SERVICE_DIR"
echo "  vLLM Setup  : ${TRUSTA_SETUP_VLLM:-auto} (installed: $SETUP_VLLM)"
echo "  DeepSpeed   : $DEEPSPEED_AIO_STATUS"
echo "  llama       : $LLAMA_STATUS"
echo "=========================================="
