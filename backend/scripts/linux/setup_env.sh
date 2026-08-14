#!/usr/bin/env bash
# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# llama (prebuilt binary + GGUF convert tooling) is installed by default, but only
# what is actually missing — an existing binary or convert checkout is left alone.
# Usage:
#   ./setup_env.sh                          # auto-detect; install llama only if missing
#   TRUSTA_ACCEL=xpu ./setup_env.sh         # force cuda | xpu
#   TRUSTA_SETUP_VLLM=0 ./setup_env.sh      # skip building the isolated vLLM environment
#   TRUSTA_INSTALL_LLAMA=0 ./setup_env.sh   # skip llama entirely
#   TRUSTA_INSTALL_LLAMA=1 ./setup_env.sh   # force reinstall even if already present
#   TRUSTA_LLAMA_BACKEND=vulkan ./setup_env.sh  # force the generic Vulkan build (sees every Intel/AMD/NVIDIA card)
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
VLLM_SERVER_DIR="$SERVICE_DIR/inference/engines/vllm_server"

# GGUF convert tooling (convert_hf_to_gguf.py / convert_lora_to_gguf.py + conversion/ + gguf-py):
# only shipped with the llama.cpp sources, not with the prebuilt binary. A sparse + blobless
# fetch pulls just these paths (~1.7MB) — no C++, nothing to compile. The pinned revision is
# maintained by hand here.
LLAMA_CONVERT_DIR="$SERVICE_DIR/utils/llama.cpp"
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_REF="c0bc8591e8815c63cb01dd3f051a8b0df02501c9"  # = tag b10107 HEAD
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
        cuda|vulkan)
            echo "$backend"
            ;;
        *)
            die "unsupported TRUSTA_LLAMA_BACKEND: $backend (use auto | cuda | vulkan)"
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

# Where the llama binary is expected. An explicit LLAMA_SERVER_BINARY wins; otherwise
# the location the official installer writes to. .env is consulted too, since that is
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
        echo "$HOME/.local/bin/llama"
    fi
}

# Echoes the path that actually holds a usable binary — the configured/default one, or
# whatever `llama` resolves to on PATH — and returns 1 when there is none. Callers report
# the path that was really found instead of the one that was merely expected.
resolve_llama_binary() {
    local configured on_path
    configured="$(llama_binary_path)"
    if [[ -x "$configured" ]]; then
        echo "$configured"
        return 0
    fi
    on_path="$(command -v llama 2>/dev/null || true)"
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
# under a service manager with a different one, and then the two disagree. Only ever adds the
# key when it is absent — an explicit value is the user's, and is never touched.
record_llama_binary() {
    local resolved="$1" env_file="$PROJECT_ROOT/.env"
    [[ -n "$resolved" ]] || return 0
    if [[ ! -f "$env_file" ]]; then
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        echo "[setup_env] no .env, so LLAMA_SERVER_BINARY was not recorded (resolved: $resolved)"
        return 0
    fi
    if grep -qE '^[[:space:]]*LLAMA_SERVER_BINARY[[:space:]]*=' "$env_file"; then
        echo "[setup_env] .env already sets LLAMA_SERVER_BINARY, leaving it alone"
        return 0
    fi
    printf '
# Recorded by setup_env: the llama binary it resolved, so the service does not
# have to re-resolve it from a possibly different PATH.
LLAMA_SERVER_BINARY=%s
'         "$resolved" >> "$env_file"
    echo "[setup_env] recorded LLAMA_SERVER_BINARY=$resolved in .env"
}

# Official prebuilt llama (ggml-org/llama-install.sh): no compiler / CUDA toolkit / Vulkan SDK / CMake.
# LLAMA_BACKEND picks the llama build (decoupled from the torch accel). The Linux installer probes
# CUDA -> ROCm -> Vulkan -> CPU and keeps the first hit, so reaching the Vulkan build takes
# SKIP_CUDA=1 *and* SKIP_ROCM=1 — with only SKIP_CUDA an AMD host silently gets the ROCm build,
# which then cannot list or select devices the way the Vulkan build does.
install_llama_prebuilt() {
    local env_args=()
    if [[ "$LLAMA_BACKEND" == "vulkan" ]]; then
        env_args+=("SKIP_CUDA=1" "SKIP_ROCM=1")
        echo "[setup_env] llama backend=vulkan -> SKIP_CUDA=1 SKIP_ROCM=1 (generic Vulkan, sees Intel/AMD/NVIDIA)"
    else
        echo "[setup_env] llama backend=cuda (native CUDA, NVIDIA only)"
    fi
    local ver="${TRUSTA_LLAMA_VERSION:-b10107}"
    env_args+=("LLAMA_VERSION=$ver")
    echo "[setup_env] pinning LLAMA_VERSION=$ver"
    echo "[setup_env] downloading the official install.sh and installing the prebuilt llama"
    # Install only when a compatible prebuilt exists: a failure (no build for this platform)
    # warns and skips instead of aborting the whole setup
    if curl -fsSL "https://raw.githubusercontent.com/ggml-org/llama-install.sh/master/install.sh" \
        | env "${env_args[@]}" sh; then
        echo "[setup_env] llama installed: \$HOME/.local/bin/llama"
        return 0
    fi
    echo "[setup_env] WARNING: llama install failed — this platform may have no compatible prebuilt. Skipped the llama binary; point LLAMA_SERVER_BINARY in .env at your own build, or re-run later." >&2
    return 1
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
XPU_CHECK_TIMEOUT="$(resolve_xpu_check_timeout)"
if [[ "$SETUP_VLLM" == "yes" && ! -f "$VLLM_SERVER_DIR/pyproject.toml" ]]; then
    die "vLLM project config not found: $VLLM_SERVER_DIR/pyproject.toml"
fi

cd "$PROJECT_ROOT"
echo "[setup_env] uv sync --extra $ACCEL"
uv sync --extra "$ACCEL"

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
    echo "[setup_env] creating isolated vLLM environment: $VLLM_SERVER_DIR"
    cd "$VLLM_SERVER_DIR"
    uv sync
else
    echo "[setup_env] skipping isolated vLLM environment (ACCEL=$ACCEL, TRUSTA_SETUP_VLLM=${TRUSTA_SETUP_VLLM:-auto})"
fi

if [[ "$LLAMA_MODE" == "skip" ]]; then
    echo "[setup_env] skipping llama (TRUSTA_INSTALL_LLAMA=0)"
    LLAMA_STATUS="skipped"
else
    # Binary
    existing_llama=""
    if [[ "$LLAMA_MODE" != "force" ]] && existing_llama="$(resolve_llama_binary)"; then
        echo "[setup_env] llama binary already present at $existing_llama — leaving it alone (TRUSTA_INSTALL_LLAMA=1 to reinstall)"
        llama_bin_status="already present"
        record_llama_binary "$existing_llama"
    elif install_llama_prebuilt; then
        llama_bin_status="prebuilt ${TRUSTA_LLAMA_VERSION:-b10107} ($LLAMA_BACKEND)"
        record_llama_binary "$(resolve_llama_binary || true)"
    else
        llama_bin_status="binary skipped (no compatible prebuilt / install failed)"
    fi

    # Convert tooling
    if [[ "$LLAMA_MODE" != "force" ]] && have_convert_tooling; then
        echo "[setup_env] convert tooling already present at $LLAMA_CONVERT_DIR — leaving it alone"
        llama_convert_status="already present"
    else
        get_llama_convert_tooling
        llama_convert_status="fetched"
    fi

    LLAMA_STATUS="binary: $llama_bin_status / convert tooling: $llama_convert_status"
fi

echo ""
echo "=========================================="
echo "  Environment setup complete"
echo "  Accelerator : $ACCEL"
echo "  XPU check   : $XPU_CHECKED"
echo "  Service Dir : $SERVICE_DIR"
echo "  vLLM Dir    : $VLLM_SERVER_DIR"
echo "  vLLM Setup  : ${TRUSTA_SETUP_VLLM:-auto}"
echo "  llama       : $LLAMA_STATUS"
echo "=========================================="
