#!/usr/bin/env bash
# Detect the GPU type and build the Python environment with the matching uv extra
# (torch cuda / xpu variant).
# Usage:
#   ./setup_env.sh                          # auto-detect
#   TRUSTA_ACCEL=xpu ./setup_env.sh         # force cuda | xpu
#   TRUSTA_SETUP_VLLM=0 ./setup_env.sh      # skip building the isolated vLLM environment
#   TRUSTA_INSTALL_LLAMA=1 ./setup_env.sh   # install the prebuilt llama + GGUF convert tooling (no compiler)
#   TRUSTA_LLAMA_BACKEND=vulkan ./setup_env.sh  # force the generic Vulkan build (sees every Intel/AMD/NVIDIA card)
set -euo pipefail

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
            echo "[setup_env] unsupported TRUSTA_LLAMA_BACKEND: $backend (use auto | cuda | vulkan)" >&2
            exit 1
            ;;
    esac
}

should_setup_vllm() {
    local mode="${TRUSTA_SETUP_VLLM:-auto}"
    case "$mode" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        0|false|FALSE|no|NO|off|OFF)
            return 1
            ;;
        auto|AUTO|"")
            [[ "$ACCEL" == "cuda" ]]
            return
            ;;
        *)
            echo "[setup_env] unsupported TRUSTA_SETUP_VLLM value: $mode (use auto / 1 / 0)" >&2
            exit 1
            ;;
    esac
}

should_install_llama() {
    local mode="${TRUSTA_INSTALL_LLAMA:-0}"
    case "$mode" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        0|false|FALSE|no|NO|off|OFF|"")
            return 1
            ;;
        *)
            echo "[setup_env] unsupported TRUSTA_INSTALL_LLAMA value: $mode (use 1 / 0)" >&2
            exit 1
            ;;
    esac
}

# Official prebuilt llama (ggml-org/llama-install.sh): no compiler / CUDA toolkit / Vulkan SDK / CMake.
# LLAMA_BACKEND picks the llama build (decoupled from the torch accel): vulkan needs SKIP_CUDA=1
# to actually take the Vulkan path (otherwise the official installer grabs CUDA whenever an
# NVIDIA card is present).
install_llama_prebuilt() {
    local env_args=()
    if [[ "$LLAMA_BACKEND" == "vulkan" ]]; then
        env_args+=("SKIP_CUDA=1")
        echo "[setup_env] llama backend=vulkan -> SKIP_CUDA=1 (generic Vulkan, sees Intel/AMD/NVIDIA)"
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
    if [[ ! -d "$LLAMA_CONVERT_DIR/.git" ]]; then
        echo "[setup_env] sparse checkout of the convert tooling: $LLAMA_CPP_URL"
        rm -rf "$LLAMA_CONVERT_DIR"
        git init "$LLAMA_CONVERT_DIR"
        git -C "$LLAMA_CONVERT_DIR" remote add origin "$LLAMA_CPP_URL"
        git -C "$LLAMA_CONVERT_DIR" sparse-checkout set --no-cone "${CONVERT_PATHS[@]}"
    fi
    echo "[setup_env] fetching the convert scripts (pinned to $LLAMA_CPP_REF, only ${CONVERT_PATHS[*]})"
    git -C "$LLAMA_CONVERT_DIR" fetch --depth 1 --filter=blob:none origin "$LLAMA_CPP_REF"
    git -C "$LLAMA_CONVERT_DIR" checkout --detach FETCH_HEAD
    echo "[setup_env] convert tooling ready (pure Python, no build step)"
}

ACCEL="${TRUSTA_ACCEL:-$(detect_accel)}"
echo "[setup_env] accelerator=$ACCEL"

case "$ACCEL" in
    cuda|xpu) ;;
    *)
        echo "[setup_env] unsupported accelerator: $ACCEL (use cuda or xpu)" >&2
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"
echo "[setup_env] uv sync --extra $ACCEL"
uv sync --extra "$ACCEL"

if should_setup_vllm; then
    if [[ ! -f "$VLLM_SERVER_DIR/pyproject.toml" ]]; then
        echo "[setup_env] vLLM project config not found: $VLLM_SERVER_DIR/pyproject.toml" >&2
        exit 1
    fi

    echo "[setup_env] creating isolated vLLM environment: $VLLM_SERVER_DIR"
    cd "$VLLM_SERVER_DIR"
    uv sync
else
    echo "[setup_env] skipping isolated vLLM environment (ACCEL=$ACCEL, TRUSTA_SETUP_VLLM=${TRUSTA_SETUP_VLLM:-auto})"
fi

LLAMA_BACKEND="$(resolve_llama_backend)"
if should_install_llama; then
    if install_llama_prebuilt; then
        llama_bin_status="prebuilt ${TRUSTA_LLAMA_VERSION:-b10107} ($LLAMA_BACKEND)"
    else
        llama_bin_status="binary skipped (no compatible prebuilt / install failed)"
    fi
    get_llama_convert_tooling
    LLAMA_STATUS="$llama_bin_status + convert tooling"
else
    echo "[setup_env] skipping llama (set TRUSTA_INSTALL_LLAMA=1 if you need inference / conversion)"
    LLAMA_STATUS="skipped"
fi

echo ""
echo "=========================================="
echo "  Environment setup complete"
echo "  Accelerator : $ACCEL"
echo "  Service Dir : $SERVICE_DIR"
echo "  vLLM Dir    : $VLLM_SERVER_DIR"
echo "  vLLM Setup  : ${TRUSTA_SETUP_VLLM:-auto}"
echo "  llama       : $LLAMA_STATUS"
echo "=========================================="
