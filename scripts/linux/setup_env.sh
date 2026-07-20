#!/usr/bin/env bash
# 偵測 GPU 種類，以對應的 cmake 旗標與 uv extra 建立 Python 環境。
# 用法：
#   ./setup_env.sh                         # 自動偵測
#   TRUSTA_ACCEL=xpu ./setup_env.sh        # 手動指定 cuda | xpu
#   TRUSTA_SETUP_VLLM=0 ./setup_env.sh     # 跳過 vLLM 隔離環境建置
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/src/service"
VLLM_SERVER_DIR="$SERVICE_DIR/inference/engines/vllm_server"

detect_accel() {
    if nvidia-smi &>/dev/null 2>&1; then
        echo "cuda"
    elif command -v clinfo &>/dev/null 2>&1 && clinfo 2>/dev/null | grep -qi "Intel"; then
        echo "xpu"
    elif [[ -d /dev/dri ]] && ls /dev/dri/renderD* &>/dev/null 2>&1; then
        # Intel Arc / iGPU 通常在 /dev/dri 下有 renderD 裝置
        echo "xpu"
    else
        echo "cuda"
    fi
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
            echo "[setup_env] 不支援的 TRUSTA_SETUP_VLLM 值: $mode（請使用 auto / 1 / 0）" >&2
            exit 1
            ;;
    esac
}

ACCEL="${TRUSTA_ACCEL:-$(detect_accel)}"
echo "[setup_env] accelerator=$ACCEL"

case "$ACCEL" in
    cuda) CMAKE_FLAGS="-DGGML_CUDA=on" ;;
    xpu)  CMAKE_FLAGS="-DGGML_VULKAN=on" ;;
    *)
        echo "[setup_env] 不支援的 accelerator: $ACCEL（請指定 cuda 或 xpu）" >&2
        exit 1
        ;;
esac

cd "$SERVICE_DIR"
echo "[setup_env] CMAKE_ARGS=$CMAKE_FLAGS  uv sync --extra $ACCEL"
CMAKE_ARGS="$CMAKE_FLAGS" uv sync --extra "$ACCEL"

if should_setup_vllm; then
    if [[ ! -f "$VLLM_SERVER_DIR/pyproject.toml" ]]; then
        echo "[setup_env] 找不到 vLLM 專案設定：$VLLM_SERVER_DIR/pyproject.toml" >&2
        exit 1
    fi

    echo "[setup_env] 建立 vLLM 隔離環境：$VLLM_SERVER_DIR"
    cd "$VLLM_SERVER_DIR"
    uv sync
else
    echo "[setup_env] 跳過 vLLM 隔離環境建置（ACCEL=$ACCEL, TRUSTA_SETUP_VLLM=${TRUSTA_SETUP_VLLM:-auto}）"
fi

echo ""
echo "=========================================="
echo "  環境設定完成"
echo "  Accelerator : $ACCEL"
echo "  CMAKE_ARGS  : $CMAKE_FLAGS"
echo "  Service Dir : $SERVICE_DIR"
echo "  vLLM Dir    : $VLLM_SERVER_DIR"
echo "  vLLM Setup  : ${TRUSTA_SETUP_VLLM:-auto}"
echo "=========================================="
