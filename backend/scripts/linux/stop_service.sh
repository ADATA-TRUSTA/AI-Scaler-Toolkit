#!/usr/bin/env bash
set -euo pipefail

# 停止 LLM 服務

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_PORT="${TRUSTA_SERVICE_PORT:-8000}"

echo "正在停止 LLM 服務..."

kill_found=false

for pattern in "python .* -m service.app" "service.app:app" "uvicorn service.app:app"; do
    if PIDS=$(pgrep -f "$pattern" || true) && [[ -n "$PIDS" ]]; then
        kill_found=true
        echo "✓ 已找到服務進程，送出終止信號: $PIDS"
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            kill "$pid" 2>/dev/null || true
        done <<< "$PIDS"
    fi
done

if [[ "$kill_found" == false ]]; then
    echo "⚠ 未找到運行中的服務進程"
fi

sleep 2

if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -ti:"$SERVICE_PORT" || true)
    if [[ -n "$PORT_PID" ]]; then
        echo "⚠ 端口 $SERVICE_PORT 仍被佔用，嘗試強制終止進程: $PORT_PID"
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$PORT_PID"
        echo "✓ 已強制終止殘留進程"
    else
        echo "✓ 端口 $SERVICE_PORT 已釋放"
    fi
else
    echo "⚠ 系統未安裝 lsof，略過端口檢查"
fi

echo ""
echo "=========================================="
echo "  服務已停止"
echo "  Project Root: $PROJECT_ROOT"
echo "=========================================="