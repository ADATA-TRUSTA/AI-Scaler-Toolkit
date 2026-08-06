#!/usr/bin/env bash
set -euo pipefail

# Stop the LLM service

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_PORT="${TRUSTA_SERVICE_PORT:-8000}"

echo "Stopping LLM service..."

kill_found=false

for pattern in "python.* -m service.app" "service.app:app" "uvicorn service.app:app"; do
    if PIDS=$(pgrep -f "$pattern" || true) && [[ -n "$PIDS" ]]; then
        kill_found=true
        echo "✓ Found service processes, sending terminate signal: $PIDS"
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            kill "$pid" 2>/dev/null || true
        done <<< "$PIDS"
    fi
done

if [[ "$kill_found" == false ]]; then
    echo "⚠ No running service process found"
fi

sleep 2

if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -ti:"$SERVICE_PORT" || true)
    if [[ -n "$PORT_PID" ]]; then
        echo "⚠ Port $SERVICE_PORT still in use, force killing process: $PORT_PID"
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$PORT_PID"
        echo "✓ Leftover process force killed"
    else
        echo "✓ Port $SERVICE_PORT released"
    fi
else
    echo "⚠ lsof not installed, skipping port check"
fi

echo ""
echo "=========================================="
echo "  Service stopped"
echo "  Project Root: $PROJECT_ROOT"
echo "=========================================="