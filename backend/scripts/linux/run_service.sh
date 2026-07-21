#!/usr/bin/env bash
set -euo pipefail

# Run the LLM service from the backend project root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/service"
VENV_PATH="$SERVICE_DIR/.venv"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
	echo "[run_service] Python environment not found at $VENV_PATH" >&2
	exit 1
fi

export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

cd "$PROJECT_ROOT"

# 啟動服務 - 由 app.py 入口統一讀取 service/settings.py 設定
exec "$VENV_PATH/bin/python" -m service.app