#!/usr/bin/env bash
set -euo pipefail

# Run the LLM service from the backend project root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
	echo "[run_service] Python environment not found at $VENV_PATH" >&2
	exit 1
fi

export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

cd "$PROJECT_ROOT"

# Start the service - the app.py entrypoint reads all settings from service/settings.py
exec "$VENV_PATH/bin/python" -m service.app