#!/usr/bin/env bash
set -euo pipefail

# Run the LLM service from the backend project root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# Not always inside the project. A one-click install provisions the environment into
# the user's data directory and tells uv so through UV_PROJECT_ENVIRONMENT -- which
# setup_env.sh honours and this did not, so running it from an installed copy reported
# "Python environment not found" about an environment that was there the whole time.
VENV_PATH="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
	echo "[run_service] Python environment not found at $VENV_PATH" >&2
	exit 1
fi

export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"
# Intel only: setup_env.sh links libze_loader.so here when libze-dev is absent, for the
# oneCCL that training starts. Inherited by the training process the service spawns.
if [[ -d "$VENV_PATH/lib/trusta-ze" ]]; then
	export LD_LIBRARY_PATH="$VENV_PATH/lib/trusta-ze${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$PROJECT_ROOT"

# Start the service - the app.py entrypoint reads all settings from service/settings.py
exec "$VENV_PATH/bin/python" -m service.app
