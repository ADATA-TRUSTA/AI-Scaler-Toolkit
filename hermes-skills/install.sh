#!/usr/bin/env bash
# Linux/macOS wrapper around install.py -- all the logic lives there so the two
# platforms cannot drift. Any arguments are passed straight through.
#
#   ./install.sh
#   ./install.sh --comfy-home /opt/ComfyUI
#   ./install.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in "${PYTHON:-}" python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then
        # 3.9+ : pathlib and the typing syntax used by the engine
        if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            PY="$c"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: need python 3.9 or newer on PATH (or set \$PYTHON)." >&2
    exit 1
fi

echo "using $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
exec "$PY" install.py "$@"
