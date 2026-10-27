#!/usr/bin/env bash
# Install the Strata engine (engine="strata") from a git repository and record it in .env.
#
# The backend always starts Strata with --sleep-mode, which only the deep-sleep build has;
# Strata's own setup would otherwise download its ready-made engine, which does not. So this
# checks out the source, runs Strata's setup with --build (the engine and the image encoder
# are compiled from that source, and the model is installed), and refuses an engine that has
# no deep sleep. setup_env.sh runs it with TRUSTA_SETUP_STRATA=1; run it alone to install or
# update Strata later.
#
# Usage:
#   scripts/linux/install_strata.sh
#   TRUSTA_STRATA_REF=v0.1.47-sleep scripts/linux/install_strata.sh      # a tag, branch or commit
#   TRUSTA_STRATA_SETUP_ARGS="--model IQ3_XXS --data-dir /mnt/models" scripts/linux/install_strata.sh
#
#   TRUSTA_STRATA_REPO        repository (default: TRUSTA-SW-FAE/strata, private: needs git
#                             credentials for it, e.g. `gh auth login` and `gh auth setup-git`)
#   TRUSTA_STRATA_REF         branch, tag or commit (default: feat/deep-sleep)
#   TRUSTA_STRATA_DIR         where the checkout goes (default: strata/ beside this repository)
#   TRUSTA_STRATA_SETUP_ARGS  passed to Strata's setup as they are (default: --model IQ3_XXS,
#                             which downloads about 76 GB unless --gguf-dir names the files)
set -euo pipefail

LOG_TAG="[install_strata]"
die() {
    echo "$LOG_TAG $*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=env_file.sh
source "$SCRIPT_DIR/env_file.sh"

STRATA_REPO="${TRUSTA_STRATA_REPO:-https://github.com/TRUSTA-SW-FAE/strata.git}"
STRATA_REF="${TRUSTA_STRATA_REF:-feat/deep-sleep}"
STRATA_DIR="${TRUSTA_STRATA_DIR:-$(dirname -- "$PROJECT_ROOT")/strata}"
STRATA_SETUP_ARGS="${TRUSTA_STRATA_SETUP_ARGS:---model IQ3_XXS}"

command -v git >/dev/null 2>&1 || die "git is needed to fetch Strata: sudo apt install git"

# Check out STRATA_REF of STRATA_REPO in STRATA_DIR. An existing checkout is updated only when
# it is a clone of the same repository with nothing uncommitted: it may hold someone's work.
checkout_strata() {
    if [[ ! -e "$STRATA_DIR" ]]; then
        echo "$LOG_TAG cloning $STRATA_REPO into $STRATA_DIR"
        git clone --no-checkout "$STRATA_REPO" "$STRATA_DIR" ||
            die "could not clone $STRATA_REPO. A private repository needs git credentials for it (gh auth login, then gh auth setup-git)."
    elif [[ ! -d "$STRATA_DIR/.git" && ! -f "$STRATA_DIR/.git" ]]; then
        die "$STRATA_DIR exists and is not a git checkout; move it away or set TRUSTA_STRATA_DIR"
    else
        local origin
        origin="$(git -C "$STRATA_DIR" remote get-url origin 2>/dev/null || true)"
        [[ "$origin" == "$STRATA_REPO" ]] ||
            die "$STRATA_DIR is a clone of '${origin:-no origin}', not $STRATA_REPO; set TRUSTA_STRATA_DIR to another folder"
        if [[ -n "$(git -C "$STRATA_DIR" status --porcelain --untracked-files=no)" ]]; then
            die "$STRATA_DIR has uncommitted changes; commit or stash them first (nothing was changed)"
        fi
        echo "$LOG_TAG updating the checkout in $STRATA_DIR"
    fi
    git -C "$STRATA_DIR" fetch --tags origin "$STRATA_REF" ||
        die "could not fetch '$STRATA_REF' from $STRATA_REPO (TRUSTA_STRATA_REF: a branch, tag or commit)"
    git -C "$STRATA_DIR" checkout -q --detach FETCH_HEAD
    echo "$LOG_TAG Strata $STRATA_REF at $(git -C "$STRATA_DIR" rev-parse --short HEAD)"
}

checkout_strata

# Strata's setup makes its own .venv, compiles the engine into engine/ and writes the model's
# run config (strata-<model>.json). --no-start: the backend starts it, not setup.
read -r -a setup_args <<< "$STRATA_SETUP_ARGS"
echo "$LOG_TAG running Strata's setup: --build --yes --no-start ${setup_args[*]}"
(cd "$STRATA_DIR" && ./setup.sh --build --yes --no-start "${setup_args[@]}") ||
    die "Strata's setup failed (see above). Nothing was recorded in .env."

engine="$STRATA_DIR/engine/strata"
python="$STRATA_DIR/.venv/bin/python"
[[ -x "$engine" ]] || die "Strata's setup finished but left no engine at $engine"
[[ -x "$python" ]] || die "Strata's setup finished but left no interpreter at $python"
# The flag's name is in the binary of every build that parses it, and in no other: the
# ready-made upstream engine has none, and the backend would fail to start it.
grep -a -q -- "--sleep-mode" "$engine" ||
    die "$engine has no deep sleep (--sleep-mode): $STRATA_REF is not the deep-sleep source. Nothing was recorded in .env."

record_env_setting STRATA_HOME "$STRATA_DIR"
record_env_setting STRATA_PYTHON "$python"
record_env_setting STRATA_ENGINE_BINARY "$engine"

echo ""
echo "$LOG_TAG Strata is installed: $STRATA_DIR ($STRATA_REF, $(git -C "$STRATA_DIR" rev-parse --short HEAD))"
# Load with engine="strata" and model_path: the backend finds the run config whose --native
# is that GGUF file. Listed here so the paths to register are on screen.
echo "$LOG_TAG installed models (load with engine=\"strata\" and model_path = the GGUF):"
for cfg in "$STRATA_DIR"/strata-*.json; do
    [[ -e "$cfg" && "$cfg" != *.shared-settings.json ]] || continue
    gguf="$("$python" -c 'import json, sys
args = json.load(open(sys.argv[1])).get("args") or []
print(args[args.index("--native") + 1] if "--native" in args[:-1] else "")' "$cfg" 2>/dev/null || true)"
    echo "    ${gguf:-(no --native)}  <- $(basename -- "$cfg")"
done
