#!/usr/bin/env bash
# Install the Strata engine (engine="strata") from a strata-sleep package and record it in .env.
#
# The backend always starts Strata with --sleep-mode, which only the deep-sleep build has;
# Strata's own setup would otherwise download its ready-made engine, which does not. The
# deep-sleep build comes as a package (TRUSTA-SW-FAE/strata-sleep releases: the engine already
# compiled, plus Strata's Python side), so nothing is cloned or compiled here and no git
# credentials are needed. This unpacks the package, checks it against its own manifest, makes
# its venv, runs Strata's setup (which installs the model and writes its run config) and
# refuses an engine that has no deep sleep. setup_env.sh runs it with TRUSTA_SETUP_STRATA=1;
# run it alone to install or update Strata later.
#
# Usage:
#   scripts/linux/install_strata.sh
#   TRUSTA_STRATA_PACKAGE=~/Downloads/strata-sleep-0.2.7-sleep.1-linux-x86_64.tar.gz scripts/linux/install_strata.sh
#   TRUSTA_STRATA_SETUP_ARGS="--model IQ3_XXS --gguf-dir /mnt/models/IQ3_XXS" scripts/linux/install_strata.sh
#
#   TRUSTA_STRATA_PACKAGE     the package: its .tar.gz or the folder it unpacks to (default: the
#                             newest vendor/strata-sleep-*-linux-x86_64.tar.gz in this repository,
#                             where the installer's payload carries it)
#   TRUSTA_STRATA_DIR         where it is installed (default: strata/ beside this repository)
#   TRUSTA_STRATA_SETUP_ARGS  passed to Strata's setup as they are (default: --model IQ3_XXS,
#                             which downloads about 76 GB unless --gguf-dir names the files)
#   TRUSTA_STRATA_GGUF_DIR    the model's GGUF files already on disk: setup's --gguf-dir, as one
#                             argument (a path may hold spaces, which _SETUP_ARGS splits on)
#   TRUSTA_STRATA_DATA_DIR    where Strata keeps what it derives from a model (the expert pack,
#                             the draft layer): setup's --data-dir, also as one argument. Recorded
#                             with --set-default-data-dir, which moves nothing that already exists
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

STRATA_PACKAGE="${TRUSTA_STRATA_PACKAGE:-}"
STRATA_DIR="${TRUSTA_STRATA_DIR:-$(dirname -- "$PROJECT_ROOT")/strata}"
STRATA_SETUP_ARGS="${TRUSTA_STRATA_SETUP_ARGS:---model IQ3_XXS}"

if [[ -z "$STRATA_PACKAGE" ]]; then
    STRATA_PACKAGE="$(ls -1 "$PROJECT_ROOT"/vendor/strata-sleep-*-linux-x86_64.tar.gz 2>/dev/null |
        sort -V | tail -n 1 || true)"
    [[ -n "$STRATA_PACKAGE" ]] ||
        die "no strata-sleep package in $PROJECT_ROOT/vendor; set TRUSTA_STRATA_PACKAGE to its .tar.gz"
fi
[[ -e "$STRATA_PACKAGE" ]] || die "TRUSTA_STRATA_PACKAGE: $STRATA_PACKAGE does not exist"
command -v uv >/dev/null 2>&1 || die "uv is needed for Strata's venv (setup_env.sh installs it)"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is needed to check the package (coreutils)"

# An install from before packages was a git checkout of the source; that folder holds
# someone's tree, not ours to overwrite.
if [[ -e "$STRATA_DIR/.git" ]]; then
    die "$STRATA_DIR is a git checkout (the old source install); move it away or set TRUSTA_STRATA_DIR"
fi
if [[ -e "$STRATA_DIR" && ! -f "$STRATA_DIR/SLEEP.json" && -n "$(ls -A "$STRATA_DIR" 2>/dev/null)" ]]; then
    die "$STRATA_DIR exists and is not a strata-sleep install; move it away or set TRUSTA_STRATA_DIR"
fi

# Unpack beside the target and check every file against the package's own manifest before
# anything is copied in: a truncated or altered package stops here, with the install untouched.
mkdir -p "$(dirname -- "$STRATA_DIR")"
staging="$(mktemp -d "$(dirname -- "$STRATA_DIR")/.strata-unpack.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
if [[ -d "$STRATA_PACKAGE" ]]; then
    cp -a "$STRATA_PACKAGE/." "$staging/"
else
    tar -xzf "$STRATA_PACKAGE" -C "$staging" --strip-components=1 ||
        die "could not unpack $STRATA_PACKAGE"
fi
[[ -f "$staging/SLEEP.json" && -f "$staging/MANIFEST.sha256" ]] ||
    die "$STRATA_PACKAGE is not a strata-sleep package (no SLEEP.json / MANIFEST.sha256)"
(cd "$staging" && sha256sum --quiet -c MANIFEST.sha256) ||
    die "$STRATA_PACKAGE does not match its MANIFEST.sha256; download it again. Nothing was installed."
version="$(cat "$staging/VERSION" 2>/dev/null || echo unknown)"

# Copied over an earlier install rather than replacing it: its venv and the run configs
# Strata's setup wrote (strata-<model>.json) stay, and setup reuses them.
mkdir -p "$STRATA_DIR"
cp -a "$staging/." "$STRATA_DIR/"
echo "$LOG_TAG strata-sleep $version unpacked into $STRATA_DIR"

python="$STRATA_DIR/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    # --seed: Strata's setup installs its packages with pip, which a bare uv venv lacks.
    uv venv -q --seed -p 3.12 "$STRATA_DIR/.venv" || die "could not make Strata's venv"
fi

# The package's engine/ is the ready-made engine setup takes as installed. --no-start: the
# backend starts it, not setup. --vision no: the package has no image encoder.
read -r -a setup_args <<< "$STRATA_SETUP_ARGS"
[[ -z "${TRUSTA_STRATA_GGUF_DIR:-}" ]] || setup_args+=(--gguf-dir "$TRUSTA_STRATA_GGUF_DIR")
[[ -z "${TRUSTA_STRATA_DATA_DIR:-}" ]] || setup_args+=(--data-dir "$TRUSTA_STRATA_DATA_DIR" --set-default-data-dir)
echo "$LOG_TAG running Strata's setup: --yes --no-start --vision no ${setup_args[*]}"
(cd "$STRATA_DIR" && "$python" setup.py --yes --no-start --vision no "${setup_args[@]}") ||
    die "Strata's setup failed (see above). Nothing was recorded in .env."

engine="$STRATA_DIR/engine/strata"
[[ -x "$engine" ]] || die "Strata's setup finished but left no engine at $engine"
# The flag's name is in the binary of every build that parses it, and in no other: the
# ready-made upstream engine has none, and the backend would fail to start it.
grep -a -q -- "--sleep-mode" "$engine" ||
    die "$engine has no deep sleep (--sleep-mode): it is not a strata-sleep engine. Nothing was recorded in .env."

record_env_setting STRATA_HOME "$STRATA_DIR"
record_env_setting STRATA_PYTHON "$python"
record_env_setting STRATA_ENGINE_BINARY "$engine"

echo ""
echo "$LOG_TAG Strata is installed: $STRATA_DIR (strata-sleep $version)"
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
