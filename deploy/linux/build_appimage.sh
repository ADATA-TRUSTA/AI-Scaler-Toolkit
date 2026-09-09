#!/usr/bin/env bash
#
# Build the Linux AppImage.
#
# Runs on the same machine as the Windows build, through WSL2 -- WSL2 has a real
# kernel so FUSE works, and no second host or paid container runtime is needed. It
# works just as well on a native Linux runner or inside an ubuntu:22.04 container.
#
# The distribution matters and is not interchangeable. Ubuntu 22.04 ships glibc 2.35,
# and the glibc of the build host is the AppImage's compatibility floor: one built on
# 24.04 (glibc 2.39) will not start on older systems, which is most of the ones this
# is meant to reach. 22.04 also matches what the official llama CUDA prebuilds expect.
#
# Only the official appimagetool is used. linuxdeploy exists to collect a C++ shared
# library tree, which a payload of Python and data files does not have, so it would be
# an extra moving part for nothing. appimage-builder is avoided outright: its
# maintenance has stalled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

APP_NAME="AI Scaler Toolkit"
APP_ID="ai-scaler-toolkit"
SERVICE_PORT="${SERVICE_PORT:-8571}"
VERSION="${APP_VERSION:-0.0.0}"

BUILD_DIR="$REPO_ROOT/build/appimage"
APPDIR="$BUILD_DIR/AppDir"
OUT_DIR="$REPO_ROOT/dist-unsigned"
TOOL_DIR="$BUILD_DIR/tools"
LAUNCHER_BUILD="$BUILD_DIR/launcher"
APPIMAGETOOL_URL="${APPIMAGETOOL_URL:-https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage}"

# To stderr, not stdout. fetch_appimagetool below returns the tool's path by printing
# it, and a progress line printed alongside lands in the same variable -- the build
# then tried to run "[appimage] downloading appimagetool" as a program.
log() { printf '[appimage] %s\n' "$*" >&2; }
die() { printf '[appimage] error: %s\n' "$*" >&2; exit 1; }

check_glibc() {
  local version
  version="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  [[ -n "$version" ]] || { log "could not determine glibc version; continuing"; return; }
  log "build host glibc: $version"
  # Louder than a comment: building on a newer glibc produces a binary that fails to
  # start on older systems, and it fails at launch on the user's machine, not here.
  if [[ "$(printf '%s\n2.35\n' "$version" | sort -V | head -1)" != "2.35" ]]; then
    log "WARNING: glibc $version is older than the expected 2.35 floor"
  elif [[ "$version" != "2.35" ]]; then
    log "WARNING: glibc $version is newer than 2.35 -- the AppImage will not run on older systems."
    log "         Build inside ubuntu:22.04 (container or WSL distribution) instead."
  fi
}

require() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but not on PATH"
}

fetch_appimagetool() {
  mkdir -p "$TOOL_DIR"
  local tool="$TOOL_DIR/appimagetool"
  if [[ ! -x "$tool" ]]; then
    log "downloading appimagetool"
    curl -fsSL -o "$tool" "$APPIMAGETOOL_URL"
    chmod +x "$tool"
  fi
  printf '%s' "$tool"
}

build_launcher() {
  # Frozen, exactly as on Windows. Without this the payload carries launcher sources
  # and no executable, AppRun has nothing to exec, and assert_payload says so and
  # stops the build -- which is what it used to do here, every time, because this
  # step did not exist.
  log "freezing the launcher"
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/build_launcher.py" \
    --out "$LAUNCHER_BUILD"
  [[ -x "$LAUNCHER_BUILD/AIScalerToolkit/AIScalerToolkit" ]] \
    || die "build_launcher.py produced no AIScalerToolkit executable"
}

build_payload() {
  log "building the Linux payload (TRUSTA build, vllm enabled)"
  # --project, not --no-project. Both scripts import make_icon, which draws the
  # application icon with Pillow; a bare interpreter has no Pillow, so the build
  # stopped on ModuleNotFoundError before it reached anything interesting.
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/make_payload.py" \
    --platform linux --out "$BUILD_DIR/payload" --port "$SERVICE_PORT" \
    --launcher-dir "$LAUNCHER_BUILD/AIScalerToolkit"
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/assert_payload.py" \
    --platform linux --payload "$BUILD_DIR/payload" --port "$SERVICE_PORT"
}

build_appdir() {
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

  cp -a "$BUILD_DIR/payload/." "$APPDIR/usr/share/$APP_ID/"

  install -m 0755 "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
  install -m 0644 "$SCRIPT_DIR/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
  install -m 0644 "$SCRIPT_DIR/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"

  # app-mark.png, which make_payload draws from the brand artwork into the payload
  # root. This used to look inside frontend/dist, where there is no PNG at all -- so
  # every AppImage built here wore appimagetool's generic icon and said so in a log
  # line nobody reads.
  local icon="$BUILD_DIR/payload/app-mark.png"
  [[ -f "$icon" ]] || icon=""
  if [[ -n "$icon" ]]; then
    install -m 0644 "$icon" "$APPDIR/$APP_ID.png"
    install -m 0644 "$icon" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
  else
    log "no app-mark.png in the payload; the AppImage will use a generic icon"
  fi
}

main() {
  require curl
  require uv
  check_glibc

  mkdir -p "$BUILD_DIR" "$OUT_DIR"
  build_launcher
  build_payload
  build_appdir

  local tool
  tool="$(fetch_appimagetool)"

  # FUSE is usually unavailable inside containers and sometimes under WSL. This flag
  # makes appimagetool unpack and run itself instead; the AppImage it produces is
  # byte-for-byte the same either way.
  export APPIMAGE_EXTRACT_AND_RUN=1

  local target="$OUT_DIR/${APP_ID}-${VERSION}-x86_64.AppImage"
  log "packaging -> $target"
  ARCH=x86_64 "$tool" "$APPDIR" "$target"
  [[ -f "$target" ]] || die "appimagetool reported success but produced no file"
  chmod +x "$target"
  log "done: $(du -h "$target" | cut -f1) $target"
}

main "$@"
