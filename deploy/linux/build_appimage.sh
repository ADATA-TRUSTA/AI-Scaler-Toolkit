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
APPIMAGETOOL_URL="${APPIMAGETOOL_URL:-https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage}"

log() { printf '[appimage] %s\n' "$*"; }
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

build_payload() {
  log "building the Linux payload (TRUSTA build, vllm enabled)"
  uv run --no-project python "$REPO_ROOT/deploy/package/make_payload.py" \
    --platform linux --out "$BUILD_DIR/payload" --port "$SERVICE_PORT"
  uv run --no-project python "$REPO_ROOT/deploy/package/assert_payload.py" \
    --platform linux --payload "$BUILD_DIR/payload" --port "$SERVICE_PORT"
}

build_appdir() {
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

  cp -a "$BUILD_DIR/payload/." "$APPDIR/usr/share/$APP_ID/"

  install -m 0755 "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
  install -m 0644 "$SCRIPT_DIR/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
  install -m 0644 "$SCRIPT_DIR/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"

  # The icon travels with the frontend package, so the AppImage shows the brand that
  # actually shipped rather than one chosen at build time.
  local icon
  icon="$(find "$BUILD_DIR/payload/frontend/dist" -maxdepth 1 -iname '*.png' -print -quit || true)"
  if [[ -n "$icon" ]]; then
    install -m 0644 "$icon" "$APPDIR/$APP_ID.png"
    install -m 0644 "$icon" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
  else
    log "no PNG icon in the frontend package; the AppImage will use a generic icon"
  fi
}

main() {
  require curl
  require uv
  check_glibc

  mkdir -p "$BUILD_DIR" "$OUT_DIR"
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
