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
# The interpreter the launcher runs on, obtained by uv rather than taken from the
# machine. Fixing it is what makes one set of packages -- and one PyGObject build --
# correct everywhere, instead of whatever the distribution happened to ship.
LAUNCHER_PYTHON="${LAUNCHER_PYTHON:-3.12}"

# The assembled layout to package. Defaults to this repository, which is where a
# developer's is; the release pipeline assembles into a directory of its own and points
# here. Every other build script already took a path -- this one did not, so the only way
# to build from a layout elsewhere was to overwrite the repository's own.
LAYOUT="${TRUSTA_LAYOUT:-$REPO_ROOT}"

# OpenPGP key to embed a signature with, by fingerprint. Unset builds an unsigned
# AppImage, which is what a developer without the release key gets. Signing happens
# here and not afterwards because appimagetool can only sign while it writes the file.
# Whatever gpg needs to reach the key (GNUPGHOME, an open token session) is the
# caller's business; nothing about the release key lives in this repository.
SIGN_KEY="${SIGN_KEY:-}"

BUILD_DIR="$REPO_ROOT/build/appimage"
APPDIR="$BUILD_DIR/AppDir"
OUT_DIR="$REPO_ROOT/dist-unsigned"
TOOL_DIR="$BUILD_DIR/tools"
# A release tag, not "continuous". That tag is a moving target: it is republished whenever the
# project builds, so the same source tree here produces AppImages made by different tools
# depending on the day, and nothing in the output records which. Not theoretical -- while this
# was being pinned, the binary under "continuous" was already a different one from the 1.9.1
# release three weeks earlier, same size, different hash.
#
# The hash is the one GitHub publishes for the asset. Overriding the URL without also setting
# the hash skips the check, which is the escape hatch for building against a local tool.
APPIMAGETOOL_VERSION="${APPIMAGETOOL_VERSION:-1.9.1}"
APPIMAGETOOL_SHA256="${APPIMAGETOOL_SHA256-ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0}"
APPIMAGETOOL_URL="${APPIMAGETOOL_URL:-https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-x86_64.AppImage}"

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

sha256_is() {
  local file="$1" expected="$2" actual
  [[ -f "$file" ]] || return 1
  actual="$(sha256sum "$file" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]]
}

fetch_appimagetool() {
  mkdir -p "$TOOL_DIR"
  # Named for the version, so changing the pin fetches rather than reuses whatever is cached.
  local tool="$TOOL_DIR/appimagetool-$APPIMAGETOOL_VERSION"
  if [[ -x "$tool" ]] && { [[ -z "$APPIMAGETOOL_SHA256" ]] || sha256_is "$tool" "$APPIMAGETOOL_SHA256"; }; then
    printf '%s' "$tool"
    return
  fi
  log "downloading appimagetool $APPIMAGETOOL_VERSION"
  # To .part first: a half-written file that happens to be executable would otherwise be cached
  # and reused by the next build, and an interrupted download is how it gets there.
  curl -fsSL -o "$tool.part" "$APPIMAGETOOL_URL" || die "could not download $APPIMAGETOOL_URL"
  if [[ -n "$APPIMAGETOOL_SHA256" ]] && ! sha256_is "$tool.part" "$APPIMAGETOOL_SHA256"; then
    local actual
    actual="$(sha256sum "$tool.part" | cut -d' ' -f1)"
    rm -f "$tool.part"
    die "appimagetool does not match its published hash.
    expected $APPIMAGETOOL_SHA256
    got      $actual
    from     $APPIMAGETOOL_URL"
  fi
  [[ -n "$APPIMAGETOOL_SHA256" ]] || log "WARNING: APPIMAGETOOL_SHA256 is unset; the tool is unverified"
  mv "$tool.part" "$tool"
  chmod +x "$tool"
  printf '%s' "$tool"
}

# The launcher is not frozen on Linux; AppRun runs it on an interpreter uv manages, and
# what it needs is the list of packages to install -- exported from the same lock the
# Windows build freezes from, so the two platforms do ship the same versions.
#
# With hashes, now that they can be. They used to be stripped and the comment here used
# to claim a version parity that did not exist: AppRun installed into whatever
# /usr/bin/python3 was, 3.12 on Ubuntu 24.04 and 3.10 on the 22.04 this build's glibc
# floor targets, and uv resolved whatever fitted. Hashes made that fail rather than
# diverge quietly -- an artifact listed for one Python, installed under another, falls
# back to the sdist, and building Pillow on a desktop with no compiler is where it
# ended. Fixing the interpreter fixes the artifacts, so the hashes can go back in.
export_requirements() {
  local out="$BUILD_DIR/payload/launcher-requirements.txt"
  log "exporting the launcher's requirements from deploy/uv.lock"
  uv export --project "$REPO_ROOT/deploy" --no-dev --no-emit-project -o "$out" >/dev/null
  [[ -s "$out" ]] || die "uv export produced no requirements"
  grep -q -- "--hash=sha256:" "$out" || die "the exported requirements carry no hashes"
  log "launcher requirements: $(grep -c -- '--hash=sha256:' "$out") hashed artifacts"
}

# PyGObject, built here for the interpreter the AppImage will run, and shipped.
#
# This is what makes fixing the interpreter possible at all. gi is an ABI-tagged C
# extension, so the distribution's own is built for the distribution's Python -- cp310
# on 22.04, cp312 on 24.04 -- and a uv-managed 3.12 can load 24.04's and not 22.04's.
# Building it here instead cuts that tie: the extension links against libglib,
# libgobject and libgirepository-1.0.so.1 and nothing else, and those are present with a
# stable ABI on both. Measured: built on 22.04 against glib 2.72, it loads Gtk 3.24 and
# WebKit2 4.1 on 24.04 with glib 2.80.
#
# Pinned below 3.52 because that release requires girepository-2.0, which wants glib
# 2.80 -- newer than the 2.72 on the oldest distribution this supports. The typelibs and
# the libraries themselves still come from the machine, as they always did; only the
# Python binding is ours.
GI_PIN="${GI_PIN:-pygobject<3.52}"

build_gi() {
  local out="$BUILD_DIR/payload/launcher-gi"
  for tool in pkg-config cc; do
    command -v "$tool" >/dev/null 2>&1 || die \
      "$tool is needed to build PyGObject. On Debian/Ubuntu:
    apt-get install build-essential pkg-config libgirepository1.0-dev libcairo2-dev"
  done
  pkg-config --exists gobject-introspection-1.0 || die \
    "libgirepository1.0-dev is needed to build PyGObject (apt-get install libgirepository1.0-dev)"

  log "building PyGObject ($GI_PIN) for the launcher's interpreter"
  local work
  work="$(mktemp -d)"
  # shellcheck disable=SC2064 - the path is wanted now, not at trap time
  trap "rm -rf '$work'" RETURN
  uv venv --python "$LAUNCHER_PYTHON" "$work/v" >/dev/null 2>&1 \
    || die "could not create a $LAUNCHER_PYTHON environment to build PyGObject in"
  uv pip install --python "$work/v/bin/python" --quiet "$GI_PIN" \
    || die "could not build PyGObject. The dev packages above are the usual cause."

  rm -rf "$out"
  mkdir -p "$out"
  cp -a "$work/v/lib/python"*/site-packages/gi "$out/"
  # The build host's glibc is this AppImage's floor already, so an extension built here
  # runs anywhere the AppImage does. Checked rather than assumed, because a missing
  # extension shows up as "no module named gi" on a user's machine and nowhere here.
  local ext
  ext="$(find "$out/gi" -name '_gi.cpython-*.so' -printf '%f\n' | head -1)"
  [[ -n "$ext" ]] || die "PyGObject built no _gi extension; the build produced nothing usable"
  log "PyGObject: $ext ($(du -sh "$out" | cut -f1))"
}

build_payload() {
  log "building the Linux payload (TRUSTA build, vllm enabled)"
  # --project, not --no-project. Both scripts import make_icon, which draws the
  # application icon with Pillow; a bare interpreter has no Pillow, so the build
  # stopped on ModuleNotFoundError before it reached anything interesting.
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/make_payload.py" \
    --platform linux --source "$LAYOUT" --out "$BUILD_DIR/payload" \
    --port "$SERVICE_PORT" --version "$VERSION"
  export_requirements
  build_gi
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
  build_payload
  build_appdir

  local tool
  tool="$(fetch_appimagetool)"

  # FUSE is usually unavailable inside containers and sometimes under WSL. This flag
  # makes appimagetool unpack and run itself instead; the AppImage it produces is
  # byte-for-byte the same either way.
  export APPIMAGE_EXTRACT_AND_RUN=1

  local sign_args=()
  if [[ -n "$SIGN_KEY" ]]; then
    require gpg
    sign_args=(--sign --sign-key "$SIGN_KEY")
  fi

  local target="$OUT_DIR/${APP_ID}-${VERSION}-x86_64.AppImage"
  log "packaging -> $target${SIGN_KEY:+ (signed with $SIGN_KEY)}"
  ARCH=x86_64 "$tool" "${sign_args[@]}" "$APPDIR" "$target"
  [[ -f "$target" ]] || die "appimagetool reported success but produced no file"
  chmod +x "$target"
  # A signing failure inside appimagetool is not always fatal to it, so the signature
  # is checked for rather than assumed. Without APPIMAGE_EXTRACT_AND_RUN, which is set
  # above for appimagetool's sake: with it, the runtime hands --appimage-signature to
  # the launcher as an ordinary argument instead of answering it, and prints nothing.
  if [[ -n "$SIGN_KEY" ]] && [[ -z "$(env -u APPIMAGE_EXTRACT_AND_RUN "$target" --appimage-signature 2>/dev/null | tr -d '\0[:space:]')" ]]; then
    die "SIGN_KEY is set but $target carries no signature"
  fi
  log "done: $(du -h "$target" | cut -f1) $target"
}

main "$@"
