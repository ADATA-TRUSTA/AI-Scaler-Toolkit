#!/usr/bin/env bash
#
# Build the Linux AppImage.
#
# Runs on the same machine as the Windows build, through WSL2 -- WSL2 has a real
# kernel so FUSE works, and no second host or paid container runtime is needed. It
# works just as well on a native Linux runner or inside an ubuntu:24.04 container.
#
# The distribution matters and is not interchangeable. Ubuntu 24.04 ships glibc 2.39,
# and the glibc of the build host is the AppImage's compatibility floor: one built here
# will not start on older systems (Ubuntu 22.04 has 2.35, Debian 12 has 2.36). 24.04 is
# the oldest release this supports; it was 22.04 until this was raised.
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
# machine, and shipped inside the AppImage with the launcher's packages already in it.
# Fixing it is what makes one set of packages -- and one PyGObject build -- correct
# everywhere, instead of whatever the distribution happened to ship; carrying it is
# what lets a first start open a window with no network. See bundle_python.
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
# Where uv puts the launcher's interpreter while it is being prepared, and where it
# sits in the AppImage. Beside the payload, not in it: the launcher copies the payload
# somewhere writable on first run (config.ensure_writable_payload), and an interpreter
# that nothing writes to has no business being copied along.
PYTHON_STORE="$BUILD_DIR/python"
PYTHON_IN_APPDIR="usr/lib/$APP_ID/python"
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
  if [[ "$(printf '%s\n2.39\n' "$version" | sort -V | head -1)" != "2.39" ]]; then
    log "WARNING: glibc $version is older than the expected 2.39 floor"
  elif [[ "$version" != "2.39" ]]; then
    log "WARNING: glibc $version is newer than 2.39 -- the AppImage will not run on 24.04."
    log "         Build inside ubuntu:24.04 (container or WSL distribution) instead."
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

# PyGObject is built against the newest release that still uses girepository-1.0. 3.52
# requires girepository-2.0, which wants glib 2.80; lifting the pin is untested.
GI_PIN="${GI_PIN:-pygobject<3.52}"

# Set by bundle_python: the interpreter's own directory, ready to copy into the AppDir.
PYTHON_ROOT=""

# The launcher's interpreter, complete: Python, the launcher's packages, PyGObject.
#
# The launcher is not frozen on Linux. pywebview draws through PyGObject, and a frozen
# build would have to carry GTK and WebKitGTK with it, which is what AppRun declines
# to do. So it runs as sources on an ordinary interpreter -- and that interpreter used
# to be made on the user's machine, on the first start: AppRun asked uv for a Python,
# built a virtual environment and installed these same packages into it. A first start
# therefore needed a network before it could draw the window that says setup needs a
# network, and showed nothing while it worked.
#
# Built here instead, once, and shipped. What is in it:
#
#   Python          uv's own build (python-build-standalone), which is made to be moved:
#                   it finds its library from where its executable is, so it runs from
#                   wherever the AppImage is mounted.
#   the packages    from deploy/uv.lock, the lock the Windows build freezes from, with
#                   their hashes. Into the interpreter itself rather than a virtual
#                   environment: a venv records the absolute path of the interpreter it
#                   was made from, and here that path changes on every start.
#   PyGObject       built for this interpreter. gi is an ABI-tagged C extension, so the
#                   distribution's own only loads in the distribution's Python. Ours
#                   links against libglib, libgobject and libgirepository-1.0.so.1 and
#                   nothing else, present with a stable ABI everywhere this runs. The
#                   typelibs and the libraries themselves still come from the machine.
#
# Everything is compiled to bytecode here, because the mount is read-only and Python
# would otherwise compile every module again on every start.
bundle_python() {
  local requirements="$BUILD_DIR/launcher-requirements.txt"
  log "exporting the launcher's requirements from deploy/uv.lock"
  # --frozen: the lock is the record of what ships. Exporting must not be what rewrites it.
  uv export --project "$REPO_ROOT/deploy" --frozen --no-dev --no-emit-project -o "$requirements" >/dev/null
  [[ -s "$requirements" ]] || die "uv export produced no requirements"
  grep -q -- "--hash=sha256:" "$requirements" || die "the exported requirements carry no hashes"
  log "launcher requirements: $(grep -c -- '--hash=sha256:' "$requirements") hashed artifacts"

  for tool in pkg-config cc; do
    command -v "$tool" >/dev/null 2>&1 || die \
      "$tool is needed to build PyGObject. On Debian/Ubuntu:
    apt-get install build-essential pkg-config libgirepository1.0-dev libcairo2-dev"
  done
  pkg-config --exists gobject-introspection-1.0 || die \
    "libgirepository1.0-dev is needed to build PyGObject (apt-get install libgirepository1.0-dev)"

  # A directory of its own, so the interpreter that ships is one nothing else on this
  # machine has installed into, and a build never inherits the last one's packages.
  log "installing Python $LAUNCHER_PYTHON for the launcher"
  rm -rf "$PYTHON_STORE"
  UV_PYTHON_INSTALL_DIR="$PYTHON_STORE" uv python install --quiet "$LAUNCHER_PYTHON" \
    || die "could not install Python $LAUNCHER_PYTHON"
  local root
  root="$(find "$PYTHON_STORE" -mindepth 1 -maxdepth 1 -type d -name "cpython-${LAUNCHER_PYTHON}*" | sort | head -1)"
  [[ -n "$root" && -x "$root/bin/python3" ]] || die "uv installed no Python $LAUNCHER_PYTHON under $PYTHON_STORE"
  local py="$root/bin/python3"
  local version site
  version="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  [[ "$site" == "$root"/* ]] || die "the interpreter's site-packages is outside it: $site"
  log "Python $("$py" -c 'import platform; print(platform.python_version())') at ${root#"$BUILD_DIR"/}"

  # uv marks its own interpreters as not to be installed into; this one exists to be.
  uv pip install --python "$py" --break-system-packages --require-hashes --quiet -r "$requirements" \
    || die "could not install the launcher's packages"

  log "building PyGObject ($GI_PIN) for it"
  local work
  work="$(mktemp -d)"
  # shellcheck disable=SC2064 - the path is wanted now, not at trap time
  trap "rm -rf '$work'" RETURN
  # In a throwaway environment on the same interpreter, so only gi itself is carried
  # over and none of what was needed to build it.
  uv venv --python "$py" "$work/v" >/dev/null 2>&1 \
    || die "could not create an environment to build PyGObject in"
  uv pip install --python "$work/v/bin/python" --quiet "$GI_PIN" \
    || die "could not build PyGObject. The dev packages above are the usual cause."
  rm -rf "$site/gi"
  cp -a "$work/v/lib/python"*/site-packages/gi "$site/"
  # The build host's glibc is this AppImage's floor already, so an extension built here
  # runs anywhere the AppImage does. Checked rather than assumed, because a missing
  # extension shows up as "no module named gi" on a user's machine and nowhere here.
  local ext
  ext="$(find "$site/gi" -name '_gi.cpython-*.so' -printf '%f\n' | head -1)"
  [[ -n "$ext" ]] || die "PyGObject built no _gi extension; the build produced nothing usable"
  log "PyGObject: $ext"

  # What only a build needs, or nothing here imports: headers and static libraries, pip,
  # Tcl/Tk and the modules on top of it, the bundled test suite.
  rm -rf "$root/include" "$root/share" "$root/lib/pkgconfig" \
         "$root/lib/python$version/config-"* \
         "$root/lib/python$version/test" "$root/lib/python$version/idlelib" \
         "$root/lib/python$version/tkinter" "$root/lib/python$version/turtledemo" \
         "$root/lib/python$version/ensurepip" "$root/lib/python$version/lib2to3" \
         "$root/lib/python$version/lib-dynload/"_tkinter*.so \
         "$root"/lib/tcl* "$root"/lib/tk* "$root"/lib/itcl* "$root"/lib/thread* \
         "$root"/lib/libtcl* "$root"/lib/libtk* \
         "$site"/pip "$site"/pip-*.dist-info "$root"/bin/pip*
  find "$root" -depth -type d -name __pycache__ -exec rm -rf {} +

  # unchecked-hash: valid whatever the source file's timestamp comes to inside the
  # squashfs, and never re-read against it.
  "$py" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$root/lib" >/dev/null \
    || log "some modules did not compile to bytecode; they will be compiled on each start"

  # The interpreter as it will run: from another path, with nothing of this machine's
  # Python in its environment. Imported rather than looked for -- a package that is
  # there and cannot be imported is the failure this catches. pystray is only looked
  # for: importing it picks a tray backend, which needs a desktop.
  local trial="$work/moved"
  cp -a "$root" "$trial"
  env -u PYTHONPATH -u PYTHONHOME "$trial/bin/python3" -c '
import importlib.util
import sys

import gi
import PIL.Image
import webview
import yaml

if importlib.util.find_spec("pystray") is None:
    sys.exit("pystray is not installed")
' || die "the bundled interpreter cannot import what the launcher needs"
  rm -rf "$trial"

  PYTHON_ROOT="$root"
  log "bundled interpreter: $(du -sh "$root" | cut -f1)"
}

build_payload() {
  log "building the Linux payload (TRUSTA build, vllm enabled)"
  # --project, not --no-project. Both scripts import make_icon, which draws the
  # application icon with Pillow; a bare interpreter has no Pillow, so the build
  # stopped on ModuleNotFoundError before it reached anything interesting.
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/make_payload.py" \
    --platform linux --source "$LAYOUT" --out "$BUILD_DIR/payload" \
    --port "$SERVICE_PORT" --version "$VERSION"
  bundle_python
  uv run --project "$REPO_ROOT/deploy" python "$REPO_ROOT/deploy/package/assert_payload.py" \
    --platform linux --payload "$BUILD_DIR/payload" --port "$SERVICE_PORT" \
    --launcher-python "$PYTHON_ROOT"
}

build_appdir() {
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

  cp -a "$BUILD_DIR/payload/." "$APPDIR/usr/share/$APP_ID/"

  mkdir -p "$APPDIR/$PYTHON_IN_APPDIR"
  cp -a "$PYTHON_ROOT/." "$APPDIR/$PYTHON_IN_APPDIR/"

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

  # Named like the Windows installer -- product, version, architecture -- with each
  # platform's own word for the architecture. APP_ID (the lowercase id) stays what the
  # desktop entry and the data directories use; the file name is not tied to it.
  local target="$OUT_DIR/AIScalerToolkit-${VERSION}-x86_64.AppImage"
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
