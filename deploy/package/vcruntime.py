"""Collect the Visual C++ runtime the payload ships beside its binaries.

Windows only. torch's DLLs and llama.cpp's both import MSVCP140, VCRUNTIME140 and
VCRUNTIME140_1 -- checked, not assumed -- and none of that is part of Windows. The
Universal CRT (api-ms-win-crt-*, ucrtbase) is, which is why only these three matter.

They used to be obtained by installing Microsoft's redistributable at first run. That
is a machine-wide component with no per-user form, so it prompted for administrator
rights -- the only such prompt in an installer built from end to end to need none.

Microsoft licenses these three files for redistribution alongside an application, and
placing them next to the executable that needs them is the supported "local
deployment". The frozen launcher already ships its own copy of two of them, which
PyInstaller collects without anyone deciding to: the precedent is in the product
already, this extends it to the two environments built on the user's machine.

Taken from Visual Studio's redistributable directory on the build machine rather than
downloaded. vc_redist.x64.exe is a self-contained Burn bundle whose payload cannot be
unpacked with anything Windows ships -- /layout copies the bundle, /extract writes
nothing, and bsdtar reaches only the bootstrapper's UI container. A build machine that
can compile the installer has the redistributable already.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# The whole CRT directory is copied, not a chosen list. It is 2 MB, and picking names
# is how this first shipped broken: msvcp140, vcruntime140 and vcruntime140_1 are what
# llama.cpp's binaries import, so those were the three collected -- and torch_python.dll
# imports MSVCP140_ATOMIC_WAIT.dll as well, which is in the same redistributable and was
# not in the list. The symptom was "Error loading torch_python.dll or one of its
# dependencies", which names neither the missing file nor the reason.
#
# These are the ones checked for, so that a directory that is not a CRT redistributable
# is rejected rather than copied as an empty set.
RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_atomic_wait.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)

# Where the payload keeps them, and where setup_env is told to look.
PAYLOAD_DIR = "vcruntime"

# Visual Studio installs the redistributable files under a version directory, per
# toolset. Both editions and the build tools use the same shape.
_VS_ROOTS = (
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
)

INSTALL_HINT = f"""The Visual C++ runtime files were not found on this machine.

They ship with any Visual Studio or Build Tools install that has the C++ workload:

  winget install Microsoft.VisualStudio.2022.BuildTools

Or point at a directory holding {", ".join(RUNTIME_DLLS)}:

  VCRUNTIME_DIR=D:\\redist\\x64\\Microsoft.VC143.CRT
"""


def find_source(override: str | None = None) -> Path | None:
    """The newest x64 CRT redistributable directory, or None.

    Newest because the runtime is backward compatible: a newer one runs everything an
    older one does, and the binaries being shipped are built with a current toolset.
    """
    if override:
        candidate = Path(override)
        return candidate if _complete(candidate) else None

    found: list[Path] = []
    for root in _VS_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        # <root>/<edition or year>/<product>/VC/Redist/MSVC/<version>/x64/Microsoft.VC<n>.CRT
        for directory in base.glob("*/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT"):
            if _complete(directory):
                found.append(directory)
    if not found:
        return None
    return max(found, key=lambda p: _version_key(p.parents[1].name))


def _complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in RUNTIME_DLLS)


def _version_key(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


def collect(payload: Path, override: str | None = None) -> Path:
    """Copy the runtime into the payload. Raises SystemExit when it cannot be found.

    Loud rather than optional: a release built without it installs and then fails on
    the user's machine with a missing-DLL box naming a file they have never heard of.
    """
    source = find_source(override)
    if source is None:
        raise SystemExit(INSTALL_HINT)

    target = payload / PAYLOAD_DIR
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for dll in sorted(source.glob("*.dll")):
        shutil.copy2(dll, target / dll.name)
        copied += 1
    print(f"[payload] vc runtime: {copied} files from {source.parents[1].name}")
    return target
