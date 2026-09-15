"""Collect the Visual C++ runtime the payload ships beside its binaries.

Windows only. torch's DLLs and llama.cpp's both import MSVCP140, VCRUNTIME140 and
VCRUNTIME140_1 -- checked, not assumed -- and none of that is part of Windows. The
Universal CRT (api-ms-win-crt-*, ucrtbase) is, which is why only these matter.

They used to be obtained by installing Microsoft's redistributable at first run. That
is a machine-wide component with no per-user form, so it prompted for administrator
rights -- the only such prompt in an installer built from end to end to need none.

Microsoft licenses these files for redistribution alongside an application, and placing
them next to the executable that needs them is the supported "local deployment". The
frozen launcher already ships its own copy of two of them, which PyInstaller collects
without anyone deciding to: the precedent is in the product already, this extends it to
the two environments built on the user's machine.

One version, named here, wherever it comes from
-----------------------------------------------
This used to take the *newest* redistributable directory on the build machine, on the
reasoning that the runtime is backward compatible. It is, but that is not the property
that matters: it made the contents of a release depend on what happened to be installed
on whoever built it, with nothing recording which. A machine with Visual Studio 18
alongside Build Tools 2022 shipped VC145 where every other build shipped VC143, and the
only way to find out was to measure a DLL's size on an installed machine.

So the toolset is pinned, a build that cannot produce exactly it fails, and what went in
is written into the payload as MANIFEST.json.

Where it comes from, in order: VCRUNTIME_DIR if set, the pinned toolset if this machine
has it, and otherwise Microsoft's own copy, downloaded. A build machine with the right
Visual Studio never reaches the network.

vc_redist.x64.exe is not one of those sources and cannot be: it is a self-contained Burn
bundle, and its payload is not reachable without running it -- /layout copies the bundle
itself, and both bsdtar and 7-Zip see only the bootstrapper's localised UI container, no
MSI, no CAB, no DLL. Running it is the machine-wide install this exists to avoid. The
.vsix below is the same files from the same vendor over the channel manifest that the
Visual Studio installer itself uses, and it is an ordinary zip.

Moving to a newer runtime
-------------------------
Deliberately more than editing a version string, because it changes what every user
runs. Fetch https://aka.ms/vs/17/release/channel, follow channelItems to the
VisualStudio manifest, find the newest Microsoft.VC.*.CRT.Redist.X64.base package, then
update TOOLSET, CRT_VERSION, VSIX_URL, VSIX_SHA256 and every hash in EXPECTED_DLLS from
that package's payload.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Two version numbers, both Microsoft's and neither a typo for the other: the
# redistributable directory is named for the toolset, the files inside it carry a build
# number of their own.
TOOLSET = "14.44.35112"
CRT_VERSION = "14.44.35211"

# Every DLL of the x64 CRT redistributable, with the SHA-256 it has at CRT_VERSION.
#
# The whole directory is shipped, not a chosen list. It is 2 MB, and picking names is how
# this first shipped broken: msvcp140, vcruntime140 and vcruntime140_1 are what
# llama.cpp's binaries import, so those were the three collected -- and torch_python.dll
# imports MSVCP140_ATOMIC_WAIT.dll as well, which is in the same redistributable and was
# not in the list. The symptom was "Error loading torch_python.dll or one of its
# dependencies", naming neither the missing file nor the reason.
#
# CXXAMP (vcamp140) and OpenMP (vcomp140) ship in the same redistributable and are not
# here: C++ AMP is deprecated and unused, and torch brings its own OpenMP as
# libiomp5md.dll. Checked on an installed machine rather than assumed.
EXPECTED_DLLS = {
    "concrt140.dll": "2405355f0a58067b258f8df33c327e3a3d716eaac5a3a5aebb757842d85bd376",
    "msvcp140.dll": "0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9",
    "msvcp140_1.dll": "bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d",
    "msvcp140_2.dll": "3ea06f0ee098b4823cb79599df3780e7f23cce52c19aac31d2a0d47efe33a5e9",
    "msvcp140_atomic_wait.dll": "640b2aefced484d0368eea5bdd06addd0658a3a70a49256e560d6923b404a479",
    "msvcp140_codecvt_ids.dll": "f2069a52880ec885ee7f0511186100eb7fada0411a2b4948fafea7735b878a18",
    "vccorlib140.dll": "19839407c3fdbc824e5bce189bf68ddf8097f12ec28b757797ffa0415c144ddd",
    "vcruntime140.dll": "d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066",
    "vcruntime140_1.dll": "1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f",
    "vcruntime140_threads.dll": "219915cf20822f34d5e7c1fdd4e21ae7f3396881096c51036225fb8f84b47afa",
}

# The names a directory must hold to be a CRT redistributable at all, used when deciding
# whether an operator's VCRUNTIME_DIR is one. Hashes are not required there -- that
# override exists for someone who means to ship something else.
RUNTIME_DLLS = tuple(EXPECTED_DLLS)

# Where the payload keeps them, and where setup_env is told to look.
PAYLOAD_DIR = "vcruntime"

# The same files from Microsoft, for a build machine without the toolset installed. This
# is one package out of the channel manifest the Visual Studio installer reads; the hash
# in the middle of the path is the file's own, so the URL cannot come to mean something
# else. Verified byte for byte against a local Visual Studio copy of CRT_VERSION.
VSIX_URL = (
    "https://download.visualstudio.microsoft.com/download/pr/"
    "45d3b8dd-bced-4b37-9974-142f748d710c/"
    "4aaf54db0bfc9435f7c3660e1a00237a4b556042bfeea64bde44c2e0194e6ee5/"
    "Microsoft.VC.14.44.17.14.CRT.Redist.X64.base.vsix"
)
VSIX_SHA256 = "4aaf54db0bfc9435f7c3660e1a00237a4b556042bfeea64bde44c2e0194e6ee5"

# Inside the .vsix, which is a zip laid out like the Redist directory it came from. The
# same tree also carries debug_nonredist/, onecore/ and spectre/ builds of these names,
# so the match is anchored rather than by filename.
_VSIX_MEMBER = re.compile(
    rf"^Contents/VC/Redist/MSVC/{re.escape(TOOLSET)}/x64/Microsoft\.VC\d+\.CRT/([^/]+\.dll)$"
)

_VS_ROOTS = (
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
)

INSTALL_HINT = f"""The Visual C++ runtime {CRT_VERSION} could not be obtained.

It ships with Visual Studio or Build Tools {TOOLSET}:

  winget install Microsoft.VisualStudio.2022.BuildTools

Without it this downloads Microsoft's own copy, so reaching this means the download
failed too -- an offline build machine, or a proxy in the way. Either install the
toolset, or point at a directory holding {", ".join(RUNTIME_DLLS[:3])} and the rest:

  VCRUNTIME_DIR=D:\\redist\\x64\\Microsoft.VC143.CRT
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_source(override: str | None = None) -> Path | None:
    """The pinned x64 CRT redistributable directory on this machine, or None.

    Not the newest one. See the module docstring: newest is what made a release depend
    on which Visual Studio the build machine happened to have.
    """
    if override:
        candidate = Path(override)
        return candidate if _complete(candidate) else None

    for root in _VS_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        # <root>/<edition or year>/<product>/VC/Redist/MSVC/<toolset>/x64/Microsoft.VC<n>.CRT
        for directory in base.glob(f"*/*/VC/Redist/MSVC/{TOOLSET}/x64/Microsoft.VC*.CRT"):
            if _complete(directory):
                return directory
    return None


def _complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in RUNTIME_DLLS)


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / ".cache" / "vcruntime"


def _download() -> Path | None:
    """Microsoft's copy of the same files, unpacked into the build cache.

    Returns a directory shaped like the one find_source returns, or None when the
    download could not be made -- the caller turns that into INSTALL_HINT, because by
    then every source has been tried and the reason no longer helps.
    """
    target = _cache_dir() / CRT_VERSION
    if _complete(target):
        return target

    archive = _cache_dir() / f"crt-{CRT_VERSION}.vsix"
    if not (archive.is_file() and _sha256(archive) == VSIX_SHA256):
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[payload] vc runtime {CRT_VERSION} is not installed here; fetching it from Microsoft")
        partial = archive.with_suffix(".part")
        try:
            with urllib.request.urlopen(VSIX_URL, timeout=120) as response, \
                    partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[payload] could not fetch the vc runtime: {exc}")
            partial.unlink(missing_ok=True)
            return None
        actual = _sha256(partial)
        if actual != VSIX_SHA256:
            # Not a retry: a wrong hash here is a different file, not a damaged one.
            partial.unlink(missing_ok=True)
            raise SystemExit(
                f"[payload] the vc runtime download does not match its recorded hash.\n"
                f"  expected {VSIX_SHA256}\n  got      {actual}\n  from     {VSIX_URL}"
            )
        partial.replace(archive)

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            match = _VSIX_MEMBER.match(member)
            if match and match.group(1) in EXPECTED_DLLS:
                (target / match.group(1)).write_bytes(bundle.read(member))
    if not _complete(target):
        raise SystemExit(
            f"[payload] {VSIX_URL} did not hold all of the CRT at {TOOLSET}. "
            "The pinned package is wrong; see 'Moving to a newer runtime' in vcruntime.py."
        )
    return target


def _verify(directory: Path) -> dict[str, str]:
    """Hash what is about to be shipped, and refuse anything but CRT_VERSION.

    Returns the hashes so the manifest records what actually went in rather than what
    was expected -- those are the same here by construction, and writing the measured
    one is what makes the manifest worth reading.
    """
    measured = {name: _sha256(directory / name) for name in RUNTIME_DLLS}
    wrong = sorted(name for name, digest in measured.items() if digest != EXPECTED_DLLS[name])
    if wrong:
        raise SystemExit(
            f"[payload] {directory} is not the CRT at {CRT_VERSION}.\n"
            + "\n".join(
                f"  {name}\n    expected {EXPECTED_DLLS[name]}\n    got      {measured[name]}"
                for name in wrong
            )
        )
    return measured


def collect(payload: Path, override: str | None = None) -> Path:
    """Copy the runtime into the payload. Raises SystemExit when it cannot be found.

    Loud rather than optional: a release built without it installs and then fails on
    the user's machine with a missing-DLL box naming a file they have never heard of.
    """
    source = find_source(override)
    origin = str(source) if source else ""
    if source is None:
        if override:
            # An operator named a directory and it is not one. Saying so beats quietly
            # downloading something else and shipping it under their instruction.
            raise SystemExit(
                f"[payload] VCRUNTIME_DIR={override} does not hold the CRT "
                f"({', '.join(RUNTIME_DLLS[:3])} and the rest)."
            )
        source = _download()
        origin = VSIX_URL
    if source is None:
        raise SystemExit(INSTALL_HINT)

    measured = _verify(source)

    target = payload / PAYLOAD_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_DLLS:
        shutil.copy2(source / name, target / name)

    # What went in, so that "which runtime does this installer carry" is a question with
    # an answer. It was previously answerable only by measuring a DLL on a machine that
    # had already installed it.
    (target / "MANIFEST.json").write_text(
        json.dumps(
            {"toolset": TOOLSET, "version": CRT_VERSION, "source": origin, "files": measured},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[payload] vc runtime: {len(measured)} files at {CRT_VERSION} from {origin}")
    return target
