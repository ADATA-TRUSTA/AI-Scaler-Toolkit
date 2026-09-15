"""Collect the uv binary the payload ships, so provisioning does not fetch one.

uv is what builds the service environment: it installs the Python, resolves the lock
and syncs the packages. Until now every install obtained it the way uv's own
documentation says to, by downloading https://astral.sh/uv/install.sh (or .ps1) and
running it -- a remote script, executed unverified, at the start of provisioning on
someone's machine.

That is a reasonable thing for a developer to type. It is a poor thing for an
installer to do on a user's behalf, and it is the largest remaining gap now that the
CRT, the llama binary, the AppImage tool and our own installer are all checked: those
are files that get verified, this was code that got run.

Carrying it also makes the install reproducible in the way the rest now is. "Which uv
built this environment" had no answer -- astral.sh serves whatever is current, so two
installs of the same release could and did resolve with different uv versions.

One version, pinned, hashed, per platform. The archives are uv's own release assets,
and the hashes are the ones GitHub publishes for them.

Moving to a newer uv
--------------------
Fetch https://api.github.com/repos/astral-sh/uv/releases/tags/<version>, take the
digest of each archive below, and update UV_VERSION and both entries.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UV_VERSION = "0.12.14"

_RELEASE = "https://github.com/astral-sh/uv/releases/download/{version}/{asset}"

# Per platform, because the payload is built for one. The member is the path inside the
# archive: uv's tarballs put the binary under a directory named for the target, its
# zips do not.
_ARCHIVES = {
    "windows": {
        "asset": "uv-x86_64-pc-windows-msvc.zip",
        "sha256": "02cecb254f37f3396da625680796d3990164a590fb5611d46f719ad83e7b9a89",
        "member": "uv.exe",
        "binary": "uv.exe",
    },
    "linux": {
        "asset": "uv-x86_64-unknown-linux-gnu.tar.gz",
        "sha256": "18ef5c3888ae59828cb13f38d57e9389b8173ecc719eff163bfafc74b38f5936",
        "member": "uv-x86_64-unknown-linux-gnu/uv",
        "binary": "uv",
    },
}

# Where the payload keeps it, and where AppRun and setup_env are told to look.
PAYLOAD_DIR = "uv"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / ".cache" / "uv"


def _archive(platform: str) -> Path:
    """The release archive, downloaded once and kept, verified either way."""
    spec = _ARCHIVES[platform]
    path = _cache_dir() / f"{UV_VERSION}-{spec['asset']}"
    if path.is_file() and _sha256(path.read_bytes()) == spec["sha256"]:
        return path

    url = _RELEASE.format(version=UV_VERSION, asset=spec["asset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[payload] fetching uv {UV_VERSION} for {platform}")
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"[payload] could not fetch uv from {url}: {exc}") from exc

    actual = _sha256(data)
    if actual != spec["sha256"]:
        # Not a retry: a wrong hash is a different file, not a damaged one.
        raise SystemExit(
            f"[payload] the uv download does not match its recorded hash.\n"
            f"  expected {spec['sha256']}\n  got      {actual}\n  from     {url}"
        )
    path.write_bytes(data)
    return path


def _extract(archive: Path, platform: str) -> bytes:
    spec = _ARCHIVES[platform]
    if platform == "windows":
        with zipfile.ZipFile(archive) as bundle:
            return bundle.read(spec["member"])
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(spec["member"])
        if member is None:
            raise SystemExit(f"[payload] {spec['member']} is not in {archive.name}")
        return member.read()


def collect(payload: Path, platform: str) -> Path:
    """Put uv in the payload. Raises SystemExit when it cannot be obtained.

    Loud rather than optional, like the CRT: a build that quietly shipped without it
    would send every install back to downloading and running a remote script, which is
    the thing this removes.
    """
    spec = _ARCHIVES[platform]
    target = payload / PAYLOAD_DIR
    target.mkdir(parents=True, exist_ok=True)
    binary = target / spec["binary"]

    data = _extract(_archive(platform), platform)
    binary.write_bytes(data)
    # Executable on Linux; harmless and meaningless on Windows. Written through the
    # archive rather than copied out of it because a tarfile member's mode is not
    # preserved by a plain read-and-write.
    binary.chmod(0o755)

    (target / "VERSION").write_text(f"{UV_VERSION}\n{_sha256(data)}  {spec['binary']}\n",
                                    encoding="utf-8")
    print(f"[payload] uv {UV_VERSION}: {spec['binary']} ({len(data) / 1e6:.1f} MB)")
    return target
