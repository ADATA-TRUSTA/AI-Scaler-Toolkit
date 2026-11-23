#!/usr/bin/env python
"""
Install FlashInfer's prebuilt kernels for this machine's GPU, so vLLM's first load does not compile.

Without them FlashInfer builds every kernel the first time a model needs it. The MoE ones are the
expensive case: a single compiler process peaks near 10 GiB, and the build takes several minutes
even when it fits in memory. FlashInfer publishes the same kernels precompiled, one wheel per GPU
family, beside a tiny "shim" wheel that FlashInfer discovers them through:

    flashinfer-jit-cache            the shim: version + provider discovery
    flashinfer-jit-cache-sm120f     one provider per GPU family (about 190 MB each)

Three properties shape this script:

* The shim lists every provider as a dependency, so it must go in with ``--no-deps`` or all six
  providers (more than a gigabyte) come with it.
* Both versions must equal FlashInfer's own, local label included (``0.7.0.post1+cu130``). A
  provider without the shim crashes ``import flashinfer``; a version mismatch makes FlashInfer
  ignore the provider with a warning. So the result is verified, and a half-installed pair is
  removed again.
* ``uv sync`` removes packages the lock does not list, so this runs after every sync, from
  scripts/linux/setup_env.sh. It does nothing when the pair is already in place.

A GPU family with no prebuilt provider (Ampere consumer cards, sm86, for one) keeps the old
behaviour: the kernels compile on first use, with the engine's MAX_JOBS cap and progress message.
That is also the outcome of any failure here -- this step never fails the setup.

    python scripts/install_flashinfer_jit_cache.py [--check] [--arch sm120f[,sm89...]]

Environment:
    TRUSTA_FLASHINFER_JIT_CACHE=0        skip entirely
    TRUSTA_FLASHINFER_JIT_CACHE_INDEX    package index (default https://flashinfer.ai/whl/<cuda tag>)
    TRUSTA_FLASHINFER_JIT_ARCH           provider list, to install without a visible GPU
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys

DEFAULT_INDEX_BASE = "https://flashinfer.ai/whl"
SHIM = "flashinfer-jit-cache"

# CUDA compute capability -> provider. Exactly the providers the shim of FlashInfer 0.7.0 lists as
# its own requirements; a capability that is not here has none to install.
CAPABILITY_TO_PROVIDER = {
    "8.0": "sm80",
    "8.9": "sm89",
    "9.0": "sm90a",
    "10.0": "sm100a",
    "10.3": "sm103a",
    "12.0": "sm120f",
}
# Exit codes. Never a failure of the setup itself; setup_env reads them for its summary line.
EXIT_OK = 0  # the pair is installed and verified (or already was)
EXIT_COMPILES = 2  # not installed: kernels compile on first use
EXIT_NOT_APPLICABLE = 3  # nothing to install here (no flashinfer, no CUDA torch, switched off)

PROVIDERS = frozenset(CAPABILITY_TO_PROVIDER.values())


def log(message: str) -> None:
    """Print one prefixed line."""
    print(f"[flashinfer-jit-cache] {message}", flush=True)


def installed_version(distribution: str) -> str | None:
    """Version of an installed distribution, or None."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda_tag() -> str | None:
    """``cu130`` from torch's ``2.13.0+cu130``, or None for a build that is not a CUDA one."""
    torch_version = installed_version("torch")
    match = re.search(r"\+(cu\d+)$", torch_version or "")
    return match.group(1) if match else None


def gpu_capabilities() -> list[str]:
    """Compute capabilities of the visible GPUs, from the driver alone (no torch, no CUDA init)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def providers_for(capabilities: list[str]) -> tuple[list[str], list[str]]:
    """(providers to install, capabilities that have none), each in first-seen order."""
    wanted: list[str] = []
    missing: list[str] = []
    for capability in capabilities:
        provider = CAPABILITY_TO_PROVIDER.get(capability)
        if provider is None:
            if capability not in missing:
                missing.append(capability)
        elif provider not in wanted:
            wanted.append(provider)
    return wanted, missing


def parse_arch_list(value: str) -> list[str]:
    """Provider names from a comma list; anything unknown is an error rather than a skipped item."""
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in PROVIDERS]
    if unknown:
        raise ValueError(
            f"unknown provider(s) {', '.join(unknown)}; choose from {', '.join(sorted(PROVIDERS))}"
        )
    return list(dict.fromkeys(names))


def wheel_version(flashinfer_version: str, tag: str) -> str:
    """The shim and provider version FlashInfer accepts: its own, plus the CUDA local label."""
    return f"{flashinfer_version}+{tag}"


def distribution_name(provider: str) -> str:
    """Distribution name of one provider, e.g. flashinfer-jit-cache-sm120f."""
    return f"{SHIM}-{provider}"


def present(providers: list[str], version: str) -> bool:
    """The shim and every requested provider are installed at exactly ``version``."""
    return all(
        installed_version(name) == version for name in [SHIM, *map(distribution_name, providers)]
    )


def find_uv() -> str | None:
    """Locate uv: PATH, then $UV, then the per-user install location."""
    found = shutil.which("uv") or os.environ.get("UV")
    if found:
        return found
    candidate = os.path.expanduser("~/.local/bin/uv")
    return candidate if os.access(candidate, os.X_OK) else None


def install(uv: str, index: str, version: str, providers: list[str]) -> bool:
    """Install the shim and providers at one version, without dependencies. True on success."""
    requirements = [
        f"{SHIM}=={version}",
        *(f"{distribution_name(p)}=={version}" for p in providers),
    ]
    command = [
        uv, "pip", "install", "--python", sys.executable, "--no-deps",
        "--index-url", index, *requirements,
    ]  # fmt: skip
    return subprocess.run(command, check=False).returncode == 0


def uninstall(uv: str, providers: list[str]) -> None:
    """Remove the shim and providers again."""
    names = [SHIM, *map(distribution_name, providers)]
    subprocess.run([uv, "pip", "uninstall", "--python", sys.executable, *names], check=False)


def verify(providers: list[str], version: str) -> str | None:
    """None when FlashInfer imports and discovers the providers; otherwise what went wrong."""
    code = (
        "import flashinfer_jit_cache as c\n"
        "import flashinfer.jit.env\n"
        "print(','.join(sorted(p.distribution for p in c.get_jit_cache_providers())))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        return lines[-1] if lines else "import flashinfer failed"
    found = set(result.stdout.strip().split(","))
    absent = [distribution_name(p) for p in providers if distribution_name(p) not in found]
    if absent:
        return f"FlashInfer did not discover {', '.join(absent)} (version mismatch with flashinfer-python?)"
    return None


def main(argv: list[str] | None = None) -> int:
    """Install what this machine needs; return one of the EXIT_ codes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report the state; install nothing")
    parser.add_argument(
        "--arch", help="provider list (sm120f,sm89,...) instead of detecting the GPU"
    )
    args = parser.parse_args(argv)

    if os.environ.get("TRUSTA_FLASHINFER_JIT_CACHE", "1") == "0":
        log("skipped (TRUSTA_FLASHINFER_JIT_CACHE=0)")
        return EXIT_NOT_APPLICABLE

    flashinfer_version = installed_version("flashinfer-python")
    if flashinfer_version is None:
        log("flashinfer-python is not installed; nothing to do")
        return EXIT_NOT_APPLICABLE
    tag = cuda_tag()
    if tag is None:
        log("this torch is not a CUDA build; there are no prebuilt kernels for it")
        return EXIT_NOT_APPLICABLE

    override = args.arch or os.environ.get("TRUSTA_FLASHINFER_JIT_ARCH")
    try:
        if override:
            wanted, missing = parse_arch_list(override), []
        else:
            wanted, missing = providers_for(gpu_capabilities())
    except ValueError as error:
        log(f"WARNING: {error}")
        return EXIT_COMPILES
    for capability in missing:
        log(
            f"no prebuilt kernels for compute capability {capability}; they will compile on first use"
        )
    if not wanted:
        if not missing:
            log("no NVIDIA GPU is visible; set TRUSTA_FLASHINFER_JIT_ARCH to install without one")
        return EXIT_COMPILES

    version = wheel_version(flashinfer_version, tag)
    if present(wanted, version) and verify(wanted, version) is None:
        log(f"already installed for {', '.join(wanted)} ({version})")
        return EXIT_OK
    if args.check:
        log(f"not installed for {', '.join(wanted)} ({version})")
        return 1

    uv = find_uv()
    if uv is None:
        log("WARNING: uv was not found; the kernels will compile on first use")
        return EXIT_COMPILES
    index = os.environ.get("TRUSTA_FLASHINFER_JIT_CACHE_INDEX", f"{DEFAULT_INDEX_BASE}/{tag}")
    log(
        f"installing {SHIM} + {', '.join(distribution_name(p) for p in wanted)} {version} from {index}"
    )
    if not install(uv, index, version, wanted):
        # uv resolves before it installs, so a failed run leaves the environment as it was.
        log("WARNING: the download failed; the kernels will compile on first use")
        return EXIT_COMPILES
    problem = verify(wanted, version)
    if problem is not None:
        uninstall(uv, wanted)
        log(f"WARNING: removed again, because {problem}; the kernels will compile on first use")
        return EXIT_COMPILES
    log(f"installed for {', '.join(wanted)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
