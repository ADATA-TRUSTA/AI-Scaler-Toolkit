"""
Probe a llama binary: which compute backend it has, and whether its CPU backend was built
with this host's vector instructions.

Both questions matter, and the second one is easy to miss. ``ggml-org/llama-install.sh``'s
CUDA/ROCm presets never set ``LLAMA_INSTALL_FLAGS``, so those binaries are genuinely CUDA
builds whose *CPU* backend has no vector ISA at all. A check that only asks "did I get
CUDA?" passes them. Measured on one host, that costs 2.1x once any weight is computed on
the CPU (partial offload, ``--n-cpu-moe``) and 5.1x on a pure-CPU workload:
https://github.com/samhong5668/llama-bench-lab

Neither probe needs a model. ``--list-devices`` prints the devices on stdout, and ggml logs
the backends it loaded on stderr:

  load_backend: loaded CPU backend from ...\\ggml-cpu-alderlake.dll

The variant name in that path is the ISA tier the runtime dispatcher chose, so a release
build reports its own answer. A build without that dispatch logs nothing there and can only
be reported as unverified, never failed - a statically linked llama.app is one such case.

So is our own Linux install, and that is the normal state there rather than a gap: setup_env
compiles it with ``GGML_NATIVE=ON``, which targets the machine doing the compiling, so there
are no run-time variants to choose between and the ISA is right by construction. In practice
this check therefore does its work on Windows, where the release ships all 14 variants and
picks one at run time.

It is gated on the shape of the build, though, not on the platform - note the ``.so`` in the
pattern below and the ``/proc/cpuinfo`` branch in :func:`host_has_avx2`. Both earn their keep
the moment someone sets ``LLAMA_SERVER_BINARY``, which setup_env itself offers as the way to
use your own binary: whatever it points at was compiled somewhere else, by someone else, with
unknown flags. That is exactly the case this check exists for, and on Linux it is also the
only route open to anyone who cannot install a toolchain to build with.

Usage:
  python scripts/llama_backend.py <binary> [--require cuda|vulkan|cpu] [--json]

Exit codes: 0 = acceptable, 1 = degraded or wrong backend (reasons on stdout).
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Probing must not hang the whole setup: an unusable binary can block on a driver call.
PROBE_TIMEOUT_SEC = 60

# ggml's CPU variants by ISA tier. Only one boundary matters here: everything from
# "haswell" up has AVX2, everything below it does not.
_CPU_VARIANT_TIERS: dict[str, int] = {
    "x64": 0,
    "sse42": 1,
    "sandybridge": 2,
    "ivybridge": 2,
    "piledriver": 2,
    "haswell": 3,
    "alderlake": 3,
    "zen4": 4,
    "skylakex": 4,
    "cannonlake": 4,
    "cascadelake": 4,
    "cooperlake": 4,
    "icelake": 4,
    "sapphirerapids": 4,
}
_TIER_AVX2 = 3

# stdout of --list-devices: "  CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15172 MiB free)"
_DEVICE_RE = re.compile(r"^\s*(?P<kind>[A-Za-z]+)(?P<index>\d+):\s*(?P<name>.+?)\s*$")
# ...and the memory suffix to drop from the name. Matched as its own trailing group rather
# than by cutting the name at its first "(", which truncated every vendor whose name
# contains parentheses - "Intel(R) Graphics (74398 MiB, ...)" was reported as "Intel".
_DEVICE_MEM_RE = re.compile(r"\s*\([^()]*MiB[^()]*\)\s*$")
# stderr: "load_backend: loaded CPU backend from <path>/ggml-cpu-alderlake.dll"
_CPU_BACKEND_RE = re.compile(r"loaded CPU backend from .*ggml-cpu-([a-z0-9]+)\.(?:dll|so)", re.I)
# llama.app prints "b10107-c0bc8591e"; a release prints "version: 10107 (c0bc8591e)".
_LLAMA_APP_VERSION_RE = re.compile(r"^b\d+-[0-9a-f]{7,}\s*$")

# llama-server -v colours its log; strip that before matching.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_ISA_LABEL = {True: "ok", False: "DEGRADED", None: "unverified"}


@dataclass
class Probe:
    """What could be established about a binary. ``None`` means "could not tell"."""

    binary: str
    ok: bool
    compute: str | None  # "cuda" | "vulkan" | "rocm" | "other" | "cpu" | None
    devices: list[str]
    cpu_variant: str | None  # e.g. "alderlake"
    cpu_isa_ok: bool | None  # None = unverified, which is not the same as failed
    flavor: str  # "release" | "llama.app" | "unknown"
    version: str | None
    reasons: list[str]


def host_has_avx2() -> bool | None:
    """
    Report whether this host supports AVX2.

    Returns True/False, or None when it cannot be determined - callers must treat None as
    "do not judge" rather than as a missing feature.
    """
    if os.name == "nt":
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40. Known since Windows 8.1; an older kernel
        # returns 0 for a feature it does not know, which would understate the host, so
        # only trust a negative answer from 8.1 upwards.
        try:
            if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
                return True
            return False if sys.getwindowsversion()[:2] >= (6, 3) else None
        except (AttributeError, OSError):
            return None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(("flags", "Features")):
                    return "avx2" in line.split(":", 1)[1].split()
    except OSError:
        return None
    return None


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a probe command; returns (-1, "", reason) when it could not run at all."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROBE_TIMEOUT_SEC,
            check=False,
            # A degraded binary is exactly where a missing DLL is likely, and on Windows
            # that pops a modal loader dialog that would hang an unattended setup.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return -1, "", f"timed out after {PROBE_TIMEOUT_SEC}s"
    except OSError as exc:
        return -1, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _is_unified(binary: str) -> bool:
    """
    Report whether this is llama.app's single multi-tool ``llama``, which needs a subcommand.

    Matched on the exact stem rather than "does the name lack 'server'", which is how
    service/inference/engines/llama_server_engine.py decides. That test is fine there - it only
    ever sees the configured server binary - but this function is handed arbitrary tools, and
    llama-bench / llama-quantize / llama-fit-params all lack "server" while taking their
    arguments directly. Passing them a `server` subcommand made the probe report no devices at
    all on a working CUDA install.
    """
    return Path(binary).stem.lower() == "llama"


def _read_version(base: list[str]) -> str | None:
    """
    Return the first line of --version, or None.

    Only a zero exit counts: llama-bench has no --version and answers with its whole usage
    block, which would otherwise be recorded as the version string.
    """
    rc, out, err = _run([*base, "--version"])
    if rc != 0:
        return None
    for text in (out, err):
        lines = [ln.strip() for ln in _ANSI_RE.sub("", text or "").strip().splitlines()]
        lines = [ln for ln in lines if ln]
        if lines:
            return lines[0]
    return None


def parse_devices(stdout: str) -> list[str]:
    """
    Turn the stdout of ``--list-devices`` into ``["CUDA0: NVIDIA GeForce RTX 5060 Ti", ...]``.

    Split out from :func:`probe_binary` so the parsing can be tested without a llama binary:
    both bugs found here so far were in this step, and neither needed a subprocess to reproduce.
    """
    devices: list[str] = []
    for line in stdout.splitlines():
        m = _DEVICE_RE.match(line)
        if m and m.group("kind").lower() not in {"available", "note"}:
            name = _DEVICE_MEM_RE.sub("", m.group("name"))
            devices.append(f"{m.group('kind')}{m.group('index')}: {name}")
    return devices


def _classify_compute(devices: list[str]) -> str:
    """Map the listed devices onto a backend name."""
    kinds = {d.split(":", 1)[0].rstrip("0123456789").lower() for d in devices}
    for name in ("cuda", "vulkan"):
        if name in kinds:
            return name
    if kinds & {"rocm", "hip"}:
        return "rocm"
    # No accelerator listed: a CPU-only build, or a GPU build that found no usable device.
    return "other" if devices else "cpu"


def probe_binary(binary: str, require: str | None = None) -> Probe:
    """
    Probe ``binary`` and return a :class:`Probe`.

    ``require`` is an optional backend name ("cuda"/"vulkan"/"rocm"/"cpu") that the binary
    must report; a mismatch sets ``ok`` to False.
    """
    reasons: list[str] = []
    if not Path(binary).is_file():
        return Probe(binary, False, None, [], None, None, "unknown", None, ["binary not found"])

    # Absolute path from here on: CreateProcess rejects a relative path written with forward
    # slashes, so "service/utils/llama-bin/llama-server.exe" passes the is_file() check above
    # and then fails to launch with WinError 2.
    binary = str(Path(binary).resolve())

    base = [binary, "server"] if _is_unified(binary) else [binary]
    version = _read_version(base)

    # -v: llama-server logs "load_backend: ..." only when verbose, and that line is the
    # only run-time evidence of which CPU variant was chosen.
    rc, out, err = _run([*base, "-v", "--list-devices"])
    if rc != 0:
        # Not every tool accepts -v; retry without it rather than reporting a false failure.
        rc, out, err = _run([*base, "--list-devices"])
    if rc == -1:
        reasons.append(f"could not run --list-devices: {err}")
        return Probe(binary, False, None, [], None, None, "unknown", version, reasons)
    if rc != 0:
        reasons.append(f"--list-devices exited {rc}")

    out = _ANSI_RE.sub("", out)
    err = _ANSI_RE.sub("", err)

    devices = parse_devices(out)
    compute = _classify_compute(devices)

    # The CPU backend: whichever variant the runtime dispatcher actually loaded.
    m = _CPU_BACKEND_RE.search(err)
    cpu_variant = m.group(1).lower() if m else None

    flavor = "unknown"
    if cpu_variant:
        flavor = "release"
    elif _is_unified(binary):
        # llama.app's own version string ("b10107-c0bc8591e") only comes from the bare
        # binary; `llama server --version` answers with llama-server's release-style line.
        app_version = _read_version([binary])
        if app_version and _LLAMA_APP_VERSION_RE.match(app_version):
            flavor = "llama.app"
            version = app_version

    cpu_isa_ok: bool | None = None
    avx2 = host_has_avx2()
    if cpu_variant is not None:
        tier = _CPU_VARIANT_TIERS.get(cpu_variant)
        if tier is None:
            reasons.append(f"unrecognised CPU variant '{cpu_variant}', not judging its ISA")
        elif avx2 is None:
            reasons.append(f"CPU variant '{cpu_variant}', but this host's AVX2 support is unknown")
        elif avx2 and tier < _TIER_AVX2:
            cpu_isa_ok = False
            reasons.append(
                f"CPU backend is the '{cpu_variant}' variant (no AVX2) on a host that has AVX2 - "
                "this binary computes CPU-side weights without vector instructions"
            )
        else:
            cpu_isa_ok = True
    elif flavor == "llama.app":
        reasons.append(
            "statically linked llama.app binary: its CPU ISA cannot be read at run time, and "
            "llama-install.sh's CUDA/ROCm builds are known to ship a baseline CPU backend. "
            "Prefer a llama.cpp release build or a local build."
        )
    else:
        reasons.append("no ggml CPU variant was logged, so the CPU ISA could not be verified")

    ok = cpu_isa_ok is not False
    if require and compute != require:
        ok = False
        reasons.append(f"expected the {require} backend but this binary reports {compute}")

    return Probe(binary, ok, compute, devices, cpu_variant, cpu_isa_ok, flavor, version, reasons)


def format_probe(probe: Probe) -> str:
    """Render a :class:`Probe` as a short human-readable block."""
    lines = [
        f"binary      : {probe.binary}",
        f"version     : {probe.version or 'unknown'}  ({probe.flavor})",
        f"compute     : {probe.compute or 'unknown'}",
        f"devices     : {', '.join(probe.devices) if probe.devices else 'none'}",
        f"CPU backend : {probe.cpu_variant or 'not reported'}"
        f"  (vector ISA: {_ISA_LABEL[probe.cpu_isa_ok]})",
    ]
    lines += [f"  - {reason}" for reason in probe.reasons]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring for the exit codes."""
    parser = argparse.ArgumentParser(
        description="Probe a llama binary's compute backend and CPU vector ISA."
    )
    parser.add_argument("binary", help="path to llama-server / llama-bench / llama")
    parser.add_argument(
        "--require",
        choices=["cuda", "vulkan", "rocm", "cpu"],
        help="backend the binary must report",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="report a degraded CPU backend but still exit 0 (unattended installs)",
    )
    args = parser.parse_args(argv)

    probe = probe_binary(args.binary, require=args.require)
    print(json.dumps(asdict(probe), indent=2) if args.json else format_probe(probe))
    return 0 if (probe.ok or args.allow_degraded) else 1


if __name__ == "__main__":
    sys.exit(main())
