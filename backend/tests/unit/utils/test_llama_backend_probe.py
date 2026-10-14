"""
Unit tests for the parsing in scripts/llama_backend.py.

The probe decides whether setup_env accepts or rejects an install, so a parsing slip makes it
either wave through a degraded binary or reject a good one. Both bugs found in it so far lived
in this layer and neither needed a real llama binary to reproduce:

- device names were cut at their first "(", so "Intel(R) Graphics (74398 MiB, ...)" was reported
  as a device named "Intel"
- every tool whose name merely lacked "server" was treated as llama.app's unified binary, so
  llama-fit-params was invoked as `llama-fit-params server ...` and reported no devices at all

scripts/ is not a package, so the module is loaded by path.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO_ROOT / "scripts" / "llama_backend.py"


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llama_backend_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# --- device parsing -------------------------------------------------------------------------

CUDA_STDOUT = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15172 MiB free)
"""

VULKAN_STDOUT = """Available devices:
  Vulkan0: NVIDIA GeForce RTX 5060 Ti (16050 MiB, 15282 MiB free)
  Vulkan1: Intel(R) Graphics (74398 MiB, 103650 MiB free)
"""


def test_a_vendor_name_containing_parentheses_survives() -> None:
    """The regression: "Intel(R) Graphics" must not be truncated to "Intel"."""
    assert probe.parse_devices(VULKAN_STDOUT) == [
        "Vulkan0: NVIDIA GeForce RTX 5060 Ti",
        "Vulkan1: Intel(R) Graphics",
    ]


def test_the_memory_suffix_is_dropped() -> None:
    """The "(16310 MiB, 15172 MiB free)" tail is noise for every caller."""
    assert probe.parse_devices(CUDA_STDOUT) == ["CUDA0: NVIDIA GeForce RTX 5060 Ti"]


def test_a_device_without_a_memory_suffix_keeps_its_whole_name() -> None:
    """Only a trailing MiB group is stripped, not any trailing parenthesis."""
    assert probe.parse_devices("  CUDA0: Some Card (Rev 2)\n") == ["CUDA0: Some Card (Rev 2)"]


def test_the_header_is_not_mistaken_for_a_device() -> None:
    """The `Available devices:` header has no index, and empty output means no devices."""
    assert probe.parse_devices("Available devices:\n") == []
    assert probe.parse_devices("") == []


# --- backend classification -----------------------------------------------------------------


def test_devices_map_onto_a_backend_name() -> None:
    """An empty list is a CPU-only build; anything unrecognised must not read as CPU."""
    assert probe._classify_compute(["CUDA0: x"]) == "cuda"
    assert probe._classify_compute(["Vulkan0: x", "Vulkan1: y"]) == "vulkan"
    assert probe._classify_compute(["ROCm0: x"]) == "rocm"
    assert probe._classify_compute([]) == "cpu"
    assert probe._classify_compute(["SYCL0: x"]) == "other"


# --- which binaries need a subcommand -------------------------------------------------------


def test_only_the_bare_llama_binary_is_unified() -> None:
    """
    The regression: a `server` subcommand belongs to llama.app's multi-tool `llama` alone.

    llama-bench / llama-quantize / llama-fit-params all lack "server" in their names while
    taking their arguments directly, and passing them a subcommand makes them report nothing.
    """
    for unified in ("llama", "llama.exe", "LLAMA.EXE", "/opt/llama.app/llama"):
        assert probe._is_unified(unified), unified
    for direct in (
        "llama-server",
        "llama-server.exe",
        "llama-bench.exe",
        "llama-quantize.exe",
        "llama-fit-params.exe",
        "/usr/local/bin/llama-cli",
    ):
        assert not probe._is_unified(direct), direct


@pytest.mark.skipif(os.name != "nt", reason="a drive-letter path only splits under WindowsPath")
def test_a_windows_path_to_the_unified_binary_is_recognised() -> None:
    """
    The same rule in the form the Windows install produces, kept off Linux deliberately.

    Path is WindowsPath only on Windows; under PosixPath a backslash is an ordinary filename
    character, so a drive-letter path has no directory part at all and its stem comes back as
    the whole string. That is a fact about pathlib, not a gap in _is_unified: probe() calls it
    only after an is_file() guard and a resolve(), so what it receives is always a real path in
    this platform's own form, and a Windows path can never reach it on Linux.
    """
    assert probe._is_unified(r"C:\tools\LLAMA.EXE")
    assert not probe._is_unified(r"C:\tools\llama-server.exe")


# --- the AVX2 boundary ----------------------------------------------------------------------


def test_every_named_cpu_variant_sits_on_the_right_side_of_avx2() -> None:
    """
    The whole point of the ISA check is the AVX2 boundary, so the table has to place it right.

    ggml gained AVX2 with Haswell; everything older must rank below _TIER_AVX2 and everything
    Haswell or newer at or above it, or a degraded install passes the check.
    """
    without_avx2 = {"x64", "sse42", "sandybridge", "ivybridge", "piledriver"}
    with_avx2 = {
        "haswell",
        "alderlake",
        "zen4",
        "skylakex",
        "cannonlake",
        "cascadelake",
        "cooperlake",
        "icelake",
        "sapphirerapids",
    }
    assert without_avx2 | with_avx2 == set(probe._CPU_VARIANT_TIERS), (
        "the variant table changed; re-check which side of AVX2 the new entries belong on"
    )
    for name in without_avx2:
        assert probe._CPU_VARIANT_TIERS[name] < probe._TIER_AVX2, name
    for name in with_avx2:
        assert probe._CPU_VARIANT_TIERS[name] >= probe._TIER_AVX2, name


def test_the_cpu_variant_is_read_out_of_the_load_backend_line() -> None:
    """The loaded-variant line is the only run-time evidence of the ISA tier; both OS forms."""
    win = r"load_backend: loaded CPU backend from C:\llama-bin\ggml-cpu-alderlake.dll"
    lin = "load_backend: loaded CPU backend from /opt/llama-bin/ggml-cpu-sse42.so"
    assert probe._CPU_BACKEND_RE.search(win).group(1) == "alderlake"
    assert probe._CPU_BACKEND_RE.search(lin).group(1) == "sse42"
    assert probe._CPU_BACKEND_RE.search("load_backend: loaded CUDA backend from x.dll") is None
