"""Layer 3 — llama.cpp XPU: llama offloading to the Intel GPU via Vulkan.

The XPU accelerator path drives llama.cpp through its Vulkan backend (no
oneAPI/SYCL toolchain required). Two binary shapes are supported:

- what setup_env installs into ``service/utils/llama-bin``: llama.cpp's own
  release zip on Windows, or a local ``-DGGML_VULKAN=ON`` build on Linux
- a unified ``llama`` invoked as ``llama serve``, for anyone who points
  ``LLAMA_SERVER_BINARY`` at one

On a machine with both an Intel and an NVIDIA GPU, Vulkan enumerates both, so
this test parses ``--list-devices`` and pins llama to the *Intel* Vulkan device
— otherwise "llama on XPU" would silently run on the NVIDIA card. It then
confirms from the startup log that layers were offloaded to the Intel GPU and
drives a real completion through the HTTP API.

This drives llama's Vulkan backend, so it deliberately does not require
torch's XPU build: Vulkan enumerates the Intel GPU whether or not torch was
installed with XPU support, and gating on torch.xpu.is_available() made this
coverage unreachable on every machine whose torch is the CUDA build - including
ones where llama on the Intel GPU works fine.

Auto-skips without any llama binary (LLAMA_SERVER_BINARY overrides discovery),
without an Intel Vulkan device, or when the GGUF cannot be provisioned
(tests/model_fixtures.py; LLAMA_GGUF overrides).

    pytest tests/xpu/test_llama_xpu.py -s
"""

import os
import re
import socket
import subprocess
import time
from pathlib import Path

import httpx2
import pytest
from model_fixtures import ModelUnavailable, ensure_gguf

from service.settings import LLAMA_BIN_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_BIN = REPO_ROOT / "service" / "utils" / "llama.cpp" / "build" / "bin"
LOG_DIR = (
    Path(os.environ.get("TESTS_LOG_DIR", str(REPO_ROOT / "tests" / "results")))
    / "xpu_logs"
)


def _find_llama_binary() -> Path | None:
    """Locate a llama binary: source-built llama-server or the unified llama."""
    override = os.environ.get("LLAMA_SERVER_BINARY", "").strip()
    if override:
        return Path(override) if Path(override).is_file() else None

    # Where setup_env puts it, then the source-build layouts. llama-install.sh's own
    # locations are not searched: its CUDA builds ship a CPU backend with no vector ISA,
    # so it has to be opted into through LLAMA_SERVER_BINARY.
    llama_bin = LLAMA_BIN_DIR
    candidates = [
        llama_bin / "llama-server",
        llama_bin / "llama-server.exe",
        BUILD_BIN / "llama-server",
        BUILD_BIN / "llama-server.exe",
        BUILD_BIN / "Release" / "llama-server.exe",
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


LLAMA_BINARY = _find_llama_binary()
# The unified binary needs a `serve` subcommand; llama-server takes args directly.
IS_UNIFIED = LLAMA_BINARY is not None and "server" not in LLAMA_BINARY.stem.lower()


def _base_cmd() -> list[str]:
    return [str(LLAMA_BINARY)] + (["serve"] if IS_UNIFIED else [])


def _intel_vulkan_device() -> str | None:
    """Return the Vulkan device id (e.g. 'Vulkan1') whose name mentions Intel."""
    try:
        out = subprocess.run(
            [*_base_cmd(), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:
        return None
    text = (out.stdout or "") + (out.stderr or "")
    for line in text.splitlines():
        # e.g. "  Vulkan1: Intel(R) Graphics (74398 MiB, 103650 MiB free)"
        m = re.match(r"\s*(\S+):\s*(.+)", line)
        if m and "intel" in m.group(2).lower():
            return m.group(1)
    return None


INTEL_DEVICE = _intel_vulkan_device() if LLAMA_BINARY is not None else None

pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(LLAMA_BINARY is None, reason="no llama binary found"),
    pytest.mark.skipif(
        INTEL_DEVICE is None, reason="no Intel Vulkan device exposed to llama"
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def llama_server():
    """Start llama on the Intel Vulkan device with full GPU offload (-ngl 99)."""
    try:
        gguf_path = ensure_gguf()
    except ModelUnavailable as e:
        pytest.skip(f"GGUF model unavailable: {e}")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"llama_xpu_{port}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    cmd = [
        *_base_cmd(),
        "-m",
        gguf_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        INTEL_DEVICE,  # pin to the Intel GPU, not a sibling NVIDIA card
        "-ngl",
        "99",  # offload every layer to the Vulkan/Intel GPU
        "-c",
        "2048",
        "-lv",
        "5",  # raise verbosity so device/offload lines are logged
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama exited early (code={proc.returncode}); "
                    f"log tail:\n{log_path.read_text(encoding='utf-8', errors='ignore')[-2000:]}"
                )
            try:
                if httpx2.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx2.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"llama never became healthy; "
                f"log tail:\n{log_path.read_text(encoding='utf-8', errors='ignore')[-2000:]}"
            )

        yield {"base_url": base_url, "log_path": log_path}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_file.close()


def test_llama_offloads_to_intel_vulkan_gpu(llama_server):
    """The startup log must show the Vulkan backend picking up the Intel GPU and
    offloading layers — otherwise it silently fell back to CPU or a non-Intel
    device."""
    log = llama_server["log_path"].read_text(encoding="utf-8", errors="ignore")
    lower = log.lower()
    dev = INTEL_DEVICE.lower()

    assert "vulkan" in lower, "no Vulkan backend in llama log (not a Vulkan build?)"
    assert "intel" in lower, f"llama did not bind to the Intel device {INTEL_DEVICE!r}"
    # llama.cpp wording varies by build ("using device Vulkan1" / layers
    # "assigned to device Vulkan1" / "offloaded N/N layers to GPU"); accept any.
    assert (
        f"using device {dev}" in lower
        or f"assigned to device {dev}" in lower
        or ("offloaded" in lower and "to gpu" in lower)
    ), (
        f"llama did not offload layers to the Intel device {INTEL_DEVICE} (CPU fallback?)"
    )


def test_llama_completion_on_xpu(llama_server):
    """A real completion through the HTTP API must return non-empty text."""
    base_url = llama_server["base_url"]
    r = httpx2.post(
        f"{base_url}/completion",
        json={
            "prompt": "The capital of France is",
            "n_predict": 16,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    assert r.status_code == 200, r.text
    content = r.json().get("content", "")
    assert content.strip(), f"empty completion from XPU llama: {r.text[:500]}"
    print(f"\n[llama-xpu] device={INTEL_DEVICE} completion: {content!r}")
