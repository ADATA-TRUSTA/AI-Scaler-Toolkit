"""
Verify that a torch XPU install can actually compute, not just report availability.

``torch.xpu.is_available()`` returning True is not evidence: the SYCL/oneAPI runtime
that ships inside the torch wheel has to match the installed Intel GPU driver, and when
it does not, the failure only surfaces on the first real kernel. Measured on one Intel
Xe iGPU (0x7D67) with driver 32.0.101.6129:

  torch 2.11.0+xpu (oneAPI 2025.3.x) -> GEMM and a full training step run fine
  torch 2.13.0+xpu (oneAPI 2026.0.x) -> is_available() is True, then the first matmul
                                        raises "could not make an engine with allocator"

An older driver has also been seen to hang kernel compilation instead of raising, so
callers must run this under a timeout (setup_env.ps1 / setup_env.sh both do).

Exit codes: 0 = usable, 1 = not usable (reason on stderr).
"""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

MATRIX_SIZE = 1024
TRAIN_STEPS = 3

# The wheel-side half of the version pair. dpcpp-cpp-rt is the DPC++/SYCL runtime torch's XPU
# build links against; it is a plain pip dependency, so its version is the one that has to be
# new enough for the installed graphics driver.
RUNTIME_PACKAGE = "dpcpp-cpp-rt"


def _fail(message: str) -> int:
    print(f"[xpu_smoke] FAIL: {message}", file=sys.stderr)
    return 1


def _bundled_runtime_version() -> str:
    """The Intel runtime version that ships inside this environment's torch wheel."""
    try:
        return version(RUNTIME_PACKAGE)
    except PackageNotFoundError:
        return "unknown"


def _installed_driver_version() -> str:
    """The Intel graphics driver version installed on this machine, or 'unknown'."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | "
                    "Where-Object { $_.Name -like '*Intel*' } | "
                    "Select-Object -First 1).DriverVersion",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        else:
            # The compute-runtime (NEO) package carries the Level Zero driver the wheel talks to.
            out = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", "intel-opencl-icd"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        reported = out.stdout.strip()
        return reported or "unknown"
    except Exception:  # noqa: BLE001 - a diagnostic must never mask the real failure
        return "unknown"


def _version_pair_note() -> str:
    """The two numbers a reader needs to decide whether their driver is too old."""
    return (
        f"[xpu_smoke] bundled Intel runtime ({RUNTIME_PACKAGE}): {_bundled_runtime_version()}\n"
        f"[xpu_smoke] installed Intel graphics driver: {_installed_driver_version()}"
    )


def _check_device(torch: ModuleType, index: int) -> None:
    """Run a real GEMM and a real optimizer step on one XPU device."""
    device = f"xpu:{index}"

    # 1. A real GEMM. This is where a runtime/driver mismatch shows up.
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)
    (a @ b).sum().item()
    torch.xpu.synchronize(index)
    print(f"[xpu_smoke] {device} GEMM ok")

    # 2. A real training step: forward + backward + optimizer, the path fine-tuning uses.
    model = torch.nn.Linear(256, 256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(16, 256, device=device)
    y = torch.randn(16, 256, device=device)
    for _ in range(TRAIN_STEPS):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    torch.xpu.synchronize(index)
    print(f"[xpu_smoke] {device} training step ok (loss {loss.item():.4f})")


def main() -> int:
    """Verify every visible XPU device can actually compute."""
    try:
        import torch
    except ImportError as exc:
        return _fail(f"torch is not installed in this environment ({exc})")

    if not torch.xpu.is_available():
        return _fail(
            "torch.xpu.is_available() is False: the build has no XPU support "
            "(is this the cuda extra?) or no Intel GPU was detected\n" + _version_pair_note()
        )

    count = torch.xpu.device_count()
    if count < 1:
        return _fail("torch.xpu.is_available() is True but device_count() is 0")

    # Every device is checked: a mismatch can be per-card (e.g. an iGPU plus a discrete Arc),
    # so proving device 0 says nothing about the one the service is configured to use.
    names = ", ".join(f"xpu:{i} {torch.xpu.get_device_name(i)}" for i in range(count))
    print(f"[xpu_smoke] torch {torch.__version__} on {count} device(s): {names}")

    for index in range(count):
        try:
            _check_device(torch, index)
        except Exception as exc:  # noqa: BLE001 - any failure here means the install is unusable
            return _fail(
                f"xpu:{index} ({torch.xpu.get_device_name(index)}) -> {type(exc).__name__}: {exc}\n"
                "[xpu_smoke] The XPU runtime bundled with this torch build cannot drive the "
                "installed Intel GPU driver. Update the Intel graphics driver, or pin a torch "
                "version whose oneAPI runtime matches it. Installing the oneAPI Base Toolkit "
                "does NOT help: the runtime comes from the wheel.\n" + _version_pair_note()
            )

    print("[xpu_smoke] XPU is usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
