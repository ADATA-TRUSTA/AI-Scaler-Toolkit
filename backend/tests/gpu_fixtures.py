"""
GPU-count detection for tests that need to know what the *machine* has.

Kept next to `model_fixtures.py` and imported the same way (`from gpu_fixtures
import ...`), which works from any tests subfolder because conftest.py puts the
tests directory on sys.path.

Why not `torch.cuda.device_count()` alone: that reports what this *process* can
see, and CUDA_VISIBLE_DEVICES changes it. `training_process.py` pins the
single-GPU worker to "0", so a test that inherits such an environment would
conclude the host has one card. Worse, torch caches the count after CUDA
initialises, so a later env change does not move it. `nvidia-smi -L` asks the
driver about physical devices and is unaffected by both.
"""

import os
import subprocess

MULTI_GPU_MIN_COUNT = 2


def nvidia_gpu_count() -> int:
    """
    Number of NVIDIA GPUs physically present, ignoring CUDA_VISIBLE_DEVICES.

    Returns 0 when there is no NVIDIA driver at all, so callers can treat
    "no GPU" and "one GPU" through the same comparison.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        pass  # no binary, or it hung -- fall through to torch

    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def visible_gpu_count() -> int:
    """
    Number of GPUs *this process* can use, i.e. after CUDA_VISIBLE_DEVICES.

    Reported alongside `nvidia_gpu_count()` in skip reasons: "the host has 2 but
    this process sees 1" is a different problem from "the host has 1", and a
    test that cannot tell them apart sends you looking in the wrong place.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        return len([part for part in raw.split(",") if part.strip() != ""])
    return nvidia_gpu_count()


def multi_gpu_skip_reason() -> str | None:
    """None when a real multi-GPU run is possible here, else why it is not."""
    present = nvidia_gpu_count()
    if present < MULTI_GPU_MIN_COUNT:
        return f"needs >= {MULTI_GPU_MIN_COUNT} NVIDIA GPUs, the host has {present}"
    visible = visible_gpu_count()
    if visible < MULTI_GPU_MIN_COUNT:
        return (
            f"the host has {present} NVIDIA GPUs but CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} leaves only {visible} visible"
        )
    return None
