"""Layer 1 — torch XPU: the Intel GPU backend that everything else sits on.

Verifies that PyTorch's native XPU backend is live and can actually compute,
using ONLY the pip `torch==2.13.0+xpu` bundled SYCL runtime — no oneAPI
`setvars` sourced (matching scripts/*/setup_env, which no longer pulls oneAPI).
If torch can talk to the Intel GPU here, the transformers and llama layers have
a foundation to stand on.

Auto-skips on any machine without a usable Intel XPU (e.g. the CUDA dev box),
so it is safe to run everywhere:

    pytest tests/xpu/test_torch_xpu.py -s
"""

import os

import pytest
import torch


def _xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(not _xpu_available(), reason="no Intel XPU"),
]


def test_torch_reports_at_least_one_xpu_device():
    count = torch.xpu.device_count()
    assert count >= 1, "torch.xpu.is_available() is True but device_count() == 0"
    name = torch.xpu.get_device_name(0)
    assert isinstance(name, str) and name.strip(), "empty XPU device name"
    print(f"\n[torch-xpu] {count} device(s); device 0 = {name!r}")


def test_torch_xpu_works_without_oneapi_setvars():
    """The whole point of the +xpu wheel: XPU runs off the bundled SYCL
    runtime, so it must work whether or not a oneAPI environment is sourced.

    We don't require oneAPI to be absent (a dev box may have it installed) —
    we require that XPU is usable regardless, i.e. we never depend on setvars.
    """
    sourced = [
        v
        for v in ("SETVARS_COMPLETED", "ONEAPI_ROOT", "CMPLR_ROOT")
        if os.environ.get(v)
    ]
    print(
        f"\n[torch-xpu] oneAPI env markers present: {sourced or 'none'} (not relied upon)"
    )

    x = torch.ones(8, device="xpu")
    assert x.device.type == "xpu"
    assert torch.equal(x.sum().cpu(), torch.tensor(8.0))


def test_xpu_matmul_matches_cpu():
    """A real GEMM on the Intel GPU must agree with the CPU reference — this is
    the kernel path that a stale Intel driver is known to hang/miscompile."""
    torch.manual_seed(0)
    a = torch.randn(128, 128)
    b = torch.randn(128, 128)

    expected = a @ b
    got = (a.to("xpu") @ b.to("xpu")).cpu()

    torch.xpu.synchronize()
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (
        "XPU matmul diverged from CPU"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xpu_half_precision_compute(dtype):
    """fp16 and bf16 are the dtypes inference/training actually load weights in;
    both must compute on the Intel GPU without falling over."""
    a = torch.randn(64, 64, dtype=dtype, device="xpu")
    b = torch.randn(64, 64, dtype=dtype, device="xpu")

    out = a @ b
    torch.xpu.synchronize()

    assert out.dtype == dtype
    assert out.device.type == "xpu"
    assert torch.isfinite(out.float()).all(), (
        f"{dtype} matmul produced non-finite values"
    )


def test_xpu_host_device_roundtrip_preserves_values():
    src = torch.arange(1000, dtype=torch.float32)
    back = src.to("xpu").cpu()
    assert torch.equal(src, back), "CPU -> XPU -> CPU round trip lost data"


def test_xpu_memory_and_cleanup_api():
    """The allocator + cleanup calls that model_utils._cleanup_memory() relies
    on for XPU must exist and run (older torch.xpu builds shipped partial APIs)."""
    before = torch.xpu.memory_allocated()
    keep = torch.empty(1024 * 1024, device="xpu")  # ~4 MiB, held past the measurement
    torch.xpu.synchronize()
    assert torch.xpu.memory_allocated() >= before

    assert torch.xpu.memory_reserved() >= torch.xpu.memory_allocated()

    del keep
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    # reset_* are optional across torch.xpu versions; call only if present.
    if hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()
