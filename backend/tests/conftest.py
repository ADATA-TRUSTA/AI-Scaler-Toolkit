import os
import platform
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Allow `import model_fixtures` from any tests subfolder. (A plain
# `import tests.model_fixtures` is unreliable: a dependency ships a regular
# top-level `tests` package in site-packages that shadows this namespace.)
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(1, str(TESTS_DIR))

# DeepSpeed JIT-builds its C++/CUDA ops (CPUAdam for CPU offload, async_io for
# NVMe offload) via torch.utils.cpp_extension, which shells out to `ninja` on
# PATH. `ninja` ships as a wheel, so it lives in the interpreter's own bin/ —
# but running `.venv/bin/python -m pytest` without activating the venv leaves
# that directory off PATH. The op builders then report "ninja not installed"
# and the affected tests *skip* rather than fail, silently zeroing DeepSpeed
# coverage. Prepend it here so a bare pytest invocation behaves like
# scripts/{linux/run_service.sh,windows/run_service.bat}, which export the same.
# sysconfig, not Path(sys.executable).parent: .venv/bin/python is a symlink to the
# system interpreter, so resolving it lands in /usr/bin. get_path("scripts") gives
# the environment's own script dir and is correct on Windows (Scripts\) too.
_SCRIPTS_DIR = sysconfig.get_path("scripts")
if _SCRIPTS_DIR and _SCRIPTS_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _SCRIPTS_DIR + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------
# Memory measurement for tests
#
# Training jobs sample memory through the job logger, but pytest never touches
# that, so a test run recorded nothing at all -- there was no way to answer
# "how much GPU/DRAM/SSD did this test use". `measure_memory` samples for the
# duration of a test and reports average and peak for each.
# --------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

MEMORY_REPORT_PATH = PROJECT_ROOT / "tests" / "results" / "memory" / "latest.jsonl"


class _DescendantSampler:
    """
    Peak GPU and RSS held by this process's *descendants*.

    `mem_probe` reads /proc/self/status, /proc/self/io and torch's own counters, so
    it measures the sampling process. That is exactly right for in-process training
    and blind for the multi-GPU path, which trains in a `deepspeed --num_gpus N`
    subprocess: the fixture reported `GPU peak=0MB` across 702 samples for a run
    that really used ~10 GB per rank.

    Zero is worse than absent. The offload matrix asserts
    `peak <= gpu_peak_ceiling_mb`, and 0 satisfies every ceiling -- so the one
    assertion written to catch "offload silently stopped offloading" passed
    unconditionally on the very path where the model is furthest from the GPU.

    Scoped to our own descendants deliberately: a machine-wide total folds in
    whatever else holds VRAM (this box runs an inference server on GPU0 holding
    27 GB), which would break the ceilings in the opposite direction.
    """

    # Deliberately slower than mem_probe's in-process cadence. Each tick shells out
    # to nvidia-smi (~50-100ms); at the 0.5s default that is a fifth of a core burning
    # for the hours the 35B cases take. This sampler exists to catch a rank that keeps
    # weights resident, which is a sustained condition, not a spike -- and it cannot
    # see spikes anyway, having no high-water mark to read.
    MIN_INTERVAL_S = 2.0

    def __init__(self, interval: float) -> None:
        self.interval = max(interval, self.MIN_INTERVAL_S)
        self.peak_proc_mb = 0.0  # largest single descendant -- comparable to a per-rank ceiling
        self.peak_total_mb = 0.0  # all descendants at once, across every card
        self.peak_rss_mb = 0.0
        self.pids: set[int] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _descendants(self) -> dict[int, float]:
        """{pid: rss_mb} for every live descendant."""
        try:
            import psutil

            out = {}
            for child in psutil.Process().children(recursive=True):
                try:
                    out[child.pid] = child.memory_info().rss / (1024.0**2)
                except psutil.Error:
                    continue  # exited between listing and reading
            return out
        except Exception:
            return {}

    def _gpu_by_pid(self) -> dict[int, float]:
        """{pid: gpu_mb} from the driver, which is the only thing that sees a subprocess."""
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if res.returncode != 0:
            return {}
        found: dict[int, float] = {}
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                # One process appears once per card it touches; sum those.
                found[int(parts[0])] = found.get(int(parts[0]), 0.0) + float(parts[1])
            except ValueError:
                continue
        return found

    def _loop(self) -> None:
        while not self._stop.is_set():
            kids = self._descendants()
            if kids:
                self.peak_rss_mb = max(self.peak_rss_mb, sum(kids.values()))
                gpu = self._gpu_by_pid()
                mine = {pid: mb for pid, mb in gpu.items() if pid in kids}
                if mine:
                    self.pids.update(mine)
                    self.peak_proc_mb = max(self.peak_proc_mb, max(mine.values()))
                    self.peak_total_mb = max(self.peak_total_mb, sum(mine.values()))
            self._stop.wait(self.interval)


class _TreeAwareAggregator:
    """
    `MemoryAggregator` plus what the descendants used.

    Wraps rather than replaces so the in-process numbers keep their exact meaning
    (the five single-GPU offload cases assert against them). `summary()` only
    substitutes the GPU peak when this process observed none *and* a descendant
    did -- i.e. when the training demonstrably happened somewhere else.

    `summary()` has to be correct mid-test, not just at teardown: the offload
    matrix reads it inside the test body.
    """

    def __init__(self, inner, sampler: _DescendantSampler) -> None:
        self._inner = inner
        self._sampler = sampler

    def __getattr__(self, name):  # delegate add(), and anything else mem_probe grows
        return getattr(self._inner, name)

    def summary(self) -> dict:
        out = self._inner.summary()
        sampler = self._sampler
        if sampler.pids or sampler.peak_rss_mb:
            out["descendants"] = {
                "gpu_peak_per_process_mb": round(sampler.peak_proc_mb, 1),
                "gpu_peak_total_mb": round(sampler.peak_total_mb, 1),
                "rss_peak_mb": round(sampler.peak_rss_mb, 1),
                "pids": sorted(sampler.pids),
            }
        in_proc = (out.get("cuda_alloc_mb") or {}).get("peak") or 0
        if not in_proc and sampler.peak_proc_mb:
            out["cuda_alloc_mb"] = {
                **(out.get("cuda_alloc_mb") or {}),
                "peak": round(sampler.peak_proc_mb, 1),
                "source": "largest descendant process, via nvidia-smi",
            }
        return out

    def format_line(self) -> str:
        line = self._inner.format_line()
        s = self._sampler
        if s.pids:
            line += (
                f"  CHILD GPU per-proc peak={s.peak_proc_mb:.0f}MB "
                f"total={s.peak_total_mb:.0f}MB over {len(s.pids)} pid(s)"
            )
        return line


@pytest.fixture
def measure_memory(request):
    """
    Sample GPU/DRAM/SSD for one test and record avg + peak.

    Yields the aggregator so a test can read the summary itself; the recorded
    line lands in tests/results/memory/latest.jsonl keyed by test id.

    Peaks come from torch's and the kernel's own high-water marks where they
    exist, so a spike between two samples is still captured -- polling alone
    would miss exactly the all-gather peaks that decide whether a run fits.

    Descendant processes are sampled too, so a test that trains through the
    multi-GPU launcher reports the ranks' usage instead of this process's zero.
    See _DescendantSampler for why a zero there was actively harmful.
    """
    from service.training.core.mem_probe import MemoryAggregator, start_memory_sampler

    try:
        import torch

        if torch.cuda.is_available():
            # Scope the GPU high-water mark to this test.
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    interval = float(os.environ.get("TESTS_MEM_SAMPLE_INTERVAL", "0.5"))
    offload_path = os.environ.get("DEEPSPEED_NVME_DIR")
    aggregator = MemoryAggregator()
    thread, stop = start_memory_sampler(
        interval=interval,
        offload_path=offload_path,
        aggregator=aggregator,
    )
    descendants = _DescendantSampler(interval)
    descendants.start()
    tracked = _TreeAwareAggregator(aggregator, descendants)
    started = time.time()
    try:
        yield tracked
    finally:
        stop.set()
        thread.join(timeout=5)
        descendants.stop()
        summary = tracked.summary()
        summary["test"] = request.node.nodeid
        summary["duration_s"] = round(time.time() - started, 2)
        try:
            MEMORY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MEMORY_REPORT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")
        except Exception:
            pass  # a reporting failure must never fail the test
        # Also surface it in -s output and in the terminal report.
        line = tracked.format_line()
        print(f"\n{line}  [{request.node.name}]")
        request.node.user_properties.append(("memory", summary))


# --------------------------------------------------------------------------
# GPU count, for tests that scale their case list by the hardware
#
# The detection itself lives in gpu_fixtures.py so a test module can decorate
# with it at import time (a fixture is too late to build a skipif).
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu_count() -> int:
    """NVIDIA GPUs physically present, regardless of CUDA_VISIBLE_DEVICES."""
    from gpu_fixtures import nvidia_gpu_count

    return nvidia_gpu_count()


# --------------------------------------------------------------------------
# Linux-only features, on a suite that runs on both platforms
#
# Fine-tuning and vLLM only run on Linux; which tests that covers, and why, is
# documented in tests/platform_gating.py. This is the half that acts on it.
#
# Two ways in, deliberately: the training tree is marked here by path, so a new
# file under tests/unit/training/ is gated without anyone remembering to mark it;
# the five vLLM modules carry `pytest.mark.linux_only` in the files themselves,
# because they sit beside ~20 cross-platform siblings and a reader opening one
# needs to see the gate in it.
# --------------------------------------------------------------------------

_SKIP_REASON = "{feature} is Linux-only (see service/platform_support.py); this host is {system}"


def pytest_collection_modifyitems(config, items):  # noqa: ANN001, ANN201
    """
    Mark the fine-tuning tree, then skip everything Linux-only when off Linux.

    The `finetune` marker is applied here too, not just `linux_only`: it had been
    hand-written into 11 of the 23 files under tests/unit/training/, so
    `-m "not finetune"` -- the other obvious way to sit out these tests -- was
    quietly still selecting the other 12. Deriving it from the path fixes that
    and keeps the two mechanisms agreeing with each other.
    """
    from platform_gating import is_finetune_path

    on_linux = platform.system() == "Linux"

    for item in items:
        if item.path is not None and is_finetune_path(item.path):
            item.add_marker(pytest.mark.finetune)
            item.add_marker(pytest.mark.linux_only)

        if on_linux or "linux_only" not in item.keywords:
            continue

        # Fine-tuning and vLLM are refused for different reasons; name the one
        # that applies so the skip line says which feature the reader lost.
        feature = "Fine-tuning" if "finetune" in item.keywords else "The vLLM engine"
        item.add_marker(
            pytest.mark.skip(reason=_SKIP_REASON.format(feature=feature, system=platform.system()))
        )
