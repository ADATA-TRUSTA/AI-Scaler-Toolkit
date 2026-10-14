"""End-to-end: a MoE model larger than the GPU, with KV offloaded to DRAM and NVMe.

What this proves, against a real vLLM server started by the backend:

1. A MoE checkpoint bigger than the card loads when only its expert weights are
   offloaded (vllm_cpu_offload_gb + vllm_cpu_offload_params=["experts"]).
2. Prefix KV is written through to the NVMe tier (vllm_kv_offloading_disk_path).
3. After an unload and reload -- a fresh vLLM process, so the GPU prefix cache and
   the DRAM tier are both empty -- the same prompt is served from the NVMe tier:
   the fs tier reports hits and bytes read, the block device under the directory
   reports real reads, and the request is much faster than recomputing.

Point 3 is what a DRAM-only check cannot show. Within one server process, a
reused prefix can come from DRAM and the disk is never read; after a restart
DRAM holds nothing, so any hit has to come off the disk. It also covers reuse
across restarts, which needs the prefix-hash seed the service pins
(PYTHONHASHSEED): with a random seed every file name changes and nothing hits.

Isolation follows test_llama_server_e2e.py: the backend runs as a uvicorn
subprocess on a free port, vLLM gets its own free port, and
service/configs/current_inference_config.json is snapshotted and restored.

Auto-skips without Linux + CUDA + vLLM, without the checkpoint in a local hub
cache (never downloads 41 GB), or without the GPU memory, host RAM and disk space
the run needs. Overrides:

- KV_OFFLOAD_E2E_MODEL  hub id of the MoE checkpoint
                        (default btbtyler09/Qwen3.5-35B-A3B-GPTQ-8bit)
- KV_OFFLOAD_E2E_DIR    directory for the NVMe tier; must be on a real block device,
                        not tmpfs (default tests/results/kv_offload_e2e)
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest

from model_fixtures import local_snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = REPO_ROOT / "service" / "configs" / "current_inference_config.json"
LOG_DIR = Path(os.environ.get("TESTS_LOG_DIR", str(REPO_ROOT / "tests" / "results"))) / "e2e_logs"

MODEL_ID = os.environ.get("KV_OFFLOAD_E2E_MODEL", "btbtyler09/Qwen3.5-35B-A3B-GPTQ-8bit")
KV_DIR = Path(
    os.environ.get("KV_OFFLOAD_E2E_DIR", str(REPO_ROOT / "tests" / "results" / "kv_offload_e2e"))
)

# Sizing, all from the measured 16 GB-card run of this model.
CPU_OFFLOAD_GB = 32  # expert weights left in pinned host memory
DRAM_TIER_GB = 4  # holds the ~16k-token prompt (~0.4 GB of offloaded KV) many times over
PROMPT_TOKENS_APPROX = 16000
LOAD_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 900


def _cuda_free_gib() -> float:
    """Free memory on GPU 0 via NVML, which -- unlike torch -- opens no CUDA context.

    A context created here would sit in the pytest process for the whole run and
    take its few hundred MB out of the budget the vLLM server is started with.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**3
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0.0


def _host_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def _block_device_of(path: Path) -> str | None:
    """The whole-disk name under /sys/block holding ``path``, or None (tmpfs, overlay...)."""
    path.mkdir(parents=True, exist_ok=True)
    dev = os.stat(path).st_dev
    sys_dev = Path(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")
    if not sys_dev.exists():
        return None
    real = sys_dev.resolve()
    # A partition's parent directory is the disk; a whole disk is its own entry.
    disk = real.parent if (real / "partition").exists() else real
    return disk.name if (Path("/sys/block") / disk.name / "stat").exists() else None


def _device_bytes_read(device: str) -> int:
    with open(f"/sys/block/{device}/stat") as f:
        return int(f.read().split()[2]) * 512


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


SNAPSHOT = local_snapshot(MODEL_ID) if sys.platform == "linux" else None

pytestmark = [
    pytest.mark.inference,
    pytest.mark.linux_only,
    pytest.mark.kv_offload,
    pytest.mark.skipif(sys.platform != "linux", reason="vLLM is Linux-only"),
    pytest.mark.skipif(not _vllm_importable(), reason="vllm is not installed"),
    pytest.mark.skipif(SNAPSHOT is None, reason=f"{MODEL_ID} is not in a local hub cache"),
    pytest.mark.skipif(_cuda_free_gib() < 13.0, reason="needs ~13 GiB of free GPU memory"),
    pytest.mark.skipif(
        _host_available_gib() < CPU_OFFLOAD_GB + DRAM_TIER_GB + 8,
        reason="needs host RAM for the pinned experts plus the DRAM KV tier",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _metrics(vllm_port: int) -> dict[str, float]:
    """vLLM Prometheus counters, summed over label sets and keyed by tier when present."""
    text = httpx2.get(f"http://127.0.0.1:{vllm_port}/metrics", timeout=10.0).text
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+\-]+)$", line)
        if not m or m.group(1).endswith(("_bucket", "_created")):
            continue
        tier = re.search(r'tier="([^"]+)"', m.group(2) or "")
        key = f"{m.group(1)}[{tier.group(1)}]" if tier else m.group(1)
        out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def _prompt() -> str:
    """A deterministic ~16k-token prompt; identical text both times is the whole point."""
    words = (
        "invoice clause warranty remedy notice term party breach liability indemnity "
        "delivery acceptance payment schedule penalty termination renewal assignment"
    ).split()
    lines = []
    for i in range(PROMPT_TOKENS_APPROX // 14):
        picked = " ".join(words[(i * 7 + k * 3) % len(words)] for k in range(10))
        lines.append(f"Section {i}: {picked}.")
    return "Contract text:\n" + "\n".join(lines) + "\n\nIn one word, what is this?"


@pytest.fixture(scope="module")
def backend():
    """An isolated backend whose vLLM child listens on its own free port."""
    backend_port = _free_port()
    vllm_port = _free_port()
    base_url = f"http://127.0.0.1:{backend_port}"
    config_backup = CONFIG_FILE.read_bytes() if CONFIG_FILE.exists() else None

    device = _block_device_of(KV_DIR)
    if device is None:
        pytest.skip(f"{KV_DIR} is not on a block device (tmpfs?); disk reads cannot be proven")
    if shutil.disk_usage(KV_DIR).free < 20 * 1024**3:
        pytest.skip(f"less than 20 GiB free under {KV_DIR}")
    shutil.rmtree(KV_DIR, ignore_errors=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"backend_kv_offload_{backend_port}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    env = dict(os.environ)
    # Left unset on purpose: the service must pin the prefix-hash seed itself,
    # or the reload phase cannot find what the first phase wrote.
    env.pop("PYTHONHASHSEED", None)
    env.pop("VLLM_USE_SIMPLE_KV_OFFLOAD", None)
    env["VLLM_PORT"] = str(vllm_port)
    env["VLLM_SWEEP_PORTS"] = str(vllm_port)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.app:app", "--host", "127.0.0.1",
         "--port", str(backend_port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"backend exited early; log tail:\n{log_path.read_text()[-2000:]}")
            try:
                if httpx2.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx2.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"backend never became healthy; log tail:\n{log_path.read_text()[-2000:]}")

        yield {"base_url": base_url, "vllm_port": vllm_port, "device": device, "log": log_path}
    finally:
        try:
            httpx2.post(f"{base_url}/inference/unload_model", timeout=120.0)
        except httpx2.HTTPError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_file.close()
        if config_backup is None:
            CONFIG_FILE.unlink(missing_ok=True)
        else:
            CONFIG_FILE.write_bytes(config_backup)
        shutil.rmtree(KV_DIR, ignore_errors=True)
        assert not _port_open(vllm_port), "vLLM server left running after the backend exited"


def _load(base_url: str) -> dict:
    config = {
        "model_name": MODEL_ID,
        "model_path": SNAPSHOT,
        "engine": "vllm",
        "vllm_gpu_memory_utilization": 0.9,
        "vllm_max_model_len": 32768,
        "vllm_max_num_seqs": 64,
        "vllm_max_num_batched_tokens": 8192,
        "vllm_cpu_offload_gb": CPU_OFFLOAD_GB,
        "vllm_cpu_offload_params": ["experts"],
        "vllm_kv_offloading_size": DRAM_TIER_GB,
        "vllm_kv_offloading_disk_path": str(KV_DIR),
    }
    r = httpx2.post(f"{base_url}/inference/load_model", json=config, timeout=60.0)
    assert r.status_code == 200, r.text
    deadline = time.time() + LOAD_TIMEOUT_S
    while time.time() < deadline:
        st = httpx2.get(f"{base_url}/inference/status", timeout=10.0).json()
        if st.get("loaded"):
            return st
        if st.get("loading_error") and not st.get("is_loading"):
            pytest.fail(f"model load failed: {st['loading_error'][:2000]}")
        time.sleep(5)
    pytest.fail(f"model load timed out after {LOAD_TIMEOUT_S}s")


def _unload(base_url: str) -> None:
    r = httpx2.post(f"{base_url}/inference/unload_model", timeout=180.0)
    assert r.status_code == 200, r.text
    deadline = time.time() + 180
    while time.time() < deadline:
        if not httpx2.get(f"{base_url}/inference/status", timeout=10.0).json().get("loaded"):
            return
        time.sleep(2)
    pytest.fail("model did not unload")


def _ask(base_url: str, prompt: str) -> float:
    body = {
        "model": MODEL_ID,
        "stream": False,
        "max_tokens": 1,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.monotonic()
    r = httpx2.post(f"{base_url}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text[:2000]
    return elapsed


def _wait_for_disk_stores(vllm_port: int, timeout_s: float = 120.0) -> float:
    """Block until the write-through to the fs tier has landed and stopped growing."""
    last, stable_since, deadline = -1.0, None, time.time() + timeout_s
    while time.time() < deadline:
        written = _metrics(vllm_port).get("vllm:kv_offload_tiering_write_bytes_total[1:fs]", 0.0)
        if written > 0 and written == last:
            stable_since = stable_since or time.time()
            if time.time() - stable_since >= 5:
                return written
        else:
            stable_since = None
        last = written
        time.sleep(1)
    pytest.fail(f"KV never finished writing to the NVMe tier (fs bytes written: {last:.0f})")


def test_moe_expert_offload_serves_kv_back_from_nvme_after_a_restart(backend):
    base_url, vllm_port, device = backend["base_url"], backend["vllm_port"], backend["device"]
    prompt = _prompt()

    # ---- phase 1: a MoE bigger than the card loads with only its experts offloaded
    status = _load(base_url)
    assert status["vllm_cpu_offload_params"] == ["experts"]
    assert status["vllm_kv_offloading_disk_path"] == str(KV_DIR)

    cold_s = _ask(base_url, prompt)
    written = _wait_for_disk_stores(vllm_port)
    block_files = list(KV_DIR.rglob("*.bin"))
    assert block_files, f"no KV block files under {KV_DIR} after {written:.0f} bytes written"

    # ---- phase 2: a fresh vLLM process -- the GPU prefix cache and DRAM tier are empty
    _unload(base_url)
    assert list(KV_DIR.rglob("*.bin")), "unloading must not wipe the NVMe tier"
    _load(base_url)
    before = _metrics(vllm_port)
    assert before.get("vllm:kv_offload_tiering_chunk_hits_total[0:primary]", 0.0) == 0.0
    assert before.get("vllm:external_prefix_cache_hits_total", 0.0) == 0.0

    # ---- phase 3: the same prompt, now only findable on disk
    disk_before = _device_bytes_read(device)
    warm_s = _ask(base_url, prompt)
    disk_read = _device_bytes_read(device) - disk_before
    after = _metrics(vllm_port)

    fs_hits = after.get("vllm:kv_offload_tiering_chunk_hits_total[1:fs]", 0.0)
    fs_read = after.get("vllm:kv_offload_tiering_read_bytes_total[1:fs]", 0.0)
    ext_hits = after.get("vllm:external_prefix_cache_hits_total", 0.0)
    prompt_tokens = after.get("vllm:prompt_tokens_total", 0.0)
    report = (
        f"cold {cold_s:.1f}s, after restart {warm_s:.1f}s; fs tier hits {fs_hits:.0f}, "
        f"fs bytes read {fs_read / 1e6:.0f} MB, device {device} read {disk_read / 1e6:.0f} MB, "
        f"external prefix hits {ext_hits:.0f} of {prompt_tokens:.0f} prompt tokens"
    )
    print(report)

    assert fs_hits > 0, f"the NVMe tier served nothing: {report}"
    assert fs_read > 0, f"the NVMe tier read no bytes: {report}"
    # O_DIRECT: the tier's reads must reach the device, so this cannot be a
    # page-cache (or tmpfs) hit passing for a disk read.
    assert disk_read >= 0.5 * fs_read, f"fs tier reads did not reach {device}: {report}"
    assert ext_hits >= 0.5 * prompt_tokens, f"most of the prompt was recomputed: {report}"
    assert warm_s < 0.5 * cold_s, f"reloading from NVMe was not faster than recompute: {report}"
