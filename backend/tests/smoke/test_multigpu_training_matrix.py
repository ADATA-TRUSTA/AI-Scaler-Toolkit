"""
Real `deepspeed --num_gpus 2` training runs, a few steps each.

Why this file exists. `num_gpus > 1` takes a completely different route from
`num_gpus == 1`: the single-GPU path trains in-process (and pins
CUDA_VISIBLE_DEVICES to "0"), while the multi-GPU path shells out to
`deepspeed --num_gpus N service/training/training_script.py` and talks to rank 0
over a stdout status channel. Every existing multi-GPU test mocks that launcher
-- test_multigpu_lifecycle.py fakes the process tree, test_entrypoint_parity.py
and test_multigpu_status_reporting.py patch it out -- so nothing in the suite
had ever started a second rank. A device-placement or collective bug therefore
could not fail anywhere except on a user's two-card machine.

The cases below drive the production launcher itself
(`_run_deepspeed_subprocess`) rather than composing an equivalent command, so a
change to how the service launches ranks cannot pass here and break in the
service.

What each case asserts:

* **Two ranks were really requested.** The launch command is logged; a case
  that silently fell back to one GPU still trains and still looks fine.
* **Every rank exited 0.** The launcher raises `_DeepSpeedTrainingError` when
  any rank fails, and one rank dying usually takes the others with it
  (collective timeout, broken pipe).
* **Rank 0 wrote the adapter.** Completing without saving is the failure mode
  where the run reports success and leaves nothing behind.
* **The status channel delivered.** Multi-GPU progress reaches the manager only
  through rank 0's `__STATUS__:` lines; a run whose stdout parsing broke trains
  correctly and reports nothing to the UI.

Auto-skips (with a reason) on a host with fewer than two NVIDIA GPUs, without
DeepSpeed, or when a checkpoint is not in the local cache. Large checkpoints are
NOT auto-downloaded.

Run just this file:
    pytest tests/smoke/test_multigpu_training_matrix.py -m multigpu -s
"""

import json
import logging
import os
import sys
import uuid
from multiprocessing import Queue
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_fixtures import MULTI_GPU_MIN_COUNT, multi_gpu_skip_reason  # noqa: E402
from model_fixtures import ModelUnavailable, ensure_vlm_model, local_snapshot  # noqa: E402
from training_fixtures import short_text_dataset  # noqa: E402

pytestmark = [
    pytest.mark.finetune,
    pytest.mark.multigpu,
    pytest.mark.skipif(multi_gpu_skip_reason() is not None, reason=str(multi_gpu_skip_reason())),
]

IMAGE_DATASET = "dataset/mixed_finetune_demo/train.jsonl"

# Kept deliberately short: this file proves the multi-GPU *route* works, not
# that a model converges -- test_image_training_effectiveness.py does that.
CASES = [
    {
        "id": "smolvlm-lora-text-2gpu",
        "model": None,  # the auto-downloaded VLM fixture
        "dataset": None,  # generated: short prompt/completion rows
        "profile": "zero3_offload_cpu_cpu",
        "max_seq_length": 512,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 2500,
        "observed": "1.27 GB per rank (2.54 GB across both)",
    },
    {
        # Same checkpoint through the image path. Multimodal is where the
        # multi-GPU route is most likely to break: the collator and the ZeRO-3
        # parameter rename both differ from the text path.
        "id": "smolvlm-lora-image-2gpu",
        "model": None,
        "dataset": IMAGE_DATASET,
        "profile": "zero3_offload_cpu_cpu",
        "max_seq_length": 2048,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 4000,
        "observed": "2.32 GB per rank (4.64 GB across both) -- images cost ~1 GB more "
        "per rank than the text path on the same checkpoint",
    },
    {
        # A checkpoint of the size actually deployed, through the image path.
        # Mirrors qwen35-35b-lora-image-cpu in the single-GPU matrix so the two
        # can be compared directly; `disk_cpu` for the same reason documented
        # there (parameters on NVMe dies in DeepSpeed's swapper).
        "id": "qwen35-35b-lora-image-2gpu",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "dataset": IMAGE_DATASET,
        "profile": "zero3_offload_disk_cpu",
        "max_seq_length": 2048,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        # Roughly double the 6.6 GB the single-GPU case peaks at: two ranks
        # all-gather from each other and the communication buffers are not free.
        #
        # DRAM goes the same way, which is the opposite of what splitting across
        # ranks suggests: 131 GB across the two ranks against 117 GB for the
        # single-GPU case, because each rank materialises its own copy of the
        # checkpoint while loading and ZeRO-3 only partitions afterwards. Adding a
        # GPU buys headroom per card, not on the host.
        "gpu_peak_ceiling_mb": 20000,
        "observed": "6 steps (12 rows / 2 ranks), 12.95 GB per rank (24.2 GB across "
        "both), 131 GB host RSS, ~4.5 min",
    },
]


def _adapter_files(output_dir: Path) -> list[Path]:
    """LoRA artefacts anywhere under the output dir (rank 0 may nest a checkpoint)."""
    return [
        p
        for p in output_dir.rglob("*")
        if p.is_file() and (p.name.startswith("adapter_model") or p.name == "adapter_config.json")
    ]


def _drain(status_q: Queue) -> list[dict]:
    """Everything the launcher forwarded, without blocking on an empty queue."""
    seen: list[dict] = []
    while True:
        try:
            seen.append(status_q.get_nowait())
        except Exception:
            return seen


@pytest.fixture(scope="module", autouse=True)
def _require_gpus_and_deepspeed():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("multi-GPU training needs CUDA")
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("deepspeed is not installed")
    # DeepSpeed JIT-builds cpu_adam for every CPU-offload profile.
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_multigpu_training_runs(case, tmp_path, caplog, measure_memory):
    """Train a few steps across two ranks through the production launcher."""
    from service.config_models import TrainingConfig
    from service.training.training_process import (
        _resolve_deepspeed_config,
        _run_deepspeed_subprocess,
    )

    if case["model"] is None:
        try:
            _, model_path = ensure_vlm_model()
        except ModelUnavailable as exc:
            pytest.skip(f"VLM fixture unavailable: {exc}")
        model_name = str(model_path)
    else:
        # The resolved snapshot path, not the repo id: see local_snapshot's docstring.
        # It matters more here than on the single-GPU path -- the ranks are separate
        # processes that re-resolve the cache from their own environment.
        model_name = local_snapshot(case["model"])
        if model_name is None:
            pytest.skip(
                f"{case['model']} has no complete local snapshot (not auto-downloaded); "
                f"`hf cache verify {case['model']}` reports what is missing"
            )

    if case["dataset"]:
        # The image demo set is gitignored (it holds photographs of real
        # people), so a fresh checkout does not have it. Skip rather than fail.
        dataset_path = str(PROJECT_ROOT / case["dataset"])
        if not Path(dataset_path).is_file():
            pytest.skip(f"{case['dataset']} is not present (gitignored local dataset)")
    else:
        dataset_path = short_text_dataset(tmp_path, rows=8)

    output_dir = tmp_path / "out"
    config = TrainingConfig(
        model_name=model_name,
        method="lora",
        dataset_path=dataset_path,
        output_dir=str(output_dir),
        prompt_field="prompt",
        completion_field="completion",
        num_train_epochs=case["epochs"],
        per_device_train_batch_size=case["batch"],
        gradient_accumulation_steps=case["accum"],
        learning_rate=2e-4,
        logging_steps=1,
        max_seq_length=case["max_seq_length"],
        use_deepspeed=True,
        deepspeed_profile=case["profile"],
        num_gpus=MULTI_GPU_MIN_COUNT,
        eval_split_ratio=0.2,
    )
    assert config.num_gpus >= MULTI_GPU_MIN_COUNT, (
        "num_gpus did not survive TrainingConfig, so this case would quietly run on one GPU"
    )

    status_q: Queue = Queue()
    session_id = f"multigpu-smoke-{uuid.uuid4().hex[:8]}"

    with caplog.at_level(logging.INFO):
        _run_deepspeed_subprocess(
            config,
            _resolve_deepspeed_config(config),
            session_id,
            status_q,
            None,  # no Redis: the stdout channel is the one under test
        )

    # The launcher logs the exact argv it ran. A case that fell back to a single
    # rank trains, saves and passes every other assertion here.
    launches = [
        r.getMessage() for r in caplog.records if "Launching multi-GPU training" in r.getMessage()
    ]
    assert launches, "the multi-GPU launcher never ran: this case took the in-process path"
    assert f"--num_gpus={MULTI_GPU_MIN_COUNT}" in launches[0], (
        f"launched with the wrong world size: {launches[0]}"
    )

    adapters = _adapter_files(output_dir)
    assert adapters, (
        f"the run completed but wrote no LoRA adapter under {output_dir}; "
        "rank 0 either skipped the save or saved somewhere else"
    )

    forwarded = _drain(status_q)
    assert any(msg.get("status") == "stage" for msg in forwarded), (
        "no stage update reached the status queue: rank 0's __STATUS__ channel is not "
        f"being parsed (got {forwarded!r})"
    )

    # The ranks are subprocesses, so this figure comes from the driver rather than
    # torch's high-water mark -- see conftest's _DescendantSampler. That makes it a
    # polled maximum: it catches a rank that stopped offloading and sat on the card,
    # not a spike between two samples. Deliberately still asserted, because the
    # alternative was the 0 MB this fixture used to report, which passed any ceiling.
    summary = measure_memory.summary()
    peak_mb = (summary.get("cuda_alloc_mb") or {}).get("peak") or 0
    assert peak_mb, (
        "no GPU usage was attributed to any rank; either the launcher never started "
        f"one or nvidia-smi could not be read (summary: {summary.get('descendants')!r})"
    )
    assert peak_mb <= case["gpu_peak_ceiling_mb"], (
        f"per-rank GPU peak {peak_mb:.0f} MB exceeds the {case['gpu_peak_ceiling_mb']} MB "
        f"ceiling for this case (previously {case['observed']}). Offload may have stopped "
        "working on the multi-GPU path."
    )
