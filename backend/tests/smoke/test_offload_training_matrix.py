"""
Real DeepSpeed ZeRO-3 offload training runs, a few steps each.

These exist to be re-run after a dependency bump (transformers, deepspeed,
torch, peft, trl). Every combination here has been observed working; if one
stops working, an upgrade broke it.

What each case asserts, and why loss alone is not enough:

* **ZeRO-3 really partitioned.** `is_deepspeed_zero3_enabled()` only says a
  config was attached. A run whose parameters are all resident is paying the
  offload's DRAM cost while keeping the model on the GPU -- it trains, it looks
  fine, and it OOMs one model size later. So every case asserts
  `resident == 0`, which is the property offload actually promises.
* **Loss is finite and non-zero.** Truncation that removes the completion
  produces `loss: 0.0` at every step and reads as perfect convergence
  (see `tests/unit/training/test_label_supervision_guard.py`).
* **GPU peak under a ceiling.** The number that catches offload silently
  ceasing to offload while everything else still passes.

Portable / self-contained in the same sense as the other smoke tests: auto-skips
(with a reason) without CUDA, without DeepSpeed, or when a checkpoint is not in
the local cache. The large checkpoints are NOT auto-downloaded -- 15-67 GB is
not something a test suite should fetch.

Run just this file:
    pytest tests/smoke/test_offload_training_matrix.py -m offload -s
"""

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model_fixtures import ModelUnavailable, ensure_vlm_model, local_snapshot  # noqa: E402
from training_fixtures import short_text_dataset  # noqa: E402

pytestmark = [pytest.mark.finetune, pytest.mark.offload]

# Every case ran to completion on 1x RTX 5060 Ti 16 GB / 125 GB DRAM.
# `gpu_peak_ceiling_mb` is roughly 1.5x the observed peak: loose enough not to
# be flaky, tight enough that offload quietly stopping is a failure.
CASES = [
    {
        "id": "smolvlm-full-image-cpu",
        "model": None,  # resolved from the auto-downloaded VLM fixture
        "method": "full",
        "dataset": "dataset/mixed_finetune_demo/train.jsonl",
        "profile": "zero3_offload_cpu_cpu",
        "max_seq_length": 4096,
        "batch": 2,
        "accum": 2,
        "epochs": 2,
        "gpu_peak_ceiling_mb": 6000,
        "observed": "6 steps, loss 5.23->4.62, GPU peak 2.97 GB",
    },
    {
        "id": "qwen3-4b-lora-text-cpu",
        "model": "Qwen/Qwen3-4B",
        "method": "lora",
        "dataset": None,  # generated: short prompt/completion rows
        "profile": "zero3_offload_cpu_cpu",
        "max_seq_length": 512,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 13000,
        "observed": "10 steps, GPU peak 7.94 GB",
    },
    {
        "id": "gemma4-e4b-lora-image-nvme",
        "model": "google/gemma-4-E4B-it",
        "method": "lora",
        "dataset": "dataset/mixed_finetune_demo/train.jsonl",
        "profile": "zero3_offload_disk_cpu",  # params in DRAM, optimizer to NVMe
        "max_seq_length": 2048,
        "batch": 2,
        "accum": 2,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 15000,
        "observed": "3 steps, loss 11.65->7.86, GPU peak 10.75 GB, 981 MB written to NVMe",
    },
    # Was xfail("~12.6 GB resident; needs a larger card, or world_size>1"). That
    # was never a property of the model: deepspeed 0.19.3 stopped releasing
    # offloaded parameters, and 0.19.5 -- the version the diagnosis was written
    # against -- made a 35B look impossible on this card. Pinning back to 0.19.2
    # takes the peak from 7.0 GB to 3.2 GB and the case passes. The ceiling below
    # is deliberately near the observed peak so the regression cannot come back
    # unnoticed; a bump that reintroduces it fails here rather than in a
    # user's job.
    {
        "id": "qwen35-35b-lora-text-cpu",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "method": "lora",
        "dataset": None,
        "profile": "zero3_offload_cpu_cpu_tight",
        "max_seq_length": 512,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 5000,
        "observed": "10 steps, loss 9.82->3.01, GPU peak 3.24 GB, DRAM peak 119 GB",
    },
    # The same checkpoint through the image path: Qwen3.5-35B-A3B is itself a
    # ConditionalGeneration model, so this loads the vision tower the text case
    # above never touches. No Qwen3.5-VL exists to point at -- the vision half
    # ships inside the main model.
    #
    # `disk_cpu` and not `cpu_disk`, which would be the better fit on memory
    # (parameters on NVMe peaks at 84 GB DRAM rather than 117 GB): putting this
    # model's parameters on NVMe trains one step and then dies inside DeepSpeed's
    # swapper with "param 0 already assigned swap buffer id 1". Not a sizing
    # problem -- _fit_nvme_buffers leaves the profile alone here because its
    # buffer_size already exceeds the largest tensor.
    #
    # This case is DRAM-bound, not GPU-bound, and the requirement does not fall
    # with max_seq_length: 512 peaked at 119 GB too, and fails outright because
    # the image placeholder tokens no longer fit. There is deliberately no
    # DRAM precondition here -- a threshold is a guess about someone else's
    # machine, and the first one written skipped the very host the number was
    # measured on. A host that cannot seat it will say so itself.
    {
        "id": "qwen35-35b-lora-image-cpu",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "method": "lora",
        "dataset": "dataset/mixed_finetune_demo/train.jsonl",
        "profile": "zero3_offload_disk_cpu",
        "max_seq_length": 2048,
        "batch": 1,
        "accum": 1,
        "epochs": 1,
        "gpu_peak_ceiling_mb": 10000,
        # The GPU peak moves more than the other cases do -- 4.91 GB and 6.62 GB
        # across two runs of the same config -- so the ceiling is 1.5x the higher
        # one rather than the first one measured. The DRAM figure is a load-time
        # spike: RSS reaches 117 GB while the checkpoint is read, then settles at
        # ~71 GB (35B x bf16) for the whole run.
        "observed": "12 steps, loss 4.75->5.93 (12 rows, one epoch -- no trend at "
        "this length), GPU peak 6.62 GB, DRAM peak 117 GB / 71 GB steady",
    },
]


def _partition_stats(trainer) -> dict:
    """
    How many parameters DeepSpeed is actually holding partitioned.

    A partitioned parameter presents `numel() == 0` and keeps its real size in
    `ds_numel`; one that is resident is fully materialised.
    """
    model = getattr(trainer, "model_wrapped", None) or trainer.model
    managed = partitioned = resident = 0
    for param in model.parameters():
        if getattr(param, "ds_status", None) is None:
            continue
        managed += 1
        if param.numel() == 0:
            partitioned += 1
        else:
            resident += 1
    return {"managed": managed, "partitioned": partitioned, "resident": resident}


RECORD_PATH = Path(__file__).resolve().parents[1] / "results" / "offload" / "latest.jsonl"


def _package_versions() -> dict:
    """The libraries an upgrade would move. Recorded so a diff can attribute a change."""
    versions = {}
    for name in ("torch", "transformers", "deepspeed", "peft", "trl", "accelerate"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _record(case_id: str, payload: dict) -> None:
    """
    Append a machine-readable result so `scripts/upgrade_regression.py` can diff runs.

    Written even when a case fails: a failure's partition state and memory are
    exactly what you want to compare against the last good run.
    """
    try:
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RECORD_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"case": case_id, "versions": _package_versions(), **payload}) + "\n"
            )
    except Exception:
        pass  # recording must never fail a test


@pytest.fixture(scope="module", autouse=True)
def _require_gpu_and_deepspeed():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("offload training needs a CUDA GPU")
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("deepspeed is not installed")
    # DeepSpeed JIT-builds cpu_adam for every CPU-offload profile.
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_offload_training_runs(case, tmp_path, measure_memory):
    """Train a few steps and assert offload is genuinely doing its job."""
    from service.config_models import TrainingConfig
    from service.training.core import build_training_components
    from service.training.training_process import _resolve_deepspeed_config

    if case["model"] is None:
        try:
            _, model_path = ensure_vlm_model()
        except ModelUnavailable as exc:
            pytest.skip(f"VLM fixture unavailable: {exc}")
        model_name = str(model_path)
    else:
        # The resolved snapshot path, not the repo id: see local_snapshot's docstring
        # for why a directory check and the loader disagree.
        model_name = local_snapshot(case["model"])
        if model_name is None:
            pytest.skip(
                f"{case['model']} has no complete local snapshot (not auto-downloaded); "
                f"`hf cache verify {case['model']}` reports what is missing"
            )

    if case["dataset"]:
        # The image demo set is gitignored (it contains photographs of real
        # people), so a fresh checkout does not have it. Skip rather than fail:
        # a missing local dataset is not a regression.
        dataset_path = str(PROJECT_ROOT / case["dataset"])
        if not Path(dataset_path).is_file():
            pytest.skip(f"{case['dataset']} is not present (gitignored local dataset)")
    else:
        dataset_path = short_text_dataset(tmp_path)

    config = TrainingConfig(
        model_name=model_name,
        method=case["method"],
        dataset_path=dataset_path,
        output_dir=str(tmp_path / "out"),
        prompt_field="prompt",
        completion_field="completion",
        num_train_epochs=case["epochs"],
        per_device_train_batch_size=case["batch"],
        gradient_accumulation_steps=case["accum"],
        learning_rate=1e-5 if case["method"] == "full" else 2e-4,
        logging_steps=1,
        max_seq_length=case["max_seq_length"],
        use_deepspeed=True,
        deepspeed_profile=case["profile"],
        eval_split_ratio=0.2,
    )

    components = build_training_components(config, None, _resolve_deepspeed_config(config))

    # The eval split must survive into the trainer -- it silently did not on the
    # multi-GPU path for a month (see tests/unit/training/test_entrypoint_parity.py).
    assert components.eval_dataset is not None, "eval_split_ratio produced no held-out split"

    trainer = components.strategy.prepare_trainer(
        components.model,
        components.processing_class,
        components.dataset,
        components.training_args,
        eval_dataset=components.eval_dataset,
    )

    stats = _partition_stats(trainer)
    outcome: dict = {"partition": stats, "profile": case["profile"], "method": case["method"]}

    try:
        assert stats["managed"] > 0, (
            "no parameter carries ds_status: the ZeRO-3 engine did not wrap this model, so the "
            "whole model stays on the GPU regardless of the offload settings"
        )
        assert stats["resident"] == 0, (
            f"{stats['resident']}/{stats['managed']} parameters are resident on the GPU; "
            "ZeRO-3 is not actually offloading"
        )
        trainer.train()
    except Exception as e:
        # Record before re-raising: a failure's partition state and memory are
        # exactly what the upgrade diff needs to attribute the regression.
        outcome.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        _record(case["id"], {**outcome, "memory": measure_memory.summary()})
        raise

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    outcome.update(
        status="passed",
        steps=int(trainer.state.global_step),
        losses=[round(float(x), 4) for x in losses[:6]],
    )
    _record(case["id"], {**outcome, "memory": measure_memory.summary()})

    assert losses, "training produced no loss entries"
    assert all(loss == loss for loss in losses), "loss went NaN"  # noqa: PLR0124 - NaN check
    assert any(loss > 0 for loss in losses), (
        "every step reported loss 0.0, which means no token was supervised -- "
        "truncation removed the completion rather than the model converging"
    )
    assert trainer.state.global_step >= 1

    peak_mb = measure_memory.summary().get("cuda_alloc_mb", {}).get("peak", 0)
    assert peak_mb <= case["gpu_peak_ceiling_mb"], (
        f"GPU peak {peak_mb:.0f} MB exceeds the {case['gpu_peak_ceiling_mb']} MB ceiling for "
        f"this case (previously {case['observed']}). Offload may have stopped working."
    )
