"""
Image fine-tuning must teach the identities on the multi-GPU path too, not just
produce a run that completes across two ranks.

`tests/smoke/test_multigpu_training_matrix.py` covers the mechanism: two ranks
start, the data really splits, ZeRO-3 partitions, the loss is finite, rank 0
saves an adapter. All of that passes on a run whose adapter turns out to do
nothing at inference -- the failure this branch was opened to explain -- so by
the same argument `test_image_training_effectiveness.py` makes for one GPU, only
asking the model a question separates the two. That test trains in-process and
therefore only ever exercised `num_gpus == 1`.

How the verdict gets here. Generation runs inside the ranks, not in this process:
the checkpoints are too large to load for inference next to the engine that
trained them (a plain load OOMs on gemma-4's weights, and device_map spilling
still OOMs on the 5.25 GiB of activations one image forward wants -- see
test_image_training_effectiveness.py). So the run sets `generation_check_path`
and reads the verdict back out of the job log, which is the same path a real
job's verdict travels.

The base-model control is deliberately not repeated here. Whether the base can
answer these probes is a property of the checkpoint and the probe set, not of
how many GPUs trained the adapter, and
`test_image_training_effectiveness.py::test_the_base_model_cannot_answer_the_probes`
already asserts it (0/3 for both checkpoints). Repeating it would mean loading
each base model a second time for no new information.

Epoch counts are trimmed to what the probes actually need, not to the reference
run's training volume. A first pass at 60/40 epochs took 98 minutes and drove
the loss to 1e-4 and 8e-3, far past the point the probes flip; the counts below
run in about 40 and answer the same 3/3.

They are floors measured rather than guessed, and the measurement is worth
knowing. Cutting gemma-4 to 15 epochs drops it to 2/3, and the probe it loses is
the one whose image carries a legible name: it answers with the caption,
"This is 陳玲娟 (Chen Lingjuan)", which is what the *base* model does. So the
floor is not "where the loss looks converged" -- it is where the model stops
reading the picture and starts recognising the person, which is the whole
distinction this probe set exists to draw. Trim further and this stops being a
multi-GPU test and starts being an under-training detector.

Opt-in, and slow (tens of minutes for the 35B):
    pytest tests/smoke/test_multigpu_image_effectiveness.py -m effectiveness -s
"""

import json
import sys
import uuid
from multiprocessing import Queue
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_fixtures import MULTI_GPU_MIN_COUNT, multi_gpu_skip_reason  # noqa: E402
from model_fixtures import local_snapshot  # noqa: E402

pytestmark = [
    pytest.mark.effectiveness,
    pytest.mark.multigpu,
    pytest.mark.skipif(multi_gpu_skip_reason() is not None, reason=str(multi_gpu_skip_reason())),
]

DATASET = "dataset/mixed_finetune_demo/train.jsonl"
PROBES = "dataset/mixed_finetune_demo/heldout.json"

CASES = [
    {
        "id": "gemma4-e4b-2gpu",
        "model": "google/gemma-4-E4B-it",
        "epochs": 30,  # 15 answers only 2/3 -- see the observed note below
        "batch": 2,
        "accum": 2,
        "observed": "two ranks at 30 epochs: 60 steps, 3/3. At 15 it answers 2/3, and the "
        "probe it misses is the one whose image carries a legible name -- it replies "
        "with the caption, 'This is 陳玲娟 (Chen Lingjuan)', instead of the person. "
        "That is the base model's behaviour, so the floor here is the point the model "
        "stops reading and starts recognising. Single-GPU reference: 120 steps in "
        "~14 min, loss 20.78 -> 0.014, 3/3",
    },
    {
        "id": "qwen35-35b-2gpu",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "epochs": 12,  # 20 on one GPU; measured 3/3 here, and it dominates the wall clock
        "batch": 1,
        "accum": 1,
        "observed": "two ranks at 12 epochs: 96 steps in ~23 min, 3/3. Single-GPU "
        "reference: 300 steps in ~115 min, loss 14.08 -> 1.17, 3/3",
    },
]


def _generation_verdicts(events_path: Path) -> list[dict]:
    """Every generation-check payload rank 0 reported, in order."""
    if not events_path.is_file():
        return []
    out = []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if str(ev.get("msg", "")).startswith("generation check"):
                out.append(ev)
    return out


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c["id"])
def trained(request, tmp_path_factory) -> dict:
    """Train one checkpoint across two ranks and return rank 0's probe verdict."""
    import torch

    from service.config_models import TrainingConfig
    from service.training.core.job_logger import JobLogWriter
    from service.training.training_process import (
        _resolve_deepspeed_config,
        _run_deepspeed_subprocess,
    )

    case = request.param
    if not torch.cuda.is_available():
        pytest.skip("offload training needs a CUDA GPU")
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("deepspeed is not installed")

    model_path = local_snapshot(case["model"])
    if model_path is None:
        pytest.skip(
            f"{case['model']} has no complete local snapshot (not auto-downloaded); "
            f"`hf cache verify {case['model']}` reports what is missing"
        )
    if not (PROJECT_ROOT / DATASET).is_file():
        pytest.skip(f"{DATASET} is not present (gitignored local dataset)")

    tmp = tmp_path_factory.mktemp(f"mg_eff_{case['id']}")
    config = TrainingConfig(
        model_name=model_path,
        method="lora",
        dataset_path=str(PROJECT_ROOT / DATASET),
        output_dir=str(tmp / "out"),
        # Named explicitly, unlike the single-GPU effectiveness test which lets
        # the loader auto-detect them. Auto-detection reports success on both
        # ranks and the row counts agree, but training then deadlocks in the
        # step-2 gradient reduce_scatter until NCCL's 600s watchdog kills the job
        # with "exited with return code 250" -- a failure that names neither the
        # cause nor the field. Reproduced on SmolVLM: identical configs differing
        # only in these two lines, one deadlocks and one passes.
        prompt_field="prompt",
        completion_field="completion",
        num_train_epochs=case["epochs"],
        per_device_train_batch_size=case["batch"],
        gradient_accumulation_steps=case["accum"],
        learning_rate=2e-4,
        logging_steps=1,
        max_seq_length=2048,
        use_deepspeed=True,
        deepspeed_profile="zero3_offload_disk_cpu",
        num_gpus=MULTI_GPU_MIN_COUNT,
        generation_check_path=str(PROJECT_ROOT / PROBES),
        eval_split_ratio=0.0,  # heldout.json is the held-out set that matters here
    )

    job_id = f"mg-eff-{case['id']}-{uuid.uuid4().hex[:8]}"
    writer = JobLogWriter(job_id, base_dir=tmp / "joblog")
    try:
        _run_deepspeed_subprocess(
            config,
            _resolve_deepspeed_config(config),
            job_id,
            Queue(),
            None,  # no Redis: the stdout channel carries the verdict
            writer,
        )
    finally:
        try:
            writer.close()
        except Exception:
            pass  # a log-close failure must not mask the run's own result

    return {"id": case["id"], "case": case, "events": _generation_verdicts(writer.events_path)}


def test_the_generation_check_runs_on_the_multi_gpu_path(trained):
    """
    A verdict has to arrive at all.

    `generation_check_path` was silently a no-op for num_gpus > 1: the multi-GPU
    branch of training_process.py returns when the subprocess exits, before its
    own generation check, and training_script.py did not run one. Setting the
    field did nothing, and nothing said so -- on the path used for the models
    where a degenerate rule is most expensive to find late.
    """
    events = trained["events"]
    assert events, (
        "no generation-check event reached the job log, so the probes never ran on "
        "the multi-GPU path"
    )
    skipped = [e for e in events if "skipped" in str(e.get("msg"))]
    assert not skipped, (
        f"the generation check was skipped rather than run: {skipped[0].get('data')!r}"
    )


def test_multigpu_training_teaches_the_held_out_identities(trained):
    """
    Every probe is an image the run never trained on, so answering is not recall
    of a seen pair. One carries `expected_not`, which fails the degenerate model
    that learned to answer "Simon" for every face.

    The control -- that the base checkpoint answers none of these -- is asserted
    by test_image_training_effectiveness.py and is independent of rank count.
    """
    verdict = trained["events"][-1].get("data") or {}
    passed, total = verdict.get("passed"), verdict.get("total")
    assert total, f"the verdict carries no probe count: {verdict!r}"
    failures = [r for r in verdict.get("results", []) if not r.get("passed")]
    assert passed == total, (
        f"{passed}/{total} held-out probes passed after two-rank training "
        f"({trained['case']['observed']}); first failure answered: "
        f"{str(failures[0].get('answer'))[:200]!r}"
        if failures
        else f"{passed}/{total} passed but no failure was reported"
    )
