"""
Image fine-tuning must produce a model that answers differently, not just a
training run that completes.

Everything else in the suite checks mechanism: parameters get selected, images
reach the loss, ZeRO-3 partitions, the loss is finite. All of that passes on a
run whose adapter turns out to do nothing at inference -- which is the run this
branch was opened to explain. Only asking the model a question can tell those
apart.

Why production-sized checkpoints. The task has to be something the base model
genuinely cannot do, or a pass proves nothing; naming a specific private
individual qualifies, and learning it needs a model of the size actually
deployed. Both families the service deploys are covered, because their image
paths are not the same code: Qwen3.5 carries its own collator fix for TRL
leaving `mm_token_type_ids` short, and its own ZeRO-3 rename workaround. A
mechanism test cannot tell whether either still works.

Neither model fits the card, so both run under ZeRO-3 offload -- for gemma-4 a
plain load OOMs on the weights, and device_map spilling still OOMs on the
5.25 GiB of activations one image forward wants. The control therefore runs on
the same engine with the adapter switched off, which is also stricter: same
weights, same offload, same processor, one variable.

Measured on 1x RTX 5060 Ti 16 GB / 125 GB DRAM, base 0/3 and tuned 3/3 for both:

    gemma-4-E4B-it        120 steps, ~14 min, loss 20.78 -> 0.014
    Qwen3.5-35B-A3B       300 steps, ~53 min, loss 14.08 -> 1.17

Opt-in for that reason:

    pytest tests/smoke/test_image_training_effectiveness.py -m effectiveness -s

Qwen answers arrive with a leading `</think>`; the probes match on substrings,
so it does not interfere, but do not read a raw answer as malformed.

What a pass does NOT establish: that the model learned *faces*. One held-out
image carries a legible name, and both base models read it -- gemma-4 answers
with a different name, Qwen quotes the caption -- so the control still holds,
but telling "recognises the person" from "reads the caption" would need images
stripped of text.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.effectiveness

DATASET = "dataset/mixed_finetune_demo/train.jsonl"
PROBES = "dataset/mixed_finetune_demo/heldout.json"

# Both families the service actually deploys. Epoch counts are what each needed
# to answer the held-out probes, not a shared default: gemma-4 accumulates to 4
# rows per step, Qwen3.5 to one, so equal epochs would not be equal training.
CASES = [
    {
        "id": "gemma4-e4b",
        "model": "google/gemma-4-E4B-it",
        "epochs": 30,
        "batch": 2,
        "accum": 2,
        "observed": "120 steps in ~14 min, loss 20.78 -> 0.014, base 0/3, tuned 3/3",
    },
    {
        "id": "qwen35-35b",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "epochs": 20,
        "batch": 1,
        "accum": 1,
        # Peaks near 117 GB of DRAM while the checkpoint loads, then holds ~71 GB
        # (35B x bf16) for the run. Not turned into a precondition: a threshold is
        # a guess about someone else's machine, and the host that cannot seat this
        # reports it more accurately than a hardcoded number can.
        "observed": "300 steps in ~53 min, loss 14.08 -> 1.17, base 0/3, tuned 3/3",
    },
]


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c["id"])
def trained(request, tmp_path_factory) -> dict:
    """Train once per model, then answer the probes with the adapter on and off."""
    import torch

    case = request.param
    model_id = case["model"]

    if not torch.cuda.is_available():
        pytest.skip("offload training needs a CUDA GPU")
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("deepspeed is not installed")

    from model_fixtures import local_snapshot

    # The resolved snapshot path, not the repo id: a directory check answers a
    # different question from the one that decides whether the load will work --
    # see local_snapshot's docstring.
    model_path = local_snapshot(model_id)
    if model_path is None:
        pytest.skip(
            f"{model_id} has no complete local snapshot (not auto-downloaded); "
            f"`hf cache verify {model_id}` reports what is missing"
        )
    if not (PROJECT_ROOT / DATASET).is_file():
        pytest.skip(f"{DATASET} is not present")

    from service.config_models import TrainingConfig
    from service.training.core import build_training_components
    from service.training.core.generation_check import load_probes, run_generation_check
    from service.training.training_process import _resolve_deepspeed_config

    probes = load_probes(str(PROJECT_ROOT / PROBES))
    config = TrainingConfig(
        model_name=model_path,
        method="lora",
        dataset_path=str(PROJECT_ROOT / DATASET),
        output_dir=str(tmp_path_factory.mktemp("image_effectiveness")),
        num_train_epochs=case["epochs"],
        per_device_train_batch_size=case["batch"],
        gradient_accumulation_steps=case["accum"],
        learning_rate=2e-4,
        logging_steps=1,
        max_seq_length=2048,
        use_deepspeed=True,
        deepspeed_profile="zero3_offload_disk_cpu",
        eval_split_ratio=0.0,  # heldout.json is the held-out set that matters here
    )

    components = build_training_components(config, None, _resolve_deepspeed_config(config))
    trainer = components.strategy.prepare_trainer(
        components.model,
        components.processing_class,
        components.dataset,
        components.training_args,
        eval_dataset=components.eval_dataset,
    )
    trainer.train()

    tuned = run_generation_check(trainer.model, components.processing_class, probes)
    peft_model = getattr(trainer.model, "module", trainer.model)
    with peft_model.disable_adapter():
        base = run_generation_check(peft_model, components.processing_class, probes)

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    return {"id": case["id"], "tuned": tuned, "base": base, "losses": losses}


def test_the_base_model_cannot_answer_the_probes(trained):
    """
    The control, asserted rather than assumed.

    If the base ever starts answering these, the case below stops testing
    training and starts testing the checkpoint -- silently, and while still
    passing. Better to fail here and rebuild the probe set.
    """
    base = trained["base"]
    assert base["passed"] == 0, (
        f"the base model already answers {base['passed']}/{base['total']} probes with the "
        f"adapter off, so a pass after training would not be evidence of anything"
    )


def test_training_teaches_the_held_out_identities(trained):
    """
    Every probe is an image the run never trained on, so answering them is not
    recall of a seen pair. One carries `expected_not`, which fails the
    degenerate model that learned to answer "Simon" for every face.
    """
    tuned = trained["tuned"]
    failures = [r for r in tuned.get("results", []) if not r.get("passed")]
    assert tuned["passed"] == tuned["total"], (
        f"{tuned['passed']}/{tuned['total']} held-out probes passed; "
        f"first failure answered: {str(failures[0].get('answer'))[:200]!r}"
        if failures
        else "probe accounting disagrees with itself"
    )


def test_the_loss_actually_moved(trained):
    """Cheap corroboration: a 3/3 with a flat loss would mean the probes are wrong."""
    losses = trained["losses"]
    assert losses, "training produced no loss entries"
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
