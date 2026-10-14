"""
LoRA adapters inside checkpointed blocks must actually receive gradients.

With gradient checkpointing on, a block's inputs have to require grad or the
recomputation graph is cut and every adapter inside it gets nothing. In a LoRA
model everything is frozen except the adapters, so without a forward hook
forcing it, nothing at the block boundary requires grad. The failure is silent:
loss barely moves and the saved adapter is near-noise.

**This repo does not install that hook, and does not need to.** transformers
5.5.3 does it inside `gradient_checkpointing_enable` itself
(`modeling_utils.py:3113-3119`): `enable_input_grads = needs_embedding_grads or
_hf_peft_config_loaded`, where `needs_embedding_grads` is
`main_input_name == "input_ids"` -- true for every causal LM here. TRL carries
its own workaround for the same thing, and `prepare_model_for_kbit_training`
covers QLoRA, so the hook gets installed on every path.

That makes this an upstream guarantee we silently depend on, which is exactly
the kind of thing worth pinning: `strategies.py` forces `use_reentrant=True`
whenever DeepSpeed is on, and reentrant is the mode that breaks without it. If
a transformers upgrade drops that block, these tests fail here rather than
producing a run that trains to a near-noise adapter.

Reviewed 2026-08-31: two independent reviews reported this as a live bug on the
`use_sft_trainer=false` path, both citing `trainer.py` calling only
`gradient_checkpointing_enable`. Measured on the installed stack, the hook is
installed *by* that call, gradients flow under reentrant checkpointing, and
adding our own call would register a second hook while overwriting the handle
of the first -- leaking it. So the correct action was to pin the behaviour, not
to "fix" it.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import model_fixtures as mf  # noqa: E402
from service.config_models import TrainingConfig  # noqa: E402
from service.training.core.model_loader import ModelLoader  # noqa: E402

pytestmark = pytest.mark.finetune


@pytest.fixture(scope="module")
def text_model_path() -> str:
    try:
        _, path = mf.ensure_hf_model()
    except mf.ModelUnavailable as exc:
        pytest.skip(f"model unavailable: {exc}")
    return str(path)


def _build_lora_model(text_model_path):
    config = TrainingConfig(
        model_name=text_model_path,
        method="lora",
        dataset_path="/tmp/unused.jsonl",
        output_dir="/tmp/unused",
        gradient_checkpointing=True,
    )
    return ModelLoader(config, None).load_model()


@pytest.fixture(scope="module")
def lora_model(text_model_path):
    """
    Shared model for the positive cases.

    Not usable for the negative control: every
    `gradient_checkpointing_enable()` call registers ANOTHER input-grad forward
    hook while `_require_grads_hook` keeps only the newest handle, so on a
    reused model `hook.remove()` leaves earlier hooks alive and the control
    silently passes for the wrong reason.
    """
    return _build_lora_model(text_model_path)


def _backward_once(model) -> dict[str, torch.Tensor | None]:
    """One forward+backward; returns each trainable parameter's gradient."""
    model.train()
    model.zero_grad(set_to_none=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    return {name: p.grad for name, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non-reentrant"])
def test_lora_gets_gradients_under_gradient_checkpointing(lora_model, use_reentrant):
    """
    Reentrant is the mode `strategies.py` forces whenever DeepSpeed is on, and
    the one that drops gradients when no input requires grad.
    """
    lora_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )

    grads = _backward_once(lora_model)

    assert grads, "expected trainable LoRA parameters"
    missing = [name for name, grad in grads.items() if grad is None]
    assert not missing, f"{len(missing)}/{len(grads)} LoRA parameters got no gradient: {missing[:3]}"
    assert any(g.abs().sum() > 0 for g in grads.values()), "gradients are all exactly zero"


def test_enabling_checkpointing_installs_the_input_grad_hook(lora_model):
    """
    Pin the upstream mechanism, so a transformers change that removes it fails
    loudly here instead of silently producing untrained adapters.
    """
    lora_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    assert getattr(lora_model, "_require_grads_hook", None) is not None, (
        "transformers.gradient_checkpointing_enable no longer enables input grads; "
        "the training pipeline now has to do it itself or LoRA will not train"
    )
    embedded = lora_model.get_input_embeddings()(torch.tensor([[1, 2, 3]]))
    assert embedded.requires_grad, "embedding output must carry grad into checkpointed blocks"


def test_removing_the_hook_breaks_training(text_model_path):
    """
    Negative control, on its OWN model so exactly one hook exists.

    Without it, nothing entering a checkpointed block requires grad, so under
    reentrant checkpointing the graph is cut and backward has nothing to walk.
    On a model where every decoder layer is checkpointed that surfaces as a
    hard RuntimeError; a model with trainable parameters outside the
    checkpointed region would instead get silently-partial gradients. Either
    way the run is broken, which is what makes the tests above meaningful.
    """
    model = _build_lora_model(text_model_path)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model._require_grads_hook.remove()

    assert not model.get_input_embeddings()(torch.tensor([[1, 2, 3]])).requires_grad

    with pytest.raises(RuntimeError, match="does not require grad"):
        _backward_once(model)
