"""
Image training must reach the image half, checked on a real checkpoint.

`test_full_finetune_scope.py` already covers the freeze logic, but against
hand-built toy modules whose names the test chooses itself. That cannot fail
when a dependency upgrade renames a real model's submodules -- which is the
failure worth guarding, because every scope decision in the training path
resolves module paths by string: which tower to freeze, where the bridge is,
which linears LoRA may target. A rename does not raise, it returns nothing. The
run then completes, the loss falls, and the result is quietly wrong.

So these run the same logic against a real SmolVLM checkpoint and a real image
batch, and assert on **gradients** rather than on `requires_grad` flags: a flag
says a parameter was selected, a gradient says the image actually reached it.

Deliberately not an effectiveness test. Nothing here asks the model to learn
anything, so there is no threshold to tune and no flakiness to manage -- a
gradient is either absent, zero, or not. Whether image fine-tuning produces a
model that works is a separate, slower question that needs a production-sized
model; see tests/smoke/.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.config_models import TrainingConfig  # noqa: E402
from service.training.core.model_loader import ModelLoader  # noqa: E402
from service.utils.model_class_resolver import find_language_tower_path  # noqa: E402

pytestmark = pytest.mark.finetune


@pytest.fixture(scope="module")
def processor(vlm_path):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(vlm_path)
    # SmolVLM tiles one image into 17 sub-images by default -- 1141 tokens rather
    # than 81, for ~17x the vision compute. These tests ask where gradients go,
    # not how well the image is read, and a single tile carries the same signal
    # through the same modules. Guarded by hasattr: the knob is Idefics3's.
    image_processor = getattr(proc, "image_processor", None)
    if hasattr(image_processor, "do_image_splitting"):
        image_processor.do_image_splitting = False
    return proc


@pytest.fixture(scope="module")
def text_only_model(vlm_path):
    """
    Shared across the default-scope cases: each backward zeroes the gradients
    first and none of them alter `requires_grad`, so there is no state to leak.
    The negative control below removes its hook explicitly for the same reason.
    """
    return _load(vlm_path, "text_only")


def _load(vlm_path: str, scope: str):
    """A full fine-tune is the path that applies the scope freeze directly."""
    config = TrainingConfig(
        model_name=vlm_path,
        method="full",
        dataset_path="/tmp/unused.jsonl",
        output_dir="/tmp/unused",
        multimodal_scope=scope,
    )
    loader = ModelLoader(config, None)
    loader.enable_image_training()
    return loader.load_model()


def _solid(colour: tuple[int, int, int]):
    from PIL import Image

    return Image.new("RGB", (64, 64), colour)


def _backward_on_image(model, processor, image) -> dict[str, torch.Tensor | None]:
    """One forward+backward over a real image batch; returns trainable grads."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Who is in this photo?"}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=prompt, images=[image], return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(model.dtype)

    model.train()
    model.zero_grad(set_to_none=True)
    model(**batch, labels=batch["input_ids"]).loss.backward()
    return {name: p.grad for name, p in model.named_parameters() if p.requires_grad}


def _partition(model, grads: dict) -> tuple[dict, dict, dict]:
    """
    Split gradients into (language tower, output head, everything else).

    The head is pulled out on its own because it is a *sibling* of the tower, so
    "outside the tower" catches it -- and a full fine-tune is supposed to keep it
    trainable. Lumping it in with `other` would let the bridge assertion below
    pass on the head alone, while the bridge lookup found nothing.
    """
    lang_path = find_language_tower_path(model)
    assert lang_path, (
        "no language tower found in a real multimodal checkpoint -- the name-keyed "
        "lookup this whole scope mechanism rests on has stopped matching"
    )

    head_names: set[str] = set()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        for module_name, module in model.named_modules():
            if module is output_embeddings and module_name:
                head_names = {
                    n for n in grads if n == module_name or n.startswith(module_name + ".")
                }
                break

    in_tower = {n: g for n, g in grads.items() if n == lang_path or n.startswith(lang_path + ".")}
    tower = {n: g for n, g in in_tower.items() if n not in head_names}
    head = {n: g for n, g in grads.items() if n in head_names}
    other = {n: g for n, g in grads.items() if n not in in_tower and n not in head_names}
    return tower, head, other


def _has_signal(grads: dict) -> bool:
    return any(g is not None and g.abs().sum() > 0 for g in grads.values())


def test_text_only_scope_trains_the_language_tower_and_nothing_else(text_only_model, processor):
    """
    The default scope. The vision half must be frozen, and -- more importantly --
    the language half must still be receiving gradients from an image batch.

    A rename that made `find_language_tower_path` return nothing would freeze
    nothing at all, so the vision tower having gradients is exactly the symptom
    to catch.
    """
    model = text_only_model
    grads = _backward_on_image(model, processor, _solid((200, 30, 30)))

    tower, head, other = _partition(model, grads)
    assert _has_signal(tower), "the language tower got no gradient from an image batch"
    assert not other, (
        f"{len(other)} parameters outside the language tower are still trainable under "
        f"multimodal_scope=text_only: {sorted(other)[:3]}"
    )
    # SmolVLM does not tie its embeddings, so a head that got frozen here would be
    # a real "trains everything except the head it is learning" run, not a no-op.
    assert _has_signal(head), (
        "the output head got no gradient; a full fine-tune must keep it trainable"
    )


def test_bridge_scope_puts_gradients_into_the_projector(vlm_path, processor):
    """
    Scope B trains the cross-modal bridge. The bridge is found by matching module
    shapes against the text hidden size, so this is the other name/shape-keyed
    lookup an upgrade can silently break.
    """
    model = _load(vlm_path, "text_and_bridge")
    grads = _backward_on_image(model, processor, _solid((30, 30, 200)))

    _, _, bridge = _partition(model, grads)
    assert bridge, (
        "multimodal_scope=text_and_bridge left nothing trainable outside the language "
        "tower -- the bridge lookup found nothing and the run would silently be text_only"
    )
    assert _has_signal(bridge), f"the bridge parameters got no gradient: {sorted(bridge)[:3]}"


def _grads_for_two_images(model, processor) -> tuple[str, torch.Tensor, torch.Tensor]:
    red = _backward_on_image(model, processor, _solid((220, 20, 20)))
    name = next(n for n, g in red.items() if g is not None and g.abs().sum() > 0)
    red_grad = red[name].detach().clone()
    blue = _backward_on_image(model, processor, _solid((20, 20, 220)))
    return name, red_grad, blue[name]


def test_the_image_content_reaches_the_loss(text_only_model, processor):
    """
    The one that catches images being dropped on the floor.

    Same prompt, two different images: if the gradients come out identical, the
    pixels never influenced the loss and training can only ever learn to answer
    from the prompt. Every other assertion here would still pass in that state --
    parameters are selected, gradients are non-zero, the loss falls.
    """
    name, red, blue = _grads_for_two_images(text_only_model, processor)

    assert not torch.allclose(red, blue), (
        f"gradients of '{name}' are identical for two different images: the image "
        f"is not reaching the loss"
    )


def test_pinning_the_vision_output_makes_the_images_indistinguishable(text_only_model, processor):
    """
    Negative control for the test above, so a pass there means something.

    Holding the vision tower's output constant is what "the model ignores the
    image" looks like from the language tower's side. If the comparison above
    cannot tell that apart from a working model, it is not testing anything.
    """
    vision_paths = [n for n, _ in text_only_model.named_modules() if "vision" in n.lower()]
    assert vision_paths, "no vision module on a multimodal checkpoint"

    def pin_output(_module, _inputs, output):
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = torch.ones_like(output.last_hidden_state)
            return output
        return torch.ones_like(output) if torch.is_tensor(output) else output

    handle = text_only_model.get_submodule(min(vision_paths, key=len)).register_forward_hook(
        pin_output
    )
    try:
        name, red, blue = _grads_for_two_images(text_only_model, processor)
    finally:
        handle.remove()

    assert torch.allclose(red, blue), (
        f"gradients of '{name}' still differ with the vision output pinned, so the "
        f"comparison in the test above is not driven by image content"
    )
