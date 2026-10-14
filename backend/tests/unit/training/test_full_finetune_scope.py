"""
A full-parameter fine-tune must not freeze the output head.

`freeze_non_language_params` confines full fine-tuning to the language tower,
the same scope LoRA gets from `exclude_modules`. But the output head is a
SIBLING of the tower in every `*ForConditionalGeneration` -- `lm_head` sits next
to `model` -- so the "outside the tower" test caught it.

With `tie_word_embeddings=True` this is invisible: `named_parameters()`
deduplicates the shared tensor, so the head stays trainable through
`embed_tokens`. Every multimodal checkpoint in the current registry ties, which
is why nothing surfaced. On an untied one the run trains everything except the
head it is meant to be learning -- no error, and a loss that still falls.

The fixtures below are deliberately synthetic: they let the untied case be
tested without a checkpoint that ties, which is the whole point.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.model_loader import freeze_non_language_params  # noqa: E402

pytestmark = pytest.mark.finetune

HIDDEN, VOCAB = 8, 16


class _LanguageTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN)])


class _Inner(nn.Module):
    """Mirrors the `model` attribute of a *ForConditionalGeneration."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageTower()
        self.vision_tower = nn.Linear(HIDDEN, HIDDEN)
        self.embed_vision = nn.Linear(HIDDEN, HIDDEN)


class _ConditionalGeneration(nn.Module):
    """`lm_head` is a sibling of `model`, exactly as in gemma-3/4 and Qwen-VL."""

    def __init__(self, *, tied: bool) -> None:
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        if tied:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def _trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("tied", [True, False], ids=["tied-embeddings", "untied-embeddings"])
def test_output_head_stays_trainable(tied):
    """
    The regression. Untied is the case that was broken; tied is included so the
    test cannot pass merely because deduplication hid the problem.
    """
    model = _ConditionalGeneration(tied=tied)

    freeze_non_language_params(model)

    assert model.lm_head.weight.requires_grad, (
        "a full fine-tune must train the output head, not just the body"
    )


@pytest.mark.parametrize("tied", [True, False], ids=["tied-embeddings", "untied-embeddings"])
def test_vision_half_is_still_frozen(tied):
    """Exempting the head must not weaken the actual purpose of the freeze."""
    model = _ConditionalGeneration(tied=tied)

    frozen = freeze_non_language_params(model)

    assert frozen > 0
    still_trainable = _trainable(model)
    assert not any(name.startswith("model.vision_tower") for name in still_trainable)
    assert not any(name.startswith("model.embed_vision") for name in still_trainable)


def test_language_tower_stays_trainable():
    model = _ConditionalGeneration(tied=False)

    freeze_non_language_params(model)

    trainable = _trainable(model)
    assert "model.language_model.embed_tokens.weight" in trainable
    assert "model.language_model.layers.0.weight" in trainable


def test_bridge_can_still_be_kept_for_scope_b():
    """keep_paths is how multimodal_scope=text_and_bridge trains the projector."""
    model = _ConditionalGeneration(tied=False)

    freeze_non_language_params(model, keep_paths=["model.embed_vision"])

    trainable = _trainable(model)
    assert "model.embed_vision.weight" in trainable
    assert "model.vision_tower.weight" not in trainable, "the encoder behind it stays frozen"
    assert model.lm_head.weight.requires_grad


def test_model_without_an_output_head_is_handled():
    """Not every architecture exposes one; the freeze must not blow up."""

    class _NoHead(_ConditionalGeneration):
        def get_output_embeddings(self):
            return None

    model = _NoHead(tied=False)
    assert freeze_non_language_params(model) > 0


def test_text_only_model_is_untouched():
    """No separate tower means nothing to confine; must stay a no-op."""

    class _TextOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _LanguageTower()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        def get_output_embeddings(self) -> nn.Module:
            return self.lm_head

    model = _TextOnly()
    before = _trainable(model)

    assert freeze_non_language_params(model) == 0
    assert _trainable(model) == before


def test_untied_head_is_a_distinct_tensor():
    """
    Guards the fixture itself: if these ever aliased, the untied cases above
    would silently degrade into the tied ones and prove nothing.
    """
    untied = _ConditionalGeneration(tied=False)
    tied = _ConditionalGeneration(tied=True)

    assert untied.lm_head.weight is not untied.model.language_model.embed_tokens.weight
    assert tied.lm_head.weight is tied.model.language_model.embed_tokens.weight
    assert len(list(untied.named_parameters())) == len(list(tied.named_parameters())) + 1
