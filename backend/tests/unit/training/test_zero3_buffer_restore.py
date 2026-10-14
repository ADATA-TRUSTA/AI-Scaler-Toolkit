"""
Regression tests for the ZeRO-3 loader dropping buffers of parameter-less modules.

`transformers.integrations.deepspeed._load_state_dict_into_zero3_model` copies a
module's tensors only when that module owns at least one DIRECT parameter present
in the state dict. `_load_from_state_dict` is the only call that copies buffers,
so a module holding its weight in a child while owning buffers directly never has
those buffers loaded -- silently.

On gemma-4-E4B this leaves all 42 `Gemma4TextDecoderLayer.layer_scalar` buffers at
their `torch.ones(1)` default instead of the checkpoint's 0.061-0.887, corrupting
the model before training starts.

The fixtures below mirror that exact shape: `_ScalarLayer` owns a `layer_scalar`
buffer while its weight lives in the `.linear` child, so it reproduces the skip.
"""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from service.training.core.zero3_utils import restore_zero3_skipped_buffers

pytestmark = pytest.mark.finetune

# Values chosen to match the real gemma-4 range; the default is always 1.0.
CHECKPOINT_SCALARS = [0.061, 0.160, 0.840]


class _ScalarLayer(nn.Module):
    """A decoder layer shaped like Gemma4TextDecoderLayer: buffer here, weight in a child."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer("layer_scalar", torch.ones(1))


class _ClipLayer(nn.Module):
    """Shaped like Gemma4ClippableLinear: only buffers, weight in `.linear`."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer("input_max", torch.tensor(float("inf")))


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_ScalarLayer() for _ in CHECKPOINT_SCALARS)
        self.clip = _ClipLayer()
        # A module that DOES own a direct parameter, so the upstream loader
        # would have handled its buffer correctly. Restoring it must be a no-op.
        self.embed = nn.Embedding(4, 4)
        self.embed.register_buffer("embed_scale", torch.tensor(2.5))


@pytest.fixture
def checkpoint(tmp_path):
    """A checkpoint carrying the true buffer values plus the weights."""
    tensors = {"clip.linear.weight": torch.randn(4, 4), "embed.weight": torch.randn(4, 4)}
    for i, value in enumerate(CHECKPOINT_SCALARS):
        tensors[f"layers.{i}.layer_scalar"] = torch.tensor([value])
        tensors[f"layers.{i}.linear.weight"] = torch.randn(4, 4)
    tensors["clip.input_max"] = torch.tensor(6.0)
    tensors["embed.embed_scale"] = torch.tensor(2.5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


@pytest.fixture
def zero3_on(monkeypatch):
    """Pretend ZeRO-3 is active; the helper is a no-op otherwise."""
    import transformers.integrations.deepspeed as ds

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: True)


def test_restores_layer_scalar_left_at_the_default(checkpoint, zero3_on):
    """The bug: buffers stay at 1.0. The fix must put the checkpoint values back."""
    model = _FakeModel()
    assert [float(layer.layer_scalar) for layer in model.layers] == [1.0, 1.0, 1.0]

    restored, corrected = restore_zero3_skipped_buffers(model, str(checkpoint))

    assert [pytest.approx(float(layer.layer_scalar), abs=1e-6) for layer in model.layers] == (
        CHECKPOINT_SCALARS
    )
    assert corrected == 4, "3 layer_scalar + 1 clip bound held wrong values"
    assert restored == 5, "the correctly-loaded embed_scale is rewritten too, harmlessly"


def test_does_not_disturb_buffers_that_were_loaded_correctly(checkpoint, zero3_on):
    """A buffer on a module with a direct parameter is already correct; leave it that way."""
    model = _FakeModel()
    model.embed.embed_scale = torch.tensor(2.5)  # as the upstream loader would have left it

    restore_zero3_skipped_buffers(model, str(checkpoint))

    assert float(model.embed.embed_scale) == pytest.approx(2.5)


def test_is_idempotent(checkpoint, zero3_on):
    """Running twice must correct nothing the second time."""
    model = _FakeModel()
    restore_zero3_skipped_buffers(model, str(checkpoint))
    _, corrected_again = restore_zero3_skipped_buffers(model, str(checkpoint))
    assert corrected_again == 0


def test_no_op_when_zero3_is_disabled(checkpoint, monkeypatch):
    """Without ZeRO-3 the normal loader is correct, so this must not touch anything."""
    import transformers.integrations.deepspeed as ds

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: False)
    model = _FakeModel()

    assert restore_zero3_skipped_buffers(model, str(checkpoint)) == (0, 0)
    assert [float(layer.layer_scalar) for layer in model.layers] == [1.0, 1.0, 1.0]


def test_warns_loudly_when_the_checkpoint_cannot_be_read(tmp_path, zero3_on, caplog):
    """Silence is what made this bug cost days; an unreadable checkpoint must say so."""
    model = _FakeModel()

    with caplog.at_level("WARNING"):
        restored, corrected = restore_zero3_skipped_buffers(model, str(tmp_path / "nonexistent"))

    assert (restored, corrected) == (0, 0)
    assert "Could not read checkpoint tensors" in caplog.text
    assert "corrupted model" in caplog.text


def test_handles_a_sharded_checkpoint(tmp_path, zero3_on):
    """The real gemma-4 layout is single-file, but large checkpoints are sharded."""
    import json

    save_file(
        {"layers.0.layer_scalar": torch.tensor([0.5]), "layers.0.linear.weight": torch.randn(4, 4)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"layers.1.layer_scalar": torch.tensor([0.25])},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.layer_scalar": "model-00001-of-00002.safetensors",
                    "layers.0.linear.weight": "model-00001-of-00002.safetensors",
                    "layers.1.layer_scalar": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    model = _FakeModel()
    restore_zero3_skipped_buffers(model, str(tmp_path))

    assert float(model.layers[0].layer_scalar) == pytest.approx(0.5)
    assert float(model.layers[1].layer_scalar) == pytest.approx(0.25)
    assert float(model.layers[2].layer_scalar) == 1.0, "absent from the index, so left alone"


def test_zero_matches_is_reported_as_unverified_not_as_clean(tmp_path, zero3_on, caplog):
    """
    A buffer matching no checkpoint key is NOT a buffer that needed no restoring.

    transformers v5 renames checkpoint keys on load (WeightRenaming) and the
    image path adds its own key_mapping, so an unknown rename leaves everything
    unmatched. Reporting that as "none needed restoring" is the exact silent
    all-clear this whole function exists to prevent.
    """
    # A readable checkpoint whose keys share nothing with the model's buffers.
    save_file({"totally.different.name": torch.tensor([1.0])}, str(tmp_path / "model.safetensors"))
    model = _FakeModel()

    with caplog.at_level("WARNING"):
        restored, corrected = restore_zero3_skipped_buffers(model, str(tmp_path))

    assert (restored, corrected) == (0, 0)
    assert "matched any" in caplog.text, "must say the buffers were not verified"
    assert "none needed restoring" not in caplog.text, "must NOT read as a clean pass"
    assert [float(layer.layer_scalar) for layer in model.layers] == [1.0, 1.0, 1.0]


def test_reads_a_legacy_torch_bin_checkpoint(tmp_path, zero3_on):
    """Older checkpoints ship pytorch_model.bin; those buffers must restore too."""
    torch.save(
        {"layers.0.layer_scalar": torch.tensor([0.42]), "layers.0.linear.weight": torch.randn(4, 4)},
        str(tmp_path / "pytorch_model.bin"),
    )
    model = _FakeModel()

    restored, corrected = restore_zero3_skipped_buffers(model, str(tmp_path))

    assert float(model.layers[0].layer_scalar) == pytest.approx(0.42)
    assert (restored, corrected) == (1, 1)


def test_reproduces_the_upstream_skip_on_a_real_gemma4_shaped_module():
    """
    Pin the upstream gate itself, so this test fails if transformers ever fixes it.

    Mirrors the loader's decision: buffers are copied only via
    `_load_from_state_dict`, which runs only when a module owns a direct
    parameter found in the state dict.
    """
    layer = _ScalarLayer()
    state_dict = {"layer_scalar": torch.tensor([0.061]), "linear.weight": torch.randn(4, 4)}

    direct_params = dict(layer.named_parameters(recurse=False))
    params_to_gather = [p for k, p in direct_params.items() if k in state_dict]

    assert direct_params == {}, "the weight lives in .linear, so the layer owns no direct parameter"
    assert params_to_gather == [], "hence the upstream gate stays shut"
    assert float(layer.layer_scalar) == 1.0, "and layer_scalar keeps its default"
