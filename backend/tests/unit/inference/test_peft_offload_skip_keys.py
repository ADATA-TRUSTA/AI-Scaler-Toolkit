"""PEFT's re-dispatch must not clear accelerate's device-placement skip list.

transformers dispatches an offloaded model with
``skip_keys=model._skip_keys_device_placement``. PEFT re-dispatches the wrapped
model and does not forward that argument, so every ``AlignDevicesHook`` is left
with ``skip_keys=None``. accelerate then sends those kwargs through
``send_to_device``, which rebuilds a dict rather than passing the same object --
so a dict the layers mutate to talk to each other gets copied per layer.

Gemma 4 shares KV across layers through such a dict: layer 22 writes
``shared_kv_states[22]`` into its own copy and layer 24 raises ``KeyError: 22``.
The base model works because transformers' own dispatch got it right; only
attaching the adapter breaks it.
"""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from service.inference.load_model.peft_loader import (  # noqa: E402
    _restore_device_placement_skip_keys,
)

SKIP_KEYS = ["past_key_values", "shared_kv_states"]


@pytest.fixture
def hooks():
    return pytest.importorskip("accelerate.hooks")


def test_wiped_hook_is_repaired(hooks):
    model = nn.Sequential(nn.Linear(2, 2))
    model[0]._hf_hook = hooks.AlignDevicesHook(skip_keys=None)

    assert _restore_device_placement_skip_keys(model, SKIP_KEYS) == 1
    assert model[0]._hf_hook.skip_keys == SKIP_KEYS


def test_hook_nested_in_sequential_is_repaired(hooks):
    """accelerate composes hooks, so the wiped one can be one level down."""
    model = nn.Sequential(nn.Linear(2, 2))
    model[0]._hf_hook = hooks.SequentialHook(
        hooks.AlignDevicesHook(skip_keys=None), hooks.AlignDevicesHook(skip_keys=None)
    )

    assert _restore_device_placement_skip_keys(model, SKIP_KEYS) == 2
    assert all(h.skip_keys == SKIP_KEYS for h in model[0]._hf_hook.hooks)


def test_explicit_skip_keys_are_left_alone(hooks):
    """Only the cleared list is repaired; a deliberate one must survive."""
    model = nn.Sequential(nn.Linear(2, 2))
    model[0]._hf_hook = hooks.AlignDevicesHook(skip_keys=["deliberate"])

    assert _restore_device_placement_skip_keys(model, SKIP_KEYS) == 0
    assert model[0]._hf_hook.skip_keys == ["deliberate"]


def test_unhooked_module_is_a_noop(hooks):
    """A single-device load carries no hooks and must not raise."""
    model = nn.Sequential(nn.Linear(2, 2))

    assert _restore_device_placement_skip_keys(model, SKIP_KEYS) == 0


@pytest.mark.parametrize("skip_keys", [None, [], ""])
def test_model_without_skip_keys_is_a_noop(hooks, skip_keys):
    """Most architectures declare nothing to skip; they must be untouched."""
    model = nn.Sequential(nn.Linear(2, 2))
    model[0]._hf_hook = hooks.AlignDevicesHook(skip_keys=None)

    assert _restore_device_placement_skip_keys(model, skip_keys) == 0
    assert model[0]._hf_hook.skip_keys is None
