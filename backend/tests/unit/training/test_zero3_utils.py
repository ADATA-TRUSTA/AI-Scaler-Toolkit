"""drain_zero3_inflight: quiesce ZeRO-3 before a manual GatheredParameters save.

Pure logic -- no model or GPU needed. Guards the fix for the
"Cannot partition a param in flight" crash that killed the adapter save
mid-training under ZeRO-3 offload.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.zero3_utils import drain_zero3_inflight  # noqa: E402

pytestmark = pytest.mark.finetune


class _Engine:
    """Stand-in DeepSpeedEngine: presence of empty_partition_cache is the signal."""

    def __init__(self, boom=False):
        self.calls = 0
        self._boom = boom

    def empty_partition_cache(self):
        self.calls += 1
        if self._boom:
            raise RuntimeError("coordinator drain blew up")


class _Trainer:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def test_noop_on_plain_object():
    # Nothing DeepSpeed-shaped -> no drain, caller proceeds unchanged.
    assert drain_zero3_inflight(object()) is False


def test_engine_passed_directly_is_drained():
    eng = _Engine()
    assert drain_zero3_inflight(eng) is True
    assert eng.calls == 1


def test_engine_found_via_model_wrapped():
    eng = _Engine()
    trainer = _Trainer(model_wrapped=eng, deepspeed=None)
    assert drain_zero3_inflight(trainer) is True
    assert eng.calls == 1


def test_engine_found_via_deepspeed_attr():
    eng = _Engine()
    trainer = _Trainer(model_wrapped=None, deepspeed=eng)
    assert drain_zero3_inflight(trainer) is True
    assert eng.calls == 1


def test_non_zero3_trainer_is_noop():
    # model_wrapped exists but is not a ZeRO-3 engine (no empty_partition_cache).
    trainer = _Trainer(model_wrapped=object(), deepspeed=None)
    assert drain_zero3_inflight(trainer) is False


def test_drain_exception_is_swallowed():
    # A failing drain must never lose the save it precedes.
    eng = _Engine(boom=True)
    assert drain_zero3_inflight(eng) is False
    assert eng.calls == 1
