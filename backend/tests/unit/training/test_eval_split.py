"""Held-out test-split behaviour: dataset splitting, metric tagging, log routing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytestmark = pytest.mark.finetune


def _bf16_gpu_available() -> bool:
    """
    Whether TrainingArguments can be built with bf16 on this host.

    Deliberately asks transformers the same question it asks itself, so the skip
    condition cannot drift from the failure condition: with no bf16-capable device
    its __post_init__ raises "Your setup doesn't support bf16/gpu".
    """
    try:
        from transformers.utils import is_torch_bf16_gpu_available

        return is_torch_bf16_gpu_available()
    except Exception:
        return False


# Only the tests that construct real TrainingArguments need this; the rest of the
# module is pure logic and must keep running everywhere. Without the skip a
# CPU-only checkout (torch installed without the cuda/xpu extra) reports failures
# that say nothing about the code under test.
requires_bf16_gpu = pytest.mark.skipif(
    not _bf16_gpu_available(),
    reason="no bf16-capable GPU; TrainingArguments cannot be constructed here",
)


# --- dataset splitting ------------------------------------------------------


def _toy_dataset(n):
    from datasets import Dataset

    return Dataset.from_dict({"text": [f"row {i}" for i in range(n)]})


def test_split_produces_disjoint_train_eval():
    from service.training.core import split_train_eval

    ds = _toy_dataset(20)
    train, ev = split_train_eval(ds, 0.25, seed=1)
    assert ev is not None
    assert len(ev) == 5
    assert len(train) == 15
    # No overlap between the two subsets.
    assert set(train["text"]).isdisjoint(set(ev["text"]))


def test_split_is_deterministic_with_seed():
    from service.training.core import split_train_eval

    ds = _toy_dataset(20)
    _, e1 = split_train_eval(ds, 0.25, seed=7)
    _, e2 = split_train_eval(ds, 0.25, seed=7)
    assert sorted(e1["text"]) == sorted(e2["text"])


def test_zero_ratio_returns_no_eval():
    from service.training.core import split_train_eval

    ds = _toy_dataset(10)
    train, ev = split_train_eval(ds, 0.0)
    assert ev is None
    assert len(train) == 10


def test_too_small_split_skips_eval():
    """A ratio that rounds to zero eval rows must not split (keeps training data)."""
    from service.training.core import split_train_eval

    ds = _toy_dataset(4)
    train, ev = split_train_eval(ds, 0.1)  # 0.4 -> rounds to 0 eval rows
    assert ev is None
    assert len(train) == 4


# --- eval metric routing in the job logger ----------------------------------


class _FakeRedis:
    def __init__(self):
        self.lists = {}

    def rpush(self, key, val):
        self.lists.setdefault(key, []).append(val)


def _writer(tmp_path, redis):
    from service.training.core.job_logger import JobLogWriter

    return JobLogWriter(job_id="jobX", base_dir=tmp_path, redis_client=redis)


def test_train_and_eval_metrics_go_to_separate_lists(tmp_path):
    import json

    redis = _FakeRedis()
    w = _writer(tmp_path, redis)
    w.metric({"step": 1, "loss": 2.0, "split": "train"})
    w.metric({"step": 1, "loss": 1.5, "split": "eval"})

    train_key = "training:history:jobX:metrics"
    eval_key = "training:history:jobX:eval_metrics"
    assert len(redis.lists[train_key]) == 1
    assert len(redis.lists[eval_key]) == 1
    assert json.loads(redis.lists[eval_key][0])["split"] == "eval"
    assert json.loads(redis.lists[train_key][0])["split"] == "train"


def test_metric_without_split_defaults_to_train(tmp_path):
    redis = _FakeRedis()
    w = _writer(tmp_path, redis)
    w.metric({"step": 1, "loss": 2.0})
    assert "training:history:jobX:metrics" in redis.lists
    assert "training:history:jobX:eval_metrics" not in redis.lists


# --- human-readable log line tagging ----------------------------------------


def test_eval_metric_line_is_tagged():
    from service.training.core.training_events import format_event_line

    line = format_event_line(
        {"type": "metric", "data": {"step": 5, "loss": 1.2, "split": "eval", "final": True}}
    )
    assert "[eval]" in line and "[final]" in line
    assert "loss=1.2" in line


def test_train_metric_line_is_not_tagged():
    from service.training.core.training_events import format_event_line

    line = format_event_line({"type": "metric", "data": {"step": 5, "loss": 1.2}})
    assert "[eval]" not in line


# --- training args wiring ---------------------------------------------------


@requires_bf16_gpu
def test_eval_enabled_sets_eval_strategy():
    from service.config_models import TrainingConfig
    from service.training.core.strategies import SFTStrategy

    cfg = TrainingConfig(
        model_name="x", method="lora", dataset_path="d.jsonl", output_dir="/tmp/o",
        eval_split_ratio=0.1, eval_steps=5,
    )
    args = SFTStrategy(cfg).get_training_args()
    assert str(args.eval_strategy) in ("IntervalStrategy.STEPS", "steps")
    assert args.eval_steps == 5


@requires_bf16_gpu
def test_no_eval_leaves_strategy_off():
    from service.config_models import TrainingConfig
    from service.training.core.strategies import SFTStrategy

    cfg = TrainingConfig(
        model_name="x", method="lora", dataset_path="d.jsonl", output_dir="/tmp/o",
    )
    args = SFTStrategy(cfg).get_training_args()
    assert str(args.eval_strategy) in ("IntervalStrategy.NO", "no")
