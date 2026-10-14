"""
A fine-tune has to be verified by what it says, not by its loss curve.

`eval_loss` is computed by the same collator with the same
`completion_only_loss` as training, so it cannot distinguish "learned the task"
from "learned a degenerate rule that fits the data" -- e.g. answering "Simon"
for every photo of a person. Both produce a smooth falling curve.

Before this existed, `compute_metrics` and `.generate(` appeared zero times in
`service/training/`, and the demo dataset's held-out probe file -- unseen
images with a question and an expected answer -- had no code reference at all.
The evaluation data was authored; the consumer never was.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.generation_check import (  # noqa: E402
    collective_world_size,
    load_probes,
    run_generation_check,
)

pytestmark = pytest.mark.finetune


def _write(tmp_path: Path, probes: list[dict]) -> str:
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(probes), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ loading


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_probes(str(tmp_path / "nope.json"))


def test_a_probe_with_no_assertion_is_rejected(tmp_path):
    """A probe that cannot fail is worse than no probe -- it reads as coverage."""
    path = _write(tmp_path, [{"question": "Who is this?"}])
    with pytest.raises(ValueError, match="expected"):
        load_probes(path)


def test_a_probe_needs_a_question(tmp_path):
    path = _write(tmp_path, [{"expected": "Simon"}])
    with pytest.raises(ValueError, match="question"):
        load_probes(path)


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_probes(_write(tmp_path, []))


def test_relative_image_paths_resolve_against_the_probe_file(tmp_path):
    """Matches how the training dataset resolves its own image paths."""
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "a.png").write_bytes(b"not really a png")
    path = _write(tmp_path, [{"image": "imgs/a.png", "question": "Who?", "expected": "Simon"}])

    probes = load_probes(path)

    assert probes[0]["image"] == str(tmp_path / "imgs" / "a.png")


def test_a_missing_image_is_caught_at_load_not_mid_run(tmp_path):
    path = _write(tmp_path, [{"image": "imgs/gone.png", "question": "Who?", "expected": "x"}])
    with pytest.raises(ValueError, match="image not found"):
        load_probes(path)


# ------------------------------------------------------------------ judging


class _FakeProcessor:
    """Returns a canned answer, so judging is tested without a model."""

    def __init__(self, answer: str):
        self._answer = answer

    def apply_chat_template(self, *_args, **_kwargs):
        import torch

        return SimpleNamespace(
            __getitem__=lambda _s, _k: torch.tensor([[1, 2, 3]]),
            to=lambda _device: {"input_ids": torch.tensor([[1, 2, 3]])},
        )

    def decode(self, *_args, **_kwargs) -> str:
        return self._answer


class _FakeModel:
    device = "cpu"

    def eval(self) -> None:
        pass

    def generate(self, **_kwargs):
        import torch

        return torch.tensor([[1, 2, 3, 9, 9]])


class _ExplodingProcessor(_FakeProcessor):
    """Fails every probe, for the two tests that differ only in how that is handled."""

    def apply_chat_template(self, *_args, **_kwargs):
        raise RuntimeError("processor blew up")


def _check(answer: str, probe: dict) -> dict:
    return run_generation_check(_FakeModel(), _FakeProcessor(answer), [probe])


def test_expected_substring_passes():
    result = _check("He is Simon from ADATA", {"question": "Who?", "expected": "Simon"})
    assert result["passed"] == 1 and result["failed"] == 0


def test_matching_is_case_insensitive():
    assert _check("he is simon", {"question": "Who?", "expected": "Simon"})["passed"] == 1


def test_missing_expected_substring_fails():
    result = _check("I cannot tell who this is", {"question": "Who?", "expected": "Simon"})
    assert result["failed"] == 1
    assert "expected" in result["results"][0]["reason"]


def test_expected_not_catches_the_degenerate_answer():
    """
    The whole point: a model that answers "Simon" for everyone passes every
    positive probe. Only a negative probe catches it.
    """
    result = _check(
        "He is Simon from the greatest Company ADATA",
        {"question": "Who?", "expected_not": "He is Simon"},
    )
    assert result["failed"] == 1


def test_both_assertions_must_hold():
    probe = {"question": "Who?", "expected": "Shalley", "expected_not": "He is Simon"}
    assert _check("This is Shalley.", probe)["passed"] == 1
    assert _check("He is Simon.", probe)["failed"] == 1
    assert _check("Some other person.", probe)["failed"] == 1


def test_a_probe_that_errors_fails_rather_than_aborting_the_rest(monkeypatch):
    """
    One bad probe must not discard the verdicts for the others.

    Only on one process, though -- the world size is pinned here rather than left
    to the ambient default, because the same input must raise across ranks and
    that contract is what the rank-safety test below asserts.
    """
    monkeypatch.setattr("service.training.core.generation_check.collective_world_size", lambda: 1)

    result = run_generation_check(
        _FakeModel(), _ExplodingProcessor("x"), [{"question": "a", "expected": "y"}]
    )

    assert result["failed"] == 1
    assert "errored" in result["results"][0]["reason"]
    assert result["results"][0]["answer"] is None


def test_summary_counts_add_up():
    probes = [
        {"question": "1", "expected": "Simon"},
        {"question": "2", "expected": "Nobody"},
    ]
    result = run_generation_check(_FakeModel(), _FakeProcessor("Simon"), probes)

    assert result["total"] == 2
    assert result["passed"] + result["failed"] == result["total"]
    assert result["passed"] == 1


def test_the_run_records_what_the_model_actually_said():
    """A verdict without the answer is unactionable when it fails."""
    result = _check("He is Simon", {"question": "Who?", "expected": "Simon", "note": "unseen"})

    entry = result["results"][0]
    assert entry["answer"] == "He is Simon"
    assert entry["note"] == "unseen"


# ── Rank safety ──────────────────────────────────────────────────────────────
#
# Recovering from a probe failure is correct on one process and unsafe across
# ranks. Under ZeRO-3 every generate() is a collective, so a rank that catches an
# error and moves on has abandoned an all-gather the others are still inside; the
# group cannot be repaired from there. Continuing only changes how the job dies --
# from "this rank, this probe, this traceback" into an NCCL watchdog timeout
# minutes later that names none of them.


def test_a_probe_error_propagates_under_a_collective(monkeypatch):
    """
    The same input the single-process test recovers from must raise here.

    Paired deliberately with test_a_probe_that_errors_fails_rather_than_aborting
    _the_rest: same processor, same probe, opposite outcome, and the world size is
    the only thing that differs.
    """
    monkeypatch.setattr("service.training.core.generation_check.collective_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="processor blew up"):
        run_generation_check(
            _FakeModel(), _ExplodingProcessor("x"), [{"question": "a", "expected": "y"}]
        )


def test_world_size_comes_from_the_live_group_not_the_environment(monkeypatch):
    """
    The launcher exports WORLD_SIZE before deepspeed.initialize builds the group,
    so trusting the variable would treat a window with no collective in flight as
    a collective one -- and, on the other side, would make this check depend on a
    variable a test or a stray export can set.
    """
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")

    assert collective_world_size() == 1
