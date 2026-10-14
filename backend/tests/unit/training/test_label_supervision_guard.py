"""
Truncation must not silently leave a run with nothing to learn from.

The completion sits at the END of the sequence, so `max_seq_length` cuts it off
first. A row whose labels are all -100 contributes no gradient; when every row
in a batch is like that, the step's loss is exactly 0.0. The run completes and
reports a flawless curve, and the adapter learned nothing -- loss of 0 reads as
spectacular convergence.

Found by actually running the matrix, not by reading code: SmolVLM +
a small mixed image-and-text dataset at `max_seq_length=512` trains for six steps with
`loss: 0` and `mean_token_accuracy: 0` at every one of them, and nothing warns.

Whether a row survives depends on its own length, so the guard samples across
the dataset. Checking only the head misses it whenever the long prompts come
later, which is exactly what happened on the first attempt at this guard.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.model_loader import verify_labels_are_supervised  # noqa: E402

pytestmark = pytest.mark.finetune

MASKED = -100


class _Dataset:
    """Rows are opaque to the guard; only the collator interprets them."""

    def __init__(self, rows: list[int]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> int:
        return self._rows[index]


def _collator(starved: set[int]):
    """Collate a single row; rows in `starved` come back fully masked."""

    def collate(batch: list[int]) -> dict:
        labels = [
            [MASKED, MASKED, MASKED] if row in starved else [MASKED, 7, 8] for row in batch
        ]
        return {"labels": torch.tensor(labels)}

    return collate


def test_all_rows_starved_is_an_error():
    """The catastrophic case: loss would be 0.0 forever."""
    rows = list(range(8))

    with pytest.raises(ValueError, match="ALL labels masked"):
        verify_labels_are_supervised(_collator(set(rows)), _Dataset(rows), max_seq_length=512)


def test_the_error_names_the_setting_to_change():
    rows = list(range(4))

    with pytest.raises(ValueError) as excinfo:
        verify_labels_are_supervised(_collator(set(rows)), _Dataset(rows), max_seq_length=512)

    message = str(excinfo.value)
    assert "max_seq_length=512" in message
    assert "image tokens count toward it" in message, "the non-obvious part must be spelled out"


def test_some_rows_starved_warns_but_proceeds(caplog):
    """
    A handful of over-long rows in a big dataset is wasteful, not fatal --
    training on the rest is still valid.
    """
    rows = list(range(8))

    with caplog.at_level("WARNING"):
        verify_labels_are_supervised(_collator({1, 2}), _Dataset(rows), max_seq_length=1024)

    assert "lose their entire completion" in caplog.text
    assert "max_seq_length=1024" in caplog.text


def test_healthy_dataset_passes_quietly():
    rows = list(range(8))
    verify_labels_are_supervised(_collator(set()), _Dataset(rows), max_seq_length=4096)


def test_rows_are_sampled_across_the_dataset_not_off_the_head():
    """
    The first version of this guard checked rows 0-1 and passed, while the
    shuffled batch the dataloader actually produced had zero supervision. Only
    the later rows were starved.
    """
    rows = list(range(100))
    seen: list[int] = []

    def recording_collate(batch: list[int]) -> dict:
        seen.extend(batch)
        return {"labels": torch.tensor([[MASKED, 7, 8] for _ in batch])}

    verify_labels_are_supervised(recording_collate, _Dataset(rows), max_seq_length=4096)

    assert len(seen) > 1
    assert max(seen) > 50, f"sampling stayed near the head: {seen}"


def test_starvation_late_in_the_dataset_is_still_caught():
    rows = list(range(100))
    # Only the back half is over-long.
    with pytest.raises(ValueError, match="ALL labels masked"):
        verify_labels_are_supervised(
            _collator(set(rows)), _Dataset(rows), max_seq_length=256
        )


def test_a_batch_without_labels_is_skipped_not_failed():
    """Some collators do not emit labels; that is not this guard's business."""
    verify_labels_are_supervised(
        lambda _batch: {"input_ids": torch.tensor([[1, 2]])}, _Dataset([0, 1]), 4096
    )


def test_a_collator_that_raises_does_not_fail_the_run(caplog):
    """A verification helper must never be the reason a good run dies."""

    def exploding(_batch):
        raise RuntimeError("collator blew up")

    with caplog.at_level("WARNING"):
        verify_labels_are_supervised(exploding, _Dataset([0, 1]), 4096)

    assert "Could not collate" in caplog.text


def test_empty_dataset_is_a_noop():
    verify_labels_are_supervised(_collator(set()), _Dataset([]), 4096)
