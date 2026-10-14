"""
Multi-GPU reporting paths that a single-GPU run never exercises.

These four went unnoticed for the same reason: the single-GPU path is the one
that gets run, and each of these is only wrong once a launcher sits between the
worker and the training code.

Nothing here needs two GPUs. The status parsing is a pure function over strings,
the payload shape is a dict comparison, and the process-tree widening is checked
against a fake psutil. The end-to-end behaviour still needs real hardware; what
is pinned here is the logic that was wrong.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytestmark = pytest.mark.finetune


# --- status extraction ------------------------------------------------------


@pytest.fixture(scope="module")
def extract():
    from service.training.training_process import extract_status_payloads

    return extract_status_payloads


def test_a_clean_status_line_is_read(extract):
    payloads, rest = extract('__STATUS__:{"status": "running", "loss": 1.5}')
    assert payloads == [{"status": "running", "loss": 1.5}]
    assert rest == ""


def test_a_progress_bar_in_front_does_not_hide_the_status(extract):
    """
    The regression. tqdm draws on stderr with a carriage return and no newline,
    stderr is merged into stdout, so one read holds a half-drawn bar and then the
    marker -- and `startswith` stops matching. Loss and accuracy silently stopped
    reaching Redis on multi-GPU runs while single-GPU runs kept working.
    """
    line = '  3%|▎         | 1/40 [00:12<08:01,  1.2s/it]__STATUS__:{"status": "progress", "step": 1}'
    payloads, rest = extract(line)
    assert payloads == [{"status": "progress", "step": 1}]
    assert "1/40" in rest, "the bar itself should still be available to log"


def test_a_status_between_two_bar_redraws_is_read(extract):
    line = '\r 50%|█████|__STATUS__:{"type": "metric", "data": {"loss": 2.0}}\r 51%|█████|'
    payloads, _ = extract(line)
    assert payloads == [{"type": "metric", "data": {"loss": 2.0}}]


def test_two_statuses_in_one_read_are_both_kept(extract):
    """Without a newline between them, one read can carry both."""
    payloads, _ = extract('__STATUS__:{"a": 1}__STATUS__:{"b": 2}')
    assert payloads == [{"a": 1}, {"b": 2}]


def test_a_brace_inside_a_value_does_not_truncate_the_payload(extract):
    """Length comes from decoding, not from scanning for the next `}`."""
    payloads, _ = extract('__STATUS__:{"error": "expected } here", "status": "error"}')
    assert payloads == [{"error": "expected } here", "status": "error"}]


def test_a_marker_followed_by_junk_is_left_as_text(extract):
    payloads, rest = extract("__STATUS__:not json at all")
    assert payloads == []
    assert "not json at all" in rest


def test_a_line_with_no_marker_is_all_remainder(extract):
    payloads, rest = extract("[deepspeed] Using /usr/bin/ld")
    assert payloads == []
    assert rest == "[deepspeed] Using /usr/bin/ld"


def test_the_stdout_reader_actually_uses_the_extractor():
    """
    The cases above prove the function is right, not that anything calls it.
    `_drain_stdout` is a closure with no seam to reach it through, so this reads
    the source: a revert to `startswith` would leave every test above passing
    while the bug came straight back.
    """
    root = Path(__file__).resolve().parents[3]
    source = (root / "service/training/training_process.py").read_text(encoding="utf-8")

    assert "startswith(_DS_STATUS_PREFIX)" not in source, (
        "the stdout reader is matching the status marker at position zero again; "
        "a tqdm bar in the same read hides it"
    )
    assert "extract_status_payloads(stripped)" in source


# --- metric payload shape ---------------------------------------------------


def _payload_keys_from(source: str, marker: str) -> set[str]:
    """Read the literal dict that follows `marker` and return its keys."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == marker for k in node.keys
        ):
            return {k.value for k in node.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"no dict literal containing {marker!r} found")


def test_both_entry_points_report_the_same_metric_keys():
    """
    The multi-GPU payload was missing `accuracy` and `split`, so the UI had no
    accuracy curve on multi-GPU runs and no way to separate eval from train.
    Both payloads feed the same writer and the same client, so a key present on
    one side and absent on the other is a silent hole rather than an error.
    """
    root = Path(__file__).resolve().parents[3]
    single = (root / "service/training/training_process.py").read_text(encoding="utf-8")
    multi = (root / "service/training/training_script.py").read_text(encoding="utf-8")

    single_keys = _payload_keys_from(single, "learning_rate")
    multi_keys = _payload_keys_from(multi, "learning_rate")

    missing = single_keys - multi_keys
    assert not missing, f"the multi-GPU metric payload is missing {sorted(missing)}"


def test_the_multi_gpu_path_separates_eval_from_train():
    """
    `loss or eval_loss` folded an evaluation point into the training curve, where
    it reads as a sudden jump. The split has to be decided before the loss is
    read, not after.
    """
    root = Path(__file__).resolve().parents[3]
    multi = (root / "service/training/training_script.py").read_text(encoding="utf-8")

    assert 'last_log.get("loss") or last_log.get("eval_loss")' not in multi, (
        "the multi-GPU path is folding eval loss back into the training curve"
    )
    assert '"split"' in multi and "eval_mean_token_accuracy" in multi


# --- process tree widening --------------------------------------------------


class _FakeProc:
    def __init__(self, pid, children=(), parent=None):
        self.pid = pid
        self._children = list(children)
        self._parent = parent

    def children(self, recursive=False):  # noqa: ARG002 - the fakes are one level deep
        return self._children

    def parent(self):
        return self._parent


def _monitor():
    from service.utils.system_monitor import SystemMonitor

    return SystemMonitor.__new__(SystemMonitor)


@pytest.mark.parametrize(
    ("world_size", "expect_sibling"),
    [("1", False), ("2", True)],
    ids=["single-gpu-stays-narrow", "distributed-includes-siblings"],
)
def test_sibling_ranks_are_counted_only_when_distributed(monkeypatch, world_size, expect_sibling):
    """
    deepspeed starts every rank as a child of the launcher, so rank 1 is a
    sibling of rank 0 rather than a descendant -- which is why GPU 1 reported 0.
    Widening unconditionally would let unrelated siblings count against the job,
    so it is gated on WORLD_SIZE.
    """
    import psutil

    monitor = _monitor()
    rank1 = _FakeProc(pid=2002)
    launcher = _FakeProc(pid=1000)
    rank0 = _FakeProc(pid=2001, parent=launcher)
    launcher._children = [rank0, rank1]

    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setattr(psutil, "Process", lambda *a, **k: rank0)
    monkeypatch.setattr("os.getpid", lambda: 2001)
    monkeypatch.setattr(
        type(monitor), "_expand_process_identity_set", lambda _self, pids: set(pids)
    )

    pids = monitor._get_process_tree_pid_candidates()

    assert 2001 in pids, "the sampling rank itself must always be counted"
    assert (2002 in pids) is expect_sibling
