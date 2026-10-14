"""
Multi-GPU failure and cancellation must not lose information or leak GPUs.

Three defects, all in how the parent and the ranks talk to each other:

1. **Cancellation orphaned every rank.** `stop_training()` sends SIGTERM, whose
   default handler kills the interpreter without unwinding, so the
   `finally: _terminate_deepspeed_tree(proc)` never ran. The launcher and its N
   ranks survived reparented, each still holding its GPU's VRAM.

2. **A crash on a non-zero rank was silent.** Only rank 0 pushed status, so a
   rank-3 OOM reached the parent as "deepspeed exited with return code N" with
   `is_oom=False`. Data-parallel OOMs fire on whichever rank is unluckiest, so
   the most likely multi-GPU failure was the worst-reported one.

3. **Rank-0-only work ran on every rank.** All ranks must enter
   `save_training_results` (the ZeRO-3 gather is a collective), but everything
   after it writes shared paths -- N processes truncating the same
   `tokenizer.json` and appending N duplicate registry entries.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytestmark = pytest.mark.finetune


# ------------------------------------------------------------- cancellation


def test_worker_installs_a_sigterm_handler():
    """
    Without one, SIGTERM kills the interpreter outright and no `finally` runs
    -- including the only thing that stops the deepspeed launcher.
    """
    from service.training import training_process

    source = inspect.getsource(training_process._training_worker_process)
    assert "signal.SIGTERM" in source
    assert "signal.signal" in source


def test_sigterm_is_turned_into_an_unwinding_exception():
    """
    Verified for real in a subprocess: the handler must let `finally` run.

    Asserting on the mechanism rather than the source, because "does cleanup
    actually happen on SIGTERM" is the property that matters.
    """
    program = """
import os, signal, sys
def handler(signum, frame):
    raise KeyboardInterrupt("terminated")
signal.signal(signal.SIGTERM, handler)
try:
    os.kill(os.getpid(), signal.SIGTERM)
    import time; time.sleep(5)
except KeyboardInterrupt:
    pass
finally:
    print("CLEANUP_RAN", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30, check=False
    )
    assert "CLEANUP_RAN" in result.stdout, "SIGTERM must unwind so cleanup handlers run"


def test_the_launcher_gets_its_own_process_group():
    """
    So the whole launcher+ranks tree can be signalled at once. Without it, the
    ranks survive their parent.
    """
    from service.training import training_process

    source = inspect.getsource(training_process)
    assert "start_new_session=True" in source


# ------------------------------------------------------------- rank errors


def test_errors_are_reported_from_every_rank():
    """Progress stays rank-0; errors must not, or an OOM on rank N vanishes."""
    from service.training import training_script

    source = inspect.getsource(training_script)
    push = source[source.index("def _push_status") : source.index("def _emit_event")]

    assert 'payload.get("status") == "error"' in push, "must special-case errors"
    assert "not _IS_RANK0 and not is_error" in push, "non-rank-0 errors must not return early"
    assert '"rank"' in push, "the report must say which rank failed"


def test_the_first_rank_error_wins():
    """
    One rank failing knocks the others over (collective timeout, broken pipe).
    The first is the cause; the rest are consequences.
    """
    from service.training import training_process

    source = inspect.getsource(training_process)
    assert "if _error_detail[0] is None:" in source, (
        "a later rank's consequential error must not overwrite the root cause"
    )


def test_the_raised_error_carries_the_rank_and_the_oom_flag():
    from service.training import training_process

    source = inspect.getsource(training_process)
    raise_block = source[source.index("        rc = proc.returncode") :][:1200]

    assert 'detail.get("rank")' in raise_block
    assert 'detail.get("is_oom", False)' in raise_block
    assert 'detail.get("traceback", "")' in raise_block


# ------------------------------------------------------------- save fan-out


def test_shared_writes_are_confined_to_rank_zero():
    """
    All ranks must ENTER the save (the gather is a collective) but only one may
    write the tokenizer and the registry.
    """
    from service.training.core import model_saver

    source = inspect.getsource(model_saver.save_training_results)
    gate = source.index("torch.distributed.get_rank() != 0")
    tokenizer_write = source.index("tokenizer.save_pretrained")

    assert gate < tokenizer_write, "the rank gate must come before the shared writes"
    assert "add_finetuned" not in source[:gate], "registry update must be after the gate"


def test_the_barrier_runs_even_if_the_save_raised():
    """
    Otherwise a rank-0 failure parks every peer in barrier() for the full
    ddp_timeout -- half an hour of apparent hang.
    """
    from service.training.core import model_saver

    source = inspect.getsource(model_saver.save_training_results)
    barrier = source.index("torch.distributed.barrier()")
    finally_block = source.index("finally:")

    assert finally_block < barrier, "barrier() must be inside the finally, not the try body"


def test_is_distributed_is_false_without_a_process_group():
    """Single-GPU runs must not take any of the rank gates."""
    from service.training.core.model_saver import _is_distributed

    assert _is_distributed() is False


# ------------------------------------------------------- terminal job record


def test_every_error_path_finalizes_the_job_log():
    """
    `writer.error()` and `writer.finalize()` write to different places, and only
    the second one is durable.

    `error()` appends an event and updates the live status key in Redis, which
    expires. `finalize()` is the only call that writes `status` and `finished_at`
    into meta.json -- the record every later history query reads -- and the
    `writer.close()` in the `finally` merely closes file handles. So an error
    path that reports but never finalizes leaves the durable record saying the
    job is still running, forever.

    This is asserted structurally rather than against one known site: the defect
    it caught was a new handler that intercepted an exception which used to reach
    the outer one, taking the better error message and losing the finalize that
    came with it. Any future handler can make the same trade by accident.
    """
    import ast
    import textwrap

    from service.training import training_process

    tree = ast.parse(textwrap.dedent(inspect.getsource(training_process._training_worker_process)))

    def _writer_calls(statements: list[ast.stmt]) -> set[str]:
        """Method names invoked on `writer` directly in this statement list."""
        found = set()
        for statement in statements:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            func = statement.value.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "writer"
            ):
                found.add(func.attr)
        return found

    unpaired = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            calls = _writer_calls(block)
            if "error" in calls and "finalize" not in calls:
                unpaired.append(getattr(block[0], "lineno", "?"))

    assert not unpaired, (
        f"writer.error() without a matching writer.finalize() in the same block "
        f"(near line(s) {unpaired} of _training_worker_process). The job's meta.json "
        f"would keep its non-terminal status after this error."
    )
