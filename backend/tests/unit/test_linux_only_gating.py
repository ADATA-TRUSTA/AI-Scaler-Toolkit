"""
The Windows gate on the suite's own Linux-only tests.

`tests/conftest.py` skips everything marked `linux_only` when the host is not
Linux, and applies that marker (plus `finetune`) to the training tree *by path*,
from the table in `tests/platform_gating.py`.

A path table rots silently in one direction only, which is the dangerous one: if
`tests/smoke/test_offload_training_matrix.py` is renamed, nothing errors -- the
entry simply stops matching, and 5 real DeepSpeed runs quietly come back on a
Windows host that cannot import deepspeed at all. So the entries are asserted to
exist rather than assumed.

The opposite failure gets a test too. `unit/test_platform_support.py` fakes the
OS with monkeypatch to assert the 501 refusals, so it is the one file that most
needs to run on Windows -- and its name and contents are full of "vllm", which is
exactly what a future path- or name-based rule would over-match on.
"""

import platform_gating as gating
import pytest

from service import platform_support


def _marker_names(item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


# --------------------------------------------------------------------------
# The path table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", gating.FINETUNE_PATHS, ids=lambda p: p.name)
def test_every_gated_path_still_exists(path):
    """A renamed or deleted entry un-gates its tests without any error."""
    assert path.exists(), (
        f"{path} is listed in tests/platform_gating.py::FINETUNE_PATHS but is not on "
        "disk; the tests it used to gate now run on Windows"
    )


def test_the_training_tree_is_gated_as_a_directory():
    """
    Gating the directory, not a file list, is what survives a new test file.

    The `finetune` marker was hand-written per module before this and had reached
    11 of 23 files; a list here would drift the same way.
    """
    training_dir = gating.TESTS_DIR / "unit" / "training"

    assert training_dir in gating.FINETUNE_PATHS
    assert training_dir.is_dir()


def test_every_training_file_matches_the_path_rule():
    """
    Asserted against the files on disk, not `session.items`, so it holds under
    `pytest tests/unit/test_linux_only_gating.py` too -- a check that only means
    something when the whole suite happens to be collected is one that passes
    vacuously exactly when someone is iterating on a narrow selection.
    """
    training_dir = gating.TESTS_DIR / "unit" / "training"
    files = sorted(training_dir.glob("test_*.py"))

    assert files, f"no test files under {training_dir} -- this test proves nothing"
    for path in files:
        assert gating.is_finetune_path(path), f"{path.name} would not be marked finetune/linux_only"


# --------------------------------------------------------------------------
# What the markers land on
# --------------------------------------------------------------------------


def test_collected_training_items_really_carry_both_markers(request):
    """
    The end-to-end half: the path rule above can be right while the hook that
    consumes it is not. Only meaningful when training tests are in this session,
    which is the case for any `pytest tests` or `pytest tests/unit` run.
    """
    training_dir = gating.TESTS_DIR / "unit" / "training"
    items = [
        item
        for item in request.session.items
        if item.path is not None and training_dir in item.path.parents
    ]
    if not items:
        pytest.skip("no training tests in this selection; covered by the path rule above")

    for item in items:
        assert {"finetune", "linux_only"} <= _marker_names(item), (
            f"{item.nodeid} is under tests/unit/training/ but is missing markers"
        )


@pytest.mark.parametrize("relative", gating.VLLM_MODULES)
def test_vllm_engine_modules_declare_the_marker(relative):
    """
    The five vLLM modules mark themselves, so the gate is visible in the file.

    Read as source rather than through collection: `pytest tests/unit` does not
    collect the smoke module, and a check that silently passes when the file was
    never collected is not a check.
    """
    source = (gating.TESTS_DIR / relative).read_text(encoding="utf-8")

    assert "pytest.mark.linux_only" in source, (
        f"{relative} exercises the vLLM engine but no longer declares linux_only"
    )


def test_the_platform_refusal_tests_are_never_gated():
    """
    These assert the 501s *as seen from Windows*; gating them deletes the only
    evidence that the refusal works on the platform it exists for.

    Both halves matter: the path table must not reach the file, and the file must
    not mark itself. It is full of the word "vllm", so it is the likeliest
    casualty of a future rule written against names instead of paths.
    """
    gate = gating.TESTS_DIR / "unit" / "test_platform_support.py"

    assert gate.exists()
    assert not gating.is_finetune_path(gate), (
        "test_platform_support.py is covered by the fine-tune path table; it must "
        "keep running on Windows -- it is what proves the Windows refusal works"
    )
    assert "linux_only" not in gate.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The skip itself
# --------------------------------------------------------------------------


def test_the_gate_agrees_with_the_service_on_what_linux_only_means(monkeypatch):
    """
    conftest gates on `platform.system() == "Linux"`; platform_support refuses on
    the same test. Two copies of one rule, so pin that they still answer alike --
    including for WSL, which reports "Linux" and is deliberately not special-cased.
    """
    # Installed, so this compares the platform rule alone: vLLM is also refused on a
    # Linux host that does not have it, which is a different question from the gate's.
    monkeypatch.setattr(platform_support, "vllm_installed", lambda: True)
    for system, supported in (("Linux", True), ("Windows", False), ("Darwin", False)):
        monkeypatch.setattr(platform_support.platform, "system", lambda s=system: s)

        assert platform_support.training_support().supported is supported
        assert platform_support.engine_support("vllm").supported is supported, (
            f"the vLLM refusal disagrees with the test gate on {system}"
        )


def test_skipping_is_decided_by_the_host_not_by_a_monkeypatch(request):
    """
    The skip is applied at collection time, before any test body runs, so a test
    that patches platform.system() cannot un-skip itself. Recorded here because
    the natural way to write a gate -- checking inside a fixture -- would behave
    differently, and the training tests do patch that function.
    """
    import inspect

    # Reached through pytest's plugin registry, which keys conftests by absolute
    # path, and not by `import conftest`: two files in this tree carry that name
    # and neither has an __init__.py, so sys.modules["conftest"] is whichever was
    # imported last -- tests/unit/training/conftest.py, when it is in the run.
    root_conftest = request.config.pluginmanager.get_plugin(str(gating.TESTS_DIR / "conftest.py"))

    assert root_conftest is not None, "tests/conftest.py is not registered as a plugin"
    source = inspect.getsource(root_conftest.pytest_collection_modifyitems)

    assert "platform.system()" in source
    assert "add_marker" in source
