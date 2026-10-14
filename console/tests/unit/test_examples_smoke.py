"""
Every example must at least start.

These run the scripts as subprocesses the way a user would, which is what
catches an import that only resolves because a test already imported the
package, or a default settings path that no longer exists.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI_EXAMPLES = [
    "examples/chat/batch.py",
    "examples/chat/image_batch.py",
    "examples/chat/parallel.py",
    "examples/model/load.py",
    "examples/model/unload.py",
    "examples/model/list.py",
    "examples/finetune/train.py",
    "examples/workflow/full.py",
]

# These two are linear demos with no argparse; importing them is the smoke test.
IMPORT_ONLY_EXAMPLES = ["examples/chat/basic.py", "examples/chat/image.py"]

INTERNAL_SCRIPTS = [
    "internal/kv_offload/kv_offload_long_prompt_test.py",
    "internal/stability/stability_test.py",
]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_example_list_matches_what_is_on_disk() -> None:
    """A new example must be added here, not silently go untested."""
    on_disk = {
        str(p).replace("\\", "/")
        for p in Path("examples").rglob("*.py")
        if "__pycache__" not in str(p)
    }
    covered = set(CLI_EXAMPLES) | set(IMPORT_ONLY_EXAMPLES)

    assert on_disk == covered, f"uncovered: {sorted(on_disk - covered)}"


@pytest.mark.parametrize("script", CLI_EXAMPLES)
def test_cli_example_shows_help(script: str) -> None:
    result = run(script, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", CLI_EXAMPLES)
def test_cli_example_documents_its_default_settings_path(script: str) -> None:
    """If --help advertises a config path, that path has to exist."""
    result = run(script, "--help")

    for token in result.stdout.split():
        cleaned = token.strip("()[],:'\"")
        if cleaned.startswith("configs/") and cleaned.endswith(".json"):
            assert Path(cleaned).is_file(), f"{script} --help offers missing {cleaned}"


@pytest.mark.parametrize("script", IMPORT_ONLY_EXAMPLES)
def test_linear_example_imports_cleanly(script: str) -> None:
    """Executes module level only; the __main__ guard keeps it off the network."""
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('smoke',{script!r});"
        "m=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "print(type(m.client).__name__)"
    )
    result = run("-c", code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OpenAI"


@pytest.mark.parametrize("script", INTERNAL_SCRIPTS)
def test_surviving_internal_script_shows_help(script: str) -> None:
    result = run(script, "--help")

    assert result.returncode == 0, result.stderr


def test_package_imports_without_touching_the_working_directory() -> None:
    """Proves the examples work from the installed package, not from cwd."""
    result = subprocess.run(
        [sys.executable, "-c", "import ai_scaler_console; print(ai_scaler_console.__file__)"],
        capture_output=True,
        text=True,
        cwd=Path(sys.executable).parent,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "ai_scaler_console" in result.stdout
