"""
Shared fixtures and collection rules for the console test suite.

Integration tests need a live backend with a model loaded, so they are
deselected unless ``--integration`` is passed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in flag for tests that need a running backend."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="also run tests marked `integration` (needs a live backend)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --integration was passed."""
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="needs a live backend; pass --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run every test from the repo root.

    The examples and configs use repo-root-relative paths, and the README tells
    users to run them from there, so the tests should agree.
    """
    monkeypatch.chdir(PROJECT_ROOT)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """
    Drop every variable settings.py reads, so defaults can be asserted.

    Also blocks python-dotenv from re-reading a developer's local .env during
    the reload that follows.
    """
    for name in ("BACKEND_URL", "API_KEY", "VERIFY_SSL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    yield monkeypatch
