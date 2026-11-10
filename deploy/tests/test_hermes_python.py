"""components._hermes_python: the interpreter the Hermes skills are installed with.

Older Hermes keeps a venv inside its checkout. Newer Hermes (v0.21 and later) has none and
runs from a Python it manages itself under <home>/tools. Both have to be found, or the
skills fail on the newer one with advice ("reinstall Hermes") that does not help.
"""

import pytest

from launcher import components, config


@pytest.fixture(params=[True, False], ids=["windows", "posix"])
def layout(request, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IS_WINDOWS", request.param)
    home = tmp_path / "hermes"
    (home / "hermes-agent").mkdir(parents=True)
    venv_python = "Scripts/python.exe" if request.param else "bin/python"
    managed_python = "python.exe" if request.param else "bin/python"
    return home, venv_python, managed_python


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_the_venv_wins_when_it_exists(layout):
    home, venv_python, managed_python = layout
    venv = _touch(home / "hermes-agent" / "venv" / venv_python)
    _touch(home / "tools" / "python-3.14.7+20260901-win32-x64" / managed_python)

    assert components._hermes_python(home) == venv


def test_a_hermes_without_a_venv_uses_its_managed_python(layout):
    home, _, managed_python = layout
    managed = _touch(home / "tools" / "python-3.14.7+20260901-win32-x64" / managed_python)
    _touch(home / "tools" / "node-26.7.0-win32-x64" / "node.exe")

    assert components._hermes_python(home) == managed


def test_the_newest_managed_python_wins_after_an_upgrade(layout):
    home, _, managed_python = layout
    _touch(home / "tools" / "python-3.13.2+20250101-win32-x64" / managed_python)
    newest = _touch(home / "tools" / "python-3.14.7+20260901-win32-x64" / managed_python)

    assert components._hermes_python(home) == newest


def test_a_python_directory_without_an_interpreter_is_not_taken(layout):
    home, _, managed_python = layout
    (home / "tools" / "python-3.14.7+20260901-win32-x64").mkdir(parents=True)
    older = _touch(home / "tools" / "python-3.13.2+20250101-win32-x64" / managed_python)

    assert components._hermes_python(home) == older


def test_nothing_to_find_is_none(layout):
    home, _, _ = layout
    (home / "tools" / "node-26.7.0-win32-x64").mkdir(parents=True)

    assert components._hermes_python(home) is None
