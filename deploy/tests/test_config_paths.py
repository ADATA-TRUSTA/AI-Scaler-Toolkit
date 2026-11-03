"""config.components_root: which directory the big installed pieces go in.

The backend's environment (~3.6 GB) and ComfyUI (~4 GB) used to sit in the data
directory whatever the installer was told, so choosing another drive still filled the
system one. They follow the program directory now, and three cases have to stay true
at once: a source checkout must not grow an environment inside itself, a fresh install
must put one on the chosen drive, and an upgrade must not strand the one already
there -- a virtual environment cannot be relocated, so stranding it means 3.6 GB
downloaded again.
"""

import pytest

from launcher import config


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A Windows-shaped install: LocalAppData, and the program beside it."""
    local = tmp_path / "LocalAppData"
    program = local / "Programs" / config.APP_NAME
    program.mkdir(parents=True)
    (local / config.APP_NAME).mkdir(parents=True)

    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(config, "IS_WINDOWS", True)
    monkeypatch.setattr(config.sys, "executable", str(program / "AIScalerToolkit.exe"))
    config.components_root.cache_clear()
    yield local / config.APP_NAME, program
    config.components_root.cache_clear()


def frozen(monkeypatch, value):
    if value:
        monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    else:
        monkeypatch.delattr(config.sys, "frozen", raising=False)
    config.components_root.cache_clear()


def test_a_source_checkout_keeps_everything_in_the_data_directory(layout, monkeypatch):
    data, _program = layout
    frozen(monkeypatch, False)
    assert config.components_root() == data
    assert config.venv_dir() == data / "service" / ".venv"


def test_an_installed_build_puts_the_components_in_the_program_directory(layout, monkeypatch):
    data, program = layout
    frozen(monkeypatch, True)
    assert config.components_root() == program
    assert config.venv_dir() == program / "service" / ".venv"
    assert data not in config.venv_dir().parents


def test_an_environment_already_installed_is_left_where_it_is(layout, monkeypatch):
    data, _program = layout
    (data / "service" / ".venv").mkdir(parents=True)
    frozen(monkeypatch, True)
    assert config.components_root() == data
    assert config.venv_dir() == data / "service" / ".venv"


def test_strata_and_the_logs_do_not_follow_the_components(layout, monkeypatch):
    data, program = layout
    frozen(monkeypatch, True)
    assert config.components_root() == program
    # Strata's setup keeps Strata-data beside this folder and the uninstaller offers to
    # keep it; the program directory is cleared wholesale, so it must not go there.
    assert config.strata_dir() == data / "strata"
    assert config.log_dir() == data / "logs"
    # state.json records where things are, so it cannot move with them.
    assert config.state_file().parent == config.config_dir()
