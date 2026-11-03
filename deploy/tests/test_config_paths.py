"""config.components_root: which directory the big installed pieces go in.

The backend's environment (~3.6 GB) and ComfyUI (~4 GB) used to sit in the data
directory whatever the installer was told, so choosing another drive still filled the
system one. They follow the program directory now, and three cases have to stay true
at once: a source checkout must not grow an environment inside itself, a fresh install
must put one on the chosen drive, and an upgrade must not strand the one already
there -- a virtual environment cannot be relocated, so stranding it means 3.6 GB
downloaded again.
"""

from pathlib import Path

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


def test_strata_moves_but_what_it_derives_from_a_model_does_not(layout, monkeypatch):
    data, program = layout
    frozen(monkeypatch, True)
    assert config.components_root() == program
    # An engine and an environment: an uninstall should take it.
    assert config.strata_dir() == program / "strata"
    # The expert pack and the draft layer: gigabytes, minutes to rebuild, and the
    # uninstaller offers to keep them -- so not in the directory cleared wholesale.
    assert config.strata_data_dir() == data / "Strata-data"
    assert program not in config.strata_data_dir().parents


def test_the_logs_and_the_settings_stay_where_they_are(layout, monkeypatch):
    data, _program = layout
    frozen(monkeypatch, True)
    assert config.log_dir() == data / "logs"
    # state.json records where things are, so it cannot move with them.
    assert config.state_file().parent == config.config_dir()


@pytest.mark.parametrize("name, fits", [("AIScalerToolkit", True), ("d" * 120, False)])
def test_an_install_path_too_long_for_strata_is_flagged_at_the_start(monkeypatch, name, fits):
    """Strata's tree is deep and install_strata refuses a folder over the budget.

    It used to be a fixed path nobody could get wrong. It follows the program directory
    now, so the person installing can choose one that will not work, and should hear
    about it before anything is installed rather than at a first Strata load.

    Not through the layout fixture: pytest's tmp_path is itself long enough to fail.
    """
    from launcher import preflight

    monkeypatch.setattr(config, "components_root", lambda: Path("C:/Programs") / name)
    check = preflight.check_path_length()
    assert check.ok is fits
    # Everything but Strata still works, so this must never stop the install.
    assert not check.required


def test_the_backend_is_told_where_strata_and_its_data_go(layout, monkeypatch):
    data, program = layout
    frozen(monkeypatch, True)
    from launcher import env_file

    keys = env_file.desired(8571)
    assert keys["STRATA_INSTALL_DIR"] == str(program / "strata")
    assert keys["STRATA_DATA_DIR"] == str(data / "Strata-data")
