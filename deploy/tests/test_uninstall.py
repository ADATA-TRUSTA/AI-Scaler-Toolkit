"""launcher.uninstall: the Linux side of removing what was installed.

Windows gets this from installer.iss. An AppImage has no uninstaller, so this module
is it, and the thing worth pinning down is what it does NOT delete: the data Strata
prepared and the model cache both cost a long rebuild, so they go only when the user
says so, and a run with nothing to answer on keeps them.
"""

import pytest

from launcher import config, uninstall


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A Linux-shaped install with everything the uninstaller touches."""
    home = tmp_path / "home"
    data = home / ".local" / "share" / config.APP_ID
    settings = home / ".config" / config.APP_ID
    models = home / ".cache" / "huggingface"
    for path in (data / "service" / ".venv", data / "components" / "comfyui",
                 data / "Strata-data", settings, models):
        path.mkdir(parents=True)
    (settings / "state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(config, "IS_WINDOWS", False)
    monkeypatch.setattr(config, "data_dir", lambda: data)
    monkeypatch.setattr(config, "config_dir", lambda: settings)
    monkeypatch.setattr(config, "hf_home", lambda: models)
    monkeypatch.setattr(uninstall, "_stop_the_service", lambda: None)
    monkeypatch.setattr(uninstall.autostart, "disable", lambda: None)
    monkeypatch.setattr(uninstall.shortcuts, "remove", lambda: None)
    return data, settings, models


def test_yes_removes_the_components_and_keeps_what_is_expensive(installed, capsys):
    data, settings, models = installed
    assert uninstall.run(assume_yes=True) == 0
    assert not (data / "service").exists()
    assert not (data / "components").exists()
    assert not settings.exists()
    assert (data / uninstall.STRATA_DATA).is_dir()
    assert models.is_dir()
    out = capsys.readouterr().out
    assert str(data / uninstall.STRATA_DATA) in out
    assert str(models) in out


def test_with_no_terminal_to_answer_on_nothing_expensive_is_removed(installed, monkeypatch):
    data, _settings, models = installed

    def no_terminal(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    assert uninstall.run() == 0
    assert (data / uninstall.STRATA_DATA).is_dir()
    assert models.is_dir()
    assert not (data / "service").exists()


def test_answering_yes_twice_removes_both(installed, monkeypatch):
    data, _settings, models = installed
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert uninstall.run() == 0
    assert not data.exists()
    assert not models.exists()


def test_on_windows_it_points_at_the_installer_instead(installed, monkeypatch, capsys):
    data, _settings, _models = installed
    monkeypatch.setattr(config, "IS_WINDOWS", True)
    assert uninstall.run(assume_yes=True) == 1
    assert data.is_dir()  # nothing touched
    assert "Start menu" in capsys.readouterr().out


def _seed_comfy(data):
    comfy = data / "components" / "comfyui" / "ComfyUI"
    (comfy / "models").mkdir(parents=True)
    (comfy / "output").mkdir(parents=True)
    (comfy / "models" / "m.safetensors").write_text("weights", encoding="utf-8")
    (comfy / "output" / "a.png").write_text("img", encoding="utf-8")


def test_comfyui_models_and_output_survive_by_default(installed):
    data, _settings, _models = installed
    _seed_comfy(data)
    assert uninstall.run(assume_yes=True) == 0
    kept = data.parent / f"{config.APP_ID}-kept"
    assert (kept / "ComfyUI-models" / "m.safetensors").is_file()
    assert (kept / "ComfyUI-output" / "a.png").is_file()
    assert not (data / "components").exists()


def test_comfyui_data_is_removed_only_when_the_user_says_so(installed, monkeypatch):
    data, _settings, _models = installed
    _seed_comfy(data)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert uninstall.run() == 0
    assert not (data.parent / f"{config.APP_ID}-kept").exists()


def test_a_foreign_listener_is_not_stopped(monkeypatch, capsys):
    from launcher import service_manager

    killed = []
    monkeypatch.setattr(service_manager, "listening_pid", lambda port: 4242)
    monkeypatch.setattr(service_manager, "probe", lambda port: "foreign")
    monkeypatch.setattr(service_manager, "terminate_pid", lambda pid: killed.append(pid))
    monkeypatch.setattr(uninstall.state, "service_port", lambda: 8571)
    uninstall._stop_the_service()
    assert killed == []
