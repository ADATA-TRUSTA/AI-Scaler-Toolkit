"""Strata: a Strata this backend installed earlier is updated from vendor/ when the build ships a newer one."""

import os
from pathlib import Path

import pytest

from service.inference.engines import strata_engine as se

pytestmark = pytest.mark.inference

PLATFORM = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
EXT = "zip" if os.name == "nt" else "tar.gz"
GYRO = "C:/models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf"


def _layout(tmp_path, monkeypatch, installed="0.2.7-sleep.3", shipped="0.2.11-sleep.2"):
    """An install made by this backend (everything inside STRATA_INSTALL_DIR) and a package in vendor/."""
    install = tmp_path / "strata"
    (install / "engine").mkdir(parents=True)
    (install / ".venv" / "bin").mkdir(parents=True)
    if installed is not None:
        (install / "VERSION").write_text(f"{installed}\n", encoding="utf-8")
    monkeypatch.setattr(se, "STRATA_INSTALL_DIR", str(install))
    monkeypatch.setattr(se, "STRATA_HOME", str(install))
    monkeypatch.setattr(se, "STRATA_PYTHON", str(install / ".venv" / "bin" / "python"))
    monkeypatch.setattr(se, "STRATA_ENGINE_BINARY", str(install / "engine" / "strata"))
    monkeypatch.setattr(se, "strata_installable", lambda: True)
    package = tmp_path / "vendor" / f"strata-{shipped}-{PLATFORM}-sycl.{EXT}"
    monkeypatch.setattr(se, "strata_package", lambda: package)
    return install


def test_versions_compare_as_numbers_not_as_text():
    assert "0.2.7-sleep.3" > "0.2.11-sleep.2"  # the text order that picked the old package
    assert se._version_key("0.2.7-sleep.3") < se._version_key("0.2.11-sleep.2")
    assert se._version_key("0.2.11-sleep.1") < se._version_key("0.2.11-sleep.2")
    assert se._version_key("0.2.11-sleep.2") == se._version_key("0.2.11-sleep.2\n")
    assert se._version_key("0.2.12") > se._version_key("0.2.11-sleep.9")
    assert se._version_key("unknown") is None


@pytest.mark.parametrize(
    "name, version",
    [
        ("strata-sleep-0.2.7-sleep.3-windows-x86_64.zip", "0.2.7-sleep.3"),
        ("strata-sleep-0.2.7-sleep.3-windows-x86_64-sycl.zip", "0.2.7-sleep.3"),
        ("strata-0.2.11-sleep.2-windows-x86_64-sycl.zip", "0.2.11-sleep.2"),
        ("strata-sleep-0.2.11-sleep.2-linux-x86_64.tar.gz", "0.2.11-sleep.2"),
        ("strata-0.2.11-sleep.2-linux-x86_64-sycl.tar.gz", "0.2.11-sleep.2"),
        ("something-else.zip", None),
    ],
)
def test_package_version_comes_from_the_file_name(name, version):
    assert se.strata_package_version(Path(name)) == version


@pytest.mark.parametrize(
    "backend, newer, older",
    [
        (
            "cuda",
            f"strata-sleep-0.2.11-sleep.2-{PLATFORM}.{EXT}",
            f"strata-sleep-0.2.7-sleep.3-{PLATFORM}.{EXT}",
        ),
        (
            "sycl",
            f"strata-0.2.11-sleep.2-{PLATFORM}-sycl.{EXT}",
            f"strata-sleep-0.2.7-sleep.3-{PLATFORM}-sycl.{EXT}",
        ),
    ],
)
def test_the_newest_package_is_taken_when_an_older_one_was_left_in_vendor(
    tmp_path, monkeypatch, backend, newer, older
):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / older).write_bytes(b"")
    (vendor / newer).write_bytes(b"")
    monkeypatch.setattr(se, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(se, "gpu_backend", lambda: backend)
    assert se.strata_package().name == newer


def test_an_older_install_of_this_backend_is_to_be_updated(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    reason = se.strata_update_needed()
    assert reason is not None
    assert "0.2.7-sleep.3" in reason and "0.2.11-sleep.2" in reason


@pytest.mark.parametrize("installed", ["0.2.11-sleep.2", "0.2.12-sleep.1", "0.3.0"])
def test_an_install_that_is_the_same_or_newer_is_left_alone(tmp_path, monkeypatch, installed):
    _layout(tmp_path, monkeypatch, installed=installed)
    assert se.strata_update_needed() is None


def test_an_engine_pinned_outside_the_install_is_left_alone(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "STRATA_ENGINE_BINARY", str(tmp_path / "dev" / "strata"))
    assert se.strata_update_needed() is None


def test_an_interpreter_outside_the_install_is_left_alone(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "STRATA_PYTHON", str(tmp_path / "dev" / "python"))
    assert se.strata_update_needed() is None


def test_a_hand_made_install_elsewhere_is_left_alone(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    other = tmp_path / "ssw" / "s5"
    (other / "engine").mkdir(parents=True)
    (other / "VERSION").write_text("0.2.11-sleep.1\n", encoding="utf-8")
    monkeypatch.setattr(se, "STRATA_HOME", str(other))
    monkeypatch.setattr(se, "STRATA_PYTHON", str(other / "python"))
    monkeypatch.setattr(se, "STRATA_ENGINE_BINARY", str(other / "engine" / "strata"))
    assert se.strata_update_needed() is None


def test_nothing_installed_is_the_first_install_not_an_update(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "STRATA_HOME", "")
    assert se.strata_update_needed() is None


@pytest.mark.parametrize("installed", [None, "", "garbage"])
def test_an_install_whose_version_cannot_be_read_is_left_alone(tmp_path, monkeypatch, installed):
    _layout(tmp_path, monkeypatch, installed=installed)
    assert se.strata_update_needed() is None


def test_no_update_when_this_host_cannot_install_a_package(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "strata_installable", lambda: False)
    assert se.strata_update_needed() is None


def _prepare(tmp_path, monkeypatch):
    """Run prepare_model with the steps replaced by recorders; return the commands it ran."""
    ran = []
    script = tmp_path / "install_strata.sh"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(se, "_install_script", lambda: script)
    monkeypatch.setattr(se, "_run_setup_step", lambda cmd, cwd, env: ran.append((cmd, env)))
    monkeypatch.setattr(se, "_adopt_recorded_install", lambda: None)
    monkeypatch.setattr(se, "strata_setup_problem", lambda: None)
    monkeypatch.setattr(se, "find_run_config", lambda path: tmp_path / "strata-gyro.json")
    se.prepare_model(GYRO)
    return script, ran


def test_a_load_runs_the_install_script_over_an_older_install(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    script, ran = _prepare(tmp_path, monkeypatch)
    assert len(ran) == 1
    cmd, env = ran[0]
    assert str(script) in cmd
    assert "setup.py" not in cmd
    assert env["TRUSTA_STRATA_PACKAGE"].endswith(f"-{PLATFORM}-sycl.{EXT}")
    assert env["TRUSTA_STRATA_DIR"] == se.STRATA_INSTALL_DIR


def test_a_load_only_runs_setup_when_the_install_is_current(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch, installed="0.2.11-sleep.2")
    script, ran = _prepare(tmp_path, monkeypatch)
    assert len(ran) == 1
    cmd, _env = ran[0]
    assert "setup.py" in cmd
    assert str(script) not in cmd


def test_a_venv_interpreter_that_is_a_link_to_the_managed_python_still_counts_as_inside(
    tmp_path, monkeypatch
):
    """On Linux the venv's python links to uv's interpreter, outside the install; the install is still this backend's."""
    install = _layout(tmp_path, monkeypatch)
    real_python = tmp_path / "uv" / "python" / "cpython-3.12" / "bin" / "python3.12"
    real_python.parent.mkdir(parents=True)
    real_python.write_bytes(b"")
    link = install / ".venv" / "bin" / "python"
    try:
        os.symlink(real_python, link)
    except (OSError, NotImplementedError):
        pytest.skip("this host cannot create symbolic links")
    monkeypatch.setattr(se, "STRATA_PYTHON", str(link))
    assert se.strata_update_needed() is not None


def _engine_with_a_run_config(tmp_path, monkeypatch, update, config_text):
    """A StrataEngine whose load finds an install and a run config; prepare_model is a sentinel."""
    from multiprocessing import Event, Queue

    from service.config_models import InferenceConfig

    config_file = tmp_path / "strata-gyro-gyro-s.json"
    config_file.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(se, "strata_setup_problem", lambda: None)
    monkeypatch.setattr(se, "strata_update_needed", lambda: update)
    monkeypatch.setattr(se, "find_run_config", lambda path: config_file)

    def prepare(path):
        raise RuntimeError("PREPARE_MODEL_RAN")

    monkeypatch.setattr(se, "prepare_model", prepare)
    engine = se.StrataEngine(Queue(), Queue(), Event(), Event())
    config = InferenceConfig(model_name="gyro-s", model_path=GYRO, engine="strata")
    return engine, config


def test_a_load_prepares_the_model_when_an_update_is_due_even_with_a_run_config(
    tmp_path, monkeypatch
):
    engine, config = _engine_with_a_run_config(
        tmp_path, monkeypatch, "the installed one is older", "{}"
    )
    with pytest.raises(RuntimeError, match="PREPARE_MODEL_RAN"):
        engine.load_model(config)


def test_a_load_uses_the_existing_run_config_when_no_update_is_due(tmp_path, monkeypatch):
    engine, config = _engine_with_a_run_config(tmp_path, monkeypatch, None, "not json")
    with pytest.raises(Exception) as raised:  # whatever comes after the run config is read
        engine.load_model(config)
    assert "PREPARE_MODEL_RAN" not in str(raised.value)
