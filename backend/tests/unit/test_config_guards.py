"""
Two settings-layer checks that used to fail without a word.

Boolean environment variables were read two ways in settings.py, so LOG_TO_FILE=on
turned file logging off while VLLM_STARTUP_SWEEP=on turned the sweep on. And the
frontend's root-level files are served at "/" from a hand-kept list that nothing
compared with the build.
"""

import importlib
import logging
import re
from pathlib import Path

import pytest

import service.app as app_module
import service.settings as settings_module
from service.settings import _parse_bool_env

pytestmark = pytest.mark.inference

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", " ON ", "True"])
def test_every_true_spelling_is_true(raw):
    assert _parse_bool_env(raw, False, "X") is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "", " Off "])
def test_every_false_spelling_is_false(raw):
    assert _parse_bool_env(raw, True, "X") is False


def test_an_unset_variable_keeps_the_default():
    assert _parse_bool_env(None, True, "X") is True
    assert _parse_bool_env(None, False, "X") is False


def test_an_unrecognised_value_keeps_the_default_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="service.settings"):
        assert _parse_bool_env("yes please", True, "LOG_TO_FILE") is True

    assert "LOG_TO_FILE='yes please'" in caplog.text


def test_log_to_file_on_turns_file_logging_on(monkeypatch):
    monkeypatch.setenv("LOG_TO_FILE", "on")
    try:
        assert importlib.reload(settings_module).LOG_TO_FILE is True
    finally:
        monkeypatch.undo()
        importlib.reload(settings_module)


def test_settings_reads_every_boolean_through_one_parser():
    """The second, narrower form is what made "on" work for some settings only."""
    source = (REPO_ROOT / "service" / "settings.py").read_text(encoding="utf-8")

    assert not re.search(r'\.lower\(\)\s*in\s*\(\s*"true"', source)


def _dist(tmp_path: Path, *names: str) -> Path:
    (tmp_path / "assets").mkdir()
    for name in names:
        (tmp_path / name).write_text("x", encoding="utf-8")
    return tmp_path


def test_a_root_file_the_list_does_not_serve_is_reported(tmp_path):
    dist = _dist(tmp_path, "index.html", "config.json", "Trusta-16.ico", "site.webmanifest")

    unserved = app_module._unserved_frontend_root_files(dist, ["config.json", "Trusta-16.ico"])

    assert unserved == ["site.webmanifest"]


def test_a_build_the_list_covers_reports_nothing(tmp_path):
    dist = _dist(tmp_path, "index.html", "config.json", "Adata.ico", ".DS_Store")

    assert app_module._unserved_frontend_root_files(dist, ["config.json", "Adata.ico"]) == []
