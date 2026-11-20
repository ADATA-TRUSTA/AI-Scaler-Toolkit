"""The service is started with the uv the payload carries on its PATH."""

import os

from launcher import config
from launcher.service_manager import with_bundled_uv


def _payload_with_uv(tmp_path, monkeypatch):
    uv = tmp_path / "uv" / ("uv.exe" if config.IS_WINDOWS else "uv")
    uv.parent.mkdir()
    uv.write_bytes(b"")
    monkeypatch.setattr(config, "uv_binary", lambda: uv)
    return uv


def test_the_payloads_uv_comes_first_on_path_and_is_named(tmp_path, monkeypatch):
    uv = _payload_with_uv(tmp_path, monkeypatch)
    environment = {"PATH": os.pathsep.join(["first", "second"])}
    assert with_bundled_uv(environment) is True
    assert environment["TRUSTA_UV"] == str(uv)
    assert environment["PATH"].split(os.pathsep) == [str(uv.parent), "first", "second"]


def test_a_path_that_was_never_set_still_gets_the_directory(tmp_path, monkeypatch):
    uv = _payload_with_uv(tmp_path, monkeypatch)
    environment: dict[str, str] = {}
    assert with_bundled_uv(environment) is True
    assert environment["PATH"].split(os.pathsep)[0] == str(uv.parent)


def test_a_source_checkout_without_a_payload_is_left_alone(monkeypatch):
    monkeypatch.setattr(config, "uv_binary", lambda: None)
    environment = {"PATH": "keep"}
    assert with_bundled_uv(environment) is False
    assert environment == {"PATH": "keep"}
