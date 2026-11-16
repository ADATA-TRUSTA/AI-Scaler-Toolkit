"""Repair clears the launcher state; the user's own choices have to come back."""

from launcher import __main__ as app
from launcher import state


def test_user_settings_are_restored_and_the_port_goes_through_set_port(monkeypatch):
    store = {}
    ports = []
    monkeypatch.setattr(state, "set", lambda key, value: store.__setitem__(key, value))
    monkeypatch.setattr(app.service_manager, "set_port", ports.append)
    app._restore_user_settings({
        "service_port": 8572, "lan_access": True, "api_key": "k",
        "autostart": False, "starter_model": None,
    })
    assert ports == [8572]
    assert store == {"lan_access": True, "api_key": "k", "autostart": False}


def test_default_port_is_left_alone(monkeypatch):
    ports = []
    monkeypatch.setattr(state, "set", lambda key, value: None)
    monkeypatch.setattr(app.service_manager, "set_port", ports.append)
    app._restore_user_settings({"service_port": app.config.DEFAULT_SERVICE_PORT})
    assert ports == []
