"""Desktop wrapper around the existing service: tray, window, and first-run setup.

This package adds packaging and lifecycle only. Training and inference logic stays in
the backend, and the UI stays in the existing WebUI -- the tray owns the environment
(service, ports, components), the WebUI owns the work.
"""

__all__ = ["config", "state", "provision", "preflight", "service_manager"]
