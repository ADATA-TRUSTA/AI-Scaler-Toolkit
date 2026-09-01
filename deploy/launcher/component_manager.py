"""List optional components, show how to install them, and detect whether they are there.

Detection only. Because we do not run the installers we never learn an outcome, so the
status shown is "found on this machine" or "not found" -- never "failed", which would
be a claim we cannot support.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config

COMPONENTS_FILE = config.resource_root() / "launcher" / "components" / "components.yaml"


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    description: str
    installed: bool
    # Either a command to copy, or a URL to open. Never executed by us.
    command: str | None
    url: str | None
    note: str | None


def _load_raw() -> list[dict]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging guarantees it
        raise RuntimeError("PyYAML is required to read the component list") from exc
    data = yaml.safe_load(COMPONENTS_FILE.read_text(encoding="utf-8")) or {}
    return list(data.get("components", []))


def _expand(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw))).expanduser()


def _detect(spec: dict) -> bool:
    on_path = spec.get("path")
    if on_path and shutil.which(on_path):
        return True
    key = "windows_paths" if config.IS_WINDOWS else "linux_paths"
    return any(_expand(candidate).exists() for candidate in spec.get(key, []) or [])


def load() -> list[Component]:
    platform_key = "windows" if config.IS_WINDOWS else "linux"
    components: list[Component] = []
    for entry in _load_raw():
        show = entry.get("show", {}) or {}
        command = None
        if show.get("type") == "command":
            command = (show.get("command") or {}).get(platform_key)
        components.append(
            Component(
                id=entry.get("id", ""),
                name=entry.get("name", entry.get("id", "")),
                description=entry.get("description", ""),
                installed=_detect(entry.get("detect", {}) or {}),
                command=command,
                url=show.get("url") if show.get("type") == "link" else None,
                note=entry.get("note"),
            )
        )
    return components


if __name__ == "__main__":
    for component in load():
        mark = "installed" if component.installed else "not installed"
        print(f"{component.name} [{mark}]")
        print(f"  {component.description}")
        if component.command:
            print(f"  run: {component.command}")
        if component.url:
            print(f"  download: {component.url}")
        if component.note:
            print(f"  note: {component.note}")
