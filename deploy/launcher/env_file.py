"""Write the backend's .env.

This is not an optimisation. The backend loads the first of `.env` and `.env.example`
that exists, and `.env.example` ships with the payload -- so without a real `.env` the
service does not fall back to code defaults, it silently adopts the example values:
port 8000, and an HF_HOME relative to the project, which puts tens of gigabytes of
models inside the install directory.

Keys are filled in, never rewritten. setup_env appends to the same file (it records
LLAMA_SERVER_BINARY there), and the user is free to edit it, so a whole-file rewrite
would throw away both.
"""

from __future__ import annotations

from pathlib import Path

from . import applog, config

log = applog.get("env")


def env_path() -> Path:
    """Fixed at the backend project root.

    The service locates it from its own file position, not from the working directory,
    so the launcher does not have to control where it is started from.
    """
    return config.backend_dir() / ".env"


def read_keys(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def desired(port: int) -> dict[str, str]:
    """The keys the launcher owns.

    LLAMA_SERVER_BINARY is absent by design: setup_env resolves and writes it, and a
    guess here would either duplicate or contradict it.
    """
    return {
        "SERVICE_HOST": config.bind_host(),
        "SERVICE_PORT": str(port),
        "HF_HOME": str(config.hf_home()),
        "LOG_DIR": str(config.log_dir()),
    }


def ensure(port: int) -> list[str]:
    """Add any missing launcher-owned keys. Returns the keys actually written."""
    path = env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_keys(path)

    added = {k: v for k, v in desired(port).items() if k not in existing}
    log.info("%s: %d existing keys, adding %s", path, len(existing),
             ", ".join(sorted(added)) if added else "nothing")
    if not added:
        return []

    lines = [] if not path.is_file() else path.read_text(encoding="utf-8").splitlines()
    if lines and lines[-1].strip():
        lines.append("")
    lines.append("# Written by the desktop launcher on first run.")
    lines.extend(f"{key}={value}" for key, value in added.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sorted(added)


def set_port(port: int) -> None:
    """Change SERVICE_PORT in place, leaving everything else untouched."""
    set_key("SERVICE_PORT", str(port))


def set_host(host: str) -> None:
    """Change the address the service binds to, leaving everything else untouched."""
    set_key("SERVICE_HOST", host)


def set_key(key: str, value: str) -> None:
    """Replace one key in place. The file is the user's as much as ours -- setup_env
    appends to it and so may they -- so nothing else in it is touched."""
    path = env_path()
    if not path.is_file():
        ensure(state_port())
        if not path.is_file():
            return

    out: list[str] = []
    replaced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def state_port() -> int:
    from . import state

    return state.service_port()
