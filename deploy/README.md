# deploy/

Packaging and desktop wrapper. Nothing in here reimplements backend behaviour.

```
deploy/
  launcher/          the tray application
  package/           payload assembly and its post-build checks
  windows/           Inno Setup script
  linux/             AppImage recipe
  dev_reset.py       clean slate / re-provision / launch (also used by CI acceptance)
```

## Running from a source checkout

```
cd deploy
uv sync
uv run python -m launcher
```

`config.payload_dir()` walks up to the repository root when `deploy/backend` does not
exist, so a checkout with the assembled layout beside it works without configuration.

## What the launcher does and does not own

| owned here | owned elsewhere |
|---|---|
| service lifecycle, port, adoption of a leftover process | training and inference settings (the WebUI) |
| `.env` keys the launcher writes (host, port, `HF_HOME`, `LOG_DIR`) | `LLAMA_SERVER_BINARY` — `setup_env` resolves and records it |
| prerequisites (WebView2, VC++, git) | the Intel GPU check — `setup_env` proves it by computing |
| autostart, updates, optional-component display | accelerator choice, `uv sync` extras, llama install |

`provision.py` calls `backend/scripts/{windows,linux}/setup_env.{ps1,sh}` and streams
its output. It does not run `uv sync` or install llama itself: a second copy of that
sequence would drift from the backend's within a release or two.

## Two details that look wrong until you hit them

**The virtual environment is not inside the payload.** `UV_PROJECT_ENVIRONMENT` moves
it to the per-user data directory, so a reinstall does not re-download several GB and
the program directory does not grow after setup. The side effect is that the backend's
own `run_service` scripts, which look for `.venv` in the project root, do not work in
an installed build — they fail loudly with "environment not found", which is the right
way for that to go wrong. Use the tray, or `dev_reset.py --gui`.

**A `.env` that does not exist is worse than an empty one.** The service falls back to
`.env.example`, which ships with the payload — so a missing `.env` does not mean code
defaults, it means the example's values: port 8000, and an `HF_HOME` relative to the
project, which puts tens of gigabytes of models under the install directory. The
launcher writes its keys on first run and only ever fills in ones that are absent,
because `setup_env` appends to the same file and the user may have edited it.

## Optional third-party components

Removed. The settings page used to list Hermes and ComfyUI with the command to install
them, as the original specification asked. Nothing in this application integrates with
either, so it amounted to showing a user two unrelated products in the settings of a
third.

The reasoning behind that section still holds for anything that does get integrated:
show the command, never run someone else's installer. It is why remedy.py is limited
to the three prerequisites setup itself cannot proceed without.
