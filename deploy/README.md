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

## The tray menu, and why it stopped being a service console

Four items: open the window, open the interface in a browser, Settings, Quit. There is
no backend status line and no Start, Stop or Restart. The specification asked for
those; they were built, and then removed.

Each one was a way to leave the application in a state nobody wants. Worse, each was
wrong in a way only visible by clicking it with a mouse:

- the status line froze, because pystray re-reads a callable menu item only when
  `update_menu()` is called, so it went on saying "Backend: running" after the service
  had stopped;
- "Start backend" was offered while it was already running, where it adopts the
  running service and does nothing;
- "Stop backend" silently failed whenever the launcher had adopted a service rather
  than spawned it, because stopping worked by process handle and there was none.

The rule now is all or nothing. Opening the window makes sure the backend is up.
Quit takes everything down. Nothing sits between those, so nothing needs managing.

Closing the window still hides to the tray rather than quitting, and that is
deliberate: the service holds loaded models and serves them to *other applications*
over its API, not only to this window. Closing a window here is not the same as
finishing with the service.

That makes Quit consequential, so two things guard it. Quitting asks first when
`/health` reports a model loaded or a training run active -- only then, because a
confirmation shown every time is one nobody reads. And the first time the window is
closed, the application says where it went: on Windows 11 a new tray icon starts in
the overflow behind the chevron, with no supported way to promote it, so an
application that hides there without saying so has effectively vanished. Settings
carries a Quit button for the same reason -- nothing essential should depend on
finding an icon Windows chose to hide.

Checking for updates moved to Settings. It belongs where someone goes to look for it.

## Optional third-party components

The Components tab in settings installs Hermes and ComfyUI. `launcher/components.py`
owns the catalogue, the detection and the install; the page polls a single in-flight
job for its log.

This has been decided twice, in opposite directions, and both are worth knowing about.

The original specification said to *show* the install command and never run it. That
bought three real things: no output panel, no elevation to handle, and nothing to
break when a vendor changed their installer. Then the section was removed outright,
on the argument that nothing here integrates with either tool. Neither version
installed anything, which is the only outcome a user was ever after — "here is a
command, go and paste it somewhere" is a to-do item dressed as a feature.

So they are installed, and the three costs are paid deliberately:

- Output is streamed to the page, because an install that takes minutes and says
  nothing is indistinguishable from one that has hung.
- Both are per-user by design — Hermes into `%LOCALAPPDATA%\hermes`, ComfyUI through
  electron-builder's per-user NSIS target — so neither raises a UAC prompt. That
  matters: this application installs without elevation and would look broken if a
  component demanded it.
- Neither installer is ours. On failure the page shows the installer's own output and
  keeps the vendor's page one click away. `_explain_failure` translates the two
  failures we know the shape of; everything else is reported as what it is.

Detection is looser than the install path on purpose: it finds copies this application
never installed, because telling someone to install what they already have is its own
kind of wrong.

`remedy.py` is a different thing and stays narrow. It installs WebView2, the Visual
C++ runtime and git — the prerequisites *setup itself* cannot proceed without. The
line is whose problem it is, not whether we are able to install it.
