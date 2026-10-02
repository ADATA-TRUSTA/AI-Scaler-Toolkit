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

That makes Quit consequential, and three guards were tried around it. Two are gone.

Quitting asked first whenever `/health` reported a model loaded or a training run
active. A model is loaded almost all of the time this is running, so the question was
not the rare warning it was designed as -- it stood between the user and the one
button whose meaning is not in doubt. Quit means quit.

Closing the window announced where it had gone, and Settings carried a Quit button,
both as insurance against a tray icon the user might not find -- on Windows 11 a new
one starts in the overflow behind the chevron with no supported way to promote it.
Both were the wrong answer: hiding to the tray is what every application with a tray
icon does without explaining itself, and a second way to quit does not fix a first one
being unreliable.

What remains is one question, and it is not "are you sure": quitting while a component
is installing warns that stopping now leaves something broken on disk. That installer
is a child of this process reading from a pipe we own, so this process going away
aborts it mid-write. Closing the settings window is safe -- the install runs here, not
there -- so quitting is what asks.

Checking for updates moved to Settings. It belongs where someone goes to look for it.

## A login is not a request to install

The autostart entry passes `--autostart`, and a launch carrying it exits immediately if
setup has not been completed.

Without that flag nothing separated a login from a person opening the application, and
bootstrap begins setup whenever it finds itself unprovisioned -- so turning the
computer on could start an eight-minute install with a window appearing and gigabytes
downloading, for a decision nobody made. Autostart is on by default, which is what
makes this worse rather than better: the user did not choose the thing that chose to
install.

Once setup has been done, a login starts the service as before. That is what autostart
is for; installing is not.

`state["setup_started"]` is written when setup begins and cleared when it ends. Finding
it still set on a later start means the last attempt was interrupted, and the window
says "Resuming setup" instead of looking like a fresh install of something the user
believed they already had. Nothing resumes a download -- `setup_env` installs only what
is missing, so finished work goes past in seconds and the llama archive starts over.
The flag exists to be honest about that, not to hide it.

## One brand in the interface, another on the icon

`BRAND_BY_PLATFORM` gives the Windows payload the ADATA frontend; `layout.ICON_BRAND`
is `trusta` and is not per-platform. So a Windows installation reads ADATA in the
interface and wears the TRUSTA mark on its executable, its tray icon, its shortcuts
and its setup screen -- all four come from the same source, so they agree with each
other and disagree with the frontend.

Left as it is deliberately, and not because the build would stop us -- it would not.
`SOURCES["adata"]` falls back to the 348x348 `adata-icon.png` in the root `assets`
directory, which is large enough that every entry down to 16 is a downscale: the icon
builds with all nine sizes, without the upscale warning, and the payload assertion
passes. What is missing is not a file but a drawing. That image serves README headers,
and nobody has decided what it should look like at 16 pixels, which is where the
taskbar and the tray draw it.

If ADATA icons arrive, put `adata-icon-1024.png` in `deploy/assets` and make
`ICON_BRAND` follow the platform the way the frontend package already does.

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
- Both are per-user by design — Hermes into `%LOCALAPPDATA%\hermes`, ComfyUI into this
  application's own data directory — so neither raises a UAC prompt. That matters:
  this application installs without elevation and would look broken if a component
  demanded it.
- Hermes's installer is not ours. On failure the page shows the installer's own output
  and keeps the vendor's page one click away. `_explain_failure` translates the two
  failures we know the shape of; everything else is reported as what it is.

### ComfyUI: no desktop shell, and the user picks the card

ComfyUI used to arrive as Comfy Desktop. It no longer does, for two reasons:

- **Desktop owns the process.** It is started from Desktop's own window with Desktop's
  arguments, and the image and video skills that sit beside this application need to
  start it themselves: the guarantee they rely on, `--reserve-vram` sized at launch to
  whatever the hosted LLM holds, only exists if they are the ones launching it. With
  Desktop already on port 8188 they connected to it and inherited no protection.
- **Desktop picks the PyTorch build by a fixed vendor order** — NVIDIA, then AMD, then
  Intel — so a machine with an NVIDIA card and an Intel GPU had no say in which ran it.

`launcher/comfyui.py` keeps ComfyUI itself and nothing around it, under
`<data dir>/components/comfyui/`:

- `ComfyUI/` — the source of the latest release, from its archive. Not a git clone:
  nothing else here needs git, and one optional component should not make it a
  prerequisite. Models, custom nodes and outputs live here, shared by every build.
- `envs/cuda`, `envs/xpu`, `envs/cpu` — one environment per PyTorch build, created the
  first time a device of that kind is started rather than all up front: each is
  gigabytes, and most machines only ever use one. They pin the backend's own torch
  versions, so uv's cache can serve what the backend already downloaded.

An environment is marked usable only after it has multiplied two matrices on its
device, under a timeout. torch reporting the device as available is not evidence —
an Intel runtime that does not match its driver passes that and fails, or hangs, on
the first kernel.

The card lists every card the machine has — NVIDIA from `nvidia-smi`, so its index is
the `--cuda-device` ComfyUI is given (launches set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to
make that true), everything else from the system's adapter list — and the CPU last,
marked as not recommended: it is the way out when no card works, not an option anyone
should pick for real work. AMD is listed and disabled; there is no ROCm build here.

Switching between two cards of the same kind is an argument; switching kinds is
another environment. Either way it is a restart, so Start refuses while a queue is
running unless the user confirms. ComfyUI runs as this application's child on
`127.0.0.1:8188` and is stopped when the application quits.

Detection is looser than the install path on purpose: it finds copies this application
never installed, because telling someone to install what they already have is its own
kind of wrong.

### Hermes on Linux: the desktop application, and Chromium's sandbox

The windowed Hermes is opt-in upstream on both platforms. Windows asked for it with
`-IncludeDesktop`; Linux did not, so it got the command line tool and nothing to open.
Linux now passes the same switch, `--include-desktop`, and success is judged by what
is on disk, as on Windows.

Open runs `hermes desktop --skip-build` rather than the built binary, because the
binary alone cannot be started safely. Ubuntu 23.10 and later restrict unprivileged
user namespaces, so Electron needs either a setuid-root `chrome-sandbox` or
`--no-sandbox`, and aborts with neither. Hermes's own launcher decides: the namespace
sandbox where the system allows it, the setuid helper through sudo where it can ask,
`--no-sandbox` where it cannot. That is the vendor's call. Launching the binary
ourselves would take it over and get it wrong on exactly the machines that matter.

From this application sudo can never ask — there is no terminal — so on a restricted
host the window opens without Chromium's sandbox, and the card says so and how to turn
it on: run `hermes desktop` once in a terminal, with the user's own password. We do not
ask for that password on their behalf. Doing it for them (a pkexec prompt installing
an AppArmor profile, as some AI desktop installers do) was considered and declined: it
would be a second place this application elevates, for someone else's program, leaving
a root-owned file that uninstalling cannot remove. Comfy Desktop, which this used to
install, ships `--no-sandbox` unconditionally, so what this does is at worst that and
usually better.

Two consequences look wrong until you know them:

- **The install prints an error and still succeeds.** The installer's last Linux step
  is that sudo, and it fails the desktop stage when sudo cannot ask — after the build
  is already on disk. The page adds a note beside the error saying so.
- **The card needs the launcher's output to know.** Open writes it to
  `logs/hermes-desktop.log`, with `PYTHONUNBUFFERED=1`: `hermes desktop` waits on
  Electron, and a buffered launcher writes nothing until the window is closed.

Hermes writes its own `hermes.desktop` the first time it is opened, so this
application writes no menu entry for it once the desktop build exists — two "Hermes"
entries that do different things would be worse than one.

`remedy.py` is a different thing and stays narrow. It installs WebView2 — the one
prerequisite *setup itself* cannot proceed without. The line is whose problem it is,
not whether we are able to install it. (It once installed the Visual C++ runtime and
git as well; both went, the first because the payload now carries it and the second
because nothing needs git any more, and each was a UAC prompt.)
