"""A model to start with, fetched once, so a new install does not open onto an empty list.

The download and the registration are the backend's: this asks for the model through
the same POST /config/models/download the Models page's Register button uses, and
waits on the same task. A second copy of that logic here -- huggingface_hub calls and
a registry writer -- would drift from the backend's within a release or two, which is
why provisioning is not reimplemented either.

It runs after the service is up and before the interface is shown, so the model is in
the list the first time the window opens. That costs about a minute at 100 Mbit/s and
several on a slow link, on the first start only.

Three rules:

- Once per machine. When the model is registered its label goes into state.json, and
  after that this never runs again: a user who deletes the model does not get it back
  on the next start.
- Already there counts. A registry entry for it -- someone registered it by hand, or a
  previous install did -- is enough, and the Hugging Face cache outlives uninstalling,
  so a reinstall finds the file and registers it in seconds.
- Never in the way. No network, a failed download, a slow one: the interface opens
  regardless. Waiting stops after a limit, but the download does not -- the backend
  owns it and finishes it, and the model appears in the list when it does.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from . import applog, config, state

log = applog.get("starter")


@dataclass(frozen=True)
class Starter:
    model_id: str
    label: str
    # The one GGUF file to fetch, or None for the repository's full weights.
    filename: str | None
    size: str
    engine: str


# MiniCPM5-1B in the two forms the two default engines load. Small enough to fetch on
# first start without making it a chore, and a LlamaForCausalLM underneath -- no
# trust_remote_code, supported by vLLM and llama.cpp as they are -- so it loads on
# either without per-model settings.
GGUF = Starter(
    model_id="openbmb/MiniCPM5-1B-GGUF",
    label="minicpm5-1b-gguf",
    filename="MiniCPM5-1B-Q4_K_M.gguf",
    size="656 MB",
    engine="llama_server",
)
WEIGHTS = Starter(
    model_id="openbmb/MiniCPM5-1B",
    label="minicpm5-1b",
    filename=None,
    size="1.08 GB",
    engine="vllm",
)

# How long the window waits before opening anyway. The download carries on without it.
_WAIT_LIMIT_SECONDS = 15 * 60
_POLL_SECONDS = 2.0
_HTTP_TIMEOUT = 15
_TERMINAL = ("completed", "failed", "cancelled")


def vllm_installed() -> bool:
    """Whether the service environment has vLLM in it.

    Read from the disk rather than from /health, which reports vLLM wherever the
    backend runs on Linux -- including on an Intel machine, where setup installs no
    vLLM because it only comes with the CUDA build.
    """
    if config.IS_WINDOWS:
        return False
    try:
        return any(config.venv_dir().glob("lib/python*/site-packages/vllm/__init__.py"))
    except OSError:
        return False


def choose() -> Starter:
    """vLLM's weights where vLLM is installed, the GGUF for llama.cpp everywhere else."""
    return WEIGHTS if vllm_installed() else GGUF


def _request(port: int, path: str, body: dict | None = None) -> dict | None:
    url = config.service_url(port) + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def registered(port: int, starter: Starter) -> bool:
    """Whether the registry already has this model, from anyone."""
    models = _request(port, "/config/models") or {}
    group = "llama_gguf_models" if starter.filename else "base_models"
    for entry in models.get(group, []):
        if entry.get("model_name") != starter.model_id:
            continue
        if starter.filename and not str(entry.get("model_path", "")).endswith(starter.filename):
            continue
        return True
    return False


def _describe(task: dict, starter: Starter) -> str:
    """What the page shows while it waits: the backend's own words, and the size."""
    words = (task.get("progress") or "").strip().rstrip(".")
    return f"{starter.label}, {starter.size}" + (f" — {words}" if words else "")


def fetch(port: int, on_update: Callable[[str, int], None]) -> str | None:
    """Make sure the starter model is registered. Returns a note for the user, or None.

    on_update(detail, elapsed_seconds) is called while waiting. The note is set when
    the interface opens without the model, saying why and what happens next.
    """
    if state.get("starter_model"):
        return None

    starter = choose()
    try:
        if registered(port, starter):
            log.info("%s is already registered", starter.label)
            state.set("starter_model", starter.label)
            return None
        log.info("fetching the starter model %s (%s, for %s)",
                 starter.model_id, starter.size, starter.engine)
        body = {"model_id": starter.model_id, "label": starter.label}
        if starter.filename:
            body["filename"] = starter.filename
        started = _request(port, "/config/models/download", body) or {}
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # The label is taken by an entry that did not match above -- the same model
            # under another name, or something else entirely. Either way it is the
            # user's, and asking again on every start would not change the answer.
            log.info("the label %s is already in use; leaving it alone", starter.label)
            state.set("starter_model", starter.label)
            return None
        log.warning("could not start the starter model download: %s", exc)
        return f"The starter model could not be requested ({exc}). It will be tried again next time."
    except (OSError, ValueError) as exc:
        log.warning("could not start the starter model download: %s", exc)
        return f"The starter model could not be requested ({exc}). It will be tried again next time."

    task_id = started.get("task_id")
    if not task_id:
        log.warning("the download was accepted without a task id: %r", started)
        return None

    began = time.monotonic()
    task: dict = {}
    while True:
        elapsed = int(time.monotonic() - began)
        try:
            task = _request(port, f"/config/models/download/{task_id}") or {}
        except (OSError, ValueError) as exc:
            # One missed poll is not a failed download; the next one will say.
            log.info("polling the starter model download failed once: %s", exc)
            task = {}
        on_update(_describe(task, starter), elapsed)
        if task.get("status") in _TERMINAL:
            break
        if elapsed >= _WAIT_LIMIT_SECONDS:
            log.info("stopped waiting for %s after %ss; it carries on in the background",
                     starter.label, elapsed)
            return (f"{starter.label} is still downloading. It will appear under Models "
                    "when it finishes.")
        time.sleep(_POLL_SECONDS)

    status = task.get("status")
    if status == "completed":
        log.info("starter model %s registered in %ss", starter.label, int(time.monotonic() - began))
        state.set("starter_model", starter.label)
        return None
    reason = (task.get("error") or status or "unknown").strip()
    log.warning("the starter model download ended %s: %s", status, reason)
    return (f"The starter model did not download ({reason}). It will be tried again next "
            "time, or add it yourself under Models.")
