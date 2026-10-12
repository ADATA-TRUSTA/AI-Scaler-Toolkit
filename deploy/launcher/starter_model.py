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
    # What the download fetches, from the Hub's own file listing. Used until the
    # backend's task reports its total, and for the log line before it does.
    total_bytes: int
    engine: str
    # What the setup window calls it, which has room for a name and not a repository.
    display: str = "MiniCPM5-1B"


# MiniCPM5-1B in the two forms the two default engines load. Small enough to fetch on
# first start without making it a chore, and a LlamaForCausalLM underneath -- no
# trust_remote_code, supported by vLLM and llama.cpp as they are -- so it loads on
# either without per-model settings.
GGUF = Starter(
    model_id="openbmb/MiniCPM5-1B-GGUF",
    label="minicpm5-1b-gguf",
    filename="MiniCPM5-1B-Q4_K_M.gguf",
    total_bytes=688_065_920,
    engine="llama_server",
)
# The whole repository, since vLLM needs the tokenizer and config beside the weights:
# 2.17 GB, nearly all of it the one BF16 safetensors file. It was written down as 1.08
# GB from a summary of the Hub's listing rather than the listing itself, and a first
# install on the VM took three times as long as that figure said it would.
WEIGHTS = Starter(
    model_id="openbmb/MiniCPM5-1B",
    label="minicpm5-1b",
    filename=None,
    total_bytes=2_171_355_254,
    engine="vllm",
)

# How long the window waits before opening anyway. The download carries on without it.
_WAIT_LIMIT_SECONDS = 15 * 60
_POLL_SECONDS = 2.0
_HTTP_TIMEOUT = 15
_TERMINAL = ("completed", "failed", "cancelled")


def vllm_installed() -> bool:
    """Whether the service environment has vLLM in it.

    Read from the disk rather than from /health, which once reported vLLM wherever the
    backend ran on Linux, whether or not setup had installed it.
    """
    if config.IS_WINDOWS:
        return False
    try:
        return any(config.venv_dir().glob("lib/python*/site-packages/vllm/__init__.py"))
    except OSError:
        return False


def choose() -> Starter:
    """vLLM's weights where vLLM is installed on NVIDIA, the GGUF for llama.cpp elsewhere.

    Intel gets vLLM too now (setup_env's vllm-xpu extra), but not this default: a laptop
    iGPU shares system memory with everything else, and llama.cpp's Vulkan build is the
    one measured there (61 tok/s on an Arc B390). Three times the download for an
    engine not shown to be faster is not a first-run default; vLLM stays one click away.
    """
    if vllm_installed() and state.get("accelerator") == "cuda":
        return WEIGHTS
    return GGUF


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


def _gb(value: int) -> str:
    return f"{value / 1e9:.1f} GB"


def _downloaded(starter: Starter) -> int:
    """Bytes of this repository in the Hugging Face cache, finished or not.

    The backend's task says which file it is on and how much there is in all, but not
    how much has arrived -- and one file is nearly all of it, so "file 7 of 11" sat
    unchanged for the whole download. The cache is where the bytes land, including
    the .incomplete file of the transfer in flight, and it is at a path that follows
    from HF_HOME, which this launcher sets. Blobs only when there are blobs: the
    snapshot directory holds links to them, and counting both would count twice.
    """
    repo = config.hf_home() / "hub" / ("models--" + starter.model_id.replace("/", "--"))
    root = repo / "blobs" if (repo / "blobs").is_dir() else repo
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    except OSError:
        pass
    return total


def _progress(task: dict, starter: Starter) -> tuple[str, float | None]:
    """What the setup window shows while it waits: a short line and a fraction.

    Short on purpose. The window is narrow, and the backend's own description names
    the file it is fetching -- model-00000-of-00001.safetensors -- which is long and
    tells a user nothing the size does not.
    """
    total = int(task.get("total_bytes") or starter.total_bytes)
    done = min(_downloaded(starter), total)
    if not done:
        return f"{starter.display} · {_gb(total)}", None
    return f"{starter.display} · {_gb(done)} of {_gb(total)}", done / total


def fetch(port: int, on_update: Callable[[str, float | None, int], None]) -> str | None:
    """Make sure the starter model is registered. Returns a note for the user, or None.

    on_update(detail, fraction, elapsed_seconds) is called while waiting; fraction is
    None until there is something to measure. The note is set when the interface
    opens without the model, saying why and what happens next.
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
                 starter.model_id, _gb(starter.total_bytes), starter.engine)
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
        detail, fraction = _progress(task, starter)
        on_update(detail, fraction, elapsed)
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
