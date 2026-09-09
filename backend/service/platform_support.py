"""
Which features this host can actually run.

Two features are Linux-only, for reasons that surface far from where the user
asked for them:

  - **Fine-tuning.** ``service/training/training_process.py`` sets WORLD_SIZE /
    RANK / LOCAL_RANK / MASTER_ADDR / MASTER_PORT at import time, so accelerate's
    ``PartialState`` always initialises a distributed process group -- even for a
    single-GPU run. On Windows that rendezvous fails inside ``TCPStore``, and torch
    reports it as ``RuntimeError: unmatched '}' in format string`` because its own
    error formatting breaks on the underlying message. The user is left with a
    string that points nowhere.

  - **vLLM.** The engine manages its server with ``os.getpgid`` / ``os.killpg`` /
    ``SIGKILL``, none of which exist on Windows, and its isolated environment is
    only created by ``scripts/linux/setup_env.sh`` -- the PowerShell setup script
    does not build one, so on Windows the venv it expects is simply absent.

Checking here means each feature is refused once, at the entry point, with a
message that says which platform is required and which one this host is. The
alternative is what used to happen: the request is accepted, work starts, and it
dies somewhere in a dependency with a traceback nobody can act on.

WSL is deliberately not special-cased. Under WSL ``platform.system()`` is
"Linux", which is the right answer -- both features work there once the GPU
runtime is in place (see scripts/xpu_smoke.py).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from .config_models import InferenceEngine


@dataclass(frozen=True)
class Support:
    """Whether a feature can run here, and why not when it cannot."""

    supported: bool
    reason: str | None = None


def current_platform() -> str:
    """
    Name of the OS this service is running on: "Linux", "Windows", "Darwin".

    Exposed over /health so a client can gate its own UI on the *backend's*
    platform rather than its own — the two differ whenever the UI is served to a
    Windows desktop from a Linux host.
    """
    return platform.system()


def _linux_only(feature: str, detail: str) -> Support:
    """Build the refusal for a feature that needs Linux, naming this host's OS."""
    if platform.system() == "Linux":
        return Support(supported=True)
    return Support(
        supported=False,
        reason=(
            f"{feature} is only supported on Linux; this backend is running on "
            f"{current_platform()}. {detail}"
        ),
    )


def training_support() -> Support:
    """Whether fine-tuning can run on this host."""
    return _linux_only(
        "Fine-tuning",
        "Run the backend on Linux (a WSL2 distro counts) to fine-tune models. "
        "Inference, chat and GGUF conversion are unaffected.",
    )


def engine_support(engine: InferenceEngine | str) -> Support:
    """Whether a given inference engine can run on this host."""
    if InferenceEngine(engine) is not InferenceEngine.VLLM:
        return Support(supported=True)
    return _linux_only(
        "The vLLM engine",
        "Use the transformers or llama_server engine instead, or run the backend "
        "on Linux (a WSL2 distro counts).",
    )


def supported_engines() -> list[InferenceEngine]:
    """
    Engines this host can actually start.

    Clients use this to build their engine picker, so an engine that cannot start
    is never offered in the first place.
    """
    return [engine for engine in InferenceEngine if engine_support(engine).supported]
