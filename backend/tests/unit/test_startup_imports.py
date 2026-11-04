"""
The API process must start without importing the ML stack.

Only the transformers worker and the training worker use torch. The API process drives
llama.cpp, vLLM and Strata over HTTP, and every one of those workers is a separate
spawned process, so torch, transformers and peft in the API process are pure startup
cost: measured on the dev machine, `python -m service.app` took 4.5 s to answer /health
and held 1003 MB RSS with them, 0.72 s and 107 MB without.

One top-level `import torch` in any module the app imports brings all of it back, and
nothing else would notice, so this checks the imported modules directly. A fresh
interpreter is used because this pytest process has long since imported torch.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.inference

REPO_ROOT = Path(__file__).resolve().parents[2]

# Heavy packages only a worker process needs. openai is not here: the API process uses
# it itself for every llama.cpp / vLLM / Strata request.
_WORKER_ONLY_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "trl",
    "bitsandbytes",
    "deepspeed",
    "torchvision",
    "sklearn",
)


def _packages_imported_by(module: str) -> list[str]:
    """Import `module` in a fresh interpreter; return which worker-only packages it loaded."""
    code = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"print(json.dumps([p for p in {list(_WORKER_ONLY_PACKAGES)!r} if p in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_api_process_does_not_import_the_ml_stack() -> None:
    """Importing the app (what the API process does at startup) loads no worker-only package."""
    assert _packages_imported_by("service.app") == []


def test_server_engine_worker_does_not_import_the_ml_stack() -> None:
    """A llama.cpp / vLLM / Strata worker imports this module and needs no torch."""
    assert _packages_imported_by("service.inference.model_inference_process") == []
