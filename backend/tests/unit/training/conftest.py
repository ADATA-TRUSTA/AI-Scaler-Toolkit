"""Test isolation for training unit tests.

`service.training.training_process` sets distributed env vars (WORLD_SIZE,
RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT) at *import* time. Once any test in
the process imports that module, those values leak into every later test, and
constructing `transformers.TrainingArguments` then attempts a torch.distributed
rendezvous — which crashes on Windows with "unmatched '}' in format string".

Clearing them before each test keeps the unit tests isolated (they construct
TrainingArguments without wanting a distributed launch). Any test that truly
needs the distributed env sets it explicitly. monkeypatch restores whatever was
present after each test.
"""

import pytest

_DIST_ENV_VARS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _isolate_distributed_env(monkeypatch):
    """Remove leaked torch.distributed env vars before each training unit test."""
    for var in _DIST_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def vlm_path() -> str:
    """
    Path to the small multimodal checkpoint, downloaded once into the repo-local
    cache and shared by every test in the module that asks for it.

    Lives here rather than in each file because it had drifted into two copies
    with identical bodies and different return types -- one handed back a `Path`
    and the other a `str` for the same value, which only stayed harmless because
    every consumer happened to accept both.
    """
    import model_fixtures as mf

    try:
        _, path = mf.ensure_vlm_model()
    except mf.ModelUnavailable as exc:
        pytest.skip(f"multimodal test model unavailable: {exc}")
    return str(path)
