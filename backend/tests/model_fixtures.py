"""Shared helpers that provision small real models for smoke/e2e tests.

Design goals (user requirements):
- Tests must not depend on machine-specific model paths or a pre-populated
  global HF cache; they download small models by themselves.
- Downloads land in a repo-local cache (tests/.cache/hf, gitignored) so runs
  are reproducible across environments and never touch the user's HF cache.
- Offline / restricted machines skip instead of failing: every helper raises
  ModelUnavailable with a reason, which callers turn into pytest.skip.

Layout: TESTS_HF_HOME (tests/.cache/hf) mirrors a standard HF_HOME, with the
hub cache at tests/.cache/hf/hub — so pointing HF_HOME at it is enough for
huggingface_hub, transformers and the backend service alike.

Environment overrides:
- LLAMA_GGUF          absolute path to a GGUF file (skips download entirely)
- SMOKE_MODEL         HF repo id for the transformers smoke model
- TESTS_MODEL_CACHE   alternative HF_HOME-style cache dir (default tests/.cache/hf)
- TESTS_NO_DOWNLOAD=1 never hit the network; use cache hits only or skip
"""

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_HOME = REPO_ROOT / "tests" / ".cache" / "hf"

GGUF_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILENAME = "qwen2.5-0.5b-instruct-q8_0.gguf"

SMOKE_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

# Smallest real image-text-to-text checkpoint we can train end to end (~500MB
# of safetensors). Used to cover the multimodal loading path; the repo also
# ships ~900MB of ONNX exports that training never touches, so they are skipped.
VLM_MODEL_DEFAULT = "HuggingFaceTB/SmolVLM-256M-Instruct"
VLM_IGNORE_PATTERNS = ["onnx/*", "*.md"]


class ModelUnavailable(Exception):
    """Raised when a test model cannot be provisioned; callers should skip."""


def tests_hf_home() -> Path:
    """Repo-local HF_HOME used by all real-run tests."""
    home = Path(os.environ.get("TESTS_MODEL_CACHE", str(DEFAULT_HF_HOME)))
    (home / "hub").mkdir(parents=True, exist_ok=True)
    return home


def hf_env() -> dict:
    """Env vars pointing HF loaders (and the backend) at the local cache."""
    home = tests_hf_home()
    return {"HF_HOME": str(home), "HF_HUB_CACHE": str(home / "hub")}


def _hub_cache_dir() -> str:
    return str(tests_hf_home() / "hub")


def _downloads_disabled() -> bool:
    return os.environ.get("TESTS_NO_DOWNLOAD", "").strip() in ("1", "true", "yes")


def _readable_hub_caches() -> list[str]:
    """Hub caches worth searching for an already-downloaded checkpoint, best first.

    The repo-local cache comes first because that is where the ``ensure_*`` helpers
    put things. HF_HUB_CACHE / HF_HOME follow: this project's own .env points HF_HOME
    at the volume holding the large checkpoints, and ignoring it made the smoke tests
    skip on exactly the machines that could run them.
    """
    caches = [_hub_cache_dir()]
    ambient = os.environ.get("HF_HUB_CACHE") or ""
    if not ambient:
        hf_home = os.environ.get("HF_HOME") or _hf_home_from_dotenv()
        ambient = str(Path(hf_home) / "hub") if hf_home else ""
    if ambient and ambient not in caches:
        caches.append(ambient)
    return caches


def _hf_home_from_dotenv() -> str:
    """HF_HOME as the service would see it, read straight from .env.

    Needed because os.environ is not enough at guard time: HF_HOME reaches the
    environment only when service.settings runs load_dotenv(), and a fixture that
    resolves a checkpoint before importing service modules would look in the wrong
    place. Parsed with dotenv_values rather than load_dotenv so the surrounding
    process keeps its environment -- .env here also holds tokens and a sudo password,
    and a test run has no business exporting either.
    """
    try:
        from dotenv import dotenv_values

        for name in (".env", ".env.example"):
            env_file = Path(__file__).resolve().parent.parent / name
            if env_file.is_file():
                return (dotenv_values(env_file).get("HF_HOME") or "").strip()
    except Exception:
        pass
    return ""


def _has_weights(snapshot_path: str) -> bool:
    """True when a snapshot directory actually holds model weights, not just metadata."""
    directory = Path(snapshot_path)
    return bool(
        list(directory.glob("*.safetensors"))
        or list(directory.glob("*.bin"))
        or list(directory.glob("*.gguf"))
    )


def local_snapshot(repo_id: str, ignore_patterns: Optional[list] = None) -> Optional[str]:
    """Path to a complete local snapshot of ``repo_id``, or None when there isn't one.

    Resolves it the way the loader does -- ``snapshot_download(local_files_only=True)``,
    which follows ``refs/<revision>`` to a snapshot directory -- and then checks the
    weights are really there. Never downloads: callers use this to decide whether to
    skip, so it must not turn a skip into a multi-gigabyte fetch.

    Why not ``Path(hub_dir).is_dir()``, which is what the smoke tests used to do: that
    answers a different question from the one that decides whether the test can run.
    Two ways it says yes and the load then fails:

    * ``refs/main`` can point at a revision that was never downloaded -- the model
      directory and its blobs exist, the snapshot directory does not. Observed on
      google/gemma-4-E4B-it, which failed with LocalEntryNotFoundError after passing
      the directory check.
    * blobs belonging to revisions that have since been detached still sit in
      ``blobs/``, so a size-based check over the whole model directory passes on
      weights no snapshot references any more.

    Return the path to the caller and load *that* rather than the repo id: an explicit
    directory bypasses hub cache resolution entirely, so the guard and the loader can
    no longer disagree about which cache they are looking at -- which is the other
    half of the same bug (a host holding the checkpoint under HF_HOME reported
    "not in the local hub cache" because the guard only consulted TESTS_MODEL_CACHE).

    Reading is opportunistic and writing is not: every cache that might hold the
    checkpoint is consulted, while the ``ensure_*`` helpers still *download* only into
    the repo-local cache. Otherwise the 15-70 GB checkpoints a developer already has
    under HF_HOME are invisible, and the smoke tests skip on the one machine that can
    actually run them -- which reads as "nothing to worry about".
    """
    from huggingface_hub import snapshot_download

    for cache_dir in _readable_hub_caches():
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        except Exception:
            continue  # nothing here resolves -- includes a dangling refs/<revision>
        if _has_weights(path):
            return path
    return None


def ensure_gguf() -> str:
    """Return an absolute path to a small chat GGUF, downloading if needed.

    Resolution order: LLAMA_GGUF env override > repo-local cache hit >
    download (~531MB Q8_0). Raises ModelUnavailable when none succeed.
    """
    override = os.environ.get("LLAMA_GGUF", "").strip()
    if override:
        if os.path.isfile(override):
            return os.path.abspath(override)
        raise ModelUnavailable(f"LLAMA_GGUF points to a missing file: {override}")

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME,
            cache_dir=_hub_cache_dir(),
            local_files_only=True,
        )
    except Exception:
        pass  # not cached yet -> try the network below

    if _downloads_disabled():
        raise ModelUnavailable(
            f"{GGUF_REPO_ID}/{GGUF_FILENAME} not cached and TESTS_NO_DOWNLOAD is set"
        )
    try:
        return hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME,
            cache_dir=_hub_cache_dir(),
        )
    except Exception as e:
        raise ModelUnavailable(f"failed to download {GGUF_REPO_ID}/{GGUF_FILENAME}: {e}") from e


def ensure_hf_model(repo_id: Optional[str] = None) -> tuple:
    """Ensure a small transformers model snapshot exists locally.

    Returns ``(repo_id, snapshot_path)``. Callers should load via the local
    ``snapshot_path`` (e.g. InferenceConfig.model_path): passing an explicit
    directory bypasses HF cache resolution entirely, which matters because a
    forked worker inherits hub cache constants computed at import time — env
    vars set later would be silently ignored. Raises ModelUnavailable when
    the snapshot cannot be obtained.

    As in ``ensure_vlm_model``, the cached fast path verifies that the weights
    are actually there: a snapshot directory can hold only a subset of the
    repo's files (say config and tokenizer from an interrupted download) and
    ``local_files_only`` happily returns that partial directory, which then
    fails at load time with "no file named model.safetensors".
    """
    repo_id = repo_id or os.environ.get("SMOKE_MODEL", SMOKE_MODEL_DEFAULT)

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=_hub_cache_dir(), local_files_only=True)
        if _has_weights(path):
            return repo_id, path
    except Exception:
        pass  # not cached (or only partially) -> try the network below

    if _downloads_disabled():
        raise ModelUnavailable(f"{repo_id} not cached and TESTS_NO_DOWNLOAD is set")
    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=_hub_cache_dir())
        return repo_id, path
    except Exception as e:
        raise ModelUnavailable(f"failed to download {repo_id}: {e}") from e


def ensure_vlm_model(repo_id: Optional[str] = None) -> tuple:
    """Ensure a small multimodal (image-text-to-text) snapshot exists locally.

    Returns ``(repo_id, snapshot_path)``. Same contract as ``ensure_hf_model``,
    but skips the repo's ONNX exports. The cached-snapshot fast path verifies
    that ``model.safetensors`` is actually present: a snapshot directory can
    exist while holding only a subset of the repo's files, and
    ``local_files_only`` happily returns that partial directory.
    """
    repo_id = repo_id or os.environ.get("VLM_MODEL", VLM_MODEL_DEFAULT)

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=_hub_cache_dir(),
            ignore_patterns=VLM_IGNORE_PATTERNS,
            local_files_only=True,
        )
        if (Path(path) / "model.safetensors").exists():
            return repo_id, path
    except Exception:
        pass  # not cached (or only partially) -> try the network below

    if _downloads_disabled():
        raise ModelUnavailable(f"{repo_id} not cached and TESTS_NO_DOWNLOAD is set")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=_hub_cache_dir(),
            ignore_patterns=VLM_IGNORE_PATTERNS,
        )
        return repo_id, path
    except Exception as e:
        raise ModelUnavailable(f"failed to download {repo_id}: {e}") from e
