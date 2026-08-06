"""Background model download task manager (HuggingFace snapshot / GGUF files)."""

import os
import threading
import time
import uuid
from typing import cast

from huggingface_hub import snapshot_download
from pydantic import BaseModel

from .model_registry import model_registry
from .settings import configure_logging
from .utils.token_utils import load_hf_token

logger = configure_logging(__name__)


class DownloadTask(BaseModel):
    """State of a single background model download task."""

    task_id: str
    model_id: str
    label: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: float
    end_time: float | None = None
    error: str | None = None
    local_path: str | None = None
    progress: str | None = None  # holds a simple progress description


class DownloadManager:
    """
    Manages background model download tasks.
    Singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "DownloadManager":
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.tasks: dict[str, DownloadTask] = {}
        self._initialized = True

    def start_download(
        self,
        model_id: str,
        label: str,
        cache_dir: str | None = None,
        force_download: bool = False,
        filename: str | None = None,
    ) -> str:
        """
        Start a new download task in a background thread.
        Returns the task_id.
        """
        task_id = str(uuid.uuid4())
        task = DownloadTask(
            task_id=task_id,
            model_id=model_id,
            label=label,
            status="pending",
            start_time=time.time(),
            progress="Initialized",
        )
        with self._lock:
            self.tasks[task_id] = task

        # Start thread
        thread = threading.Thread(
            target=self._download_worker,
            args=(task_id, model_id, label, cache_dir, force_download, filename),
            daemon=True,
        )
        thread.start()

        return task_id

    def _download_worker(
        self,
        task_id: str,
        model_id: str,
        label: str,
        cache_dir: str | None,
        force_download: bool,
        filename: str | None,
    ) -> None:
        task = self.get_task(task_id)
        if not task:
            return

        try:
            task.status = "running"
            task.progress = "Downloading from HuggingFace..."
            logger.info(
                f"[DownloadManager] Starting download task {task_id} for {model_id} (filename={filename})"
            )

            hf_token = load_hf_token()

            downloaded_path = None
            max_context = None
            size_gb = "unknown"

            if filename:
                # GGUF single-file / multi-file download logic
                import re

                from huggingface_hub import hf_hub_download, list_repo_files

                # 1. Check whether this is a split file (e.g., model-00001-of-00005.gguf)
                target_files = [filename]

                # List repo files to find sibling shards and mmproj
                all_files = []
                try:
                    task.progress = "Fetching repository file list..."
                    all_files = list_repo_files(repo_id=model_id, token=hf_token)
                except Exception as list_err:
                    logger.warning(f"[DownloadManager] Failed to list repo files: {list_err}")

                # Simple heuristic on the "-of-" pattern: if the user picked one
                # shard, download all of them
                if "-of-" in filename and ".gguf" in filename and all_files:
                    # Extract the prefix and directory, e.g. "BF16/gemma-3-27b-it-BF16"
                    # given a filename like "BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf",
                    # then match every *-of-*.gguf file in the same directory.

                    # Directory part
                    dir_part = os.path.dirname(filename)
                    base_name_part = os.path.basename(filename)

                    # Derive the common prefix (strip -00001-of-XXXXX.gguf);
                    # digit width is not fixed (e.g. 01-of-05 or 00001-of-00005)
                    match = re.match(r"(.*)-\d+-of-\d+\.gguf", base_name_part)
                    if match:
                        prefix = match.group(1)
                        # Keep every file matching the prefix and containing -of-
                        related_files = []
                        for f in all_files:
                            # Check that it lives in the same directory
                            f_dir = os.path.dirname(f)
                            f_name = os.path.basename(f)
                            if (
                                f_dir == dir_part
                                and f_name.startswith(prefix)
                                and "-of-" in f_name
                                and f_name.endswith(".gguf")
                            ):
                                related_files.append(f)

                        if related_files:
                            target_files = sorted(related_files)
                            logger.info(
                                f"[DownloadManager] Detected split files ({len(target_files)} parts), downloading all..."
                            )

                # Add mmproj files too (they provide multimodal/image support):
                # any ".gguf" file in the repo whose name contains "mmproj"
                if all_files:
                    mmproj_files = [
                        f
                        for f in all_files
                        if "mmproj" in f.lower() and f.lower().endswith(".gguf")
                    ]
                    for mf in mmproj_files:
                        if mf not in target_files:
                            logger.info(
                                f"[DownloadManager] Detected Vision/Multimodal projection file: {mf}, adding to download list..."
                            )
                            target_files.append(mf)

                # 2. Download every file in turn
                total_files = len(target_files)

                for idx, f_name in enumerate(target_files):
                    task.progress = f"Downloading file {idx + 1}/{total_files}: {f_name}..."
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=f_name,
                        cache_dir=cache_dir,
                        token=hf_token,
                        force_download=force_download,
                    )
                    # Keep the first file's path as the main one (loaders usually need only that, or find the rest)
                    if idx == 0:
                        downloaded_path = path

                logger.info(
                    f"[DownloadManager] GGUF Model {model_id} downloaded. Main path: {downloaded_path}"
                )

                # 3. Read GGUF metadata (via the gguf library)
                task.progress = "Reading GGUF metadata..."
                try:
                    # Compute the total size
                    total_size = 0
                    for f_name in target_files:
                        # hf_hub_download returns absolute paths, and the loop above
                        # did not keep them all, so just re-resolve each path here
                        # (cached, so fast)
                        p = hf_hub_download(
                            repo_id=model_id, filename=f_name, cache_dir=cache_dir, token=hf_token
                        )
                        total_size += os.path.getsize(p)
                    size_gb = f"{total_size / (1024**3):.1f}GB"

                    # Read the context length
                    try:
                        import gguf

                        # downloaded_path is set by the loop above (target_files is non-empty
                        # since filename is truthy here), so it is never None at this point.
                        reader = gguf.GGUFReader(
                            cast(str, downloaded_path)
                        )  # the first shard normally carries the metadata

                        # Keys to look up, in priority order
                        ctx_keys = [
                            "llama.context_length",
                            "qwen.context_length",
                            "context_length",
                            "n_ctx",
                        ]
                        found_ctx = None

                        # 1. Try the standard keys first
                        for key in ctx_keys:
                            if key in reader.fields:
                                field = reader.fields[key]
                                if field.parts:
                                    # parts[-1] holds the data, as a list or numpy array
                                    raw_val = field.parts[-1]
                                    if isinstance(raw_val, list) and len(raw_val) > 0:
                                        found_ctx = raw_val[0]
                                    elif hasattr(raw_val, "item"):  # numpy scalar
                                        found_ctx = raw_val.item()
                                    elif isinstance(raw_val, (int, float)):
                                        found_ctx = raw_val

                                    if found_ctx:
                                        logger.info(
                                            f"[DownloadManager] Found GGUF context length via key '{key}': {found_ctx}"
                                        )
                                        break

                        # 2. Generic fallback search
                        if not found_ctx:
                            for key in reader.fields:
                                if (
                                    "context_length" in key or "n_ctx" in key
                                ) and "rope" not in key:
                                    field = reader.fields[key]
                                    if field.parts:
                                        raw_val = field.parts[-1]
                                        val = None
                                        if isinstance(raw_val, list) and len(raw_val) > 0:
                                            val = raw_val[0]
                                        elif hasattr(raw_val, "item"):
                                            val = raw_val.item()

                                        if isinstance(val, (int, float)):
                                            found_ctx = val
                                            logger.info(
                                                f"[DownloadManager] Found GGUF context length via heuristic key '{key}': {found_ctx}"
                                            )
                                            break

                        if found_ctx:
                            max_context = int(found_ctx)

                    except ImportError:
                        logger.warning(
                            "[DownloadManager] 'gguf' library not found, skipping metadata read. Please install with `pip install gguf`"
                        )
                    except Exception as meta_err:
                        logger.warning(
                            f"[DownloadManager] Failed to read GGUF metadata: {meta_err}"
                        )

                except Exception as e:
                    logger.warning(f"[DownloadManager] Error processing GGUF size/metadata: {e}")

            else:
                # Standard snapshot download
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=hf_token,
                    force_download=force_download,
                )
                logger.info(f"[DownloadManager] Model {model_id} downloaded to {downloaded_path}")

                task.progress = "Analyzing model configuration..."

                # Context length logic - only read config.json for non-GGUF models
                if not filename:
                    try:
                        from transformers import AutoConfig

                        config = AutoConfig.from_pretrained(
                            downloaded_path,
                            token=hf_token,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        max_context = getattr(config, "max_position_embeddings", None)
                        if max_context is None:
                            max_context = getattr(config, "n_positions", None)
                        if max_context is None:
                            max_context = getattr(config, "max_sequence_length", None)
                    except Exception as e:
                        logger.warning(f"[DownloadManager] Failed to get context length: {e}")

            task.progress = "Registering model..."

            if filename:
                # GGUF registration flow
                # Per the requested mapping:
                # base_model_name -> repo_id (model_id)
                # label -> user custom label
                # filename -> specific filename
                # source -> "hf"
                # local_path -> downloaded_path

                model_registry.add_llama_gguf_model(
                    label=label,
                    base_model_name=model_id,  # repo_id
                    size=size_gb,
                    max_context_length=max_context,
                    source="hf",
                    local_path=downloaded_path,
                    filename=filename,
                )
            else:
                # Register Standard HF Model
                model_registry.add_base_model(
                    label=label,
                    hf_model_name=model_id,
                    local_path=downloaded_path,
                    max_context_length=max_context,
                )

            task.status = "completed"
            task.progress = "Done"
            task.local_path = downloaded_path
            task.end_time = time.time()
            logger.info(f"[DownloadManager] Task {task_id} completed successfully")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress = "Failed"
            task.end_time = time.time()
            logger.exception(f"[DownloadManager] Task {task_id} failed: {e}")
            import traceback

            logger.exception(traceback.format_exc())

    def get_task(self, task_id: str) -> DownloadTask | None:
        """Return the task with the given id, or None if unknown."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> list[DownloadTask]:
        """Return all tasks ordered by start time (most recent first)."""
        # Sort by start time desc
        return sorted(self.tasks.values(), key=lambda x: x.start_time, reverse=True)


# Global instance
download_manager = DownloadManager()
