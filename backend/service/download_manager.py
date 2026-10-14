"""Background model download task manager (HuggingFace snapshot / GGUF files)."""

import contextlib
import os
import threading
import time
import uuid
from pathlib import Path
from typing import cast

from huggingface_hub import snapshot_download
from pydantic import BaseModel

from .model_registry import model_registry
from .settings import HF_XET_ENABLED, configure_logging
from .utils.token_utils import load_hf_token

logger = configure_logging(__name__)


class DownloadCancelled(Exception):
    """Raised inside a worker when the task has been asked to stop."""


def _abort_on_cancel(event: threading.Event) -> type:
    """
    A progress-bar class that ends the transfer once `event` is set.

    huggingface_hub hands the caller the bar it drives while a file downloads, and
    raising from its update ends the transfer -- which is the only way to stop one
    part way through, because there is no cancellation in the API.

    It works on the classic HTTP path only. Xet drives the same bar from its own
    reporting thread, so the exception lands there and the transfer carries on: the
    task reports itself stopping, the bytes keep coming, and the only trace is a stack
    in the log. That is why this is installed only when Xet is off -- see
    settings.HF_XET_ENABLED, which is what decides that.
    """
    from huggingface_hub.utils.tqdm import tqdm as hf_tqdm

    class _AbortingTqdm(hf_tqdm):  # type: ignore[misc, valid-type]
        def update(self, n: float = 1) -> bool | None:
            if event.is_set():
                raise DownloadCancelled
            return super().update(n)

    return _AbortingTqdm


class DownloadTask(BaseModel):
    """State of a single background model download task."""

    task_id: str
    model_id: str
    label: str
    # "pending", "running", "cancelling", "completed", "failed", "cancelled".
    #
    # "cancelling" is here so that a cancel that has been asked for is visible to
    # everyone, not just to the browser that asked. The button used to be driven by
    # local state, so a second screen -- or the same screen after a refresh -- was
    # still offered it on a task that was already cancelling, and pressing it again
    # was the only way to find out. The status is the one place all of them read.
    status: str
    start_time: float
    end_time: float | None = None
    error: str | None = None
    local_path: str | None = None
    progress: str | None = None  # holds a simple progress description
    # What the task is actually going to fetch, known before the first byte.
    #
    # Asking for a repository without naming a file downloads all of it, and for a
    # GGUF repository that means every quantisation: nine files and 5.1 GB for a 0.5B
    # model, tens of gigabytes for a 7B one. That is a legitimate thing to want, but
    # "Downloading from HuggingFace..." gave no way to tell it apart from a 500 MB one.
    total_files: int | None = None
    completed_files: int = 0
    total_bytes: int | None = None


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
        self._cancels: dict[str, threading.Event] = {}
        self._initialized = True

    def cancel(self, task_id: str) -> bool:
        """
        Ask a running task to stop. False if there is no such task running.

        A request, not a kill, and how quickly it lands depends on the transfer in
        use. With Xet off, which is the default, the file in flight is ended from its
        progress bar and the task stops in a second or two. With HF_XET_ENABLED=1 the
        transfer happens inside a Rust extension that cannot be interrupted, so the
        stop can only take effect before the next file: measured on a 3B model, that
        was still going 150 seconds later, against 2 seconds with Xet off.

        What the download put on the disk is then discarded rather than kept -- all
        of it, the completed files included. See _discard_this_download for why, and
        for what stops it reaching anything the download did not add.
        """
        task = self.tasks.get(task_id)
        if task is None or task.status not in ("pending", "running"):
            return False
        event = self._cancels.get(task_id)
        if event is None:
            return False
        logger.info(f"[DownloadManager] Cancellation requested for task {task_id}")
        event.set()
        # Recorded on the task, not only in the event: the event is internal to this
        # process, and the status is what every client polls.
        task.status = "cancelling"
        task.progress = "Cancelling..."
        return True

    @staticmethod
    def _repo_cache_folders(model_id: str, cache_dir: str | None) -> tuple[Path, ...]:
        """
        Everywhere huggingface_hub puts something for this repository.

        Two places, not one. The repository itself -- blobs, snapshots, refs -- and a
        directory of its own under .locks, which is where the per-file download locks
        go. Cleaning only the first leaves an empty directory named after the model
        someone just cancelled, which is nothing in bytes and still the wrong answer
        to "it should be as though I never started".
        """
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.file_download import repo_folder_name

        root = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
        name = repo_folder_name(repo_id=model_id, repo_type="model")
        return (root / name, root / ".locks" / name)

    @staticmethod
    def _cache_contents(folders: tuple[Path, ...]) -> set[Path]:
        """
        Every file under `folders`, skipping any that is not there yet.

        Symlinks count as files without being followed: a snapshot entry is a link to
        a blob on a system that allows them, and a link left dangling by a removed
        blob still has to be seen here.
        """
        found: set[Path] = set()
        for folder in folders:
            if folder.is_dir():
                found.update(p for p in folder.rglob("*") if p.is_symlink() or p.is_file())
        return found

    def _another_task_is_downloading(self, model_id: str, task_id: str) -> bool:
        return any(
            other.model_id == model_id
            and other.task_id != task_id
            and other.status in ("pending", "running")
            for other in self.tasks.values()
        )

    def _discard_this_download(
        self, task_id: str, model_id: str, folders: tuple[Path, ...], before: set[Path]
    ) -> tuple[int, int]:
        """
        Undo a cancelled download: remove what it added, keep what was already there.

        A cancel registers nothing, so whatever it leaves in the cache is invisible to
        the application -- not in the model list, and on no screen where someone could
        find it and delete it. A cancelled 20 GB download that quietly keeps 14 GB is
        worse than one that costs the bytes again, so the whole of it goes: the files
        that completed as well as the fragment of the one in flight.

        `before` is the listing taken ahead of the first byte, and only what is absent
        from it is removed. That is the part that matters: someone can ask for a model
        they already have and then cancel, and deleting the repository wholesale would
        take a working model with it. Directories left empty are pruned afterwards, so
        a repository this download created goes away entirely.

        Best effort, and never allowed to turn a cancel into a failure -- a file the
        system will not release is logged and left behind.
        """
        if self._another_task_is_downloading(model_id, task_id):
            logger.info(
                f"[DownloadManager] {model_id} is also being downloaded by another "
                "task; leaving the cache untouched"
            )
            return 0, 0

        removed = 0
        freed = 0
        for path in self._cache_contents(folders) - before:
            try:
                # Not path.stat() on a symlink: that reports the size of the blob it
                # points at, which is counted in its own right.
                size = 0 if path.is_symlink() else path.stat().st_size
                path.unlink()
            except OSError as exc:
                logger.warning(f"[DownloadManager] Could not remove {path}: {exc}")
                continue
            removed += 1
            freed += size

        # Deepest first, because a directory only goes once the ones inside it have.
        # rmdir on a directory that still holds something raises and is suppressed,
        # which is what keeps this from removing anything that was already here.
        for folder in folders:
            if not folder.is_dir():
                continue
            for directory in sorted(
                (p for p in folder.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                with contextlib.suppress(OSError):
                    directory.rmdir()
            with contextlib.suppress(OSError):
                folder.rmdir()

        return removed, freed

    def _check_cancelled(self, task_id: str) -> None:
        event = self._cancels.get(task_id)
        if event is not None and event.is_set():
            raise DownloadCancelled

    @staticmethod
    def _repo_size(model_id: str, files: list[str], hf_token: str | None) -> int | None:
        """
        Total size of the named files, or None if the repository will not say.

        Best effort on purpose. This is here so a task can say how big it is before it
        starts, and a size that cannot be fetched is not a reason to refuse to
        download -- it just means the task reports a file count and no total.
        """
        try:
            from huggingface_hub import HfApi

            wanted = set(files)
            info = HfApi().model_info(repo_id=model_id, files_metadata=True, token=hf_token)
            sizes = [
                s.size
                for s in (info.siblings or [])
                if s.rfilename in wanted and s.size is not None
            ]
            return sum(sizes) if sizes else None
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.warning(f"[DownloadManager] Could not size {model_id}: {exc}")
            return None

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
            self._cancels[task_id] = threading.Event()

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

        # Set before the try, because the cancellation handler reads them and an
        # exception can be raised before the assignments inside it are reached.
        repo_folders: tuple[Path, ...] | None = None
        cached_before: set[Path] = set()

        try:
            # Checked before the task is marked running, so a stop asked for while it
            # was still pending is not undone by the line below.
            self._check_cancelled(task_id)
            task.status = "running"
            task.progress = "Downloading from HuggingFace..."
            logger.info(
                f"[DownloadManager] Starting download task {task_id} for {model_id} (filename={filename})"
            )

            hf_token = load_hf_token()

            # Assigned here, not exported as an environment variable: the variable
            # is read when huggingface_hub is imported, and this module cannot count
            # on running first. The constant is read per download.
            from huggingface_hub import constants as hf_constants

            hf_constants.HF_HUB_DISABLE_XET = not HF_XET_ENABLED

            # What the cache holds for this repository before the first byte. A
            # cancel removes the difference and nothing else.
            repo_folders = self._repo_cache_folders(model_id, cache_dir)
            cached_before = self._cache_contents(repo_folders)

            # The abort hook only goes on where it can actually stop the transfer;
            # under Xet it raises on a reporting thread and leaves a stack in the log
            # for nothing.
            abort_kwargs: dict = {}
            if not HF_XET_ENABLED:
                abort_kwargs["tqdm_class"] = _abort_on_cancel(
                    self._cancels.get(task_id) or threading.Event()
                )

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
                task.total_files = total_files
                task.total_bytes = self._repo_size(model_id, target_files, hf_token)

                for idx, f_name in enumerate(target_files):
                    self._check_cancelled(task_id)
                    task.progress = f"Downloading file {idx + 1}/{total_files}: {f_name}..."
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=f_name,
                        cache_dir=cache_dir,
                        token=hf_token,
                        force_download=force_download,
                        **abort_kwargs,
                    )
                    # Keep the first file's path as the main one (loaders usually need only that, or find the rest)
                    if idx == 0:
                        downloaded_path = path
                    task.completed_files = idx + 1

                # The loop only checks before each file, so a stop asked for during
                # the last one would fall straight through to registering the model
                # the user had just said they did not want.
                self._check_cancelled(task_id)

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
                # Standard snapshot download, one file at a time.
                #
                # snapshot_download does the same thing in one call, and that call
                # cannot be interrupted: a repository asked for without a file name
                # fetches all of it, so someone who wanted one quantisation and got
                # nine had no way to stop what they had started -- short of killing
                # the service and clearing the cache by hand. Downloading file by
                # file is what the GGUF branch above already does; it costs nothing
                # and it makes the task answerable and stoppable.
                from huggingface_hub import hf_hub_download, list_repo_files

                task.progress = "Fetching repository file list..."
                repo_files = list_repo_files(repo_id=model_id, token=hf_token)
                task.total_files = len(repo_files)
                task.total_bytes = self._repo_size(model_id, repo_files, hf_token)
                logger.info(
                    f"[DownloadManager] {model_id}: {len(repo_files)} file(s)"
                    + (f", {task.total_bytes / (1024**3):.1f}GB" if task.total_bytes else "")
                )

                for idx, f_name in enumerate(repo_files):
                    self._check_cancelled(task_id)
                    task.progress = f"Downloading file {idx + 1}/{len(repo_files)}: {f_name}..."
                    hf_hub_download(
                        repo_id=model_id,
                        filename=f_name,
                        cache_dir=cache_dir,
                        token=hf_token,
                        force_download=force_download,
                        **abort_kwargs,
                    )
                    task.completed_files = idx + 1

                # As above: without this, a stop asked for during the last file is
                # lost and the download registers itself as though nothing happened.
                self._check_cancelled(task_id)

                # Resolved rather than derived from a file path: a repository with
                # files in subdirectories would otherwise give the wrong directory.
                # Everything is cached by now, so this does not touch the network.
                downloaded_path = snapshot_download(
                    repo_id=model_id, cache_dir=cache_dir, local_files_only=True
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

        except DownloadCancelled:
            # Not a failure. Nothing is registered, because a partly fetched
            # repository is not a model anyone should be offered -- and because
            # nothing is registered, what was fetched is removed with it.
            logger.info(
                f"[DownloadManager] Task {task_id} cancelled after "
                f"{task.completed_files}/{task.total_files} file(s)"
            )
            # Removing first, and only then reporting the task cancelled. Deleting
            # tens of gigabytes is not instant, and the other order has the interface
            # announce that the download was removed while it is still being removed.
            if repo_folders is not None:
                task.progress = "Discarding what was downloaded..."
                try:
                    removed, freed = self._discard_this_download(
                        task_id, model_id, repo_folders, cached_before
                    )
                    logger.info(
                        f"[DownloadManager] Discarded {removed} file(s), "
                        f"{freed / (1024**3):.2f}GB, from the cancelled download"
                    )
                except Exception as cleanup_err:  # noqa: BLE001 - cleanup is not the task
                    logger.warning(
                        f"[DownloadManager] Could not clear the cancelled download: {cleanup_err}"
                    )
            task.status = "cancelled"
            task.progress = "Cancelled"
            task.end_time = time.time()

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress = "Failed"
            task.end_time = time.time()
            logger.exception(f"[DownloadManager] Task {task_id} failed: {e}")
            import traceback

            logger.exception(traceback.format_exc())

        finally:
            self._cancels.pop(task_id, None)

    def get_task(self, task_id: str) -> DownloadTask | None:
        """Return the task with the given id, or None if unknown."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> list[DownloadTask]:
        """Return all tasks ordered by start time (most recent first)."""
        # Sort by start time desc
        return sorted(self.tasks.values(), key=lambda x: x.start_time, reverse=True)


# Global instance
download_manager = DownloadManager()
