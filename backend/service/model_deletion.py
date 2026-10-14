"""
Removing a model's files, after the request that asked for it has been answered.

Deleting a model is two things that take very different amounts of time: dropping a
line from the registry, and removing what can be tens of gigabytes from the disk. The
registry entry is therefore marked "deleting" and kept until the files are actually
gone -- see service.model_registry -- and this is the part that does the removing.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from .model_registry import model_registry
from .settings import configure_logging
from .utils.path_safety import is_protected_system_path

logger = configure_logging(__name__)


def resolve_model_files_path(model_info: dict[str, Any]) -> str | None:
    """
    The path on disk holding this model, or None if the entry does not name one.

    For a model living in the Hugging Face cache the answer is the repository
    directory rather than the snapshot inside it: the snapshot is a directory of
    links, and removing it alone leaves every blob behind -- all of the bytes, none
    of the model.
    """

    def _expand_to_hf_cache_root(raw_path: str) -> str:
        """Walk up from a snapshot path to the models--Org--Repo directory."""
        try:
            current = Path(raw_path)
            if "snapshots" not in current.parts:
                return raw_path
            while len(current.parts) > 1:
                if current.name.startswith("models--"):
                    logger.info(f"Detected HF cache, expanding deletion path to: {current}")
                    return str(current)
                current = current.parent
        except Exception as e:
            logger.warning(f"Error parsing HF cache path: {e}")
        return raw_path

    model_type = model_info.get("type")

    if model_type == "base":
        raw_path = model_info.get("local_path")
        if raw_path:
            return _expand_to_hf_cache_root(raw_path)
        # Fallback: without a local_path, infer the default path from HF_HOME
        hf_id = model_info.get("base_model_name")
        if hf_id and "/" in hf_id:
            try:
                hf_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
                # HF cache directory naming convention: models--Org--Repo
                potential_path = Path(hf_home) / ("models--" + hf_id.replace("/", "--"))
                if potential_path.exists():
                    logger.info(f"No local_path found, using default HF cache: {potential_path}")
                    return str(potential_path)
            except Exception as e:
                logger.warning(f"Error inferring default HF cache path: {e}")
        return None

    if model_type == "finetuned":
        return model_info.get("output_dir")

    if model_type == "llama_gguf":
        # local_path, or a legacy entry where filename holds an absolute path
        raw_path = model_info.get("local_path") or model_info.get("filename")
        return _expand_to_hf_cache_root(raw_path) if raw_path else None

    return None


async def delete_model_files(label: str, path_to_delete: str | None) -> None:
    """
    Remove a model's files, then drop it from the registry.

    Runs after the response has been sent, because removing tens of gigabytes takes
    long enough that no client should be made to hold a request open for it. Until it
    finishes the entry stays in the registry marked "deleting", which is what tells
    every client -- not only the one that asked -- that the model is on its way out.
    A failure leaves the entry behind marked "delete_failed" rather than removing it,
    so files that are still on the disk still have something to delete them with.
    """
    if path_to_delete is None:
        model_registry.delete_model(label)
        logger.info(f"Model {label} removed from registry; no local path to delete")
        return

    # Guard against deleting OS system directories via a poisoned registry entry
    # (e.g. output_dir="/etc" from a malicious config). Off the event loop with the
    # rest of the filesystem work: the guard resolves the path, and resolving one on
    # an unreachable network mount waits out an SMB/DNS timeout of ten seconds or
    # more -- long enough to stall every other request the service is serving.
    if await asyncio.to_thread(is_protected_system_path, path_to_delete):
        logger.error(f"Blocked attempt to delete protected system path: {path_to_delete}")
        model_registry.mark_deletion_failed(
            label, "Deletion blocked: path resolves to a protected system directory"
        )
        return

    try:
        if await asyncio.to_thread(os.path.lexists, path_to_delete):
            if await asyncio.to_thread(os.path.isdir, path_to_delete):
                await asyncio.to_thread(shutil.rmtree, path_to_delete)
            else:
                await asyncio.to_thread(os.remove, path_to_delete)
            logger.info(f"Deleted local files for model {label} at {path_to_delete}")
        else:
            # Already gone is the state that was asked for, not an error. This is also
            # what makes a second attempt after an interrupted deletion work.
            logger.info(f"Model files for {label} already missing at {path_to_delete}")
    except FileNotFoundError:
        logger.info(f"Model files for {label} disappeared during deletion at {path_to_delete}")
    except Exception as e:
        logger.exception(f"Failed to delete files at {path_to_delete}: {e}")
        model_registry.mark_deletion_failed(label, str(e))
        return

    model_registry.delete_model(label)
    logger.info(f"Model {label} fully deleted")
