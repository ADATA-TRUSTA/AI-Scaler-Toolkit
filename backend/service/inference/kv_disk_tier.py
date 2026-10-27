"""
Size cap and cleanup for vLLM's disk (fs) KV offload tier.

vLLM's fs tier never evicts: every prompt block offloaded to DRAM is written through
to disk and kept across restarts, so the directory only grows (a benchmark directory
on the test box reached 65 GB). This module measures it and deletes the least
recently used blocks, which vLLM tolerates while running:

- Lookups are plain file-existence checks with no index to go stale, and a block
  deleted between lookup and load is reported as a miss and recomputed
  (vllm/v1/kv_offload/tiering/fs/manager.py, get_finished_jobs).
- Stores write ``<hash>.bin_<random>.tmp`` and ``os.replace`` it into place, so a
  ``.bin`` is always complete and an in-flight write is never a ``.bin``.

Only files in vLLM's own layout are touched:
``<root>/<model>_<12 hex>_r<rank>/<3 hex>/<2 hex>_g<group>/<hash>.bin``. The run's
``config.json`` (in the directory without ``_r<rank>``) and anything else under the
root are left alone.
"""

import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_RANK_DIR = re.compile(r"^.+_[0-9a-f]{12}_r\d+$")
_SHARD_DIR = re.compile(r"^[0-9a-f]{3}$")
_GROUP_DIR = re.compile(r"^[0-9a-f]{2}_g\d+$")
_BLOCK_FILE = re.compile(r"^([0-9a-f]{16,})\.bin$")
_TEMP_FILE = re.compile(r"^[0-9a-f]{16,}\.bin_\d+\.tmp$")

# A .tmp this old is a write that died with its process, not one in flight.
STALE_TEMP_SECONDS = 3600
# Evicting stops this far under the cap, so the next store does not trip it again.
LOW_WATER_FRACTION = 0.9
GIB = 1 << 30


class DiskTierNotConfigured(RuntimeError):
    """No vLLM model with a disk KV tier is loaded or loading."""


@dataclass
class _Block:
    """All files of one block: the same hash under every KV cache group."""

    paths: list[str] = field(default_factory=list)
    size: int = 0
    last_used: float = 0.0


@dataclass
class _Scan:
    blocks: dict[tuple[str, str], _Block]
    temps: list[tuple[str, int, float]]  # path, size, mtime

    @property
    def block_bytes(self) -> int:
        return sum(b.size for b in self.blocks.values())

    @property
    def temp_bytes(self) -> int:
        return sum(size for _, size, _ in self.temps)


def _subdirs(path: str, pattern: re.Pattern[str]) -> list[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return [e for e in it if e.is_dir(follow_symlinks=False) and pattern.match(e.name)]
    except OSError:
        return []


def _scan(root: str) -> _Scan:
    blocks: dict[tuple[str, str], _Block] = defaultdict(_Block)
    temps: list[tuple[str, int, float]] = []
    for rank in _subdirs(root, _RANK_DIR):
        for shard in _subdirs(rank.path, _SHARD_DIR):
            for group in _subdirs(shard.path, _GROUP_DIR):
                try:
                    entries = list(os.scandir(group.path))
                except OSError:
                    continue
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue  # deleted under us (vLLM replacing it, or another sweep)
                    if match := _BLOCK_FILE.match(entry.name):
                        block = blocks[(rank.name, match.group(1))]
                        block.paths.append(entry.path)
                        block.size += st.st_size
                        # relatime refreshes atime at most daily, so this is LRU to
                        # within a day; mtime covers a block written but never read.
                        block.last_used = max(block.last_used, st.st_atime, st.st_mtime)
                    elif _TEMP_FILE.match(entry.name):
                        temps.append((entry.path, st.st_size, st.st_mtime))
    return _Scan(blocks=dict(blocks), temps=temps)


def disk_tier_usage(root: str) -> dict[str, int]:
    """Bytes and counts of what vLLM's disk tier holds under ``root``."""
    scan = _scan(root)
    return {
        "total_bytes": scan.block_bytes + scan.temp_bytes,
        "blocks": len(scan.blocks),
        "block_files": sum(len(b.paths) for b in scan.blocks.values()),
        "temp_files": len(scan.temps),
    }


def _remove(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False  # already gone: vLLM rewrote it, or another sweep got there first
    except OSError as e:
        logger.warning("[KV disk] cannot delete %s: %s", path, e)
        return False


def _prune_empty_dirs(root: str) -> None:
    """Drop shard/group directories emptied by eviction; vLLM recreates them on store."""
    for rank in _subdirs(root, _RANK_DIR):
        for shard in _subdirs(rank.path, _SHARD_DIR):
            for group in _subdirs(shard.path, _GROUP_DIR):
                try:
                    os.rmdir(group.path)
                except OSError:
                    pass  # not empty, or just repopulated
            try:
                os.rmdir(shard.path)
            except OSError:
                pass


def evict_disk_tier(root: str, target_bytes: int, *, now: float | None = None) -> dict[str, int]:
    """
    Delete the least recently used blocks until the tier holds at most ``target_bytes``.

    Stale ``.tmp`` files go first; a block goes whole (every group file of its hash),
    so no block is left half on disk. ``target_bytes=0`` empties the tier.
    """
    now = time.time() if now is None else now
    scan = _scan(root)
    total = scan.block_bytes + scan.temp_bytes
    freed = deleted = 0

    for path, size, mtime in scan.temps:
        if now - mtime >= STALE_TEMP_SECONDS and _remove(path):
            freed, deleted, total = freed + size, deleted + 1, total - size

    for block in sorted(scan.blocks.values(), key=lambda b: b.last_used):
        if total <= target_bytes:
            break
        for path in block.paths:
            if _remove(path):
                deleted += 1
        freed, total = freed + block.size, total - block.size

    if deleted:
        _prune_empty_dirs(root)
        logger.info(
            "[KV disk] evicted %d file(s), %.2f GiB from %s; %.2f GiB left",
            deleted,
            freed / GIB,
            root,
            max(total, 0) / GIB,
        )
    return {"deleted_files": deleted, "freed_bytes": freed, "remaining_bytes": max(total, 0)}


class DiskTierJanitor:
    """
    Keep one disk tier under its cap from a background thread.

    Checks right away (a directory that is already over the cap is trimmed at load,
    not one interval later) and then every ``interval_s``. Over the cap it evicts down
    to LOW_WATER_FRACTION of it. ``evict`` runs an on-demand sweep under the same lock,
    so a manual cleanup and a scheduled one never walk the directory at once.
    """

    def __init__(self, root: str, max_bytes: int, interval_s: float) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.interval_s = interval_s
        self.last_usage_bytes: int | None = None
        self._stop = threading.Event()
        self._sweep_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="kv-disk-janitor", daemon=True)

    def start(self) -> "DiskTierJanitor":
        """Start the background check; returns self for chaining."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the background check and wait briefly for an in-progress sweep."""
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def check_once(self) -> None:
        """Measure the tier and trim it to the low-water mark when over the cap."""
        with self._sweep_lock:
            usage = disk_tier_usage(self.root)["total_bytes"]
            self.last_usage_bytes = usage
            if usage > self.max_bytes:
                result = evict_disk_tier(self.root, int(self.max_bytes * LOW_WATER_FRACTION))
                self.last_usage_bytes = result["remaining_bytes"]

    def evict(self, target_bytes: int) -> dict[str, int]:
        """On-demand eviction down to ``target_bytes``, serialized with the scheduled sweep."""
        with self._sweep_lock:
            result = evict_disk_tier(self.root, target_bytes)
            self.last_usage_bytes = result["remaining_bytes"]
            return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:  # a failed sweep must not end the janitor
                logger.exception("[KV disk] cap check failed for %s", self.root)
            self._stop.wait(self.interval_s)
