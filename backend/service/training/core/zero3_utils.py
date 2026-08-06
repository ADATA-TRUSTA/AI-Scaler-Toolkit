"""
Helpers for saving LoRA adapters under DeepSpeed ZeRO-3.

The training pipeline saves a LoRA adapter by manually gathering only the
trainable (adapter) parameters with ``deepspeed.zero.GatheredParameters`` --
deliberately avoiding a full 16-bit model gather that would OOM on large models.

That manual gather is fragile mid-training: ZeRO-3 with ``overlap_comm`` and a
non-zero ``stage3_prefetch_bucket_size`` keeps parameter all-gathers running
asynchronously. When a checkpoint fires during the loop (e.g. the end-of-epoch
save transformers forces once ``global_step == max_steps``), a parameter can
still be in flight while ``GatheredParameters.__exit__`` re-partitions it,
raising::

    AssertionError: ... Cannot partition a param in flight

Draining the ZeRO-3 parameter coordinator first waits on every in-flight
all-gather handle and releases all params back to their partitioned state, so
the subsequent manual gather starts from a quiescent point. Training can resume
afterwards: the forward hooks re-fetch parameters lazily.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def _find_deepspeed_engine(obj: object) -> object | None:
    """
    Return the DeepSpeedEngine reachable from a trainer/model, or None.

    ``empty_partition_cache`` is only defined on the DeepSpeedEngine, so its
    presence doubles as the "is this a DeepSpeed engine" test.
    """
    for candidate in (
        obj,
        getattr(obj, "model_wrapped", None),
        getattr(obj, "deepspeed", None),
    ):
        if candidate is not None and hasattr(candidate, "empty_partition_cache"):
            return candidate
    return None


def drain_zero3_inflight(trainer_or_engine: object) -> bool:
    """
    Quiesce ZeRO-3 before a manual ``GatheredParameters`` save.

    Waits on any in-flight parameter all-gathers and releases all params, so the
    following adapter gather cannot trip the "Cannot partition a param in flight"
    assertion. A no-op (returns False) when this is not a ZeRO-3 engine -- e.g.
    ZeRO-1/2 or DeepSpeed disabled -- so callers can invoke it unconditionally.
    """
    engine = _find_deepspeed_engine(trainer_or_engine)
    if engine is None:
        return False
    try:
        # Delegates to the ZeRO-3 optimizer -> parameter_offload -> coordinator
        # release_and_reset_all, which pops and waits on every in-flight handle.
        # DeepSpeedEngine is untyped (object); presence checked via hasattr above.
        cast(Any, engine).empty_partition_cache()
        return True
    except Exception as e:  # defensive: never let the drain itself lose a save
        logger.warning("[ZeRO3] in-flight drain skipped (%s); proceeding with save", e)
        return False
