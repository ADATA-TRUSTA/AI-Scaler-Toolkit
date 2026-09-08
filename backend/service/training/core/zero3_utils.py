"""
Helpers for training under DeepSpeed ZeRO-3.

Two independent ZeRO-3 hazards live here: buffers that the loader silently
drops (``restore_zero3_skipped_buffers``) and in-flight parameter gathers that
break a mid-loop save (``drain_zero3_inflight``).

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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _checkpoint_buffer_sources(model_name_or_path: str) -> "dict[str, Path]":
    """
    Map every checkpoint tensor name to the file holding it.

    Handles both the sharded layout (an index json naming each shard) and the
    single-file layout. Returns an empty mapping when the checkpoint cannot be
    located, which callers must treat as "could not verify", not as "clean".
    """
    import json

    from ...utils.model_class_resolver import _resolve_checkpoint_file

    # Sharded layouts first, safetensors before torch .bin in each case.
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _resolve_checkpoint_file(model_name_or_path, index_name)
        if index is None:
            continue
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        # Resolve each shard ONCE, not once per tensor -- a 14-shard checkpoint
        # names thousands of keys and every one hit the filesystem.
        shard_paths: dict[str, Path | None] = {}
        sources: dict[str, Path] = {}
        for key, shard_name in weight_map.items():
            if shard_name not in shard_paths:
                shard_paths[shard_name] = _resolve_checkpoint_file(model_name_or_path, shard_name)
            shard = shard_paths[shard_name]
            if shard is not None:
                sources[key] = shard
        return sources

    single = _resolve_checkpoint_file(model_name_or_path, "model.safetensors")
    if single is not None:
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            return dict.fromkeys(f.keys(), single)

    # Older checkpoints ship a single torch pickle. Read the keys without
    # materialising tensors; `_read_tensor` handles loading later.
    legacy = _resolve_checkpoint_file(model_name_or_path, "pytorch_model.bin")
    if legacy is not None:
        import torch

        state = torch.load(str(legacy), map_location="meta", weights_only=True)
        return dict.fromkeys(state.keys(), legacy)

    return {}


@contextmanager
def _open_checkpoint(path: "Path") -> "Iterator[Any]":
    """
    Uniform ``get_tensor(key)`` reader over a safetensors file or a torch pickle.

    safetensors reads one tensor at a time; a .bin has to be loaded whole, so it
    is wrapped to present the same interface rather than branching at each use.
    """
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            yield f
        return

    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    try:
        yield SimpleNamespace(get_tensor=lambda key: state[key])
    finally:
        state.clear()


def restore_zero3_skipped_buffers(
    model: "PreTrainedModel", model_name_or_path: str
) -> tuple[int, int]:
    """
    Re-load checkpoint buffers that the ZeRO-3 loader silently skipped.

    ``transformers.integrations.deepspeed._load_state_dict_into_zero3_model``
    copies a module's tensors only inside::

        params_to_gather = [p for k, p in module.named_parameters(recurse=False) if k in state_dict]
        if len(params_to_gather) > 0:
            with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                module._load_from_state_dict(*args)

    ``_load_from_state_dict`` is the only call that copies a module's **buffers**,
    and it is gated on that module owning at least one **direct parameter**. Any
    module whose weights live in a child but which owns buffers directly never
    gets its buffers loaded -- silently, and with no missing-keys warning.

    On gemma-4-E4B that is 970 tensors: 928 benign ``Gemma4ClippableLinear``
    clip bounds (whose ``±inf`` defaults make clamping a no-op) and, fatally,
    the 42 ``Gemma4TextDecoderLayer.layer_scalar`` values. Those default to
    ``torch.ones(1)`` while the real values run 0.061-0.887, so every decoder
    layer contributes far too much to the residual stream and the model is
    wrecked before the first step. A LoRA run then spends its epochs learning to
    compensate for the damage, producing an adapter that is meaningless once the
    model is loaded correctly at inference.

    Rather than reproducing the loader's gate -- which would drift as upstream
    changes -- this restores *every* checkpoint-backed buffer. Buffers the
    loader already handled are rewritten with the identical value, so the
    operation is idempotent and safe to call unconditionally. ZeRO-3 partitions
    parameters but never buffers, so a plain copy is correct on every rank.

    Returns ``(restored, corrected)``: how many buffers were written, and how
    many of those actually held a wrong value -- ``corrected > 0`` means the
    upstream bug was live for this checkpoint.
    """
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        return (0, 0)

    import torch

    owned: dict[str, tuple[Any, str, torch.Tensor]] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                owned[f"{prefix}{buffer_name}"] = (module, buffer_name, buffer)
    if not owned:
        return (0, 0)

    sources = _checkpoint_buffer_sources(model_name_or_path)
    if not sources:
        # Silently skipping is how this bug cost days in the first place.
        logger.warning(
            f"[ZeRO3] Could not read checkpoint tensors for '{model_name_or_path}'; "
            f"{len(owned)} buffers were left as loaded. If this architecture keeps "
            "buffers on parameter-less modules (gemma-4 does), they are wrong and "
            "training will optimise against a corrupted model."
        )
        return (0, 0)

    by_file: dict[Path, list[str]] = {}
    for key in owned:
        src = sources.get(key)
        if src is not None:
            by_file.setdefault(src, []).append(key)

    # A buffer name that matches no checkpoint key is NOT the same thing as a
    # buffer that needed no restoring, and conflating them is how the original
    # bug hid. transformers v5 rewrites checkpoint keys via WeightRenaming, and
    # the image path adds its own key_mapping, so a rename this function does
    # not know about would leave every buffer unmatched -- while a naive
    # "corrected == 0" reads like a clean bill of health.
    matched = sum(len(keys) for keys in by_file.values())
    if matched == 0:
        logger.warning(
            f"[ZeRO3] None of this model's {len(owned)} buffers matched any of the "
            f"{len(sources)} checkpoint tensor names. Buffers were NOT verified -- most "
            "likely the checkpoint keys are renamed on load. If this architecture keeps "
            "buffers on parameter-less modules (gemma-4 does), they are still wrong."
        )
        return (0, 0)

    restored = corrected = 0
    for path, keys in by_file.items():
        with _open_checkpoint(path) as f:
            for key in keys:
                module, buffer_name, buffer = owned[key]
                reference = f.get_tensor(key)
                if buffer.is_meta:
                    # Never materialised; copy_ would fail, so rebind instead.
                    module.register_buffer(
                        buffer_name, reference.clone().to(dtype=buffer.dtype), persistent=True
                    )
                    restored += 1
                    corrected += 1
                    continue
                target = reference.to(dtype=buffer.dtype, device=buffer.device)
                if buffer.shape != target.shape:
                    logger.warning(
                        f"[ZeRO3] Shape mismatch for buffer '{key}': model {tuple(buffer.shape)} "
                        f"vs checkpoint {tuple(target.shape)}; leaving it alone."
                    )
                    continue
                if not torch.equal(buffer, target):
                    corrected += 1
                buffer.copy_(target)
                restored += 1

    if corrected:
        logger.warning(
            f"[ZeRO3] Restored {corrected} of {restored} checkpoint buffers that the ZeRO-3 "
            f"loader had skipped (e.g. gemma-4's per-layer 'layer_scalar'). Training would "
            f"otherwise have run against a corrupted model."
        )
    else:
        logger.info(
            f"[ZeRO3] Verified {restored} of {len(owned)} model buffers against the checkpoint; "
            "none needed restoring."
        )
    return (restored, corrected)


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
