"""
Shared resolution of the correct transformers model class for a checkpoint.

Multimodal checkpoints (Gemma 3/4, Qwen3.5, SmolVLM, ...) keep their language
weights under ``model.language_model.*`` and must be loaded with
``AutoModelForImageTextToText``. Loading them with ``AutoModelForCausalLM``
maps only part of the weights, which yields a model that trains and generates
but produces garbage -- a failure mode that is easy to miss because nothing
raises.

Detection is driven by the checkpoint's own ``model_type`` checked against the
transformers registry (``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``) instead
of matching against model names. Newly supported architectures are therefore
picked up as soon as the installed transformers version knows about them, with
no changes here.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig, AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForImageTextToText

    IMAGE_TEXT_TO_TEXT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None
    IMAGE_TEXT_TO_TEXT_AVAILABLE = False
    logger.warning(
        "[ModelClassResolver] AutoModelForImageTextToText not available; "
        "multimodal checkpoints will fall back to AutoModelForCausalLM"
    )

try:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    _IMAGE_TEXT_TO_TEXT_MODEL_TYPES = frozenset(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys())
    _CAUSAL_LM_MODEL_TYPES = frozenset(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.keys())
except ImportError:  # pragma: no cover - depends on transformers version
    _IMAGE_TEXT_TO_TEXT_MODEL_TYPES = frozenset()
    _CAUSAL_LM_MODEL_TYPES = frozenset()
    logger.warning(
        "[ModelClassResolver] auto mappings unavailable; "
        "falling back to vision_config detection only"
    )


# Attribute names transformers uses for the text half of a multimodal model.
LANGUAGE_TOWER_NAMES = ("language_model", "text_model")

# Attribute names used for the module bridging an encoder into the text
# embedding space. Purely structural detection proved too loose (every text-side
# projection also emits hidden_size), so a name is required *and* the structural
# check must agree -- see find_bridge_module_paths.
BRIDGE_MODULE_NAMES = (
    "connector",
    "merger",
    "projector",
    "multi_modal_projector",
    "mm_projector",
    "modality_projection",
    "embed_vision",
    "embed_audio",
)


def find_language_tower_path(model: "PreTrainedModel") -> str | None:
    """
    Return the module path of the text tower, or None if there isn't one.

    Multimodal checkpoints put the text stack under e.g. ``model.language_model``
    and the other modalities in sibling subtrees (``vision_tower``,
    ``audio_tower``, ``embed_vision``, projectors, ...).
    """
    for name, _ in model.named_modules():
        if name and name.split(".")[-1] in LANGUAGE_TOWER_NAMES:
            return name
    return None


def build_non_language_exclude_pattern(model: "PreTrainedModel") -> str | None:
    """
    Regex matching every module *outside* the text tower.

    Used as peft's ``exclude_modules`` so LoRA only ever touches the language
    stack. Stated as "keep the text tower" rather than as a blocklist of
    modality names: Gemma 4 alone carries vision *and* audio towers plus two
    embedding bridges, and any hand-maintained list of such names goes stale
    with the next architecture.

    Returns None when the model has no separate text tower (text-only models),
    which leaves LoRA targeting untouched.
    """
    lang_path = find_language_tower_path(model)
    if not lang_path:
        return None
    # Ancestors of the text tower match too, but they are containers rather
    # than adaptable layers, so excluding them costs nothing.
    return rf"^(?!{re.escape(lang_path)}\.).*"


def is_outside_language_tower(module_name: str, lang_path: str | None) -> bool:
    """True when a module path sits outside the given text tower."""
    if not lang_path:
        return False
    return not module_name.startswith(lang_path + ".")


def find_bridge_module_paths(model: "PreTrainedModel", text_hidden_size: int | None) -> list:
    """
    Locate the projector(s) mapping encoder output into the text embedding space.

    The bridge is the small module that makes an image (or audio clip) look like
    text to the language model: it emits vectors of exactly ``text_hidden_size``,
    which then overwrite placeholder-token embeddings. It is the part worth
    training when the input domain is far from what the encoder was pretrained
    on, while the encoder itself stays frozen.

    Its name and depth vary by architecture -- ``connector`` (SmolVLM),
    ``embed_vision``/``embed_audio`` (Gemma 4, one per modality), or nested as
    ``visual.merger`` (Qwen3.5) -- so it is identified structurally rather than
    by name: a module outside the text tower holding a Linear that emits
    ``text_hidden_size``, with no deeper module inside it doing the same.

    Returns module paths, innermost-first. Empty when the model is text-only or
    the bridge cannot be identified.
    """
    import torch.nn as nn

    if not text_hidden_size:
        return []

    lang_path = find_language_tower_path(model)
    if not lang_path:
        # No separate text tower means no cross-modal bridge to train. Without
        # this guard every text-model projection emitting hidden_size (q_proj,
        # o_proj, down_proj, ...) would qualify.
        return []

    def _outside(name: str) -> bool:
        # The tower itself, not just its descendants, must be excluded.
        return name != lang_path and not name.startswith(lang_path + ".")

    def _emits_text_dim(module: nn.Module) -> bool:
        return any(
            isinstance(child, nn.Linear) and child.out_features == text_hidden_size
            for _, child in module.named_modules()
        )

    candidates = [
        name
        for name, module in model.named_modules()
        if name
        and _outside(name)
        and name.split(".")[-1] in BRIDGE_MODULE_NAMES
        and _emits_text_dim(module)
    ]

    # Keep the outermost hit per subtree: the bridge is a unit (projection plus
    # its norms/pooling), and descending to the final Linear alone would train
    # only part of it.
    return [n for n in candidates if not any(o != n and n.startswith(o + ".") for o in candidates)]


def load_model_config(
    model_name_or_path: str,
    token: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> "PretrainedConfig | None":
    """
    Load a checkpoint's config, returning None when it cannot be read.

    Never raises: callers treat an unreadable config as "assume text-only",
    which preserves the previous behaviour rather than failing the job.
    """
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path,
            token=token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as e:
        logger.warning(f"[ModelClassResolver] Could not read config for {model_name_or_path}: {e}")
        return None


def is_multimodal_config(config: "PretrainedConfig | None") -> bool:
    """True when the config describes an image-text-to-text architecture."""
    if config is None:
        return False

    model_type = getattr(config, "model_type", None)
    if model_type and model_type in _IMAGE_TEXT_TO_TEXT_MODEL_TYPES:
        return True

    # Secondary signal: architectures registered via trust_remote_code are
    # absent from the mapping, but a nested vision_config is a reliable tell.
    if getattr(config, "vision_config", None) is not None:
        return True

    return False


def is_multimodal_model(
    model_name_or_path: str,
    token: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> bool:
    """True when the checkpoint at the given path is multimodal."""
    config = load_model_config(
        model_name_or_path,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return is_multimodal_config(config)


def _read_checkpoint_keys(model_dir: str) -> set:
    """Read weight names from a local checkpoint without loading any tensors."""
    import json
    from pathlib import Path

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text()).get("weight_map", {}))
    single = d / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            return set(f.keys())
    return set()


def build_zero3_key_mapping(
    model_name_or_path: str,
    model_class: type | None = None,
    token: str | None = None,
    local_files_only: bool = True,
) -> dict[str, str] | None:
    """
    Return a key_mapping that works around a broken ZeRO-3 weight rename, or None.

    Some multimodal architectures (Qwen3.5) register a checkpoint rename meant
    for their *text-only* submodel -- ``^model.language_model.`` -> ``model.`` --
    that, under DeepSpeed ZeRO-3, is applied unconditionally to the
    ConditionalGeneration model too. There the language weights already live at
    ``model.language_model.*``, so the rename maps them to keys that do not
    exist and ~all language weights fail to bind (see the load probe). The
    normal (non-ZeRO-3) loader avoids this by matching against the real model
    keys; the ZeRO-3 path does not.

    The fix re-expresses the source keys as ``language_model.`` so the broken
    rename no longer matches, and the loader's prefix-restoration step puts the
    ``model.`` prefix back. Returns None when the checkpoint has no such rename,
    so unaffected architectures are left untouched.

    Detection reads the config's model types and the checkpoint's key names
    directly -- it never instantiates the model, because this runs while the
    ZeRO-3 ``zero.Init`` context is active, under which building even a meta
    model raises "Cannot copy out of meta tensor". ``model_class`` is accepted
    for API symmetry but unused.
    """
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
    except Exception as e:  # pragma: no cover - depends on transformers internals
        logger.warning(f"[ModelClassResolver] Cannot inspect conversion mapping: {e}")
        return None

    config = load_model_config(model_name_or_path, token=token, local_files_only=local_files_only)
    if config is None:
        return None

    # Collect the model types that contribute renames: the top-level one and any
    # nested text/vision sub-configs (the buggy rename is registered under the
    # text sub-model type, e.g. "qwen3_5_text").
    model_types = []
    for cfg in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        mt = getattr(cfg, "model_type", None)
        if mt and mt not in model_types:
            model_types.append(mt)

    checkpoint_keys = _read_checkpoint_keys(model_name_or_path)
    if not checkpoint_keys:
        return None

    for mt in model_types:
        try:
            conversions = get_checkpoint_conversion_mapping(mt) or []
        except Exception:
            continue
        for entry in conversions:
            if not isinstance(entry, WeightRenaming):
                continue
            for src in getattr(entry, "_original_source_patterns", []) or getattr(
                entry, "source_patterns", []
            ):
                if "language_model" not in src:
                    continue
                stripped = src.lstrip("^").rstrip("$")
                # The rename fires only if checkpoint keys actually carry this path.
                if any(stripped in k for k in checkpoint_keys):
                    mapping = {r"^model\.language_model\.": "language_model."}
                    logger.info(
                        f"[ModelClassResolver] Applying ZeRO-3 key_mapping workaround "
                        f"(rename '{src}' from model_type '{mt}' would unbind language "
                        f"weights): {mapping}"
                    )
                    return mapping
    return None


def resolve_model_class(
    model_name_or_path: str,
    token: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> tuple[Any, bool]:
    """
    Return ``(model_class, is_multimodal)`` for a checkpoint.

    ``AutoModelForCausalLM`` stays the default, including for multimodal
    checkpoints. For an architecture that registers both (Qwen3.5, Gemma 3/4)
    it resolves to a class that loads the text stack with every weight bound,
    which is what a text-dataset fine-tune wants -- and for Qwen3.5 it is also
    the only one that loads at all under DeepSpeed ZeRO-3, where the
    ConditionalGeneration variant leaves the language weights unbound and sends
    transformers into re-initializing ~64GB of them.

    ``AutoModelForImageTextToText`` is used only as a fallback, for
    architectures with no causal-LM mapping at all (Idefics3/SmolVLM), where
    ``AutoModelForCausalLM`` raises outright.

    Note the returned flag reports what the *checkpoint* is, not what the
    loaded class covers: a multimodal checkpoint loaded via CausalLM may expose
    no vision tower at all. Callers must inspect the built model rather than
    assume, which is what the LoRA targeting below does.
    """
    config = load_model_config(
        model_name_or_path,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    multimodal = is_multimodal_config(config)
    model_type = getattr(config, "model_type", None)

    if model_type and model_type in _CAUSAL_LM_MODEL_TYPES:
        return AutoModelForCausalLM, multimodal

    if multimodal and IMAGE_TEXT_TO_TEXT_AVAILABLE:
        logger.info(
            f"[ModelClassResolver] {model_name_or_path} (model_type={model_type}) "
            "has no causal-LM mapping; using AutoModelForImageTextToText"
        )
        return AutoModelForImageTextToText, True

    return AutoModelForCausalLM, multimodal
