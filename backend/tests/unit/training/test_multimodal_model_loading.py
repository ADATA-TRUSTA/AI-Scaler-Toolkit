"""Multimodal checkpoint handling in the training model loader.

Covers two things the text-only pipeline got wrong:
1. Multimodal checkpoints must load via AutoModelForImageTextToText. Loading
   e.g. Qwen3.5 with AutoModelForCausalLM resolves to a *different* class
   (Qwen3_5MoeForCausalLM vs Qwen3_5MoeForConditionalGeneration) and maps only
   part of the weights, with nothing raising.
2. LoRA must stay off the vision tower / projector. peft matches
   target_modules by suffix, so a bare "q_proj" otherwise also adapts the
   vision attention.
"""

import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import model_fixtures as mf  # noqa: E402
from service.utils.model_class_resolver import (  # noqa: E402
    build_non_language_exclude_pattern,
    find_bridge_module_paths,
    find_language_tower_path,
    is_multimodal_config,
    is_outside_language_tower,
    resolve_model_class,
)

pytestmark = pytest.mark.finetune


def _bf16_gpu_available() -> bool:
    """
    Whether TrainingArguments can be built with bf16 on this host.

    Asks transformers the same question it asks itself, so the skip condition cannot
    drift from the failure condition: with no bf16-capable device its __post_init__
    raises "Your setup doesn't support bf16/gpu".
    """
    try:
        from transformers.utils import is_torch_bf16_gpu_available

        return is_torch_bf16_gpu_available()
    except Exception:
        return False


# Only the end-to-end training test needs a GPU; the class-resolution tests above it
# are pure logic and must keep running on a CPU-only checkout.
requires_bf16_gpu = pytest.mark.skipif(
    not _bf16_gpu_available(),
    reason="no bf16-capable GPU; TrainingArguments cannot be constructed here",
)


@pytest.fixture(scope="module")
def text_path():
    try:
        _, path = mf.ensure_hf_model()
    except mf.ModelUnavailable as e:
        pytest.skip(f"text test model unavailable: {e}")
    return path


# --- pure logic, no model needed ------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        # Gemma 4 carries an audio tower and two embedding bridges alongside
        # vision; a blocklist of vision-only names would let these through.
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_1",
        "model.embed_vision.embedding_projection",
        "model.embed_audio.input_proj_linear",
        "model.multi_modal_projector.linear_1",
        "lm_head",
    ],
)
def test_modules_outside_text_tower_are_excluded(name):
    assert is_outside_language_tower(name, "model.language_model")


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.7.mlp.down_proj",
    ],
)
def test_text_tower_modules_are_kept(name):
    assert not is_outside_language_tower(name, "model.language_model")


def test_text_only_model_has_no_exclusions():
    """A model without a separate text tower must keep current behaviour."""
    assert is_outside_language_tower("model.layers.0.self_attn.q_proj", None) is False


# --- full-FT vision freeze (train the language tower only) ------------------


def _fake_multimodal_module():
    import torch.nn as nn

    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Linear(4, 4)
    root.model.vision_tower = nn.Linear(4, 4)
    root.model.multi_modal_projector = nn.Linear(4, 4)
    return root


def test_full_ft_freezes_vision_keeps_language():
    from service.training.core.model_loader import freeze_non_language_params

    m = _fake_multimodal_module()
    frozen = freeze_non_language_params(m)
    assert frozen > 0
    grads = {n: p.requires_grad for n, p in m.named_parameters()}
    # language tower stays trainable; vision / projector are frozen.
    assert grads["model.language_model.weight"] is True
    assert grads["model.vision_tower.weight"] is False
    assert grads["model.multi_modal_projector.weight"] is False


def test_full_ft_freeze_is_noop_on_text_only_model():
    """No separate language tower -> nothing frozen (train everything)."""
    import torch.nn as nn

    from service.training.core.model_loader import freeze_non_language_params

    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.Linear(4, 4)  # plain text model, no language_model subtree
    assert freeze_non_language_params(root) == 0
    assert all(p.requires_grad for p in root.parameters())


def test_unreadable_config_is_treated_as_text_only():
    # load_model_config returns None on failure; callers must not blow up.
    assert is_multimodal_config(None) is False


def test_config_with_vision_config_is_multimodal():
    # Covers trust_remote_code architectures absent from the auto mapping.
    class _Cfg:
        model_type = "some_custom_arch"
        vision_config = object()

    assert is_multimodal_config(_Cfg()) is True


# --- what the trainer processes examples with ------------------------------


class _FakeDataset:
    def __init__(self, *columns):
        self.column_names = list(columns)


class _FakeConfig:
    dataset_path = "train.jsonl"
    model_name = "some-model"

    def __init__(self, use_sft_trainer=True, packing=False):
        self.use_sft_trainer = use_sft_trainer
        self.packing = packing


def test_text_dataset_keeps_the_tokenizer():
    from service.training.core import select_processing_class

    tok, proc = object(), object()
    ds = _FakeDataset("prompt", "completion")
    assert select_processing_class(ds, tok, proc, _FakeConfig()) is tok


@pytest.mark.parametrize("image_column", ["images", "image"])
def test_image_dataset_switches_to_the_processor(image_column):
    from service.training.core import select_processing_class

    tok, proc = object(), object()
    ds = _FakeDataset(image_column, "prompt", "completion")
    assert select_processing_class(ds, tok, proc, _FakeConfig()) is proc


def test_images_without_a_processor_is_rejected():
    from service.training.core import select_processing_class

    ds = _FakeDataset("images", "messages")
    with pytest.raises(ValueError, match="no multimodal processor"):
        select_processing_class(ds, object(), None, _FakeConfig())


def test_images_with_causal_lm_path_is_rejected():
    """The Causal LM path tokenizes text only; images would be dropped silently."""
    from service.training.core import select_processing_class

    ds = _FakeDataset("images", "prompt", "completion")
    with pytest.raises(ValueError, match="SFT trainer"):
        select_processing_class(
            ds, object(), object(), _FakeConfig(use_sft_trainer=False)
        )


def test_images_with_packing_is_rejected():
    """TRL raises on packing for VLMs; fail here with the reason instead."""
    from service.training.core import select_processing_class

    ds = _FakeDataset("images", "prompt", "completion")
    with pytest.raises(ValueError, match="packing"):
        select_processing_class(ds, object(), object(), _FakeConfig(packing=True))


# --- images must actually reach the model ----------------------------------


class _FakeModelConfig:
    def __init__(self, image_token_id=None):
        if image_token_id is not None:
            self.image_token_id = image_token_id


class _FakeProcessor:
    image_token = "<|image|>"


def _collator_returning(ids):
    return lambda samples: {"input_ids": torch.tensor(ids)}


def test_missing_image_tokens_is_rejected():
    """The failure this guards is silent: pixel_values are produced and dropped."""
    from service.training.core.model_loader import verify_images_reach_the_model

    collator = _collator_returning([[1, 2, 3, 4]])  # no image token present
    with pytest.raises(ValueError, match=r"none of them reach the model"):
        verify_images_reach_the_model(
            collator, [{}], _FakeModelConfig(image_token_id=999), _FakeProcessor()
        )


def test_error_names_the_marker_the_user_must_add():
    from service.training.core.model_loader import verify_images_reach_the_model

    with pytest.raises(ValueError, match=re.escape("<|image|>")):
        verify_images_reach_the_model(
            _collator_returning([[1, 2]]),
            [{}],
            _FakeModelConfig(image_token_id=999),
            _FakeProcessor(),
        )


def test_present_image_tokens_pass():
    from service.training.core.model_loader import verify_images_reach_the_model

    collator = _collator_returning([[1, 999, 999, 4]])
    verify_images_reach_the_model(
        collator, [{}], _FakeModelConfig(image_token_id=999), _FakeProcessor()
    )


def test_model_without_image_token_id_is_skipped_not_failed():
    """No id to check against must not block training."""
    from service.training.core.model_loader import verify_images_reach_the_model

    verify_images_reach_the_model(
        _collator_returning([[1, 2]]), [{}], _FakeModelConfig(), _FakeProcessor()
    )


def test_collation_failure_does_not_mask_the_real_error():
    """Collation problems surface with better context during training itself."""
    from service.training.core.model_loader import verify_images_reach_the_model

    def _boom(_samples):
        raise RuntimeError("collator exploded")

    verify_images_reach_the_model(
        _boom, [{}], _FakeModelConfig(image_token_id=999), _FakeProcessor()
    )


@pytest.mark.parametrize("attr", ["image_token_id", "image_token_index"])
def test_both_image_token_attribute_spellings_are_read(attr):
    from service.training.core.model_loader import resolve_image_token_id

    config = type("C", (), {attr: 4242})()
    assert resolve_image_token_id(config) == 4242


# --- mm_token_type_ids alignment (TRL collator bug on Qwen3.5-VL) -----------


def test_mm_token_type_ids_realigned_to_input_ids():
    """TRL leaves mm_token_type_ids short for prompt/completion; realign it."""
    from service.training.core.model_loader import MMTokenTypeAlignmentCollator

    IMG = 999
    # input_ids has 8 tokens (2 image, rest text); inner returns a stale
    # mm_token_type_ids of length 5, as TRL does for the prompt-only part.
    def inner(_examples):
        return {
            "input_ids": torch.tensor([[1, IMG, IMG, 4, 5, 6, 7, 8]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]]),
        }

    out = MMTokenTypeAlignmentCollator(inner, image_token_id=IMG)([{}])
    assert out["mm_token_type_ids"].shape == out["input_ids"].shape
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0, 0, 0, 0]]


def test_alignment_is_idempotent_when_already_correct():
    from service.training.core.model_loader import MMTokenTypeAlignmentCollator

    IMG = 7

    def inner(_examples):
        return {
            "input_ids": torch.tensor([[1, IMG, 3]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        }

    out = MMTokenTypeAlignmentCollator(inner, image_token_id=IMG)([{}])
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 0]]


def test_alignment_marks_video_tokens_when_configured():
    from service.training.core.model_loader import MMTokenTypeAlignmentCollator

    def inner(_examples):
        return {
            "input_ids": torch.tensor([[5, 9, 3]]),  # 5=image, 9=video
            "mm_token_type_ids": torch.tensor([[1]]),
        }

    out = MMTokenTypeAlignmentCollator(inner, image_token_id=5, video_token_id=9)([{}])
    assert out["mm_token_type_ids"].tolist() == [[1, 2, 0]]


def test_alignment_noop_without_mm_token_type_ids():
    """Models that don't produce the field (SmolVLM) are left untouched."""
    from service.training.core.model_loader import MMTokenTypeAlignmentCollator

    def inner(_examples):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    out = MMTokenTypeAlignmentCollator(inner, image_token_id=2)([{}])
    assert "mm_token_type_ids" not in out


# --- resolver against real checkpoints -------------------------------------


def test_resolver_falls_back_when_no_causal_lm_mapping(vlm_path):
    """Idefics3/SmolVLM has no causal-LM mapping; AutoModelForCausalLM raises."""
    cls, multimodal = resolve_model_class(vlm_path)
    assert multimodal is True
    assert cls.__name__ == "AutoModelForImageTextToText"


def test_multimodal_checkpoint_with_causal_lm_mapping_stays_on_causal_lm(monkeypatch):
    """Qwen3.5 / Gemma 3-4 register both mappings and must keep using CausalLM.

    Switching them to AutoModelForImageTextToText regresses two things: it
    pulls in a vision tower a text fine-tune does not need, and for Qwen3.5 it
    breaks weight binding under DeepSpeed ZeRO-3 entirely -- transformers then
    tries to re-initialize ~64GB of language weights on the GPU and OOMs.
    """
    from service.utils import model_class_resolver as mcr

    class _Cfg:
        model_type = "qwen3_5_moe"  # registered in both auto mappings
        vision_config = object()

    monkeypatch.setattr(mcr, "load_model_config", lambda *a, **k: _Cfg())
    cls, multimodal = mcr.resolve_model_class("irrelevant/path")
    assert multimodal is True, "checkpoint is still reported as multimodal"
    assert cls.__name__ == "AutoModelForCausalLM"


def test_resolver_leaves_text_checkpoint_on_causal_lm(text_path):
    cls, multimodal = resolve_model_class(text_path)
    assert multimodal is False
    assert cls.__name__ == "AutoModelForCausalLM"


def test_enable_image_training_switches_class_for_causal_lm_multimodal(vlm_path, tmp_path):
    """A multimodal checkpoint loaded as CausalLM for text must switch for images."""
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    loader.enable_image_training()
    assert loader.model_class.__name__ == "AutoModelForImageTextToText"


def test_enable_image_training_rejects_text_only_model(text_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(text_path, tmp_path), None)
    with pytest.raises(ValueError, match="not multimodal"):
        loader.enable_image_training()


def test_zero3_key_mapping_only_triggers_for_affected_arch(vlm_path):
    """The workaround must be a no-op for architectures without the broken rename."""
    from transformers import AutoModelForImageTextToText

    from service.utils.model_class_resolver import build_zero3_key_mapping

    # SmolVLM/Idefics3 has no language_model rename -> no workaround.
    assert build_zero3_key_mapping(
        vlm_path, AutoModelForImageTextToText, local_files_only=True
    ) is None


# --- image pixel cap -------------------------------------------------------
#
# Qwen-VL sizes images by pixel *area* and is unbounded by default, so one photo
# can outweigh the whole rest of the dataset and OOM in backward. Idefics3 reuses
# the same `longest_edge` key for an *edge length*, so the cap must tell them
# apart or it would shrink those images to a handful of pixels.


def _fake_processor(*, merge_size, size):
    return SimpleNamespace(image_processor=SimpleNamespace(merge_size=merge_size, size=dict(size)))


def test_pixel_cap_applies_to_area_based_processor():
    from service.training.core.model_loader import cap_image_pixels

    proc = _fake_processor(merge_size=2, size={"longest_edge": 16777216, "shortest_edge": 65536})
    cap_image_pixels(proc, 768 * 768)

    assert proc.image_processor.size == {"longest_edge": 589824, "shortest_edge": 65536}


def test_pixel_cap_keeps_the_floor_under_the_ceiling():
    """A cap below the existing minimum must drag the minimum down, not invert it."""
    from service.training.core.model_loader import cap_image_pixels

    proc = _fake_processor(merge_size=2, size={"longest_edge": 16777216, "shortest_edge": 65536})
    cap_image_pixels(proc, 32 * 32)

    assert proc.image_processor.size == {"longest_edge": 1024, "shortest_edge": 1024}


def test_pixel_cap_never_upscales():
    from service.training.core.model_loader import cap_image_pixels

    proc = _fake_processor(merge_size=2, size={"longest_edge": 262144})
    cap_image_pixels(proc, 1024 * 1024)

    assert proc.image_processor.size == {"longest_edge": 262144}


@pytest.mark.parametrize("max_pixels", [None, 0])
def test_pixel_cap_unset_is_a_noop(max_pixels):
    from service.training.core.model_loader import cap_image_pixels

    proc = _fake_processor(merge_size=2, size={"longest_edge": 16777216})
    cap_image_pixels(proc, max_pixels)

    assert proc.image_processor.size == {"longest_edge": 16777216}


def test_pixel_cap_skips_edge_based_processor():
    """Idefics3's longest_edge is an edge length; writing an area there ruins it."""
    from service.training.core.model_loader import cap_image_pixels

    proc = _fake_processor(merge_size=None, size={"longest_edge": 2048})
    cap_image_pixels(proc, 768 * 768)

    assert proc.image_processor.size == {"longest_edge": 2048}


def test_pixel_cap_leaves_a_real_fixed_resolution_processor_alone(vlm_path):
    """The discriminator has to hold against the real SmolVLM processor."""
    from transformers import AutoProcessor

    from service.training.core.model_loader import cap_image_pixels

    processor = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)
    before = dict(processor.image_processor.size)

    cap_image_pixels(processor, 768 * 768)

    assert dict(processor.image_processor.size) == before


# --- checkpoint key lookup -------------------------------------------------
#
# The ZeRO-3 workaround is gated on reading the checkpoint's key names. Models
# are usually requested by repo id, not by directory, so a lookup that only
# understands directories silently disables the workaround and the language
# weights are re-initialised instead of loaded.


def _write_index(directory: Path, keys: tuple[str, ...]) -> Path:
    """Write a minimal safetensors index listing ``keys``."""
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": dict.fromkeys(keys, "model-00001.safetensors")}))
    return index


def test_read_checkpoint_keys_from_local_directory(tmp_path):
    from service.utils.model_class_resolver import _read_checkpoint_keys

    _write_index(tmp_path, ("model.language_model.layers.0.mlp.down_proj.weight",))

    assert _read_checkpoint_keys(str(tmp_path)) == {
        "model.language_model.layers.0.mlp.down_proj.weight"
    }


def test_read_checkpoint_keys_from_repo_id_in_hub_cache(tmp_path, monkeypatch):
    """A repo id must resolve through the hub cache, not be treated as a directory."""
    import huggingface_hub

    from service.utils.model_class_resolver import _read_checkpoint_keys

    index = _write_index(tmp_path, ("model.language_model.embed_tokens.weight",))

    def fake_try_to_load_from_cache(repo_id, filename, **kwargs):
        if repo_id == "Qwen/Qwen3.5-35B-A3B" and filename == "model.safetensors.index.json":
            return str(index)
        return None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_try_to_load_from_cache)

    assert _read_checkpoint_keys("Qwen/Qwen3.5-35B-A3B") == {
        "model.language_model.embed_tokens.weight"
    }


def test_read_checkpoint_keys_empty_for_uncached_repo_id(monkeypatch):
    import huggingface_hub

    from service.utils.model_class_resolver import _read_checkpoint_keys

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo_id, filename, **kwargs: None
    )

    assert _read_checkpoint_keys("some-org/never-downloaded") == set()


QWEN_CHECKPOINT_KEYS = {
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "lm_head.weight",
}


def _fake_transformers_rename(monkeypatch, source_pattern, *, scoped):
    """
    Stand in for the installed transformers conversion registry.

    ``scoped`` mirrors the upstream fix: transformers >=5.15 scopes each
    sub-model's transforms via ``WeightRenaming.scope_prefix``, which is what
    makes the workaround unnecessary (and harmful) there.
    """
    import transformers.conversion_mapping as cm
    import transformers.core_model_loading as cml
    from service.utils import model_class_resolver

    class FakeWeightRenaming:
        if scoped:
            scope_prefix = ""

        def __init__(self, patterns):
            self._original_source_patterns = patterns

    entry = FakeWeightRenaming([source_pattern])
    monkeypatch.setattr(cml, "WeightRenaming", FakeWeightRenaming)
    monkeypatch.setattr(
        cm,
        "get_checkpoint_conversion_mapping",
        lambda mt: [entry] if mt == "qwen3_5_moe_text" else [],
    )
    monkeypatch.setattr(
        model_class_resolver,
        "load_model_config",
        lambda *a, **kw: SimpleNamespace(
            model_type="qwen3_5_moe", text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
        ),
    )
    monkeypatch.setattr(
        model_class_resolver, "_read_checkpoint_keys", lambda _: QWEN_CHECKPOINT_KEYS
    )


@pytest.mark.parametrize(
    "source_pattern",
    [
        "^model.language_model.",  # transformers 5.5.x spelling
        r"^model\.language_model\.(.+)$",  # transformers 5.15.x spelling
    ],
)
def test_zero3_workaround_fires_on_unscoped_transformers(monkeypatch, source_pattern):
    """
    The source pattern is a regex; both spellings must be recognised.

    Matching it as a literal substring silently disabled the workaround when
    transformers changed the spelling, which costs a ZeRO-3 run its language
    weights.
    """
    from service.utils.model_class_resolver import build_zero3_key_mapping

    _fake_transformers_rename(monkeypatch, source_pattern, scoped=False)

    assert build_zero3_key_mapping("Qwen/Qwen3.5-35B-A3B") == {
        r"^model\.language_model\.": "language_model."
    }


def test_zero3_workaround_skipped_once_transformers_scopes_renames(monkeypatch):
    """Scoped renames cannot reach the composite model, so re-adding one is wrong."""
    from service.utils.model_class_resolver import build_zero3_key_mapping

    _fake_transformers_rename(monkeypatch, r"^model\.language_model\.(.+)$", scoped=True)

    assert build_zero3_key_mapping("Qwen/Qwen3.5-35B-A3B") is None


def test_unreadable_checkpoint_keys_warn_instead_of_silently_skipping(monkeypatch, caplog):
    """A skipped workaround must be visible; silence reads as 'no rename needed'."""
    from service.utils import model_class_resolver

    # Everything resolves except the key names, on a transformers that needs the
    # workaround -- so the skip is a real loss, not a no-op.
    _fake_transformers_rename(monkeypatch, "^model.language_model.", scoped=False)
    monkeypatch.setattr(model_class_resolver, "_read_checkpoint_keys", lambda _: set())

    with caplog.at_level(logging.WARNING, logger=model_class_resolver.__name__):
        assert model_class_resolver.build_zero3_key_mapping("Qwen/Qwen3.5-35B-A3B") is None

    assert "Could not read checkpoint keys" in caplog.text


# --- model loader ----------------------------------------------------------


def _training_config(model_path, tmp_path, **overrides):
    from service.config_models import TrainingConfig

    kwargs = dict(
        model_name=model_path,
        method="lora",
        dataset_path=str(tmp_path / "train.jsonl"),
        output_dir=str(tmp_path / "out"),
    )
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


def test_loader_picks_image_text_to_text_class(vlm_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    assert loader.is_multimodal is True
    assert loader.model_class.__name__ == "AutoModelForImageTextToText"


def test_lora_stays_inside_the_text_tower(vlm_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    model = loader.load_model()

    lang_path = find_language_tower_path(model.base_model.model)
    assert lang_path, "expected a separate text tower on a multimodal checkpoint"

    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    assert adapted, "expected LoRA adapters to be installed"
    # Adapter paths carry peft's "base_model.model." prefix.
    stripped = [n.split("base_model.model.", 1)[-1] for n in adapted]
    assert not [n for n in stripped if is_outside_language_tower(n, lang_path)]


def test_suffix_matching_alone_would_escape_the_text_tower(vlm_path, tmp_path):
    """Guards the reason exclude_modules exists.

    Projection names are shared between the text tower and the other modality
    towers, so peft's suffix matching reaches outside the text stack unless
    exclude_modules confines it. On Gemma 4 this is not merely undesirable: the
    vision and audio towers wrap their linears in Gemma4ClippableLinear, which
    peft cannot adapt at all, so the injection raises.
    """
    from peft import LoraConfig, get_peft_model

    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    base = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    lang_path = find_language_tower_path(base)

    naive = get_peft_model(
        base,
        LoraConfig(task_type="CAUSAL_LM", r=4, target_modules=["q_proj", "v_proj"]),
    )
    adapted = [
        n.split("base_model.model.", 1)[-1]
        for n, _ in naive.named_modules()
        if n.endswith("lora_A.default")
    ]
    assert [n for n in adapted if is_outside_language_tower(n, lang_path)], (
        "modality towers no longer share projection names with the text tower"
    )


def test_bridge_detection_on_multimodal_checkpoint(vlm_path, tmp_path):
    from transformers import AutoConfig

    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    model = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    config = AutoConfig.from_pretrained(vlm_path, local_files_only=True)
    text_hidden = getattr(getattr(config, "text_config", config), "hidden_size", None)

    bridges = find_bridge_module_paths(model, text_hidden)
    assert bridges, "expected a bridge on a multimodal checkpoint"
    for path in bridges:
        # Resolvable, outside the text tower, and a module rather than a Linear.
        module = model.get_submodule(path)
        assert not isinstance(module, torch.nn.Linear), (
            f"{path} is a bare Linear; the surrounding bridge module was expected"
        )
        assert is_outside_language_tower(path, find_language_tower_path(model))


def test_no_bridge_on_text_only_model(text_path, tmp_path):
    """Guards the failure this check originally had.

    With no text tower there is no bridge, but a purely structural rule matched
    every text-side projection emitting hidden_size (q_proj, o_proj, ...).
    """
    from transformers import AutoConfig

    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(text_path, tmp_path), None)
    model = loader.model_class.from_pretrained(text_path, local_files_only=True)
    hidden = AutoConfig.from_pretrained(text_path, local_files_only=True).hidden_size

    assert find_bridge_module_paths(model, hidden) == []


def test_bridge_detection_without_hidden_size_is_empty(vlm_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    model = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    assert find_bridge_module_paths(model, None) == []


def test_exclude_pattern_is_none_for_text_only_models(text_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(text_path, tmp_path), None)
    base = loader.model_class.from_pretrained(text_path, local_files_only=True)
    assert build_non_language_exclude_pattern(base) is None


# --- multimodal_scope ------------------------------------------------------
#
# text_and_bridge has to survive exclude_modules, which matches the bridge too
# (it is outside the text tower). If peft ever let the exclusion win, the run
# would train nothing extra and report success -- so assert on the trainable
# parameters rather than on the config we asked for.


def _bridge_paths_for(loader, model):
    from service.utils.model_class_resolver import resolve_text_hidden_size

    return find_bridge_module_paths(model, resolve_text_hidden_size(model.config))


def test_scope_b_makes_the_bridge_trainable(vlm_path, tmp_path):
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(
        _training_config(vlm_path, tmp_path, multimodal_scope="text_and_bridge"), None
    )
    base = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    bridges = _bridge_paths_for(loader, base)
    assert bridges, "fixture model has no bridge; the test cannot say anything"

    peft_model = loader._apply_lora(base)
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}

    for path in bridges:
        assert any(path in n and "modules_to_save" in n for n in trainable), (
            f"bridge {path} is not trainable; exclude_modules likely swallowed modules_to_save"
        )


def test_scope_a_leaves_the_bridge_frozen(vlm_path, tmp_path):
    """The default must not start training the vision half by accident."""
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(vlm_path, tmp_path), None)
    base = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    bridges = _bridge_paths_for(loader, base)

    peft_model = loader._apply_lora(base)
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}

    for path in bridges:
        assert not any(path in n for n in trainable), (
            f"bridge {path} trains under the default scope"
        )


def test_scope_b_refuses_a_checkpoint_with_no_bridge(text_path, tmp_path):
    """Falling back to text_only would burn a whole run on the wrong scope."""
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(
        _training_config(text_path, tmp_path, multimodal_scope="text_and_bridge"), None
    )
    base = loader.model_class.from_pretrained(text_path, local_files_only=True)
    with pytest.raises(ValueError, match="no cross-modal bridge"):
        loader._apply_lora(base)


def test_full_finetune_scope_b_keeps_the_bridge_trainable(vlm_path, tmp_path):
    """The freeze for full fine-tunes has to honour the same scope as LoRA."""
    from service.training.core.model_loader import freeze_non_language_params

    from service.training.core.model_loader import ModelLoader  # noqa: I001

    loader = ModelLoader(_training_config(vlm_path, tmp_path, method="full"), None)
    model = loader.model_class.from_pretrained(vlm_path, local_files_only=True)
    bridges = _bridge_paths_for(loader, model)
    assert bridges

    freeze_non_language_params(model, keep_paths=bridges)

    bridge_params = [
        (n, p) for n, p in model.named_parameters() if any(n.startswith(b + ".") for b in bridges)
    ]
    assert bridge_params
    assert all(p.requires_grad for _, p in bridge_params)

    # The output head is a SIBLING of the language tower (`lm_head` next to
    # `model`), so "outside the tower" catches it -- but it belongs to the
    # language model and a full fine-tune must train it. SmolVLM does not tie
    # its embeddings, so this was a live freeze, not a latent one: the run
    # trained everything except the head it was learning.
    assert model.lm_head.weight.requires_grad, "a full fine-tune must train the output head"

    # The encoder behind the bridge must still be frozen.
    lang = find_language_tower_path(model)
    head_names = {n for n, p in model.named_parameters() if p is model.lm_head.weight}
    encoder = [
        p
        for n, p in model.named_parameters()
        if is_outside_language_tower(n, lang)
        and not any(n.startswith(b + ".") for b in bridges)
        and n not in head_names
    ]
    assert encoder and not any(p.requires_grad for p in encoder)


@requires_bf16_gpu
def test_multimodal_checkpoint_trains_end_to_end(vlm_path, tmp_path, measure_memory):
    """LoRA fine-tune of a multimodal checkpoint on a text dataset.

    Takes `measure_memory` so the one test that actually trains records avg and
    peak GPU/DRAM/SSD; the summary lands in tests/results/memory/latest.jsonl.

    Deliberately stops short of save_training_results(): that helper writes to
    the hardcoded service/configs/ paths, so calling it from a test corrupts
    the developer's real registry and last_finetuned_model.json.
    """
    import json

    from service.training.core.dataset_loader import load_training_dataset
    from service.training.core.model_loader import ModelLoader
    from service.training.core.strategies import StrategyFactory

    ds_path = tmp_path / "train.jsonl"
    ds_path.write_text(
        "\n".join(
            json.dumps({"prompt": f"Q{i}: what is {i}+{i}?", "completion": f"A: {i + i}."})
            for i in range(8)
        ),
        encoding="utf-8",
    )

    config = _training_config(
        vlm_path,
        tmp_path,
        prompt_field="prompt",
        completion_field="completion",
        text_field=None,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=64,
        warmup_steps=0,
        use_deepspeed=False,
        gradient_checkpointing=False,
    )

    loader = ModelLoader(config, None)
    tokenizer = loader.load_tokenizer()
    dataset = load_training_dataset(str(ds_path))
    strategy = StrategyFactory.get_strategy(config, None)
    dataset = strategy.preprocess_dataset(dataset, tokenizer)
    trainer = strategy.prepare_trainer(loader.load_model(), tokenizer, dataset)

    result = trainer.train()
    assert result.global_step > 0
    assert result.training_loss > 0

    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_text_checkpoint_lora_is_unchanged(text_path, tmp_path):
    """The multimodal branch must not alter behaviour for text-only models."""
    from service.training.core.model_loader import ModelLoader

    loader = ModelLoader(_training_config(text_path, tmp_path), None)
    assert loader.is_multimodal is False

    model = loader.load_model()
    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    assert adapted
    assert model.peft_config["default"].exclude_modules is None
