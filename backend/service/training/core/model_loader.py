"""Model, tokenizer, and processor loading for text and multimodal training."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ...config_models import TrainingConfig, TrainingMethod
from ...utils.model_class_resolver import (
    build_non_language_exclude_pattern,
    build_zero3_key_mapping,
    find_language_tower_path,
    is_outside_language_tower,
    resolve_model_class,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

logger = logging.getLogger(__name__)


def dataset_has_images(dataset: "Dataset") -> bool:
    """True when the dataset carries an image column TRL recognises."""
    columns = getattr(dataset, "column_names", None) or []
    return any(field in columns for field in ("images", "image"))


def select_processing_class(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    processor: "ProcessorMixin | None",
    config: TrainingConfig,
) -> "PreTrainedTokenizerBase | ProcessorMixin":
    """
    Choose what the trainer processes examples with.

    TRL turns on its whole vision-language path from the *type* of this object
    (``ProcessorMixin`` -> image collator), so the processor is handed over only
    when the dataset actually has images. A text dataset keeps the plain
    tokenizer and behaves exactly as before.

    Raises ValueError with an actionable message when images are present but the
    run cannot support them, rather than letting it surface later as a collator
    KeyError or as images being silently dropped.
    """
    if not dataset_has_images(dataset):
        return tokenizer

    if processor is None:
        raise ValueError(
            f"Dataset '{getattr(config, 'dataset_path', '?')}' contains images but model "
            f"'{getattr(config, 'model_name', '?')}' has no multimodal processor. "
            "Use a vision-language checkpoint for image training."
        )
    if not getattr(config, "use_sft_trainer", True):
        raise ValueError(
            "Image training requires the SFT trainer (use_sft_trainer=true); the Causal LM "
            "path tokenizes text only and would drop the images."
        )
    if getattr(config, "packing", False):
        raise ValueError(
            "packing=true is not supported for vision-language training. Set packing=false."
        )

    logger.info(
        f"[ModelLoader] Vision-language training enabled (processor={type(processor).__name__})"
    )
    return processor


class MMTokenTypeAlignmentCollator:
    """
    Wrap a VLM collator so ``mm_token_type_ids`` matches ``input_ids`` length.

    TRL 0.26's ``DataCollatorForVisionLanguageModeling`` builds
    ``mm_token_type_ids`` from the prompt only. With prompt/completion data the
    completion tokens are then appended to ``input_ids`` but not to
    ``mm_token_type_ids``, leaving it shorter. Qwen3.5-VL's ``get_rope_index``
    indexes ``mm_token_type_ids`` by the attention mask and raises on the length
    mismatch; SmolVLM and Gemma 4 forward paths don't use it, so they were
    unaffected.

    ``mm_token_type_ids`` is exactly the multimodal-token mask (1 = image,
    2 = video, 0 = text), so it is recomputed from ``input_ids`` at full length.
    This is idempotent for already-aligned batches (the completion tokens are
    text -> 0), so it is safe to apply on every vision batch.
    """

    def __init__(
        self,
        inner: Callable[..., dict],
        image_token_id: int,
        video_token_id: int | None = None,
    ) -> None:
        self.inner = inner
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def __call__(self, examples: list) -> dict:
        """Recompute ``mm_token_type_ids`` from ``input_ids`` at full length."""
        batch = self.inner(examples)
        input_ids = batch.get("input_ids")
        if input_ids is not None and "mm_token_type_ids" in batch:
            mm = (input_ids == self.image_token_id).long()
            if self.video_token_id is not None:
                mm = mm + 2 * (input_ids == self.video_token_id).long()
            batch["mm_token_type_ids"] = mm
        return batch


def resolve_image_token_id(model_config: "PretrainedConfig") -> int | None:
    """Return the token id that stands in for image content, if the model has one."""
    for attr in ("image_token_id", "image_token_index"):
        value = getattr(model_config, attr, None)
        if isinstance(value, int):
            return value
    return None


def verify_images_reach_the_model(
    data_collator: Callable[..., dict],
    samples: list,
    model_config: "PretrainedConfig",
    processor: "ProcessorMixin",
) -> None:
    """
    Fail fast when images are processed but never enter the token sequence.

    Processors accept text containing no image placeholder without complaining:
    they still return ``pixel_values``, but ``input_ids`` holds no image tokens,
    so the model has nowhere to attach the visual features. Training then runs
    to completion with a falling loss while the model never sees the images --
    a silent failure that is far more expensive than an early error.

    Conversational datasets get the placeholder inserted by TRL. Plain-string
    prompts do not, which is the case for base checkpoints with no chat
    template, where the marker has to be written into the prompt by hand.
    """
    image_token_id = resolve_image_token_id(model_config)
    if image_token_id is None:
        logger.warning(
            "[ModelLoader] Model config exposes no image token id; skipping the "
            "check that images reach the token sequence"
        )
        return

    try:
        batch = data_collator(list(samples))
    except Exception as e:
        # Collation problems surface with better context during training itself.
        logger.warning(f"[ModelLoader] Could not pre-collate a batch to verify images: {e}")
        return

    input_ids = batch.get("input_ids")
    if input_ids is None:
        return
    if int((input_ids == image_token_id).sum()) > 0:
        return

    marker = getattr(processor, "image_token", None) or f"<token id {image_token_id}>"
    raise ValueError(
        "Dataset provides images but none of them reach the model: the collated "
        f"input_ids contain no image token ({marker}). The images are processed into "
        "pixel_values and then ignored, so training would silently learn nothing from "
        f"them. Add '{marker}' to the prompt text where the image belongs, or use a "
        "conversational prompt/completion dataset with a checkpoint that has a chat "
        "template."
    )


def freeze_non_language_params(model: "PreTrainedModel") -> int:
    """
    Freeze every parameter outside the language tower (vision/audio/projector).

    For full-parameter fine-tuning of a multimodal checkpoint, keep training
    confined to the language tower -- the same scope the LoRA path enforces via
    ``exclude_modules``. This keeps the vision half identical to the base so the
    base-derived mmproj stays faithful, and avoids overfitting the pretrained
    vision encoder on a small SFT set.

    A no-op (returns 0) when the model has no separate language tower -- e.g. a
    text-only model, or a multimodal checkpoint loaded as a plain Causal LM for
    text training -- so callers can invoke it unconditionally.
    """
    lang_path = find_language_tower_path(model)
    if not lang_path:
        return 0
    frozen = 0
    for name, param in model.named_parameters():
        if is_outside_language_tower(name, lang_path):
            if param.requires_grad:
                param.requires_grad = False
                frozen += 1
    return frozen


class ModelLoader:
    """Handles model and tokenizer loading for training."""

    def __init__(self, config: TrainingConfig, hf_token: str | None) -> None:
        self.config = config
        self.hf_token = hf_token
        # Resolved once so the model class and the LoRA targeting policy below
        # agree on whether this checkpoint is multimodal.
        self.model_class, self.is_multimodal = resolve_model_class(
            config.model_name,
            token=hf_token,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Extra kwargs (e.g. key_mapping) merged into from_pretrained; populated
        # by enable_image_training() when the vision tower is required.
        self._extra_load_kwargs: dict = {}

    def enable_image_training(self) -> None:
        """
        Force loading the full multimodal model (with vision tower) for images.

        For text training, an architecture that also has a causal-LM mapping
        (Qwen3.5, Gemma 3/4) is loaded through AutoModelForCausalLM, which for
        Qwen3.5 resolves to a text-only class with no vision tower. Image
        training needs the tower, so switch to AutoModelForImageTextToText and,
        where required, apply the ZeRO-3 key_mapping workaround so the language
        weights still bind under DeepSpeed.
        """
        if not self.is_multimodal:
            raise ValueError(
                f"Model '{self.config.model_name}' is not multimodal; it cannot be "
                "trained on images."
            )
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as e:
            raise ValueError(
                "Image training needs AutoModelForImageTextToText, unavailable in this "
                f"transformers version: {e}"
            ) from e
        if self.model_class is not AutoModelForImageTextToText:
            logger.info(
                f"[ModelLoader] Image training: switching load class from "
                f"{self.model_class.__name__} to AutoModelForImageTextToText"
            )
            self.model_class = AutoModelForImageTextToText

        key_mapping = build_zero3_key_mapping(
            self.config.model_name,
            self.model_class,
            token=self.hf_token,
            local_files_only=True,
        )
        if key_mapping:
            self._extra_load_kwargs["key_mapping"] = key_mapping

    def _get_attn_implementation(self) -> str | None:
        """
        Return the attention implementation to use based on config and available packages.

        Priority: flash_attention_2 (requires flash-attn) → sdpa (PyTorch built-in Triton kernel) → None.
        """
        if not getattr(self.config, "use_flash_attention", False):
            return None
        try:
            import flash_attn  # noqa: F401  # pyright: ignore[reportMissingImports]  # optional dep

            logger.info("[ModelLoader] flash-attn detected, using flash_attention_2")
            return "flash_attention_2"
        except ImportError:
            logger.info(
                "[ModelLoader] flash-attn not installed, falling back to sdpa (PyTorch Triton kernel)"
            )
            return "sdpa"

    def load_tokenizer(self) -> "PreTrainedTokenizerBase":
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=self.hf_token,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_processor(self) -> "ProcessorMixin | None":
        """
        Load the multimodal processor, or None when the model is text-only.

        The processor bundles the tokenizer with an image processor. TRL keys
        its whole vision-language path off ``isinstance(processing_class,
        ProcessorMixin)``, so returning this instead of a bare tokenizer is what
        enables image training; returning None keeps the text-only behaviour.
        """
        if not self.is_multimodal:
            return None
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                token=self.hf_token,
                local_files_only=True,
            )
        except Exception as e:
            logger.warning(
                f"[ModelLoader] Multimodal checkpoint but no usable AutoProcessor ({e}); "
                "image training will not be available for this model"
            )
            return None

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"[ModelLoader] Loaded processor {type(processor).__name__}")
        return processor

    def load_model(self) -> "PreTrainedModel | PeftModel":
        """Load model based on configuration (Full, LoRA, QLoRA)."""
        if self.config.method == TrainingMethod.QLORA:
            return self._prepare_qlora_model()
        else:
            return self._prepare_standard_model()

    def _prepare_standard_model(self) -> "PreTrainedModel | PeftModel":
        """Load standard model (Full or LoRA)."""
        attn_impl = self._get_attn_implementation()
        load_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        load_kwargs.update(self._extra_load_kwargs)

        try:
            base_model = self.model_class.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
        except (AttributeError, ValueError, TypeError) as e:
            # Fallback for custom models (like gpt-oss) where low_cpu_mem_usage causes meta-device initialization issues
            # specifically 'AttributeError: ... object has no attribute ...' during load_shard_file
            logger.warning(
                f"[ModelLoader] Fast loading failed ({e}). Retrying with low_cpu_mem_usage=False..."
            )
            load_kwargs["low_cpu_mem_usage"] = False
            base_model = self.model_class.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )

        base_model.config.use_cache = False

        if self.config.method == TrainingMethod.LORA:
            return self._apply_lora(base_model)
        else:
            # Full fine-tune: on a multimodal checkpoint, confine training to the
            # language tower (freeze vision/audio/projector) -- same scope as LoRA.
            # Keeps the base vision half intact so the base mmproj stays faithful,
            # and avoids wrecking the pretrained vision encoder on a small set.
            frozen = freeze_non_language_params(base_model)
            if frozen:
                logger.info(
                    f"[ModelLoader] Full multimodal fine-tune: froze {frozen} non-language "
                    "parameter tensors (vision/audio/projector); training the language tower only"
                )
            return base_model

    def _prepare_qlora_model(self) -> "PeftModel":
        """Create a 4-bit QLoRA model."""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        attn_impl = self._get_attn_implementation()
        qlora_kwargs = {
            "quantization_config": bnb_config,
            "token": self.hf_token,
            "dtype": torch.bfloat16,
            "device_map": None,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if attn_impl:
            qlora_kwargs["attn_implementation"] = attn_impl
        qlora_kwargs.update(self._extra_load_kwargs)

        logger.info("[ModelLoader] Loading 4-bit quantized base model for QLoRA...")
        base_model = self.model_class.from_pretrained(
            self.config.model_name,
            **qlora_kwargs,
        )

        base_model.config.use_cache = False

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(base_model)

        return self._apply_lora(model)

    def _find_linear_modules(self, model: "PreTrainedModel") -> list[str]:
        """
        Dynamically find all linear modules for LoRA targeting.

        Refs: https://github.com/artidoro/qlora/blob/main/qlora.py
        """

        # Add basic Linear
        cls_set = {torch.nn.Linear}

        # Add Quantized Linear types if available
        if self.config.method == TrainingMethod.QLORA:
            try:
                # bitsandbytes re-exports these at runtime; stubs mark them private.
                from bitsandbytes.nn import (
                    Linear4bit,  # pyright: ignore[reportPrivateImportUsage]
                )

                cls_set.add(Linear4bit)
            except ImportError:
                pass
            try:
                from bitsandbytes.nn import (
                    Linear8bitLt,  # pyright: ignore[reportPrivateImportUsage]
                )

                cls_set.add(Linear8bitLt)
            except ImportError:
                pass

        # Keyed off the built model, not self.is_multimodal: a multimodal
        # checkpoint loaded through AutoModelForCausalLM may expose only the
        # text stack, in which case there is no separate tower and nothing to
        # exclude. Text-only models return None here too.
        lang_path = find_language_tower_path(model)

        lora_module_names = set()
        skipped_non_language = 0
        for name, module in model.named_modules():
            if any(isinstance(module, cls) for cls in cls_set):
                # On multimodal checkpoints, only the text tower is adapted.
                # The other modality towers may also wrap their linears in
                # architecture-specific classes peft cannot target at all
                # (e.g. Gemma4ClippableLinear), so collecting their names would
                # both mis-target and crash adapter injection.
                if lang_path and is_outside_language_tower(name, lang_path):
                    skipped_non_language += 1
                    continue
                names = name.split(".")
                module_name = names[-1]
                # Avoid targeting the output head for stability unless explicitly requested
                if module_name != "lm_head":
                    lora_module_names.add(module_name)

        if skipped_non_language:
            logger.info(
                f"[ModelLoader] Skipped {skipped_non_language} linear modules outside "
                f"the text tower '{lang_path}' during LoRA target detection"
            )

        return list(lora_module_names)

    def _apply_lora(self, base_model: "PreTrainedModel") -> "PeftModel":
        """Apply LoRA adapters."""
        target_modules = self.config.lora_target_modules

        # If no target modules specified, try dynamic detection first, then fallback
        if not target_modules:
            target_modules = self._find_linear_modules(base_model)
            if not target_modules:
                # Fallback to defaults if dynamic detection finds nothing
                if self.config.method == TrainingMethod.QLORA:
                    target_modules = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                else:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            logger.info(f"[ModelLoader] Auto-detected LoRA target modules: {target_modules}")

        lora_kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "inference_mode": False,
        }

        # target_modules are matched by suffix, so a bare "q_proj" also matches
        # the vision and audio towers' attention. Filtering during detection is
        # therefore not enough on its own -- exclude_modules is what actually
        # confines LoRA to the text tower, and it also covers user-supplied
        # target_modules, which bypass detection entirely.
        exclude_pattern = build_non_language_exclude_pattern(base_model)
        if exclude_pattern:
            lora_kwargs["exclude_modules"] = exclude_pattern
            logger.info(
                "[ModelLoader] Model exposes a separate text tower: confining LoRA "
                f"to it via exclude pattern {exclude_pattern}"
            )

        lora_cfg = LoraConfig(**lora_kwargs)

        model = get_peft_model(base_model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        # get_peft_model returns PeftModel | PeftMixedModel; this path builds a
        # standard PeftModel (mixed adapters are not used here).
        return cast("PeftModel", model)
