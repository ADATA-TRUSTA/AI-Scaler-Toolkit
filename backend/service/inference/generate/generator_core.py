"""
Generator Core - Core utilities for text generation
Contains parameter validation, tokenization helpers, and common generation logic
"""
import math
import torch
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from io import BytesIO
import base64
import requests
from transformers import AutoTokenizer
from PIL import Image

logger = logging.getLogger(__name__)


def _convert_messages_to_text_parts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert plain string chat content into multimodal text parts format."""
    converted_messages: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            converted_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": content}],
            })
        else:
            converted_messages.append(msg)
    return converted_messages


def _extract_image_url_from_part(part: Dict[str, Any]) -> Optional[str]:
    """Extract normalized image source from a multimodal content part."""
    if not isinstance(part, dict):
        return None

    part_type = str(part.get("type", "")).strip().lower()
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
        else:
            url = image_url
        if isinstance(url, str) and url.strip():
            return url.strip()

    if part_type == "image":
        image_value = part.get("image")
        if isinstance(image_value, str) and image_value.strip():
            return image_value.strip()

    return None


def _collect_message_image_sources(messages: List[Dict[str, Any]]) -> List[str]:
    """Collect image sources from message content parts in traversal order."""
    image_sources: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            image_src = _extract_image_url_from_part(part)
            if image_src:
                image_sources.append(image_src)
    return image_sources


def _build_messages_with_loaded_images(
    messages: List[Dict[str, Any]],
    pil_images: List[Image.Image],
    *,
    embed_images: bool,
) -> List[Dict[str, Any]]:
    """Replace image_url parts with embedded images or placeholders, preserving order."""
    normalized = _convert_messages_to_text_parts(messages)
    if not pil_images:
        return normalized

    converted: List[Dict[str, Any]] = []
    image_index = 0
    last_user_idx = -1

    for msg in normalized:
        role = msg.get("role", "user")
        if role == "user":
            last_user_idx = len(converted)

        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        new_parts: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                image_src = _extract_image_url_from_part(part)
                if image_src is not None:
                    if image_index >= len(pil_images):
                        raise ValueError("Image placeholder count exceeds loaded images")
                    if embed_images:
                        new_parts.append({"type": "image", "image": pil_images[image_index]})
                    else:
                        new_parts.append({"type": "image"})
                    image_index += 1
                    continue

                part_type = str(part.get("type", "")).strip().lower()
                if part_type == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        new_parts.append({"type": "text", "text": text})
                else:
                    new_parts.append(part)

        converted.append({"role": role, "content": new_parts})

    extra_images = pil_images[image_index:]
    if extra_images and last_user_idx >= 0:
        extra_parts = (
            [{"type": "image", "image": img} for img in extra_images]
            if embed_images
            else [{"type": "image"} for _ in extra_images]
        )
        existing_content = converted[last_user_idx].get("content", [])
        if isinstance(existing_content, list):
            converted[last_user_idx]["content"] = extra_parts + existing_content
        else:
            converted[last_user_idx]["content"] = extra_parts

    return converted


def _inject_image_placeholders_into_messages(
    messages: List[Dict[str, Any]],
    num_images: int,
) -> List[Dict[str, Any]]:
    """Normalise all message content to list-of-parts and inject
    ``{"type": "image"}`` placeholders into the last user message.
    """
    # Step 1: normalise ALL messages – string content → [{"type": "text", ...}]
    normalised = _convert_messages_to_text_parts(messages)

    if num_images <= 0:
        return normalised

    # Step 2: inject image placeholders before the last user message's text
    last_user_idx = -1
    for i, msg in enumerate(normalised):
        if msg.get("role") == "user":
            last_user_idx = i

    if last_user_idx < 0:
        return normalised

    injected = list(normalised)
    last_user = dict(injected[last_user_idx])
    content = last_user.get("content", [])
    image_parts = [{"type": "image"} for _ in range(num_images)]

    if isinstance(content, list):
        has_image_part = any(
            isinstance(p, dict) and p.get("type") in {"image", "image_url"}
            for p in content
        )
        if not has_image_part:
            last_user["content"] = image_parts + list(content)
    else:
        last_user["content"] = image_parts

    injected[last_user_idx] = last_user
    return injected


def _build_messages_with_embedded_images(
    messages: List[Dict[str, Any]],
    pil_images: list,
) -> List[Dict[str, Any]]:
    """Build messages where PIL Image objects are embedded directly into the
    last user message content as ``{"type": "image", "image": <PIL.Image>}``.

    This is the *official* Gemma 4 API (HF model card) and avoids the
    ``images=`` kwarg to ``apply_chat_template`` which triggers
    ``string indices must be integers, not 'str'`` inside the Jinja template.
    """
    if not pil_images:
        return _convert_messages_to_text_parts(messages)

    # First normalise all string content to list-of-parts
    normalised = _convert_messages_to_text_parts(messages)

    last_user_idx = -1
    for i, msg in enumerate(normalised):
        if msg.get("role") == "user":
            last_user_idx = i

    if last_user_idx < 0:
        return normalised

    injected = list(normalised)
    last_user = dict(injected[last_user_idx])
    content = last_user.get("content", [])
    # Embed PIL Image objects directly – processor reads them from here
    image_parts = [{"type": "image", "image": img} for img in pil_images]

    if isinstance(content, list):
        has_image_part = any(
            isinstance(p, dict) and p.get("type") in {"image", "image_url"}
            for p in content
        )
        if not has_image_part:
            last_user["content"] = image_parts + list(content)
    else:
        last_user["content"] = image_parts

    injected[last_user_idx] = last_user
    return injected


def _apply_processor_chat_template_with_fallback(
    processor,
    prompt: List[Dict[str, Any]],
    *,
    tokenize: bool,
    extra_kwargs: Dict[str, Any],
    model_device,
    max_length: int,
):
    """Apply processor chat template with content-format fallback.

    Some text-only processors expect plain string `content`, while some multimodal
    processors require OpenAI-style content parts. Try the original payload first,
    then fallback to text parts only if needed.
    """
    variants = [("original", prompt)]
    converted_prompt = _convert_messages_to_text_parts(prompt)
    if converted_prompt != prompt:
        variants.append(("text_parts", converted_prompt))

    last_error: Optional[Exception] = None
    for variant_name, candidate_prompt in variants:
        try:
            if tokenize:
                # 不傳 processor_kwargs 以避免 Gemma 4 等 processor 的 kwarg 警告；
                # truncation/max_length 對 apply_chat_template 的文字路徑影響有限。
                return processor.apply_chat_template(
                    candidate_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    **extra_kwargs,
                ).to(model_device)

            return processor.apply_chat_template(
                candidate_prompt,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except Exception as exc:
            last_error = exc
            logger.debug(
                "[Generator] processor.apply_chat_template failed with %s variant: %s",
                variant_name,
                exc,
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError("processor.apply_chat_template failed without a captured exception")


def _safe_float_param(
    params: Dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Parse float generation param safely, rejecting NaN/Inf and invalid values."""
    raw_value = params.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "[Generator] Invalid float param %s=%r, fallback to default=%s",
            key,
            raw_value,
            default,
        )
        return default

    if not math.isfinite(value):
        logger.warning(
            "[Generator] Non-finite float param %s=%r, fallback to default=%s",
            key,
            raw_value,
            default,
        )
        return default

    return max(minimum, min(value, maximum))


def _safe_int_param(
    params: Dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Parse integer generation param safely."""
    raw_value = params.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "[Generator] Invalid int param %s=%r, fallback to default=%s",
            key,
            raw_value,
            default,
        )
        return default

    return max(minimum, min(value, maximum))


def _load_images(image_sources: List[str]) -> List[Image.Image]:
    """Load images from local path/http(s)/data URL and convert to RGB PIL images."""
    if not image_sources:
        return []

    if len(image_sources) > 8:
        raise ValueError("Too many images. Maximum supported images per request is 8.")

    loaded: List[Image.Image] = []
    for raw in image_sources:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Each image input must be a non-empty string")

        src = raw.strip()
        try:
            if src.startswith("data:image/") and ";base64," in src:
                _, b64_payload = src.split(";base64,", 1)
                image_bytes = base64.b64decode(b64_payload)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                loaded.append(image)
                continue

            if src.startswith("http://") or src.startswith("https://"):
                response = requests.get(src, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                loaded.append(image)
                continue

            local_path = Path(src).expanduser().resolve()
            if not local_path.exists() or not local_path.is_file():
                raise ValueError(f"Image path does not exist or is not a file: {src}")
            image = Image.open(local_path).convert("RGB")
            loaded.append(image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {src}, error: {e}") from e

    return loaded


def validate_and_prepare_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize generation parameters to safe ranges
    
    Args:
        params: Raw generation parameters from request
        
    Returns:
        Validated and normalized parameters
    """
    # Safe limits for parameters to prevent CUDA errors
    temperature = _safe_float_param(params, "temperature", 0.7, 0.0, 2.0)
    top_p = _safe_float_param(params, "top_p", 0.9, 0.0, 1.0)
    top_k = _safe_int_param(params, "top_k", 50, 0, 1000)
    repetition_penalty = _safe_float_param(params, "repetition_penalty", 1.1, 1.0, 2.0)
    max_new_tokens = _safe_int_param(params, "max_new_tokens", 512, 1, 8192)
    total_timeout = params.get("total_timeout", 300)
    do_sample = temperature > 0.0

    if do_sample and top_p == 0.0 and top_k == 0:
        logger.warning(
            "[Generator] Sampling requested but both top_p and top_k disable candidate selection; fallback to greedy decoding"
        )
        do_sample = False
    
    validated = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "total_timeout": total_timeout,
        "do_sample": do_sample,
    }
    
    logger.info(f"[Generator] Validated params: temp={temperature}, top_p={top_p}, "
                f"top_k={top_k}, rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}, do_sample={do_sample}")
    
    return validated


def tokenize_prompt(
    prompt: Any,
    tokenizer: AutoTokenizer,
    model_device,
    params: Optional[Dict[str, Any]] = None,
    processor=None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompt with proper max_length handling.
    Supports both string prompt and list of messages (chat).
    
    Args:
        prompt: Input text prompt (str) or list of messages (List[Dict])
        tokenizer: The tokenizer instance
        model_device: Device to move tensors to
        params: Optional dictionary of parameters (including enable_thinking)
        
    Returns:
        Tokenized inputs as tensors
    """
    # Get model's maximum length
    max_length = getattr(tokenizer, 'model_max_length', None)
    if max_length is None or max_length > 1000000:
        # If not defined or set to a very large value, use reasonable default
        max_length = 8192
    
    param_images = params.get("images") if params else None
    prompt_image_sources: List[str] = []
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        prompt_image_sources = _collect_message_image_sources(prompt)

    extra_images = [img for img in (param_images or []) if isinstance(img, str) and img.strip()]
    image_sources = prompt_image_sources + extra_images
    has_images = len(image_sources) > 0

    if has_images:
        if processor is None:
            raise ValueError(
                "This model does not expose an image processor. "
                "Please load a multimodal model that supports images."
            )

        pil_images = _load_images(image_sources)

        extra_kwargs = {}
        if params:
            enable_thinking = params.get("enable_thinking")
            if enable_thinking is not None:
                extra_kwargs["enable_thinking"] = enable_thinking

        # 記錄圖片尺寸，方便診斷圖片是否被正確處理
        for i, img in enumerate(pil_images):
            logger.info("[Generator] Image[%d]: size=%s mode=%s", i, img.size, img.mode)

        # 建立兩種 messages——它們對應不同的 Path
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            # Path A1 用：將 PIL Image 內嵌入 content
            messages_embedded = _build_messages_with_loaded_images(
                prompt,
                pil_images,
                embed_images=True,
            )
            # Path A2 / B 用：僅 {"type":"image"} placeholder
            messages_placeholder = _build_messages_with_loaded_images(
                prompt,
                pil_images,
                embed_images=False,
            )
        else:
            text_content = str(prompt) if prompt else ""
            base_content = [{"type": "text", "text": text_content}] if text_content else []
            messages_embedded = [{"role": "user", "content": [{"type": "image", "image": img} for img in pil_images] + base_content}]
            messages_placeholder = [{"role": "user", "content": [{"type": "image"} for _ in pil_images] + base_content}]

        # ── Path A1：內嵌 PIL Image，不使用 images= kwarg ─────────────────────────
        # Gemma 4 官方 HF model card 建議的方式：
        # {"type": "image", "image": <PIL.Image>} 小 不使用 images= kwarg
        # 避免 apply_chat_template 內部訕圖 images kwarg 時 Jinja 對
        # 字串做 string key 索引導致的「string indices must be integers」錯誤。
        try:
            inputs = processor.apply_chat_template(
                messages_embedded,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **extra_kwargs,
            ).to(model_device)
            logger.info("[Generator] Tokenized via apply_chat_template + embedded PIL images (path A1)")
            return inputs
        except Exception as e_a1:
            logger.warning(
                "[Generator] Path A1 (embedded PIL images) failed: %s – trying A2",
                e_a1,
            )

        # ── Path A2：placholder + images= kwarg ──────────────────────────────────
        try:
            inputs = processor.apply_chat_template(
                messages_placeholder,
                images=pil_images,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **extra_kwargs,
            ).to(model_device)
            logger.info("[Generator] Tokenized via apply_chat_template + images kwarg (path A2)")
            return inputs
        except Exception as e_a2:
            logger.warning(
                "[Generator] Path A2 (images= kwarg) failed: %s – falling back to path B",
                e_a2,
            )

        # ── Path B：圖文分離兩步驟 ─────────────────────────────────────────
        # Step B-1: 用 placeholder messages 產生文字 template
        # 必須用 messages_placeholder（含 {"type":"image"}）不能用原始 prompt，
        # 否則文字中無 <start_of_image> 標記 → processor 不知道圖片放哪。
        try:
            prompt_text = processor.apply_chat_template(
                messages_placeholder,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
            logger.info(
                "[Generator] Path B text template: %d chars | first 120: %r",
                len(prompt_text),
                prompt_text[:120],
            )
        except Exception as e_b:
            logger.warning(
                "[Generator] Path B apply_chat_template (text-only) failed: %s",
                e_b,
            )
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages_placeholder,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
            except Exception:
                prompt_text = str(prompt) if not isinstance(prompt, list) else ""

        # Step B-2: processor(text_with_image_markers, images=pil_images)
        # 不傳 truncation/processor_kwargs 避免 Gemma 4 的矛盾警告
        inputs = processor(
            text=prompt_text,
            images=pil_images,
            return_tensors="pt",
        ).to(model_device)
        logger.info("[Generator] Tokenized via processor(text, images) (path B)")
        return inputs

    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Prepare extra kwargs for chat template (e.g. for thinking/reasoning)
        extra_kwargs = {}
        if params:
            enable_thinking = params.get("enable_thinking")
            if enable_thinking is not None:
                extra_kwargs["enable_thinking"] = enable_thinking
        
        # Prefer processor.apply_chat_template when available, but keep a
        # fallback because some text-only processors expect plain string content
        # while some multimodal processors expect OpenAI-style text parts.
        if processor is not None:
            inputs = _apply_processor_chat_template_with_fallback(
                processor,
                prompt,
                tokenize=True,
                extra_kwargs=extra_kwargs,
                model_device=model_device,
                max_length=max_length,
            )
            return inputs
        
        # Fallback: use tokenizer directly
        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_length,
            **extra_kwargs
        ).to(model_device)
        return inputs
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model_device)
    
    return inputs


def get_generation_kwargs(
    inputs: Dict[str, torch.Tensor],
    params: Dict[str, Any],
    tokenizer: AutoTokenizer,
    eos_token_ids,
    streamer=None
) -> Dict[str, Any]:
    """
    Build generation kwargs for model.generate()
    
    Args:
        inputs: Tokenized input tensors
        params: Validated generation parameters
        tokenizer: Tokenizer instance
        eos_token_ids: End of sequence token ID(s)
        streamer: Optional text streamer for streaming generation
        
    Returns:
        Dictionary of generation arguments
    """
    generation_kwargs = {
        **inputs,
        "max_new_tokens": params["max_new_tokens"],
        "repetition_penalty": params["repetition_penalty"],
        "do_sample": params["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids,
        "use_cache": True,
        "output_scores": False,
        "return_dict_in_generate": False,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }

    if params["do_sample"]:
        generation_kwargs["temperature"] = max(params["temperature"], 1e-5)
        if params["top_p"] > 0.0:
            generation_kwargs["top_p"] = params["top_p"]
        if params["top_k"] > 0:
            generation_kwargs["top_k"] = params["top_k"]
    
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    
    return generation_kwargs


def decode_generated_tokens(
    outputs: torch.Tensor,
    input_length: int,
    tokenizer: AutoTokenizer,
    skip_special_tokens: bool = True
) -> str:
    """
    Decode generated tokens (excluding input prompt)
    
    Args:
        outputs: Generated token tensor from model
        input_length: Length of input tokens to skip
        tokenizer: Tokenizer instance
        skip_special_tokens: Whether to skip special tokens in decoding
        
    Returns:
        Decoded text string
    """
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=skip_special_tokens
    )
    return generated_text
