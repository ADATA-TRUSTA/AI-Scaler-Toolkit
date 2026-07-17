"""
Generator Worker - Worker process handlers for generation commands
Handles 'generate' and 'generate_stream' commands in the worker process loop
"""
import torch
import logging
import time
import queue
from typing import Dict, Any, Optional
from threading import Thread
from transformers import TextIteratorStreamer

from .generator_core import (
    validate_and_prepare_params,
    tokenize_prompt,
    get_generation_kwargs,
    decode_generated_tokens
)
from .gpt_parser import create_gpt_parser, is_gpt_model, create_stream_parser, TokenIDStreamer
from ...settings import get_response_queue_debug

logger = logging.getLogger(__name__)


def _sync_cuda():
    """Synchronize CUDA for accurate timing if available."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _is_xpu_model_device(model) -> bool:
    """Best-effort check whether the loaded model is running on Intel XPU."""
    try:
        model_device = getattr(model, "device", None)
        if model_device is not None and getattr(model_device, "type", None) == "xpu":
            return True
    except Exception:
        pass

    try:
        first_param = next(model.parameters())
        if getattr(first_param.device, "type", None) == "xpu":
            return True
    except Exception:
        pass

    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                if str(dev).lower().startswith("xpu"):
                    return True
    except Exception:
        pass

    return False


def _split_stream_text_chunks(text: str, max_chars: int = 96) -> list[str]:
    """Split text into small chunks so fallback paths can still emit visible SSE streaming."""
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    for ch in text:
        current += ch
        if ch in {"\n", ".", "!", "?", ",", "，", "。", "！", "？", "；", ";", ":", "："} or len(current) >= max_chars:
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _normalize_eos_token_ids(eos_token_ids) -> set[int]:
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {int(eos_token_ids)}
    out = set()
    try:
        for token_id in eos_token_ids:
            out.add(int(token_id))
    except Exception:
        pass
    return out


def _apply_repetition_penalty_(logits: torch.Tensor, seen_token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not seen_token_ids:
        return logits

    # outputs.logits may be an inference tensor; clone before any in-place update.
    logits = logits.clone()

    unique_ids = set(int(token_id) for token_id in seen_token_ids)
    for token_id in unique_ids:
        token_logit = logits[..., token_id]
        logits[..., token_id] = torch.where(
            token_logit < 0,
            token_logit * penalty,
            token_logit / penalty,
        )
    return logits


def _sample_next_token_id(logits: torch.Tensor, params: Dict[str, Any]) -> int:
    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 50))
    do_sample = bool(params.get("do_sample", temperature > 0.01))

    next_token_logits = logits
    if temperature > 0:
        next_token_logits = next_token_logits / max(temperature, 1e-5)

    if top_k > 0 and top_k < next_token_logits.shape[-1]:
        topk_values, _ = torch.topk(next_token_logits, top_k)
        kth_value = topk_values[..., -1, None]
        next_token_logits = torch.where(
            next_token_logits < kth_value,
            torch.full_like(next_token_logits, float("-inf")),
            next_token_logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

    if do_sample:
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    return int(next_token.item())


def _stream_generate_xpu_text(
    *,
    model,
    tokenizer,
    inputs,
    validated_params: Dict[str, Any],
    eos_token_ids,
    data_queue,
    request_id: str,
    stop_generation_flag,
) -> Dict[str, Any]:
    """Manual token-by-token streaming fallback for XPU text generation."""
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    token_type_ids = inputs.get("token_type_ids")

    if input_ids is None:
        raise ValueError("XPU text streaming requires input_ids")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    max_new_tokens = int(validated_params["max_new_tokens"])
    total_timeout = float(validated_params["total_timeout"])
    repetition_penalty = float(validated_params.get("repetition_penalty", 1.0))
    eos_token_id_set = _normalize_eos_token_ids(eos_token_ids)

    perf_start = time.perf_counter()
    first_token_time: Optional[float] = None
    prompt_tokens = int(input_ids.shape[1])
    gen_token_count = 0
    start_time = time.time()
    stopped = False

    generated_token_ids: list[int] = []
    pending_token_ids: list[int] = []
    past_key_values = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    current_token_type_ids = token_type_ids

    for _ in range(max_new_tokens):
        if stop_generation_flag.is_set():
            logger.info("[Worker] Stop generation signal received during XPU token streaming")
            stopped = True
            break

        if time.time() - start_time > total_timeout:
            logger.error("[Worker] XPU token streaming total generation timeout exceeded")
            raise TimeoutError("Total generation timeout")

        model_inputs = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if current_token_type_ids is not None:
            model_inputs["token_type_ids"] = current_token_type_ids
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        with torch.inference_mode():
            outputs = model(**model_inputs)

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = _apply_repetition_penalty_(next_token_logits, generated_token_ids, repetition_penalty)
        next_token_id = _sample_next_token_id(next_token_logits, validated_params)

        generated_token_ids.append(next_token_id)
        pending_token_ids.append(next_token_id)
        gen_token_count += 1

        if first_token_time is None:
            first_token_time = time.perf_counter()

        next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device, dtype=input_ids.dtype)
        current_input_ids = next_token_tensor
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype)],
            dim=1,
        )
        if current_token_type_ids is not None:
            next_token_type = current_token_type_ids[:, -1:].clone()
            current_token_type_ids = torch.cat([current_token_type_ids, next_token_type], dim=1)
            current_token_type_ids = current_token_type_ids[:, -1:]

        if next_token_id in eos_token_id_set:
            break

        decoded_text = tokenizer.decode(
            pending_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded_text:
            _put_response(data_queue, {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": decoded_text,
                "done": False,
                "chunk_tokens": len(pending_token_ids),
            })
            pending_token_ids = []

    if pending_token_ids and not stopped:
        decoded_text = tokenizer.decode(
            pending_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded_text:
            _put_response(data_queue, {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": decoded_text,
                "done": False,
                "chunk_tokens": len(pending_token_ids),
            })

    perf_end = time.perf_counter()
    return {
        "stopped": stopped,
        "perf_stats": _log_transformers_perf(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_token_count,
            prompt_time_s=(first_token_time - perf_start) if first_token_time else None,
            gen_time_s=(perf_end - first_token_time) if first_token_time else None,
            total_time_s=(perf_end - perf_start) if perf_end and perf_start else None,
        ),
    }


def _log_transformers_perf(
    *,
    prompt_tokens: Optional[int],
    gen_tokens: Optional[int],
    prompt_time_s: Optional[float],
    gen_time_s: Optional[float],
    total_time_s: Optional[float],
    tag: str = "Transformers",
) -> Dict[str, Any]:
    """Log transformers performance stats when timing/token info is available."""
    stats: Dict[str, Any] = {
        "total_tokens": None,
        "gen_tps": None,
        "prompt_tokens": None,
        "prompt_tps": None,
        "gen_tokens": None,
    }

    try:
        if isinstance(prompt_tokens, int):
            stats["prompt_tokens"] = prompt_tokens
        if isinstance(gen_tokens, int):
            stats["gen_tokens"] = gen_tokens
        if isinstance(prompt_tokens, int) and isinstance(gen_tokens, int):
            stats["total_tokens"] = prompt_tokens + gen_tokens

        if isinstance(prompt_tokens, int) and prompt_time_s and prompt_time_s > 0:
            stats["prompt_tps"] = prompt_tokens / float(prompt_time_s)

        if isinstance(gen_tokens, int):
            base_time = None
            if gen_time_s and gen_time_s > 0:
                base_time = gen_time_s
            elif total_time_s and total_time_s > 0:
                base_time = total_time_s
            if base_time:
                stats["gen_tps"] = gen_tokens / float(base_time)

        if stats.get("prompt_tps") is not None and stats.get("gen_tps") is not None:
            logger.info(
                "[%s][perf] prompt %s tok @ %.2f tok/s | gen %s tok @ %.2f tok/s | total %s tok",
                tag,
                stats.get("prompt_tokens"),
                stats.get("prompt_tps"),
                stats.get("gen_tokens"),
                stats.get("gen_tps"),
                stats.get("total_tokens"),
            )
        elif stats.get("gen_tps") is not None and stats.get("gen_tokens") is not None:
            logger.info(
                "[%s][perf] gen %s tok @ %.2f tok/s | total %s tok",
                tag,
                stats.get("gen_tokens"),
                stats.get("gen_tps"),
                stats.get("total_tokens"),
            )

        return stats
    except Exception as e:
        logger.debug(f"[{tag}] Failed to compute perf stats: {e}")
        return stats


def _count_tokens_safe(tokenizer, text: str) -> int:
    """Count tokens for a text chunk safely (best-effort)."""
    try:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as e:
        logger.debug(f"[Worker] Token count failed for chunk: {e}")
        return 0


def _get_extended_eos_token_ids(tokenizer, model, model_name: str):
    """Resolve complete eos token ids from tokenizer/model generation config.

    Qwen3.5 publishes multiple eos ids in generation_config (e.g. [248046, 248044]).
    If we only keep tokenizer.eos_token_id, generation may miss the actual stop token
    and continue sampling into garbled multilingual text.
    """
    try:
        eos_ids = []

        generation_config = getattr(model, "generation_config", None)
        generation_eos = getattr(generation_config, "eos_token_id", None)
        if isinstance(generation_eos, (list, tuple, set)):
            for token_id in generation_eos:
                if isinstance(token_id, int) and token_id not in eos_ids:
                    eos_ids.append(token_id)
        elif isinstance(generation_eos, int):
            eos_ids.append(generation_eos)

        if getattr(tokenizer, "eos_token_id", None) is not None:
            if tokenizer.eos_token_id not in eos_ids:
                eos_ids.append(tokenizer.eos_token_id)
        
        # Check if Gemma3 model
        if any(pattern in model_name.lower() for pattern in ["gemma-3", "gemma3", "gemma-2-3"]):
            try:
                end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if end_turn_id is not None and end_turn_id not in eos_ids:
                    eos_ids.append(end_turn_id)
            except Exception as e:
                logger.debug(f"[Worker] Could not resolve <end_of_turn> id: {e}")

        logger.info("[Worker] Resolved eos_token_id(s) for %s: %s", model_name, eos_ids)
        
        if not eos_ids:
            return None
        return eos_ids[0] if len(eos_ids) == 1 else eos_ids
    except Exception as e:
        logger.debug(f"[Worker] _get_extended_eos_token_ids error: {e}")
        return getattr(tokenizer, "eos_token_id", None)


def _cleanup_generation_inputs(inputs):
    """Clean up generation inputs to free memory"""
    try:
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"[Worker] Cleanup error: {e}")


def _put_response(data_queue, response: dict, debug: bool = None):
    """輔助函數：發送響應到隊列並打印（用於調試）
    
    Args:
        data_queue: 目標隊列
        response: 要發送的數據字典
        debug: 是否啟用調試輸出（None 時使用全局設定）
    """
    if debug is None:
        debug = get_response_queue_debug()
    
    if debug:
        # 打印響應內容（縮短 chunk 以避免過長）
        debug_data = response.copy()
        if "chunk" in debug_data and len(str(debug_data["chunk"])) > 50:
            debug_data["chunk"] = str(debug_data["chunk"])[:50] + "..."
        print(f"[Response Queue] {debug_data}", flush=True)
    
    try:
        data_queue.put(response)
    except Exception as e:
        logger.error(f"[Worker] Failed to put response to queue: {e}")


def handle_generate_request(
    request: Dict[str, Any],
    model,
    tokenizer,
    processor,
    config,
    data_queue,
    stop_generation_flag
):
    """
    Handle non-streaming generation request
    
    Args:
        request: Request dictionary with prompt, params, and request_id
        model: Loaded model instance
        tokenizer: Tokenizer instance
        config: Model configuration
        data_queue: Queue to send response data
        stop_generation_flag: Event flag for stopping generation
    """
    request_id = request.get("request_id")
    prompt = request.get("prompt")
    params = request.get("params", {})
    
    try:
        # Clear stop flag for new generation
        stop_generation_flag.clear()
        
        # Validate and prepare parameters
        validated_params = validate_and_prepare_params(params)
        
        # Tokenize prompt
        inputs = tokenize_prompt(prompt, tokenizer, model.device, params, processor)
        
        # Get EOS token IDs
        model_name = config.model_name if config else "unknown"
        eos_token_ids = _get_extended_eos_token_ids(tokenizer, model, model_name)
        
        # Build generation kwargs
        generation_kwargs = get_generation_kwargs(
            inputs, validated_params, tokenizer, eos_token_ids
        )
        
        # Generate
        # Use inference_mode for better performance and memory usage (supports offload)
        _sync_cuda()
        gen_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**generation_kwargs)
        _sync_cuda()
        gen_end = time.perf_counter()
        
        # Check if GPT model and use structured parsing
        gpt_parser = create_gpt_parser(model_name)
        
        if gpt_parser and gpt_parser.should_parse():
            # GPT model: parse structured response with <think></think> tags
            logger.info("[Worker] Using GPT Harmony parser for structured response")
            
            # Extract generated tokens (excluding input)
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:].tolist()
            
            # Parse with Harmony
            parsed = gpt_parser.parse_generated_tokens(generated_tokens, strict=False)
            
            if parsed["parsed"] and parsed["formatted_text"]:
                # Successfully parsed - return formatted text
                perf_stats = _log_transformers_perf(
                    prompt_tokens=int(inputs.input_ids.shape[1]),
                    gen_tokens=int(outputs[0].shape[0] - inputs.input_ids.shape[1]),
                    prompt_time_s=None,
                    gen_time_s=None,
                    total_time_s=(gen_end - gen_start) if gen_end and gen_start else None,
                )

                response = {
                    "type": "result",
                    "request_id": request_id,
                    "result": parsed["formatted_text"],
                }
                if perf_stats.get("total_tokens") is not None:
                    response["total_tokens"] = perf_stats.get("total_tokens")
                if perf_stats.get("gen_tokens") is not None:
                    response["gen_tokens"] = perf_stats.get("gen_tokens")
                if perf_stats.get("gen_tps") is not None:
                    response["gen_tps"] = perf_stats.get("gen_tps")
                if perf_stats.get("prompt_tokens") is not None:
                    response["prompt_tokens"] = perf_stats.get("prompt_tokens")
                if perf_stats.get("prompt_tps") is not None:
                    response["prompt_tps"] = perf_stats.get("prompt_tps")
                _put_response(data_queue, response)
                logger.info(f"[Worker] GPT response parsed successfully")
            else:
                # Parsing failed - fallback to normal decode
                logger.warning("[Worker] GPT parsing failed, falling back to normal decode")
                generated_text = decode_generated_tokens(
                    outputs, inputs.input_ids.shape[1], tokenizer
                )
                perf_stats = _log_transformers_perf(
                    prompt_tokens=int(inputs.input_ids.shape[1]),
                    gen_tokens=int(outputs[0].shape[0] - inputs.input_ids.shape[1]),
                    prompt_time_s=None,
                    gen_time_s=None,
                    total_time_s=(gen_end - gen_start) if gen_end and gen_start else None,
                )
                response = {
                    "type": "result",
                    "request_id": request_id,
                    "result": generated_text,
                }
                if perf_stats.get("total_tokens") is not None:
                    response["total_tokens"] = perf_stats.get("total_tokens")
                if perf_stats.get("gen_tokens") is not None:
                    response["gen_tokens"] = perf_stats.get("gen_tokens")
                if perf_stats.get("gen_tps") is not None:
                    response["gen_tps"] = perf_stats.get("gen_tps")
                if perf_stats.get("prompt_tokens") is not None:
                    response["prompt_tokens"] = perf_stats.get("prompt_tokens")
                if perf_stats.get("prompt_tps") is not None:
                    response["prompt_tps"] = perf_stats.get("prompt_tps")
                _put_response(data_queue, response)
        else:
            # Non-GPT model or parser not available: standard decode
            generated_text = decode_generated_tokens(
                outputs, inputs.input_ids.shape[1], tokenizer
            )
            perf_stats = _log_transformers_perf(
                prompt_tokens=int(inputs.input_ids.shape[1]),
                gen_tokens=int(outputs[0].shape[0] - inputs.input_ids.shape[1]),
                prompt_time_s=None,
                gen_time_s=None,
                total_time_s=(gen_end - gen_start) if gen_end and gen_start else None,
            )

            response = {
                "type": "result",
                "request_id": request_id,
                "result": generated_text,
            }
            if perf_stats.get("total_tokens") is not None:
                response["total_tokens"] = perf_stats.get("total_tokens")
            if perf_stats.get("gen_tokens") is not None:
                response["gen_tokens"] = perf_stats.get("gen_tokens")
            if perf_stats.get("gen_tps") is not None:
                response["gen_tps"] = perf_stats.get("gen_tps")
            if perf_stats.get("prompt_tokens") is not None:
                response["prompt_tokens"] = perf_stats.get("prompt_tokens")
            if perf_stats.get("prompt_tps") is not None:
                response["prompt_tps"] = perf_stats.get("prompt_tps")
            _put_response(data_queue, response)
    
    except Exception as e:
        logger.error(f"[Worker] Generation error: {e}")
        error_str = str(e)
        error_type = type(e).__name__
        
        is_oom = ("out of memory" in error_str.lower() or
                  "oom" in error_str.lower() or
                  "OutOfMemoryError" in error_type or
                  isinstance(e, torch.cuda.OutOfMemoryError))
        
        if is_oom:
            logger.error("[Worker] Recoverable OOM during generation – soft cleanup")
            try:
                _cleanup_generation_inputs(inputs)
            except:
                pass
            
            _put_response(data_queue, {
                "type": "error",
                "request_id": request_id,
                "error": f"OOM Error: {error_str}",
                "is_oom": True,
                "recoverable": True,
                "suggestions": [
                    "Lower max_new_tokens",
                    "Reduce prompt length / history",
                    "Increase offload / quantization"
                ]
            })
        else:
            _put_response(data_queue, {
                "type": "error",
                "request_id": request_id,
                "error": error_str
            })
    finally:
        # Cleanup
        try:
            _cleanup_generation_inputs(inputs)
        except:
            pass


def handle_generate_stream_request(
    request: Dict[str, Any],
    model,
    tokenizer,
    processor,
    config,
    data_queue,
    stop_generation_flag
):
    """
    Handle streaming generation request
    
    Args:
        request: Request dictionary with prompt, params, and request_id
        model: Loaded model instance
        tokenizer: Tokenizer instance
        config: Model configuration
        data_queue: Queue to send response chunks
        stop_generation_flag: Event flag for stopping generation
    """
    request_id = request.get("request_id")
    prompt = request.get("prompt")
    params = request.get("params", {})
    
    try:
        # Clear stop flag for new generation
        stop_generation_flag.clear()
        
        # Validate and prepare parameters
        validated_params = validate_and_prepare_params(params)
        
        # Tokenize prompt
        inputs = tokenize_prompt(prompt, tokenizer, model.device, params, processor)
        
        # Get EOS token IDs
        model_name = config.model_name if config else "unknown"
        eos_token_ids = _get_extended_eos_token_ids(tokenizer, model, model_name)
        
        # Check if GPT model
        is_gpt = is_gpt_model(model_name)
        is_xpu = _is_xpu_model_device(model)

        if is_xpu and isinstance(prompt, list) and not params.get("images"):
            logger.info("[Worker] XPU streaming token loop enabled")
            xpu_stream_result = _stream_generate_xpu_text(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                validated_params=validated_params,
                eos_token_ids=eos_token_ids,
                data_queue=data_queue,
                request_id=request_id,
                stop_generation_flag=stop_generation_flag,
            )

            response = {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": "",
                "done": True,
                "stopped": bool(xpu_stream_result.get("stopped")),
            }
            perf_stats = xpu_stream_result.get("perf_stats") or {}
            if perf_stats.get("total_tokens") is not None:
                response["total_tokens"] = perf_stats.get("total_tokens")
            if perf_stats.get("gen_tokens") is not None:
                response["gen_tokens"] = perf_stats.get("gen_tokens")
            if perf_stats.get("gen_tps") is not None:
                response["gen_tps"] = perf_stats.get("gen_tps")
            if perf_stats.get("prompt_tokens") is not None:
                response["prompt_tokens"] = perf_stats.get("prompt_tokens")
            if perf_stats.get("prompt_tps") is not None:
                response["prompt_tps"] = perf_stats.get("prompt_tps")
            _put_response(data_queue, response)
            _cleanup_generation_inputs(inputs)
            return
        elif is_xpu:
            logger.info("[Worker] XPU streaming fallback enabled; generating full response before emitting stream chunks")

            generation_kwargs = get_generation_kwargs(
                inputs, validated_params, tokenizer, eos_token_ids, streamer=None
            )

            result_holder: Dict[str, Any] = {"outputs": None, "exception": None}

            def generate_xpu_fallback():
                try:
                    with torch.inference_mode():
                        result_holder["outputs"] = model.generate(**generation_kwargs)
                except Exception as e:
                    result_holder["exception"] = e
                    logger.error(f"[Worker] XPU fallback generation error: {e}")

            perf_start = time.perf_counter()
            thread = Thread(target=generate_xpu_fallback)
            thread.daemon = True
            thread.start()

            total_timeout = validated_params["total_timeout"]
            start_time = time.time()
            stopped = False

            while thread.is_alive():
                if stop_generation_flag.is_set():
                    logger.info("[Worker] Stop generation signal received during XPU fallback")
                    stopped = True
                    break

                if time.time() - start_time > total_timeout:
                    logger.error("[Worker] XPU fallback total generation timeout exceeded")
                    raise TimeoutError("Total generation timeout")

                time.sleep(0.1)

            thread.join(timeout=1.0)

            if result_holder["exception"] is not None:
                raise result_holder["exception"]

            if stopped:
                _put_response(data_queue, {
                    "type": "stream_chunk",
                    "request_id": request_id,
                    "chunk": "",
                    "done": True,
                    "stopped": True,
                })
                return

            outputs = result_holder.get("outputs")
            if outputs is None:
                raise RuntimeError("XPU fallback generation completed without outputs")

            generated_text = decode_generated_tokens(
                outputs, inputs.input_ids.shape[1], tokenizer
            )
            gen_tokens = int(outputs[0].shape[0] - inputs.input_ids.shape[1])
            perf_end = time.perf_counter()
            perf_stats = _log_transformers_perf(
                prompt_tokens=int(inputs.input_ids.shape[1]),
                gen_tokens=gen_tokens,
                prompt_time_s=None,
                gen_time_s=(perf_end - perf_start) if perf_end and perf_start else None,
                total_time_s=(perf_end - perf_start) if perf_end and perf_start else None,
            )

            if generated_text:
                text_chunks = _split_stream_text_chunks(generated_text)
                if not text_chunks:
                    text_chunks = [generated_text]

                remaining_tokens = gen_tokens
                for idx, text_chunk in enumerate(text_chunks):
                    if stop_generation_flag.is_set():
                        logger.info("[Worker] Stop generation signal received during XPU fallback chunk emission")
                        stopped = True
                        break

                    chunk_tokens = _count_tokens_safe(tokenizer, text_chunk)
                    if chunk_tokens <= 0:
                        chunk_tokens = max(0, remaining_tokens) if idx == len(text_chunks) - 1 else 0
                    remaining_tokens = max(0, remaining_tokens - chunk_tokens)

                    _put_response(data_queue, {
                        "type": "stream_chunk",
                        "request_id": request_id,
                        "chunk": text_chunk,
                        "done": False,
                        "chunk_tokens": chunk_tokens,
                    })

                    time.sleep(0.01)

            response = {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": "",
                "done": True,
                "stopped": stopped,
            }
            if perf_stats.get("total_tokens") is not None:
                response["total_tokens"] = perf_stats.get("total_tokens")
            if perf_stats.get("gen_tokens") is not None:
                response["gen_tokens"] = perf_stats.get("gen_tokens")
            if perf_stats.get("gen_tps") is not None:
                response["gen_tps"] = perf_stats.get("gen_tps")
            if perf_stats.get("prompt_tokens") is not None:
                response["prompt_tokens"] = perf_stats.get("prompt_tokens")
            if perf_stats.get("prompt_tps") is not None:
                response["prompt_tps"] = perf_stats.get("prompt_tps")
            _put_response(data_queue, response)
            _cleanup_generation_inputs(inputs)
            return
        
        # Create streamer
        if is_gpt and not is_xpu:
            stream_parser = create_stream_parser(model_name)
            if stream_parser:
                streamer = TokenIDStreamer(
                    skip_prompt=True
                )
            else:
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                stream_parser = None
        else:
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            stream_parser = None

        if is_gpt and is_xpu:
            logger.info("[Worker] XPU detected for GPT streaming; using TextIteratorStreamer fallback")
        
        # Build generation kwargs
        generation_kwargs = get_generation_kwargs(
            inputs, validated_params, tokenizer, eos_token_ids, streamer
        )
        
        # Generation exception tracking
        generation_exception = None
        
        def generate_with_exception_handling():
            nonlocal generation_exception
            try:
                # Use inference_mode for better performance and memory usage (supports offload)
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except Exception as e:
                generation_exception = e
                logger.error(f"[Worker] Generation thread error: {e}")
                # Signal exception to streamer if it's a TokenIDStreamer
                if is_gpt and isinstance(streamer, TokenIDStreamer):
                    streamer.set_exception(e)
        
        # Start generation in separate thread
        perf_start = time.perf_counter()
        first_token_time: Optional[float] = None
        gen_token_count: int = 0
        early_done_sent = False
        thread = Thread(target=generate_with_exception_handling)
        thread.daemon = True
        thread.start()
        
        # Stream generated tokens with timeout
        total_timeout = validated_params["total_timeout"]
        start_time = time.time()
        first_token_timeout = min(45.0, max(10.0, float(total_timeout) * 0.1))
        stopped = False
        
        # For GPT models with StreamableParser: track state
        if is_gpt and stream_parser:
            shown_headers = set()
            last_channel = None
        
        try:
            dead_thread_check_count = 0
            
            while True:
                # 1. Check for generation exception
                if generation_exception is not None:
                    logger.error(f"[Worker] Generation thread exception: {generation_exception}")
                    raise generation_exception
                
                # 2. Check stop flag
                if stop_generation_flag.is_set():
                    logger.info("[Worker] Stop generation signal received")
                    stopped = True
                    break
                
                # 3. Check total timeout
                if time.time() - start_time > total_timeout:
                    logger.error("[Worker] Total generation timeout exceeded")
                    raise TimeoutError("Total generation timeout")

                # 3.5. First token watchdog: surface hangs before the full request timeout.
                if first_token_time is None and (time.time() - start_time) > first_token_timeout:
                    logger.error("[Worker] First token timeout exceeded (%.1fs)", first_token_timeout)
                    raise TimeoutError(f"No first token produced within {first_token_timeout:.1f} seconds")
                
                # 4. Check if generation thread finished
                if not thread.is_alive():
                    if generation_exception is not None:
                        raise generation_exception
                    else:
                        dead_thread_check_count += 1
                        if dead_thread_check_count > 3:
                            logger.info("[Worker] Thread finished and queue empty after multiple checks")
                            break
                
                # 5. Get next token from streamer
                try:
                    if is_gpt and isinstance(streamer, TokenIDStreamer):
                        # GPT model: get token ID and process with StreamableParser
                        try:
                            token_id = next(iter(streamer))
                        except StopIteration:
                            logger.info("[Worker] Token stream completed")
                            break
                        except (TimeoutError, Exception) as stream_error:
                            logger.error(f"[Worker] Streamer error: {stream_error}")
                            raise
                        
                        dead_thread_check_count = 0
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        gen_token_count += 1
                        
                        # Process token with StreamableParser
                        stream_parser.process(token_id)
                        
                        current_channel = stream_parser.current_channel
                        last_delta = stream_parser.last_content_delta
                        
                        # Detect channel switch
                        if current_channel != last_channel:
                            # Close previous channel
                            if last_channel == "analysis" and 'think_end' not in shown_headers:
                                _put_response(data_queue, {
                                    "type": "stream_chunk",
                                    "request_id": request_id,
                                    "chunk": "\n</think>\n\n",
                                    "done": False
                                })
                                shown_headers.add('think_end')
                            
                            # Open new channel
                            if current_channel == "analysis" and 'think' not in shown_headers:
                                _put_response(data_queue, {
                                    "type": "stream_chunk",
                                    "request_id": request_id,
                                    "chunk": "<think>\n",
                                    "done": False
                                })
                                shown_headers.add('think')
                            
                            last_channel = current_channel
                        
                        # Send delta (new content)
                        if last_delta:
                            chunk_tokens = 1
                            _put_response(data_queue, {
                                "type": "stream_chunk",
                                "request_id": request_id,
                                "chunk": last_delta,
                                "done": False,
                                "chunk_tokens": chunk_tokens
                            })
                    
                    else:
                        # Non-GPT: standard text streaming
                        chunk_tokens = 0
                        if hasattr(streamer, 'text_queue'):
                            text = streamer.text_queue.get(timeout=1.0)
                        else:
                            text = next(iter(streamer))
                        
                        dead_thread_check_count = 0
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        
                        if text == streamer.stop_signal if hasattr(streamer, 'stop_signal') else text is None:
                            logger.info("[Worker] Stream completed (stop signal)")
                            break
                        
                        # Handle <end_of_turn> for Gemma3
                        if "<end_of_turn>" in text:
                            before, _, _ = text.partition("<end_of_turn>")
                            if before:
                                chunk_tokens = _count_tokens_safe(tokenizer, before)
                                gen_token_count += chunk_tokens
                                _put_response(data_queue, {
                                    "type": "stream_chunk",
                                    "request_id": request_id,
                                    "chunk": before,
                                    "done": False,
                                    "chunk_tokens": chunk_tokens
                                })
                            perf_end = time.perf_counter()
                            perf_stats = _log_transformers_perf(
                                prompt_tokens=int(inputs.input_ids.shape[1]),
                                gen_tokens=gen_token_count,
                                prompt_time_s=(first_token_time - perf_start) if first_token_time else None,
                                gen_time_s=(perf_end - first_token_time) if first_token_time else None,
                                total_time_s=(perf_end - perf_start) if perf_end and perf_start else None,
                            )
                            response = {
                                "type": "stream_chunk",
                                "request_id": request_id,
                                "chunk": "",
                                "done": True,
                                "stopped": False
                            }
                            if perf_stats.get("total_tokens") is not None:
                                response["total_tokens"] = perf_stats.get("total_tokens")
                            if perf_stats.get("gen_tokens") is not None:
                                response["gen_tokens"] = perf_stats.get("gen_tokens")
                            if perf_stats.get("gen_tps") is not None:
                                response["gen_tps"] = perf_stats.get("gen_tps")
                            if perf_stats.get("prompt_tokens") is not None:
                                response["prompt_tokens"] = perf_stats.get("prompt_tokens")
                            if perf_stats.get("prompt_tps") is not None:
                                response["prompt_tps"] = perf_stats.get("prompt_tps")
                            _put_response(data_queue, response)
                            early_done_sent = True
                            break
                        
                        if text:
                            chunk_tokens = _count_tokens_safe(tokenizer, text)
                            gen_token_count += chunk_tokens
                        _put_response(data_queue, {
                            "type": "stream_chunk",
                            "request_id": request_id,
                            "chunk": text,
                            "done": False,
                            "chunk_tokens": chunk_tokens
                        })
                
                except queue.Empty:
                    continue
                except StopIteration:
                    logger.info("[Worker] Streamer iteration complete (StopIteration)")
                    break
        
        except Exception as e:
            if not isinstance(e, (TimeoutError, RuntimeError)):
                logger.error(f"[Worker] Unexpected error in stream loop: {e}")
            # Wait for thread to finish on exception
            logger.warning("[Worker] Exception occurred, waiting for generation thread to finish...")
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.error("[Worker] Generation thread still alive after exception, will be left as daemon")
            raise
        
        # Wait for thread to finish
        wait_timeout = 5.0 if stopped else 3.0
        thread.join(timeout=wait_timeout)
        
        # Check for thread exception
        if generation_exception is not None:
            raise generation_exception
        
        # Check if thread still running
        if thread.is_alive():
            if stopped:
                logger.warning("[Worker] Generation thread still running after user stop (normal, will terminate as daemon)")
            else:
                logger.error("[Worker] Generation thread still alive after timeout!")
                raise TimeoutError("Generation thread timeout")
        else:
            logger.info("[Worker] Generation thread completed successfully")
        
        # Cleanup generation resources
        _cleanup_generation_inputs(inputs)
        
        # For GPT models: close any remaining tags
        if is_gpt and stream_parser and not stopped:
            try:
                # Close think tag if still open
                if last_channel == "analysis" and 'think_end' not in shown_headers:
                    _put_response(data_queue, {
                        "type": "stream_chunk",
                        "request_id": request_id,
                        "chunk": "\n</think>\n\n",
                        "done": False
                    })
            except Exception as e:
                logger.debug(f"[Worker] Error closing GPT tags: {e}")
        
        # Send completion signal
        if not early_done_sent:
            perf_end = time.perf_counter()
            perf_stats = _log_transformers_perf(
                prompt_tokens=int(inputs.input_ids.shape[1]),
                gen_tokens=gen_token_count,
                prompt_time_s=(first_token_time - perf_start) if first_token_time else None,
                gen_time_s=(perf_end - first_token_time) if first_token_time else None,
                total_time_s=(perf_end - perf_start) if perf_end and perf_start else None,
            )
            response = {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": "",
                "done": True,
            }
            if perf_stats.get("total_tokens") is not None:
                response["total_tokens"] = perf_stats.get("total_tokens")
            if perf_stats.get("gen_tokens") is not None:
                response["gen_tokens"] = perf_stats.get("gen_tokens")
            if perf_stats.get("gen_tps") is not None:
                response["gen_tps"] = perf_stats.get("gen_tps")
            if perf_stats.get("prompt_tokens") is not None:
                response["prompt_tokens"] = perf_stats.get("prompt_tokens")
            if perf_stats.get("prompt_tps") is not None:
                response["prompt_tps"] = perf_stats.get("prompt_tps")
            _put_response(data_queue, response)
    
    except Exception as e:
        logger.error(f"[Worker] Stream generation error: {e}")
        error_str = str(e)
        error_type = type(e).__name__
        
        is_oom = ("out of memory" in error_str.lower() or
                  "oom" in error_str.lower() or
                  "OutOfMemoryError" in error_type or
                  isinstance(e, torch.cuda.OutOfMemoryError))
        
        if is_oom:
            logger.error("[Worker] Recoverable OOM during stream generation – soft cleanup")
            try:
                _cleanup_generation_inputs(inputs)
            except:
                pass
            
            _put_response(data_queue, {
                "type": "error",
                "request_id": request_id,
                "error": f"OOM Error: {error_str}",
                "is_oom": True,
                "recoverable": True,
                "suggestions": [
                    "Lower max_new_tokens",
                    "Reduce prompt length / history",
                    "Increase offload / quantization",
                    "Invoke /inference/cleanup_generation_memory if needed"
                ]
            })
        else:
            _put_response(data_queue, {
                "type": "error",
                "request_id": request_id,
                "error": error_str
            })
            if isinstance(e, TimeoutError) and thread.is_alive():
                logger.warning("[Worker] Timeout with thread still running - thread is daemon and will not block unload")
    finally:
        # Final cleanup
        try:
            _cleanup_generation_inputs(inputs)
        except:
            pass
