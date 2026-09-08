"""
GPU Memory Estimator for Model Inference
Estimates GPU memory requirements for models under different configurations.
Supports reading model configs from Hugging Face and computing parameter counts automatically.
Supports MoE (Mixture of Experts) models.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# Optional torch import for environment detection
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - estimator should still work without torch
    torch = None

# Optional transformers for config loading
try:
    from transformers import AutoConfig  # type: ignore
except Exception:
    AutoConfig = None

# Optional accelerate for meta-device instantiation (the parameter-counting path)
try:
    from accelerate import init_empty_weights  # type: ignore
except Exception:
    init_empty_weights = None


def _hf_token() -> str | None:
    """
    Token for gated repos (Gemma, Llama), or None.

    Resolved per call rather than at import, so a token added to .env takes effect on
    restart without this module having to be the first thing that reads it. Shares the
    helper download_manager uses, so one place decides where the token comes from.
    """
    try:
        from ..utils.token_utils import load_hf_token

        return load_hf_token() or None
    except Exception:  # pragma: no cover - estimation must not fail over a missing token
        return None


def _hf_endpoint() -> str:
    """
    Base URL for the Hub, honouring HF_ENDPOINT so a mirror or proxy is respected.

    Hardcoding huggingface.co would ignore the redirect an on-premise install relies
    on, which is the same class of mistake as assuming the Hub is reachable at all.
    """
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


# Whether the Hub answered the last time we tried. None means untested.
#
# Without this, an air-gapped host pays the timeout on every model it is asked about:
# huggingface_hub retries five times with backoff, which measured 159 seconds for a
# single un-downloaded model. The endpoint is supposed to answer "will this fit", not
# hang for two and a half minutes before admitting it does not know.
_hub_reachable: bool | None = None


def _hub_is_reachable(timeout: float = 2.0) -> bool:
    """
    Whether the Hub can be reached, probed once per process.

    HF_HUB_OFFLINE is honoured first: it is the standard way to tell this stack not to
    touch the network, and a deployment that sets it should never wait on a socket.
    """
    global _hub_reachable

    if os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "True"}:
        return False
    if _hub_reachable is not None:
        return _hub_reachable

    try:
        import httpx2

        httpx2.head(f"{_hf_endpoint()}/api/models", timeout=timeout, follow_redirects=True)
        _hub_reachable = True
    except Exception as e:
        logger.info(
            f"Hugging Face Hub unreachable ({type(e).__name__}); estimates will use local "
            "data only for the rest of this process"
        )
        _hub_reachable = False
    return _hub_reachable


def _mark_hub_unreachable() -> None:
    """Record that the Hub failed, so the next model does not wait on it again."""
    global _hub_reachable
    if _hub_reachable is not False:
        logger.info("Hub request failed; skipping remote lookups for the rest of this process")
    _hub_reachable = False


def _remote_config_available(model_name: str, timeout: float = 3.0) -> bool:
    """
    Whether the Hub will serve this model's config, decided within ``timeout``.

    Asked before handing the fetch to transformers, because huggingface_hub cannot be
    bounded from outside: it retries five times, and against an unreachable address
    each attempt waits on the OS connect timeout — 159 seconds in total, measured, and
    HF_HUB_ETAG_TIMEOUT does not shorten it. Once this has answered, the window in
    which the link can die before transformers starts is milliseconds rather than
    minutes.

    A refusal (gated repo, unknown model) means the network is fine and only this
    model is out of reach, so the Hub is not marked down for it.
    """
    try:
        import httpx2

        headers = {"Authorization": f"Bearer {_hf_token()}"} if _hf_token() else {}
        response = httpx2.head(
            f"{_hf_endpoint()}/{model_name}/resolve/main/config.json",
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
    except Exception as e:
        logger.info(f"Hub did not answer for {model_name} ({type(e).__name__})")
        _mark_hub_unreachable()
        return False

    if response.status_code >= 400:
        logger.info(
            f"Hub will not serve the config for {model_name} (HTTP {response.status_code}); "
            "estimating from the model name"
        )
        return False
    return True


# Names whose *size* cannot be read off them, for the last-resort path only.
#
# There used to be a table of ~20 popular models here, used whenever the config could
# not be read. It is gone: the parameter count now comes from the checkpoint itself
# (see _count_parameters), which is exact for every architecture transformers can
# build, including the MoE and multimodal cases a name or a hand-written table gets
# badly wrong.
#
# What remains is a guard for the fallback. Reading a size out of a model id works for
# "Llama-3-70B" but is catastrophically wrong for MoE naming conventions:
#
#   Mixtral-8x7B        -> reads 7B,   actually 46.7B   (6.7x under)
#   Qwen1.5-MoE-A2.7B   -> reads 2.7B, actually 14.3B   (5.3x under)
#
# "A3B" and "8x7B" describe *active* or *per-expert* size; the total is not in the
# name at all and no arithmetic recovers it. Under-reporting is the direction that
# hurts — it says a model fits when it does not — so the fallback refuses rather than
# guesses when it sees one of these.
MOE_NAME_MARKERS = (
    r"\d+x\d+\.?\d*b",  # Mixtral-8x7B, Mixtral-8x22B
    r"-a\d+\.?\d*b",  # Qwen1.5-MoE-A2.7B, Qwen3-235B-A22B
    r"\bmoe\b",  # explicit MoE in the name
)

# How much to add to a size read off a model name. Names carry the rounded marketing
# figure, which sits below the real total because it excludes embeddings: measured
# against safetensors totals, gpt-oss-20b is really 21.5B and gemma-3-4b is 4.3B,
# both 7.5% over their names. Erring high is the safe direction here.
NAME_ESTIMATE_MARGIN = 1.10


class MemoryEstimator:
    """Memory requirement estimator - supports HuggingFace configs and MoE models."""

    def __init__(self) -> None:
        self._config_cache = {}  # Cache for loaded configs
        # model id -> was the config read off local disk rather than fetched from the
        # Hub. Gates trust_remote_code; see _load_model_config.
        self._config_is_local: dict[str, bool] = {}
        self._param_cache: dict[str, float] = {}  # model id -> parameter count (B)

    def _load_model_config(self, model_name: str) -> Any | None:  # noqa: ANN401 - transformers PretrainedConfig or None
        """
        Load a model config from Hugging Face.

        Args:
            model_name: model name or path

        Returns:
            Model config object, or None
        """
        if model_name in self._config_cache:
            return self._config_cache[model_name]

        if AutoConfig is None:
            logger.warning("transformers is not installed; cannot load model config automatically")
            return None

        # Local first, so an already-downloaded model never touches the network. Falling
        # through to the Hub is what makes the accurate path reachable at all: this
        # endpoint answers "will it fit" *before* downloading, and local_files_only=True
        # meant that case always missed. config.json is a few KB, not the weights, and
        # transformers caches it.
        #
        # The remote attempt is skipped entirely when the Hub is known to be out of
        # reach. huggingface_hub retries five times with backoff, which on an air-gapped
        # host means 159 seconds of waiting per un-downloaded model — measured — before
        # falling back to a guess it could have made immediately.
        attempts = (True, False) if _hub_is_reachable() else (True,)
        for local_only in attempts:
            try:
                # Ask about the file ourselves first, with a timeout we control. The
                # process-level probe cannot cover a link that dies between checks, and
                # transformers cannot be bounded from outside once it starts.
                if not local_only and not _remote_config_available(model_name):
                    return None
                config = AutoConfig.from_pretrained(
                    model_name,
                    # from_pretrained executes Python from the repo when the config
                    # declares an auto_map, so this flag is a code-execution decision,
                    # not a compatibility one. It is granted only for a checkpoint that
                    # is already on this disk: the operator put it there and loading it
                    # would run the same code anyway. A repo id that merely arrived in a
                    # request has cleared no such bar — this endpoint takes an arbitrary
                    # model_name, has no authentication, and the service runs with
                    # allow_origins=["*"], so any page the operator visits can reach it.
                    # A remote checkpoint whose architecture needs custom code therefore
                    # falls back to the name estimate (size_source="model_name") instead.
                    trust_remote_code=local_only,
                    local_files_only=local_only,
                    token=_hf_token(),
                )
            except ValueError as e:
                # transformers parsed a real response and refused it — an unknown
                # architecture, or one whose custom code this will not run. The network
                # is fine, so the Hub must not be marked down over it.
                if local_only:
                    logger.debug(f"Config for {model_name} not usable from disk: {e}")
                    continue
                logger.warning(
                    f"Could not build a config for {model_name} from the Hub ({e}); "
                    "falling back to estimating from the model name"
                )
                return None
            except Exception as e:
                if local_only:
                    logger.debug(f"Config for {model_name} not in the local cache: {e}")
                    continue
                # A fetch that failed after the probe said yes means the link is not
                # dependable; stop trying for the rest of this process rather than
                # paying the timeout again on the next model.
                _mark_hub_unreachable()
                # Gated repos (Gemma, Llama) answer 401 without a token. Say so: what
                # follows is markedly less accurate.
                logger.warning(
                    f"Could not read the config for {model_name} ({e}); "
                    "falling back to estimating from the model name"
                )
                return None
            self._config_cache[model_name] = config
            self._config_is_local[model_name] = local_only
            logger.info(
                f"Loaded model config from the {'local cache' if local_only else 'Hub'}: "
                f"{model_name}"
            )
            return config

        if len(attempts) == 1:
            logger.info(
                f"{model_name} is not in the local cache and the Hub is unreachable; "
                "estimating from the model name"
            )
        return None

    def _count_parameters(self, model_name: str, config: Any) -> float | None:  # noqa: ANN401 - transformers PretrainedConfig
        """
        Total parameter count in billions, or None when it cannot be established.

        Builds the model on PyTorch's meta device: every tensor gets a shape and a
        dtype but no storage, so a 120B checkpoint costs nothing to instantiate and
        no weights are downloaded. Counting the result is then exact for whatever
        transformers builds — MoE experts, vision and audio towers, tied embeddings —
        rather than re-deriving each architecture's arithmetic here and getting it
        wrong in a different way for each new family. This is the technique behind
        `accelerate estimate-memory`.

        Cross-checked against the checkpoint's own safetensors index when the Hub can
        be reached, and the larger of the two wins. num_parameters() counts
        nn.Parameters only, so an architecture that keeps weights in buffers reports
        low (Qwen3-Next by 2%); the index is what is actually on disk. Erring high is
        the safe direction: under-reporting says a model fits when it does not.

        Building the model runs the repo's modeling code for a custom architecture, so
        that is allowed only for a config that came off local disk — the same bar
        _load_model_config applies. A remote checkpoint needing custom code therefore
        yields no meta-device count, but the safetensors index still does: that half
        reads published metadata and executes nothing, so such a model keeps an exact
        figure. Only when both halves come back empty does the caller reach the name
        estimate.
        """
        if model_name in self._param_cache:
            return self._param_cache[model_name]

        counted = self._count_via_meta_device(
            config, trust_remote_code=self._config_is_local.get(model_name, False)
        )
        on_disk = self._count_via_safetensors_index(model_name)

        if counted is None and on_disk is None:
            return None
        best = max(v for v in (counted, on_disk) if v is not None)

        if counted is not None and on_disk is not None:
            drift = abs(counted - on_disk) / on_disk
            if drift > 0.01:
                logger.info(
                    f"{model_name}: meta-device count {counted:.2f}B vs checkpoint "
                    f"{on_disk:.2f}B ({drift:.1%}); using {best:.2f}B"
                )

        self._param_cache[model_name] = best
        return best

    def _count_via_meta_device(
        self,
        config: Any,  # noqa: ANN401 - transformers PretrainedConfig
        *,
        trust_remote_code: bool = False,
    ) -> float | None:
        """
        Instantiate on the meta device and count. None if the class cannot be built.

        ``trust_remote_code`` decides whether a checkpoint carrying its own modelling
        code may have that code imported and run. init_empty_weights() only skips
        allocating tensor storage; it does not sandbox the import. Pass True only for a
        checkpoint already on local disk.
        """
        if init_empty_weights is None:
            logger.warning("accelerate is not installed; cannot count parameters exactly")
            return None

        try:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            from ..utils.model_class_resolver import is_multimodal_config

            # Deliberately *not* resolve_model_class(): that prefers CausalLM even for
            # multimodal checkpoints, because a text-dataset fine-tune wants the text
            # stack and ZeRO-3 needs it. Its own docstring notes the consequence — such
            # a load "may expose no vision tower at all" — which is exactly the weights
            # this has to account for. Counting wants the whole checkpoint, so the
            # preference is inverted here.
            #
            # AutoModel is not an option either: it drops the LM head, which on a small
            # model is a whole vocabulary table (SmolVLM-256M came out 11% light).
            candidates = (
                (AutoModelForImageTextToText, AutoModelForCausalLM)
                if is_multimodal_config(config)
                else (AutoModelForCausalLM, AutoModelForImageTextToText)
            )
            model = None
            for model_class in candidates:
                try:
                    with init_empty_weights():
                        model = model_class.from_config(config, trust_remote_code=trust_remote_code)
                    break
                except Exception as e:
                    logger.debug(f"{model_class.__name__} cannot build this config: {e}")
            if model is None:
                logger.warning("No auto class could build this config on the meta device")
                return None
            # from_config does not tie on the meta device, so a tied lm_head would be
            # counted as a second copy of the embedding table — 27% over on
            # Qwen2.5-0.5B, whose vocabulary is a quarter of the model.
            model.tie_weights()
            return model.num_parameters() / 1e9
        except Exception as e:
            logger.warning(f"Could not build the model on the meta device: {e}")
            return None

    def _count_via_safetensors_index(self, model_name: str) -> float | None:
        """
        Parameter count from the Hub's safetensors index — what is actually on disk.

        Skipped for local paths (no repo to ask about) and whenever the Hub is out of
        reach — this is a cross-check on the meta-device count, never the only source,
        so an offline host loses a little accuracy rather than waiting on a socket.
        """
        if "/" not in model_name or Path(model_name).exists():
            return None
        if not _hub_is_reachable():
            return None

        try:
            import httpx2

            headers = {"Authorization": f"Bearer {_hf_token()}"} if _hf_token() else {}
            response = httpx2.get(
                f"{_hf_endpoint()}/api/models/{model_name}",
                headers=headers,
                timeout=3.0,
            )
            response.raise_for_status()
            total = (response.json().get("safetensors") or {}).get("total")
            return total / 1e9 if total else None
        except Exception as e:
            logger.debug(f"No safetensors index for {model_name}: {e}")
            return None

    def _describe_architecture(self, config: Any) -> dict:  # noqa: ANN401 - transformers PretrainedConfig
        """
        Read the shape of the model off its config, for the parts of the estimate that
        are not the weights: activations and KV cache need the hidden size, layer count
        and head layout, and the response reports the MoE layout.
        """
        info = {
            "architecture": getattr(config, "architectures", ["unknown"])[0]
            if getattr(config, "architectures", None)
            else "unknown",
            "is_moe": False,
            "num_experts": 0,
            "experts_per_token": 0,
        }

        # Multimodal checkpoints keep the language model under text_config; the towers
        # beside it are already in the parameter count, but the KV cache is the text
        # tower's alone.
        text_config = getattr(config, "text_config", None) or config

        for experts_attr in ("num_local_experts", "n_routed_experts", "num_experts"):
            experts = getattr(text_config, experts_attr, 0) or 0
            if experts > 1:
                info["is_moe"] = True
                info["num_experts"] = experts
                info["experts_per_token"] = getattr(text_config, "num_experts_per_tok", 2)
                break

        info["hidden_size"] = getattr(text_config, "hidden_size", 4096)
        info["num_layers"] = getattr(text_config, "num_hidden_layers", 32)
        info["intermediate_size"] = getattr(
            text_config, "intermediate_size", info["hidden_size"] * 4
        )
        info["vocab_size"] = getattr(text_config, "vocab_size", 32000)

        # The KV cache is sized by the *key/value* head count, not the model's hidden
        # size. Under GQA those differ by the grouping factor — Qwen3-32B has 64
        # attention heads sharing 8 KV heads — so reading hidden_size here overstates
        # the cache by that factor.
        heads = getattr(text_config, "num_attention_heads", 32) or 32
        info["num_attention_heads"] = heads
        info["num_key_value_heads"] = getattr(text_config, "num_key_value_heads", heads) or heads
        info["head_dim"] = getattr(text_config, "head_dim", None) or info["hidden_size"] // heads
        return info

    def extract_model_size(self, model_name: str) -> float | None:
        """
        Extract the parameter count from a model name, preferring the HuggingFace config.

        Args:
            model_name: model name, e.g. "meta-llama/Llama-2-7b-chat-hf"

        Returns:
            Parameter count (Billion) or None
        """
        config = self._load_model_config(model_name)
        if config is not None:
            counted = self._count_parameters(model_name, config)
            if counted is not None:
                return counted

        return self._guess_size_from_name(model_name)

    def _guess_size_from_name(self, model_name: str) -> float | None:
        """
        Last resort: read a size out of the model id.

        Only reached when the config could not be read at all — a gated repo with no
        HF_TOKEN, or a private model offline. Returns None rather than a number
        whenever the name cannot be trusted, because a confident wrong answer here
        becomes a confident wrong "it fits".
        """
        name = model_name.lower()

        # MoE names advertise active or per-expert size, never the total, and the gap
        # is 5-7x rather than a few percent. No margin covers that, so do not guess.
        for marker in MOE_NAME_MARKERS:
            if re.search(marker, name):
                logger.warning(
                    f"{model_name} looks like an MoE checkpoint and its config could not "
                    "be read; refusing to estimate from the name, which would report the "
                    "active size rather than the total"
                )
                return None

        match = re.search(r"(\d+\.?\d*)b", name) or re.search(r"(\d+\.?\d*)-?billion", name)
        if not match:
            logger.warning(f"Cannot extract the parameter count from the model name: {model_name}")
            return None

        size = float(match.group(1)) * NAME_ESTIMATE_MARGIN
        logger.warning(
            f"Estimating {model_name} from its name: {match.group(1)}B plus a "
            f"{NAME_ESTIMATE_MARGIN:.0%} margin -> {size:.2f}B. Provide HF_TOKEN or "
            "download the model for an exact figure."
        )
        return size

    def estimate_memory_requirements(
        self,
        model_name: str,
        quantization: str = "none",
        include_activations: bool = True,
        batch_size: int = 1,
        sequence_length: int = 2048,
    ) -> dict:
        """
        Estimate the memory requirements of a model (MoE models supported).

        Args:
            model_name: model name
            quantization: quantization type (none, int8, int4, nf4, fp4)
            include_activations: whether to include activation memory
            batch_size: batch size
            sequence_length: sequence length

        Returns:
            Dict with the memory estimation result
        """
        config = self._load_model_config(model_name)
        params_info = {}
        model_size_b = None

        if config is not None:
            params_info = self._describe_architecture(config)
            model_size_b = self._count_parameters(model_name, config)
            if model_size_b is not None:
                params_info["total_params"] = model_size_b
                logger.info(
                    f"Counted from the checkpoint: {model_size_b:.2f}B params "
                    f"(MoE: {params_info['is_moe']})"
                )

        if model_size_b is None:
            # No usable config: everything below is inferred from the name.
            params_info = {}
            model_size_b = self._guess_size_from_name(model_name)

        if model_size_b is None:
            return {
                "error": "Cannot determine the model size",
                "model_name": model_name,
                "suggestion": (
                    "The config could not be read and the name does not carry a reliable "
                    "size. Set HF_TOKEN for a gated repo, or download the model first."
                ),
            }

        is_moe = bool(params_info.get("is_moe"))
        # The count is the total across all experts, which is what the weights cost:
        # inference keeps every expert resident regardless of routing.
        model_memory = self._calculate_model_memory(model_size_b, quantization)

        # Activation memory (intermediate results during inference)
        # Use the actual config when available, otherwise estimate.
        # When params_info is truthy, hidden_size/num_layers are already ints
        # (see _describe_architecture).
        hidden_size = cast(
            int,
            params_info.get("hidden_size")
            if params_info
            else self._estimate_hidden_size(model_size_b),
        )
        num_layers = cast(
            int,
            params_info.get("num_layers")
            if params_info
            else self._estimate_num_layers(model_size_b),
        )

        activation_memory = 0
        if include_activations:
            # The logits tensor dominates this term, so the vocabulary size matters more
            # than anything else here; fall back to the largest in common use when the
            # config could not be read.
            activation_memory = self._calculate_activation_memory_with_config(
                hidden_size,
                num_layers,
                batch_size,
                sequence_length,
                quantization,
                vocab_size=cast(int, params_info.get("vocab_size", 151936)),
            )

        # KV cache memory. With a config the cached width is num_key_value_heads x
        # head_dim. Without one the head layout is unknown and the geometry is guessed,
        # so the fallback helper owns that case — it assumes MHA and adds the band-edge
        # margin. Re-deriving it here is what let the two drift apart before.
        if params_info:
            kv_cache_memory = self._calculate_kv_cache_memory_with_config(
                num_layers,
                batch_size,
                sequence_length,
                kv_width=cast(int, params_info["num_key_value_heads"] * params_info["head_dim"]),
            )
        else:
            kv_cache_memory = self._calculate_kv_cache_memory(
                model_size_b, batch_size, sequence_length
            )

        # Runtime overhead. Only the device share belongs in a GPU budget — the Python
        # process, the framework and the library code live in system RAM, and adding
        # them here charged every VRAM figure about 1.3 GB it does not need.
        overhead_breakdown = self._estimate_runtime_overhead(quantization)
        overhead_memory = overhead_breakdown["device_total"]

        # Total memory requirement
        total_memory = model_memory + activation_memory + kv_cache_memory + overhead_memory

        # Recommended minimum GPU memory
        recommended_gpu_memory = total_memory * 1.1  # reserve a 10% safety margin

        # Minimum GPU memory in the hybrid offload mode
        min_gpu_with_offload = (
            activation_memory + kv_cache_memory + overhead_memory + (model_memory * 0.1)
        )

        result = {
            "model_name": model_name,
            "model_size_billions": round(model_size_b, 2),
            # Whether the figures were counted from the checkpoint or inferred from the
            # model's name. On the name path the parameter count, hidden size and layer
            # count are all guesses, so every number below inherits that uncertainty;
            # clients should present it as an approximation, not a measurement.
            "size_source": "config" if params_info else "model_name",
            "quantization": quantization,
            "memory_breakdown_gb": {
                "model_weights": round(model_memory, 2),
                "activations": round(activation_memory, 2),
                "kv_cache": round(kv_cache_memory, 2),
                "overhead": round(overhead_memory, 2),
                "total": round(total_memory, 2),
            },
            # Split by where the memory lives: only device_total is part of the GPU
            # requirement above, the host entries are system RAM.
            "overhead_details_gb": {
                "python_runtime": round(overhead_breakdown["python_runtime"], 2),
                "pytorch_framework": round(overhead_breakdown["pytorch_framework"], 2),
                "transformers_lib": round(overhead_breakdown["transformers_lib"], 2),
                "host_total": round(overhead_breakdown["host_total"], 2),
                "device_context": round(overhead_breakdown["device_context"], 2),
                "quantization_lib": round(overhead_breakdown["quantization_lib"], 2),
                "device_total": round(overhead_breakdown["device_total"], 2),
                "total": round(overhead_breakdown["total"], 2),
            },
            "recommendations": {
                "full_gpu_memory_gb": round(recommended_gpu_memory, 2),
                "min_gpu_with_cpu_offload_gb": round(min_gpu_with_offload, 2),
                "min_gpu_with_disk_offload_gb": round(min_gpu_with_offload * 0.7, 2),
            },
            "offload_strategies": self._generate_offload_strategies(
                model_memory, activation_memory, kv_cache_memory, overhead_memory
            ),
            "notes": [
                f"Estimate based on a {model_size_b:.2f}B parameter model",
                f"Batch size: {batch_size}, sequence length: {sequence_length}",
            ],
        }

        # MoE-specific info. Active params are deliberately absent: they describe how
        # much compute a token costs, not how much memory the weights need, and every
        # expert is resident either way. Reporting them next to a memory estimate
        # invited sizing a GPU against the active figure.
        if is_moe:
            result["moe_info"] = {
                "is_moe": True,
                "num_experts": params_info.get("num_experts", 0),
                "experts_per_token": params_info.get("experts_per_token", 0),
                "total_params_billions": round(model_size_b, 2),
            }
            result["notes"].append(
                f"MoE model: {params_info.get('num_experts')} experts, "
                f"{params_info.get('experts_per_token')} active per token. "
                f"All {model_size_b:.2f}B parameters stay resident."
            )

        # Config info
        if params_info:
            result["model_config"] = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "intermediate_size": params_info.get("intermediate_size"),
                "vocab_size": params_info.get("vocab_size"),
                "architecture": params_info.get("architecture", "unknown"),
            }
            result["notes"].append(
                f"Config source: Hugging Face ({params_info.get('architecture')})"
            )
        else:
            result["notes"].append(
                "Config source: estimated (install transformers for the exact config)"
            )

        result["notes"].extend(
            [
                "Actual memory use varies with the framework and runtime (CUDA/driver/libraries)",
                "Reserving a 20% safety margin is recommended",
                (
                    f"{overhead_breakdown['accelerator']} accelerator detected; its "
                    f"context is counted ({overhead_breakdown['device_total']:.2f} GB). "
                    f"A further {overhead_breakdown['host_total']:.2f} GB of framework "
                    "overhead sits in system RAM, not on the device"
                    if overhead_breakdown["accelerator"] != "cpu"
                    else "No accelerator detected; the estimate covers device memory only"
                ),
            ]
        )

        return result

    def _estimate_runtime_overhead(self, quantization: str) -> dict[str, float]:
        """
        Non-weight memory the runtime needs, split by where it actually lives.

        Only ``device_total`` belongs in a GPU budget. The host entries — the Python
        process, the framework and library code — sit in system RAM and were previously
        summed into the same figure, adding about 1.3 GB of RAM costs to every VRAM
        requirement.

        The device figures were 10x too large as well. Measured on CUDA (RTX 5060 Ti,
        torch 2.11+cu130) as the gap between what nvidia-smi reports and what torch has
        allocated for tensors, after loading a model and running a forward pass:

            Qwen2.5-0.5B    134 MiB          SmolLM2-135M     90 MiB
            bare context + one matmul: 181 MiB

        against the 2.15 GB the old constants claimed (0.8 context + 1.1 libraries +
        0.25 driver, scaled by compute capability). 0.35 GB below leaves roughly 2x
        headroom over the largest measurement without dominating the estimate.

        Anything that is not CUDA but is a GPU — Intel XPU here — used to fall into a
        "CPU-only" branch and be charged nothing at all, which is the wrong direction.
        It now takes the same figure.

        On XPU only a lower bound could be measured: torch 2.11+xpu on an Intel iGPU
        reserves 2 MiB for a bare context and 41.8 MiB beyond the weights once a model
        is loaded. The driver-side total is not readable there — the backend already
        carries a workaround for XPU lacking mem_get_info — so this is under-counted
        rather than measured. 0.35 GB sits above that bound and inside the range CUDA
        measured, which is the best that can be said without a working counter. Note an
        integrated GPU draws from system RAM rather than dedicated VRAM, so the figure
        matters far less there than on a discrete card.
        """
        accelerator = self._detect_accelerator()

        overhead = {
            # Host-side: system RAM, not VRAM.
            "python_runtime": 0.3,  # Python process and core libraries
            "pytorch_framework": 0.8 if accelerator != "cpu" else 0.5,
            "transformers_lib": 0.2,
            # Device-side: what the GPU itself holds beyond the weights.
            "device_context": 0.0,
            "quantization_lib": 0.0,
        }

        if accelerator != "cpu":
            # Context, kernels and library workspaces, measured rather than assumed.
            overhead["device_context"] = 0.35
            # bitsandbytes loads its kernels onto the device.
            if quantization.lower() in {"int8", "int4", "nf4", "fp4"}:
                overhead["quantization_lib"] = 0.4
        elif quantization.lower() in {"int8", "int4", "nf4", "fp4"}:
            overhead["quantization_lib"] = 0.1

        host_total = (
            overhead["python_runtime"]
            + overhead["pytorch_framework"]
            + overhead["transformers_lib"]
        )
        device_total = overhead["device_context"] + overhead["quantization_lib"]

        overhead["host_total"] = round(host_total, 2)
        overhead["device_total"] = round(device_total, 2)
        # Kept for callers that reported a single number; it is host + device, so it
        # is not a GPU figure.
        overhead["total"] = round(host_total + device_total, 2)
        overhead["accelerator"] = accelerator  # type: ignore[assignment]
        return overhead

    @staticmethod
    def _detect_accelerator() -> str:
        """Which kind of accelerator torch can see: "cuda", "xpu" or "cpu"."""
        if torch is None:
            return "cpu"
        try:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except Exception:  # pragma: no cover - a broken driver must not fail the estimate
            logger.warning("Accelerator detection failed; assuming CPU")
        return "cpu"

    def _calculate_model_memory(self, model_size_b: float, quantization: str) -> float:
        """
        Compute the model weight memory requirement.

        Args:
            model_size_b: model size (Billion parameters)
            quantization: quantization type

        Returns:
            Memory requirement (GB)
        """
        # Bits per parameter
        bits_per_param = {
            "none": 16,  # FP16/BF16
            "fp16": 16,
            "bf16": 16,
            "int8": 8,
            "int4": 4,
            "nf4": 4,
            "fp4": 4,
        }

        bits = bits_per_param.get(quantization.lower(), 16)

        # Compute memory (GB)
        # 1B parameters * bits_per_param / 8 (bytes) / 1024^3 (GB)
        memory_gb = (model_size_b * 1e9 * bits / 8) / (1024**3)

        return memory_gb

    def _calculate_activation_memory(
        self, model_size_b: float, batch_size: int, sequence_length: int, quantization: str
    ) -> float:
        """
        Activation memory without a config (fallback path).

        The term is dominated by the logits tensor, whose width is the vocabulary — and
        with no config that is unknown. 151936 is the largest in common use (the Qwen
        families); assuming it over-reports a smaller vocabulary rather than under, in
        keeping with the rest of this path.
        """
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        return self._calculate_activation_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length, quantization, vocab_size=151936
        )

    def _calculate_activation_memory_with_config(
        self,
        hidden_size: int,
        num_layers: int,
        batch_size: int,
        sequence_length: int,
        quantization: str,
        vocab_size: int = 32000,
    ) -> float:
        """
        Peak activation memory for a forward pass, which the logits tensor dominates.

        The output projection produces one score per vocabulary entry per position, so
        it is batch x sequence x vocab — far larger than anything the layers keep live.
        Measured with torch's allocator on CUDA, weights subtracted:

            model             vocab    seq    measured    batch*seq*vocab*2 bytes
            Qwen2.5-0.5B     151936    128     47.3 MiB       37.1 MiB
                                       512    158.4 MiB      148.4 MiB
                                      2048    606.6 MiB      593.5 MiB
            SmolLM2-135M      49152    128     12.1 MiB       12.0 MiB
                                      2048    195.0 MiB      192.0 MiB

        The same model on an Intel XPU (torch 2.11+xpu) tracks the logits tensor even
        more closely — 38.2 / 149.3 / 597.5 MiB at the three lengths — so the shape is
        not a CUDA artefact.

        Re-measured on Linux (WSL2, torch 2.13+cu129, same RTX 5060 Ti) the logits term
        holds but the residual does not. SmolLM2 matches Windows almost exactly — 0.1 /
        0.6 / 2.3 / 9.1 MiB at 128 / 512 / 2048 / 8192 — while Qwen2.5-0.5B carries a
        roughly 33 MiB constant workspace where Windows measured 10-13:

            seq     measured   logits   residual
            128         70.2     37.1       33.1
            512        181.3    148.4       32.9
            2048       629.5    593.5       36.0
            8192      2420.1   2374.0       46.1

        A 16 MiB floor did not cover that: at 128 and 512 tokens the estimate came out
        below the measurement, which is the direction that says a model fits when it
        does not. The floor is 64 MiB, about twice the largest constant seen.

        Whether the constant is the OS or the torch version cannot be separated here —
        the Windows figures were taken on torch 2.11+cu and that install has since been
        replaced by the XPU build the training tests need. It does not change the fix:
        the formula has to cover both, and one of them needs 33 MiB.

        The residual grows slowly with sequence length on top of that constant, and the
        10% margin covers the growth from 2048 tokens up — the floor is what carries the
        short-sequence cases, where the whole term is small in absolute terms anyway.

        The previous formula (sequence x hidden x layers / 30, no recorded derivation)
        had no logits term at all and came out 216x low: 2.8 MiB against 606.6 measured.

        Assumes a single forward over the whole sequence, which is what prefill does.
        A decode step computes logits for one position only, so this is the peak rather
        than the steady state — the right figure for "will it fit".
        """
        # 16-bit dtypes (none/fp16/bf16) use 2 bytes per element; quantized
        # weights use a rough 1-byte approximation for activations.
        bytes_per_element = 2 if quantization.lower() in ("none", "fp16", "bf16") else 1

        logits = batch_size * sequence_length * vocab_size * bytes_per_element / (1024**3)
        # 10% plus a 64 MiB floor covers the measured residual on both platforms above.
        # The floor is set by Linux's ~33 MiB constant workspace, not by Windows' 10-13.
        return logits * 1.10 + 64 / 1024

    def _calculate_kv_cache_memory(
        self, model_size_b: float, batch_size: int, sequence_length: int
    ) -> float:
        """
        KV cache without a config (fallback path).

        With no config the head layout is unknown, so this assumes MHA — every
        attention head keeping its own K and V, i.e. a per-token width of hidden_size.
        That is the largest the cache can be, and over-estimating is the safe
        direction for a "will it fit" answer.

        For a GQA model that ceiling leaves 3-15x of headroom, so the geometry guess
        underneath it barely matters. For a model that really is MHA it leaves almost
        none — measured against the guessed geometry, phi-2 came out at 1.00x, falcon-7b
        and Phi-3-mini at 1.11x, gpt-neox-20b and bloom-1b7 at 1.3x. Those are the models
        where the step functions are the only thing standing between the answer and an
        under-estimate, and a band's lower edge is where they are thinnest: the tables
        return one value for everything from 13B to 30B, so a deep, narrow model sitting
        just under a threshold is the case that can go negative.

        Hence the margin. It is not a fudge for the tables being poor — they measure
        1.00-2.67x across 18 models and never under (see _estimate_hidden_size) — it is
        cover for the band edges, on a path whose parameter count is already a guess
        carrying NAME_ESTIMATE_MARGIN for the same reason.
        """
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        exact = self._calculate_kv_cache_memory_with_config(
            num_layers, batch_size, sequence_length, kv_width=hidden_size
        )
        return exact * NAME_ESTIMATE_MARGIN

    def _calculate_kv_cache_memory_with_config(
        self, num_layers: int, batch_size: int, sequence_length: int, kv_width: int
    ) -> float:
        """
        Exact KV cache size: 2 (K and V) x layers x batch x tokens x kv_width x 2 bytes.

        ``kv_width`` is num_key_value_heads x head_dim, which is what an attention layer
        actually caches. This used to pass hidden_size and then scale the result by 0.6
        ("about 60% is used on average in practice"). Both parts were wrong, in opposite
        directions:

          - hidden_size assumes MHA. Under GQA the cache is smaller by the grouping
            factor, so the figure came out 2.4-4.2x high on every model measured
            (Qwen2.5-0.5B 4.2x, Qwen3-32B 3.0x, gpt-oss-20b 3.4x, Mixtral 2.4x).
          - the 0.6 then removed 40% regardless, which for a genuine MHA model turns
            the answer into an under-estimate — the direction that says a model fits
            when it does not.

        There is no averaging to do here: the caller asks for a specific batch size and
        sequence length, and the cache for those has to be allocated in full.
        """
        bytes_per_element = 2  # FP16 / BF16

        return (2 * num_layers * batch_size * sequence_length * kv_width * bytes_per_element) / (
            1024**3
        )

    def _estimate_hidden_size(self, model_size_b: float) -> int:
        """
        Hidden size for a model whose config could not be read.

        The tables here and in _estimate_num_layers arrived with no recorded derivation,
        so they were checked against the real configs of 18 models spanning 0.1B-72B and
        8 families (Qwen2.5, Qwen3, SmolLM2, Phi, Llama-derived, Falcon, GPT-NeoX, BLOOM).
        What matters is the product layers x hidden, since that is what the KV cache is
        proportional to. Against the real product:

            worst over-estimate   2.67x  (SmolLM2-135M, deliberately deep and narrow)
            worst under-estimate  1.00x  (phi-2, landing exactly on a table row)
            median                ~1.25x

        Never under, which is the property that matters for "will it fit".

        A scaling law was fitted as a replacement — params = 12 x layers x hidden^2 with
        a fixed aspect ratio, the usual dense-transformer relation, calibrated over the
        same 18 models. It came out at 1.06-2.67x: no better. MoE models are why nothing
        derived from the parameter count can do much better here, as their count is
        inflated by experts that add no attention layers, and on the name path there is
        no way to tell. The tables stay.
        """
        if model_size_b <= 1:
            return 2048
        elif model_size_b <= 3:
            return 2560
        elif model_size_b <= 7:
            return 4096
        elif model_size_b <= 13:
            return 5120
        elif model_size_b <= 30:
            return 6656
        elif model_size_b <= 70:
            return 8192
        else:
            return 12288

    def _estimate_num_layers(self, model_size_b: float) -> int:
        """
        Layer count for a model whose config could not be read.

        Only meaningful together with _estimate_hidden_size — see the measurement there.
        """
        if model_size_b <= 1:
            return 22
        elif model_size_b <= 3:
            return 32
        elif model_size_b <= 7:
            return 32
        elif model_size_b <= 13:
            return 40
        elif model_size_b <= 30:
            return 60
        elif model_size_b <= 70:
            return 80
        else:
            return 120

    def _generate_offload_strategies(
        self,
        model_memory: float,
        activation_memory: float,
        kv_cache_memory: float,
        overhead_memory: float,
    ) -> list:
        """
        Generate offload strategy suggestions for different GPU memory sizes.
        """
        strategies = []

        # Strategy 1: full GPU (no offload)
        full_gpu = model_memory + activation_memory + kv_cache_memory + overhead_memory
        strategies.append(
            {
                "name": "Full GPU (No Offload)",
                "min_gpu_gb": round(full_gpu * 1.1, 2),
                "description": "All model weights and compute stay on the GPU",
                "performance": "fastest",
                "config": {"offload": "none"},
            }
        )

        # Strategy 2: CPU offload (some weights)
        cpu_offload_50 = (
            (model_memory * 0.5) + activation_memory + kv_cache_memory + overhead_memory
        )
        strategies.append(
            {
                "name": "CPU Offload (50% weights)",
                "min_gpu_gb": round(cpu_offload_50 * 1.1, 2),
                "description": "50% of the model weights offloaded to CPU",
                "performance": "moderate",
                "config": {"offload": "cpu", "device_map": "auto"},
            }
        )

        # Strategy 3: CPU offload (most weights)
        cpu_offload_80 = (
            (model_memory * 0.2) + activation_memory + kv_cache_memory + overhead_memory
        )
        strategies.append(
            {
                "name": "CPU Offload (80% weights)",
                "min_gpu_gb": round(cpu_offload_80 * 1.1, 2),
                "description": "80% of the weights offloaded to CPU; only key layers on GPU",
                "performance": "slower",
                "config": {"offload": "cpu", "device_map": "auto"},
            }
        )

        # Strategy 4: disk offload
        disk_offload = (model_memory * 0.1) + activation_memory + kv_cache_memory + overhead_memory
        strategies.append(
            {
                "name": "Disk Offload (90% weights)",
                "min_gpu_gb": round(disk_offload * 1.1, 2),
                "description": "Most weights offloaded to disk (NVMe recommended)",
                "performance": "slowest, but the lowest GPU requirement",
                "config": {"offload": "disk", "offload_dir": "./offload"},
            }
        )

        return strategies


# Create the global instance
memory_estimator = MemoryEstimator()
