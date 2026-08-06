"""
Memory estimator for GGUF models served by llama.cpp.

Answers, for a given ``-ngl`` / ``-c`` / ``-ctk`` / ``-ctv`` / ``--n-cpu-moe`` combination,
how many MiB land on each GPU and how many on host RAM, and which combination fits the
machine.

Two tracks:

* **analytic** (default) - reproduces llama.cpp's tensor placement and KV-cache sizing in
  pure Python from the GGUF tensor table. Weights and KV cache are exact; the compute
  buffer uses a calibrated formula that errs on the high side. Needs no binary, so a full
  parameter sweep costs nothing.
* **llama_cpp** - shells out to ``llama-fit-params -fitp on``, which performs a ``no_alloc``
  dry-run load and prints llama.cpp's own numbers. Exact, but one process per data point.

The placement rules are transcribed from the vendored llama.cpp:

* ``i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`` (``src/llama-model.cpp``). There are
  ``n_layer + 1`` slots and the GPU takes the *last* ``n_gpu_layers`` of them, so the output
  layer is offloaded before any block.
* The input layer (token embeddings) always stays on the CPU.
* ``--n-cpu-moe N`` installs one CPU override per layer with the pattern
  ``blk\\.{i}\\.ffn_(up|down|gate|gate_up)_(ch|)exps`` for ``i`` in ``0..N-1``; overrides are
  matched per tensor with a substring regex search and applied after slot placement.
* KV cache for a layer lives on the device that owns the layer, which is why ``--n-cpu-moe``
  keeps attention and KV on the GPU while pushing expert weights to RAM.
* ``n_ctx`` is padded up to a multiple of 256 (``src/llama-context.cpp``).
"""

import bisect
import logging
import math
import os
import re
import shlex
import subprocess
from typing import Any

from .. import settings
from .gguf_inspect import GgufModelInfo, read_gguf_model_info

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

# llama.cpp pads the KV cache to a multiple of this many cells.
N_CTX_PAD = 256

# Compute-buffer calibration, measured against llama-fit-params on two models with very
# different vocabularies: Qwen3-32B-Q8_0 (n_vocab=151936) and a 128-token synthetic MoE,
# across n_ubatch in {128, 512, 1024, 2048} and n_ctx in {4096 .. 131072}.
#
# ggml's allocator reuses the space freed by the f32 logits tensor for the attention mask,
# so the two do not add up - the buffer tracks whichever is larger. Modelling it as a sum
# over-estimates a large-vocab model by up to 60%; taking the max and applying a safety
# factor lands within roughly -1% to +12% on both. Slightly high is the safe direction.
_COMPUTE_SAFETY_FACTOR = 1.15
_COMPUTE_MASK_BYTES_PER_CELL = 6

# The attention mask is padded along the batch axis, so measurements at n_ubatch 128 match
# those at 256. Below this floor the mask term stops shrinking.
_COMPUTE_MASK_UBATCH_FLOOR = 256

# Under partial offload, host-resident weights are staged into VRAM per op, double-buffered.
_COMPUTE_STAGING_BUFFERS = 2

# The only types llama.cpp accepts for -ctk / -ctv, with their bytes per element. Used both
# as the accepted-value set and as a fallback when the gguf quantization table is unreadable.
KV_CACHE_TYPES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
    "q4_0": 18 / 32,
    "q4_1": 20 / 32,
    "iq4_nl": 18 / 32,
}

# Default margin left free on each device, matching llama-fit-params' -fitt default.
DEFAULT_MARGIN_MIB = 1024

# Upper bound on sweep() grid size, to keep responses bounded.
MAX_SWEEP_ROWS = 2000

# KV cache types plan() offers by default, ordered best-quality first. q4_0 roughly quarters
# the KV cache against f16, which is what buys a long context on a small budget.
DEFAULT_KV_VARIANTS: tuple[str, ...] = ("f16", "q8_0", "q4_0")

# Upper bound on plan() cells (budgets x KV types); each cell is a full search plus probes.
MAX_PLAN_CELLS = 32

# Candidates whose share of the model on the GPU is within this many percentage points of the
# best are treated as equally fast, and ranked by context instead. Without it, one extra
# offloaded layer out of thirty (3% of the weights, a few percent of tokens/s) outranks a ten
# times longer context, which is not the trade the caller would make.
_OFFLOAD_TIE_PCT = 5.0

# How many times recommend() may re-solve against a target tightened by the measured error.
_VERIFY_CORRECTION_ROUNDS = 3


def kv_bytes_per_element(cache_type: str) -> float:
    """
    Bytes per element for a KV cache type.

    Only the types llama.cpp accepts for ``-ctk`` / ``-ctv`` are allowed; other ggml types
    (Q3_K and friends) exist but cannot be used for the cache. The size itself comes from
    the ``gguf`` quantization table when readable, so it tracks upstream automatically.

    Raises:
        ValueError: the type is not a valid KV cache type.
    """
    name = cache_type.strip().lower()
    if name not in KV_CACHE_TYPES:
        raise ValueError(
            f"Unsupported KV cache type: {cache_type}. "
            f"llama.cpp accepts: {', '.join(sorted(KV_CACHE_TYPES))}"
        )
    try:
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType[name.upper()]]
        return type_size / block_size
    except Exception:
        return KV_CACHE_TYPES[name]


def _to_mib(n_bytes: float) -> float:
    """Convert bytes to MiB, rounded to two decimals."""
    return round(n_bytes / MIB, 2)


# ---------------------------------------------------------------------- CLI parsing

# llama-server flags that change memory use, mapped to estimate() keyword arguments.
# Every alias llama.cpp accepts is listed so a pasted command line parses as-is.
_VALUE_FLAGS: dict[str, str] = {
    "-m": "model_path",
    "--model": "model_path",
    "-ngl": "n_gpu_layers",
    "--gpu-layers": "n_gpu_layers",
    "--n-gpu-layers": "n_gpu_layers",
    "-c": "n_ctx",
    "--ctx-size": "n_ctx",
    "-b": "n_batch",
    "--batch-size": "n_batch",
    "-ub": "n_ubatch",
    "--ubatch-size": "n_ubatch",
    "-np": "n_parallel",
    "--parallel": "n_parallel",
    "-ctk": "cache_type_k",
    "--cache-type-k": "cache_type_k",
    "-ctv": "cache_type_v",
    "--cache-type-v": "cache_type_v",
    "-ncmoe": "n_cpu_moe",
    "--n-cpu-moe": "n_cpu_moe",
    "-ts": "tensor_split",
    "--tensor-split": "tensor_split",
    "-ot": "extra_overrides",
    "--override-tensor": "extra_overrides",
}

_BOOL_FLAGS: dict[str, tuple[str, bool]] = {
    "-cmoe": ("cpu_moe", True),
    "--cpu-moe": ("cpu_moe", True),
    "-nkvo": ("no_kv_offload", True),
    "--no-kv-offload": ("no_kv_offload", True),
    "--swa-full": ("swa_full", True),
}

# Flags that do not change the totals but change how host memory is held, which decides
# whether the host figure is evictable page cache or real resident memory.
_NOTE_FLAGS: dict[str, str] = {
    "--mlock": (
        "--mlock locks the weights in RAM, so the host figure is resident memory that "
        "cannot be evicted under pressure."
    ),
    "--no-mmap": (
        "--no-mmap loads the weights into anonymous memory, so the host figure is real RSS "
        "rather than evictable page cache."
    ),
}

_TRUTHY = {"on", "1", "true", "yes", "enabled"}
_FALSEY = {"off", "0", "false", "no", "disabled"}
_INT_FIELDS = {"n_gpu_layers", "n_ctx", "n_batch", "n_ubatch", "n_parallel", "n_cpu_moe"}


def parse_llama_server_args(args: str | list[str]) -> dict[str, Any]:
    """
    Parse a llama-server command line into :meth:`GgufMemoryEstimator.estimate` arguments.

    Accepts a shell string or an argv list, with or without a leading binary name, and
    handles every alias llama.cpp does (``-ngl`` / ``--gpu-layers`` / ``--n-gpu-layers``,
    ``--ctx-size=8192`` as well as ``-c 8192``). Later occurrences win, matching llama.cpp,
    which is also why ``llama_server_extra_args`` overrides the structured config at runtime.

    Flags that do not affect memory (``--host``, ``--jinja``, ...) are collected under
    ``ignored`` rather than dropped silently, so callers can see what was skipped.

    Returns:
        ``{"settings": {...}, "ignored": [...], "warnings": [...]}`` where ``settings`` is
        directly usable as ``**kwargs`` (it may contain ``model_path``).
    """
    raw_tokens = shlex.split(args) if isinstance(args, str) else [str(a) for a in args]
    settings: dict[str, Any] = {}
    ignored: list[str] = []
    warnings: list[str] = []

    # Split --flag=value into two tokens, then everything below is a flat scan.
    tokens: list[str] = []
    for token in raw_tokens:
        if token.startswith("--") and "=" in token:
            flag, _, value = token.partition("=")
            tokens.extend([flag, value])
        else:
            tokens.append(token)

    # Drop a leading binary path such as "llama-server" or "./build/bin/llama-server".
    if tokens and not tokens[0].startswith("-"):
        tokens = tokens[1:]

    index = 0
    while index < len(tokens):
        flag = tokens[index]
        index += 1
        if not flag.startswith("-"):
            ignored.append(flag)
            continue

        # A following token is this flag's value unless it is itself a flag. Negative
        # numbers such as "-ngl -1" look like flags, so they are checked numerically.
        raw: str | None = None
        if index < len(tokens) and _is_value_token(tokens[index]):
            raw = tokens[index]

        if flag in ("-fa", "--flash-attn"):
            # The value is optional on older builds, where a bare -fa means "on".
            if raw is not None and raw.lower() in _TRUTHY | _FALSEY | {"auto"}:
                index += 1
            else:
                raw = None
            if raw is None or raw.lower() in _TRUTHY:
                settings["flash_attn"] = True
            elif raw.lower() in _FALSEY:
                settings["flash_attn"] = False
            else:
                settings["flash_attn"] = True
                warnings.append(
                    "-fa auto resolves at runtime; assuming flash attention ends up enabled."
                )
            continue

        if flag in _NOTE_FLAGS:
            warnings.append(_NOTE_FLAGS[flag])
            continue

        if flag in _BOOL_FLAGS:
            key, value = _BOOL_FLAGS[flag]
            settings[key] = value
            continue

        key = _VALUE_FLAGS.get(flag)
        if key is None:
            # Consume the value too, so it is not mistaken for a positional argument.
            ignored.append(flag)
            if raw is not None:
                index += 1
            continue

        if raw is None:
            warnings.append(f"{flag} was given without a value and was ignored.")
            continue
        index += 1

        if key == "extra_overrides":
            patterns, skipped = _parse_tensor_overrides(raw)
            warnings.extend(skipped)
            if patterns:
                settings.setdefault("extra_overrides", []).extend(patterns)
        elif key == "tensor_split":
            try:
                settings[key] = [float(p) for p in raw.replace(";", ",").split(",") if p]
            except ValueError:
                warnings.append(f"Could not parse {flag} {raw}; ignoring the tensor split.")
        elif key in _INT_FIELDS:
            parsed = _parse_int_arg(flag, raw, warnings)
            if parsed is not None:
                settings[key] = parsed
        else:
            settings[key] = raw

    return {"settings": settings, "ignored": ignored, "warnings": warnings}


# estimate() arguments that sweep() sweeps over and recommend() solves for, so a value
# supplied on the command line cannot be honoured by those two endpoints.
SWEEP_SOLVED_KEYS = frozenset({"n_gpu_layers", "n_ctx", "cache_type_k", "cache_type_v"})
RECOMMEND_SOLVED_KEYS = frozenset(
    {"n_gpu_layers", "n_cpu_moe", "cpu_moe", "cache_type_k", "cache_type_v"}
)


def apply_cli_args(
    base: dict[str, Any],
    args: str | list[str] | None,
    *,
    solved_keys: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Overlay a parsed llama-server command line onto structured keyword arguments.

    Command-line values win, mirroring llama-server, which appends
    ``llama_server_extra_args`` after the arguments it builds from the config and so lets
    them override it. Keys the caller solves for are reported instead of applied.

    Returns:
        ``(merged_kwargs, report)`` where ``report`` carries ``parsed``, ``ignored`` and
        ``warnings`` for the response.
    """
    if not args:
        return dict(base), {}

    parsed = parse_llama_server_args(args)
    warnings = list(parsed["warnings"])
    merged = dict(base)
    applied: dict[str, Any] = {}
    for key, value in parsed["settings"].items():
        if key in solved_keys:
            warnings.append(
                f"'{key}' was given on the command line but this endpoint solves for it; "
                f"the supplied value ({value}) is ignored."
            )
            continue
        merged[key] = value
        applied[key] = value

    return merged, {
        "parsed": applied,
        "ignored": parsed["ignored"],
        "warnings": warnings,
    }


def _is_value_token(token: str) -> bool:
    """Whether a token is an argument value rather than the next flag."""
    if not token.startswith("-"):
        return True
    try:  # negative numbers, e.g. "-ngl -1"
        float(token)
    except ValueError:
        return False
    return True


def _parse_int_arg(flag: str, raw: str, warnings: list[str]) -> int | None:
    """Parse an integer flag value, translating llama.cpp's -ngl auto/all keywords."""
    lowered = raw.lower()
    if flag in ("-ngl", "--gpu-layers", "--n-gpu-layers") and lowered in ("auto", "all"):
        if lowered == "auto":
            warnings.append(
                "-ngl auto lets llama.cpp fit the layer count at startup; estimating full "
                "offload instead. Use the recommend endpoint to see what would be chosen."
            )
        return -1
    try:
        return int(raw)
    except ValueError:
        warnings.append(f"Could not parse {flag} {raw} as an integer; ignoring it.")
        return None


def _parse_tensor_overrides(raw: str) -> tuple[list[str], list[str]]:
    """
    Split an ``-ot pattern=buffer_type,...`` value into CPU-bound regex patterns.

    Only overrides that move tensors to the CPU change the GPU/host split in a way this
    estimator models; overrides onto another accelerator are reported instead.
    """
    patterns: list[str] = []
    warnings: list[str] = []
    for entry in raw.split(","):
        pattern, separator, buft = entry.rpartition("=")
        if not separator:
            warnings.append(f"-ot entry '{entry}' has no '=<buffer type>'; ignoring it.")
            continue
        if buft.strip().upper().startswith("CPU"):
            patterns.append(pattern)
        else:
            warnings.append(
                f"-ot '{entry}' targets {buft} rather than CPU; it is not modelled and the "
                f"affected tensors are counted on their default device."
            )
    return patterns, warnings


class GgufMemoryEstimator:
    """Estimates llama.cpp memory usage for GGUF models."""

    def __init__(self) -> None:
        self._info_cache: dict[tuple[str, float, int], GgufModelInfo] = {}

    # ------------------------------------------------------------------ model info

    def load_info(self, model_path: str) -> GgufModelInfo:
        """
        Read (and cache) the GGUF header, keyed by path plus mtime and size.

        Raises:
            FileNotFoundError: the path does not point at a readable file.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF file not found: {model_path}")
        stat = os.stat(model_path)
        key = (os.path.abspath(model_path), stat.st_mtime, stat.st_size)
        cached = self._info_cache.get(key)
        if cached is None:
            cached = read_gguf_model_info(model_path)
            self._info_cache[key] = cached
        return cached

    # ------------------------------------------------------------------ placement

    @staticmethod
    def resolve_n_gpu_layers(n_gpu_layers: int, n_block: int) -> int:
        """Resolve ``-ngl`` to the effective slot count, clamped to ``n_block + 1``."""
        effective = n_block + 1 if n_gpu_layers < 0 else n_gpu_layers
        return min(effective, n_block + 1)

    @staticmethod
    def _cumulative_splits(n_gpu: int, tensor_split: list[float] | None) -> list[float]:
        """Normalized cumulative split points, mirroring llama_model_base::load_tensors."""
        raw = list(tensor_split[:n_gpu]) if tensor_split else []
        if len(raw) < n_gpu or sum(raw) <= 0:
            raw = [1.0] * n_gpu
        total = 0.0
        cumulative: list[float] = []
        for value in raw:
            total += value
            cumulative.append(total)
        return [c / total for c in cumulative]

    def _slot_devices(
        self,
        n_block: int,
        n_gpu_layers: int,
        n_gpu: int,
        tensor_split: list[float] | None,
    ) -> dict[int, int]:
        """
        Map each slot onto a GPU index.

        Slots absent from the result live on the host. Slot ``n_block`` is the output layer;
        the input layer is never included because llama.cpp always keeps it on the CPU.
        """
        if n_gpu <= 0:
            return {}
        effective = self.resolve_n_gpu_layers(n_gpu_layers, n_block)
        if effective <= 0:
            return {}

        i_gpu_start = max(n_block + 1 - effective, 0)
        splits = self._cumulative_splits(n_gpu, tensor_split)
        devices: dict[int, int] = {}
        for slot in range(i_gpu_start, n_block + 1):
            position = (slot - i_gpu_start) / effective
            index = min(bisect.bisect_right(splits, position), n_gpu - 1)
            devices[slot] = index
        return devices

    @staticmethod
    def _cpu_override_patterns(
        n_cpu_moe: int, cpu_moe: bool, extra_overrides: list[str] | None
    ) -> list[re.Pattern[str]]:
        """Build the CPU tensor-override regexes for --cpu-moe / --n-cpu-moe / -ot."""
        exps = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
        patterns: list[str] = []
        if cpu_moe:
            patterns.append(exps)
        elif n_cpu_moe > 0:
            patterns.extend(rf"blk\.{i}{exps}" for i in range(n_cpu_moe))
        if extra_overrides:
            patterns.extend(extra_overrides)
        return [re.compile(p) for p in patterns]

    # ------------------------------------------------------------------ estimation

    def estimate(  # noqa: PLR0913 - mirrors the llama.cpp CLI surface
        self,
        model_path: str,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 0,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        n_cpu_moe: int = 0,
        cpu_moe: bool = False,
        flash_attn: bool = True,
        n_parallel: int = 1,
        no_kv_offload: bool = False,
        swa_full: bool = False,
        n_gpu: int = 1,
        tensor_split: list[float] | None = None,
        extra_overrides: list[str] | None = None,
        include_per_layer: bool = True,
    ) -> dict[str, Any]:
        """
        Estimate per-device memory for one llama.cpp configuration.

        Returns a dict with ``memory_breakdown_mib``, ``placement``, ``model_info`` and,
        when requested, a ``per_layer_mib`` table. Returns ``{"error": ...}`` on failure.
        """
        try:
            info = self.load_info(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e), "model_path": model_path}

        try:
            bpe_k = kv_bytes_per_element(cache_type_k)
            bpe_v = kv_bytes_per_element(cache_type_v)
        except ValueError as e:
            return {"error": str(e), "model_path": model_path}

        notes = list(info.warnings)
        n_gpu = max(n_gpu, 0)
        effective_ngl = self.resolve_n_gpu_layers(n_gpu_layers, info.n_block)
        slot_devices = self._slot_devices(info.n_block, n_gpu_layers, n_gpu, tensor_split)
        overrides = self._cpu_override_patterns(n_cpu_moe, cpu_moe, extra_overrides)

        gpu_model = [0] * max(n_gpu, 1)
        host_model = 0
        overridden_bytes = 0
        largest_host_tensor = 0
        for tensor in info.tensors:
            slot_device = slot_devices.get(tensor.slot)
            device = slot_device
            if device is not None and overrides and any(p.search(tensor.name) for p in overrides):
                overridden_bytes += tensor.n_bytes
                device = None
            if device is None:
                host_model += tensor.n_bytes
                # Staging applies only to blocks that -ngl left off the GPU: those run on the
                # CPU backend and their weights are copied in per op. A buffer-type override
                # (--n-cpu-moe / -ot) keeps the layer on the GPU device, and measurement
                # confirms it does not change the compute buffer at all. Input and output
                # projections are accounted for separately.
                if slot_device is None and 0 <= tensor.slot < info.n_block:
                    largest_host_tensor = max(largest_host_tensor, tensor.n_bytes)
            else:
                gpu_model[device] += tensor.n_bytes

        resolved_ctx = n_ctx if n_ctx > 0 else info.n_ctx_train
        n_ctx_pad = max(int(math.ceil(resolved_ctx / N_CTX_PAD)) * N_CTX_PAD, N_CTX_PAD)
        n_swa_cells = n_ctx_pad
        if info.n_swa > 0 and not swa_full:
            raw = min(n_ctx_pad, info.n_swa * max(n_parallel, 1) + n_ubatch)
            n_swa_cells = max(int(math.ceil(raw / N_CTX_PAD)) * N_CTX_PAD, N_CTX_PAD)

        gpu_kv = [0.0] * max(n_gpu, 1)
        host_kv = 0.0
        per_layer_kv: list[float] = []
        for il in range(info.n_block):
            if info.is_recurrent_layer(il):
                # Linear-attention layer: fixed-size state, no growth with the context.
                layer_kv = float(info.recurrent_state_bytes(n_parallel))
            else:
                heads_kv = info.n_head_kv_at(il) if info.has_kv(il) else 0
                key_len, value_len = info.kv_dims_at(il)
                cells = n_swa_cells if info.is_swa(il) else n_ctx_pad
                layer_kv = cells * heads_kv * (key_len * bpe_k + value_len * bpe_v)
            per_layer_kv.append(layer_kv)
            device = slot_devices.get(il)
            if device is None or no_kv_offload:
                host_kv += layer_kv
            else:
                gpu_kv[device] += layer_kv

        gpu_compute, host_compute = self._compute_buffers(
            info,
            n_ctx_pad=n_ctx_pad,
            n_ubatch=n_ubatch,
            flash_attn=flash_attn,
            n_gpu=n_gpu,
            output_device=slot_devices.get(info.n_block),
            largest_host_tensor=largest_host_tensor,
        )

        if info.n_swa > 0 and not swa_full:
            swa_dims = ""
            if info.key_length_swa and info.key_length_swa != info.key_length:
                swa_dims = (
                    f" and a {info.key_length_swa}-wide key/value head instead of {info.key_length}"
                )
            notes.append(
                f"SWA model: {sum(info.swa_layers)}/{info.n_block} layers use a "
                f"{n_swa_cells}-cell sliding-window cache instead of {n_ctx_pad} cells"
                f"{swa_dims}."
            )
        n_recurrent = sum(1 for il in range(info.n_block) if info.is_recurrent_layer(il))
        if n_recurrent:
            state_mib = _to_mib(info.recurrent_state_bytes(n_parallel) * n_recurrent)
            notes.append(
                f"Hybrid attention: {n_recurrent}/{info.n_block} layers use linear attention "
                f"and hold {state_mib} MiB of recurrent state in total, which does not grow "
                f"with the context. Only the other "
                f"{info.n_block - n_recurrent} layers hold a KV cache."
            )
        shared_kv = sum(
            1
            for il in range(info.n_block)
            if not info.has_kv(il) and not info.is_recurrent_layer(il)
        )
        if shared_kv > 0:
            notes.append(
                f"The last {shared_kv} of {info.n_block} layers share an earlier layer's KV "
                f"cache and allocate none of their own."
            )
        if info.tied_embedding:
            tied_mib = _to_mib(info.output_weight_bytes)
            notes.append(
                f"Tied embeddings: the model has no output.weight, so llama.cpp allocates a "
                f"second copy of token_embd ({tied_mib} MiB) as the output projection. It is "
                f"counted here even though it is not in the file."
            )
        if overridden_bytes:
            sources = []
            if cpu_moe:
                sources.append("MoE expert weights of all layers")
            elif n_cpu_moe:
                sources.append(f"MoE expert weights of the first {n_cpu_moe} layer(s)")
            if extra_overrides:
                sources.append(f"tensors matching {', '.join(extra_overrides)}")
            notes.append(
                f"{' and '.join(sources).capitalize()} ({_to_mib(overridden_bytes)} MiB) were "
                f"moved to host RAM; attention weights and the KV cache for those layers stay "
                f"on the GPU."
            )
        if (n_cpu_moe > 0 or cpu_moe) and not info.is_moe:
            notes.append("Model has no routed experts, so --n-cpu-moe / --cpu-moe has no effect.")
        if n_gpu > 0 and effective_ngl == 0:
            notes.append(
                "With -ngl 0 no weights are resident on the GPU, but llama.cpp still stages "
                "the output projection there for batch processing, so the GPU compute buffer "
                "is not zero. Pass n_gpu=0 to model a CPU-only build."
            )
        if effective_ngl >= 1:
            notes.append(
                "llama.cpp offloads the output layer first: -ngl N places the output layer "
                f"plus the last {max(effective_ngl - 1, 0)} of {info.n_block} blocks on GPU."
            )
        notes.append("Token embeddings (token_embd.weight) always stay on the host.")
        if flash_attn and n_gpu > 0:
            without_fa = _to_mib(4.0 * n_ubatch * n_ctx_pad * info.n_head)
            notes.append(
                f"Assumes flash attention is active (llama.cpp's -fa default is 'auto'). If it "
                f"falls back to the non-flash path, the GPU compute buffer grows by roughly "
                f"{without_fa} MiB at this context and ubatch."
            )

        # Slot index of the first GPU-resident layer; n_block + 1 means "nothing offloaded".
        i_gpu_start = max(info.n_block + 1 - effective_ngl, 0) if n_gpu else info.n_block + 1
        if cpu_moe:
            moe_cpu_layers = list(range(info.n_block))
        elif n_cpu_moe:
            moe_cpu_layers = list(range(min(n_cpu_moe, info.n_block)))
        else:
            moe_cpu_layers = []

        gpus = [
            {
                "index": i,
                "model": _to_mib(gpu_model[i]),
                "context": _to_mib(gpu_kv[i]),
                "compute": _to_mib(gpu_compute[i]),
                "total": _to_mib(gpu_model[i] + gpu_kv[i] + gpu_compute[i]),
            }
            for i in range(n_gpu)
        ]
        host = {
            "model": _to_mib(host_model),
            "context": _to_mib(host_kv),
            "compute": _to_mib(host_compute),
            "total": _to_mib(host_model + host_kv + host_compute),
        }

        result: dict[str, Any] = {
            "source": "analytic",
            "model_path": info.path,
            "settings": {
                "n_gpu_layers": n_gpu_layers,
                "n_gpu_layers_effective": effective_ngl,
                "n_ctx": resolved_ctx,
                "n_ctx_padded": n_ctx_pad,
                "n_batch": n_batch,
                "n_ubatch": n_ubatch,
                "cache_type_k": cache_type_k,
                "cache_type_v": cache_type_v,
                "n_cpu_moe": n_cpu_moe,
                "cpu_moe": cpu_moe,
                "flash_attn": flash_attn,
                "n_parallel": n_parallel,
                "no_kv_offload": no_kv_offload,
                "swa_full": swa_full,
                "n_gpu": n_gpu,
                "extra_overrides": extra_overrides or [],
            },
            "model_info": self._model_info_dict(info),
            "memory_breakdown_mib": {
                "gpu": gpus,
                "gpu_total": {
                    "model": _to_mib(sum(gpu_model[:n_gpu])),
                    "context": _to_mib(sum(gpu_kv[:n_gpu])),
                    "compute": _to_mib(sum(gpu_compute[:n_gpu])),
                    "total": sum(g["total"] for g in gpus),
                },
                "host": host,
            },
            "placement": {
                "i_gpu_start": i_gpu_start,
                "blocks_on_gpu": sum(1 for s in slot_devices if s < info.n_block),
                "output_on_gpu": info.n_block in slot_devices,
                "input_on_gpu": False,
                "moe_cpu_layers": moe_cpu_layers,
            },
            "notes": notes,
        }

        if include_per_layer:
            layer_bytes = info.layer_bytes()
            result["per_layer_mib"] = [
                {
                    "index": il,
                    "device": f"gpu{slot_devices[il]}" if il in slot_devices else "host",
                    "attn": _to_mib(layer_bytes[il]["attn"]),
                    "ffn": _to_mib(layer_bytes[il]["ffn"]),
                    "moe_exps": _to_mib(layer_bytes[il]["moe_exps"]),
                    "shexp": _to_mib(layer_bytes[il]["shexp"]),
                    "other": _to_mib(layer_bytes[il]["other"]),
                    "weights": _to_mib(layer_bytes[il]["total"]),
                    "kv": _to_mib(per_layer_kv[il]),
                    "swa": info.is_swa(il),
                }
                for il in range(info.n_block)
            ]
        return result

    def _compute_buffers(
        self,
        info: GgufModelInfo,
        *,
        n_ctx_pad: int,
        n_ubatch: int,
        flash_attn: bool,
        n_gpu: int,
        output_device: int | None,
        largest_host_tensor: int = 0,
    ) -> tuple[list[float], float]:
        """
        Estimate the ggml compute buffers.

        The host term is exact against llama-fit-params: two f16 masks plus an f32 residual
        pair. The device term is the larger of the f32 logits tensor and the attention mask,
        because the allocator reuses one buffer for both, plus the attention scores when
        flash attention is off. It is charged to whichever device computes the final
        projection and carries a safety factor so it errs high.

        When a GPU exists but holds no layer, llama.cpp still stages the output projection
        matrix in VRAM (measured: 1094 MiB at -ngl 0 on Qwen3-32B, of which 788 MiB is
        ``output.weight``), so that copy is counted too.

        Under partial offload, host-resident weights are copied into VRAM op by op, which
        needs a double-buffered staging area sized by the largest such tensor. Without this
        term the estimate under-reports partial offload by up to 50%, and an under-report is
        what makes a budget-targeted recommendation overshoot its cap.
        """
        host = 4.0 * n_ctx_pad * n_ubatch + 8.0 * n_ubatch * info.n_embd

        logits = 4.0 * info.n_vocab * n_ubatch
        # The mask width is padded, so a ubatch below the floor costs the same as the floor.
        mask_width = max(n_ubatch, _COMPUTE_MASK_UBATCH_FLOOR)
        mask = float(_COMPUTE_MASK_BYTES_PER_CELL) * n_ctx_pad * mask_width
        scores = 0.0 if flash_attn else 4.0 * n_ubatch * n_ctx_pad * info.n_head
        device_term = _COMPUTE_SAFETY_FACTOR * (max(logits, mask) + scores)

        gpu = [0.0] * max(n_gpu, 1)
        if n_gpu <= 0:
            return gpu, host + device_term

        index = output_device if output_device is not None else 0
        gpu[index] += device_term + _COMPUTE_STAGING_BUFFERS * largest_host_tensor
        if output_device is None:
            gpu[index] += info.output_weight_bytes
        return gpu, host

    @staticmethod
    def _model_info_dict(info: GgufModelInfo) -> dict[str, Any]:
        """Serializable summary of the GGUF hyper-parameters."""
        return {
            "architecture": info.arch,
            "n_block": info.n_block,
            "n_embd": info.n_embd,
            "n_head": info.n_head,
            "n_head_kv": info.n_head_kv[0] if info.n_head_kv else 0,
            "key_length": info.key_length,
            "value_length": info.value_length,
            "n_vocab": info.n_vocab,
            "n_ctx_train": info.n_ctx_train,
            "n_expert": info.n_expert,
            "n_expert_used": info.n_expert_used,
            "key_length_swa": info.key_length_swa or info.key_length,
            "value_length_swa": info.value_length_swa or info.value_length,
            "n_swa": info.n_swa,
            "n_swa_layers": sum(info.swa_layers),
            "n_layer_kv": sum(1 for il in range(info.n_block) if info.has_kv(il)),
            "is_moe": info.is_moe,
            "is_recurrent": info.is_recurrent,
            "tied_embedding": info.tied_embedding,
            "file_size_mib": _to_mib(info.file_bytes),
            "weights_mib": _to_mib(info.total_bytes),
            "n_shards": len(info.shard_paths),
        }

    # ------------------------------------------------------------------ exact probe

    def _resolve_fit_binary(self) -> str | None:
        """
        Locate a binary that can run the fit-params probe.

        Mirrors ConversionManager._resolve_quantize_binary: an explicit ``LLAMA_FIT_PARAMS_BIN``
        wins, then a legacy standalone ``llama-fit-params`` from a source build, then the
        official prebuilt unified ``llama`` (from ggml-org/llama-install.sh), which exposes it
        as a ``fit-params`` subcommand — see `_fit_cmd_prefix`. No source build required, which
        matters because setup_env only sparse-checks out the Python convert tooling, never a
        C++ build tree.
        """
        llama_cpp_dir = str(settings.LLAMA_CPP_DIR)
        names = ["llama-fit-params.exe"] if os.name == "nt" else ["llama-fit-params"]
        candidates: list[str] = []

        configured = os.getenv("LLAMA_FIT_PARAMS_BIN", "").strip()
        if configured:
            candidates.append(configured)
        for name in names:
            candidates.extend(
                [
                    os.path.join(llama_cpp_dir, "build", "bin", "Release", name),
                    os.path.join(llama_cpp_dir, "build", "bin", name),
                    os.path.join(llama_cpp_dir, "bin", name),
                    os.path.join(llama_cpp_dir, name),
                ]
            )

        # Prebuilt unified binary (fit-params is a subcommand there). LLAMA_SERVER_BINARY
        # already resolves it, but only trust that path when it really is the unified
        # `llama` — a self-built `llama-server` has no fit-params subcommand.
        server_binary = str(settings.LLAMA_SERVER_BINARY)
        if os.path.basename(server_binary).lower() in {"llama", "llama.exe"}:
            candidates.append(server_binary)
        if os.name == "nt":
            local_app = os.environ.get("LOCALAPPDATA")
            if local_app:
                candidates.append(os.path.join(local_app, "Microsoft", "WindowsApps", "llama.exe"))
        else:
            candidates.append(os.path.join(os.path.expanduser("~"), ".local", "bin", "llama"))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
        return None

    @staticmethod
    def _fit_cmd_prefix(binary: str) -> list[str]:
        """
        Command prefix for invoking the fit-params probe.

        The legacy standalone ``llama-fit-params`` takes args directly; the unified
        ``llama`` binary needs a ``fit-params`` subcommand.
        """
        if "fit-params" in os.path.basename(binary).lower():
            return [binary]
        return [binary, "fit-params"]

    @staticmethod
    def _fit_env(binary: str) -> dict[str, str]:
        """Environment for the probe, with the llama.cpp shared libraries on the path."""
        env = os.environ.copy()
        lib_dir = os.path.dirname(binary)
        if os.name != "nt" and os.path.isdir(lib_dir):
            existing = env.get("LD_LIBRARY_PATH", "").strip()
            env["LD_LIBRARY_PATH"] = ":".join([lib_dir, *([existing] if existing else [])])
        return env

    def probe_exact(  # noqa: PLR0913 - mirrors the llama.cpp CLI surface
        self,
        model_path: str,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 0,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        n_cpu_moe: int = 0,
        cpu_moe: bool = False,
        flash_attn: bool = True,
        n_parallel: int = 1,
        no_kv_offload: bool = False,
        swa_full: bool = False,
        extra_overrides: list[str] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any] | None:
        """
        Ask llama.cpp itself for the exact memory breakdown.

        Runs ``llama-fit-params -fitp on``, which loads the model with ``no_alloc`` and prints
        ``<device> <model> <context> <compute>`` in MiB per device followed by a ``Host`` row.
        Returns ``None`` if the binary is missing or the probe fails, so callers can fall
        back to the analytic estimate.
        """
        binary = self._resolve_fit_binary()
        if binary is None:
            logger.debug("[GgufEstimator] llama-fit-params not found; skipping exact probe")
            return None

        cmd = [
            *self._fit_cmd_prefix(binary),
            "-m", model_path,
            "-fitp", "on",
            "-ngl", str(n_gpu_layers),
        ]  # fmt: skip
        if n_ctx > 0:
            cmd += ["-c", str(n_ctx)]
        cmd += [
            "-b", str(n_batch),
            "-ub", str(n_ubatch),
            "-ctk", cache_type_k,
            "-ctv", cache_type_v,
            "-np", str(n_parallel),
            "-fa", "on" if flash_attn else "off",
        ]  # fmt: skip
        if cpu_moe:
            cmd.append("--cpu-moe")
        elif n_cpu_moe > 0:
            cmd += ["-ncmoe", str(n_cpu_moe)]
        if no_kv_offload:
            cmd.append("--no-kv-offload")
        if swa_full:
            cmd.append("--swa-full")
        if extra_overrides:
            cmd += ["-ot", ",".join(f"{p}=CPU" for p in extra_overrides)]

        try:
            proc = subprocess.run(  # noqa: S603 - fixed binary, numeric/enum arguments
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=self._fit_env(binary),
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"[GgufEstimator] llama-fit-params probe failed: {e}")
            return None

        if proc.returncode != 0:
            logger.warning(
                f"[GgufEstimator] llama-fit-params exited {proc.returncode}: "
                f"{proc.stderr.strip()[-500:]}"
            )
            return None

        row = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
        gpus: list[dict[str, Any]] = []
        host: dict[str, Any] | None = None
        for line in proc.stdout.splitlines():
            match = row.match(line.strip())
            if match is None:
                continue
            device, model, context, compute = match.groups()
            entry = {
                "model": float(model),
                "context": float(context),
                "compute": float(compute),
                "total": float(model) + float(context) + float(compute),
            }
            if device == "Host":
                host = entry
            else:
                gpus.append({"index": len(gpus), "device": device, **entry})

        if host is None:
            logger.warning("[GgufEstimator] Could not parse llama-fit-params output")
            return None

        return {
            "source": "llama_cpp",
            "binary": binary,
            "memory_breakdown_mib": {
                "gpu": gpus,
                "gpu_total": {
                    "model": sum(g["model"] for g in gpus),
                    "context": sum(g["context"] for g in gpus),
                    "compute": sum(g["compute"] for g in gpus),
                    "total": sum(g["total"] for g in gpus),
                },
                "host": host,
            },
        }

    # ------------------------------------------------------------------ sweep

    def sweep(
        self,
        model_path: str,
        *,
        n_gpu_layers_grid: list[int] | None = None,
        n_ctx_grid: list[int] | None = None,
        kv_quant_grid: list[str] | None = None,
        gpu_budget_mib: float | None = None,
        host_budget_mib: float | None = None,
        margin_mib: float = DEFAULT_MARGIN_MIB,
        **estimate_kwargs: Any,  # noqa: ANN401 - forwarded to estimate()
    ) -> dict[str, Any]:
        """
        Evaluate a grid of ``-ngl`` x ``-c`` x KV-quant combinations.

        Analytic only, so a full grid costs no subprocesses. Each row reports the breakdown
        and whether it fits the given budgets. Returns ``{"error": ...}`` on failure.
        """
        try:
            info = self.load_info(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e), "model_path": model_path}

        slots = info.n_block + 1
        if n_gpu_layers_grid is None:
            n_gpu_layers_grid = sorted({0, 1, *(round(slots * i / 8) for i in range(1, 9))})
        if n_ctx_grid is None:
            ceiling = info.n_ctx_train or 32768
            n_ctx_grid = [c for c in (4096, 8192, 16384, 32768, 65536, 131072) if c <= ceiling]
            n_ctx_grid = n_ctx_grid or [ceiling]
        if kv_quant_grid is None:
            kv_quant_grid = ["f16", "q8_0"]

        budgets = self.resolve_budgets(gpu_budget_mib, host_budget_mib)
        gpu_budget = budgets["gpu_budget_mib"]
        host_budget = budgets["host_budget_mib"]

        rows: list[dict[str, Any]] = []
        truncated = False
        for ngl in n_gpu_layers_grid:
            for ctx in n_ctx_grid:
                for kv in kv_quant_grid:
                    if len(rows) >= MAX_SWEEP_ROWS:
                        truncated = True
                        break
                    est = self.estimate(
                        model_path,
                        n_gpu_layers=ngl,
                        n_ctx=ctx,
                        cache_type_k=kv,
                        cache_type_v=kv,
                        include_per_layer=False,
                        **estimate_kwargs,
                    )
                    if "error" in est:
                        return est
                    breakdown = est["memory_breakdown_mib"]
                    gpu_total = breakdown["gpu_total"]["total"]
                    host_total = breakdown["host"]["total"]
                    fits_gpu = gpu_budget is None or gpu_total + margin_mib <= gpu_budget
                    fits_host = host_budget is None or host_total + margin_mib <= host_budget
                    rows.append(
                        {
                            "n_gpu_layers": ngl,
                            "n_ctx": ctx,
                            "kv_quant": kv,
                            "gpu_mib": gpu_total,
                            "host_mib": host_total,
                            "gpu_model": breakdown["gpu_total"]["model"],
                            "gpu_context": breakdown["gpu_total"]["context"],
                            "gpu_compute": breakdown["gpu_total"]["compute"],
                            "fits_gpu": fits_gpu,
                            "fits_host": fits_host,
                        }
                    )

        if truncated:
            logger.warning(
                f"[GgufEstimator] Sweep truncated at {MAX_SWEEP_ROWS} rows; narrow the grid"
            )

        return {
            "source": "analytic",
            "model_path": info.path,
            "model_info": self._model_info_dict(info),
            "grid": {
                "n_gpu_layers": n_gpu_layers_grid,
                "n_ctx": n_ctx_grid,
                "kv_quant": kv_quant_grid,
            },
            "budgets_mib": budgets,
            "margin_mib": margin_mib,
            "rows": rows,
            "truncated": truncated,
        }

    # ------------------------------------------------------------------ recommend

    @staticmethod
    def resolve_budgets(
        gpu_budget_mib: float | None, host_budget_mib: float | None
    ) -> dict[str, Any]:
        """Fill in missing budgets from live hardware, via the shared system monitor."""
        result: dict[str, Any] = {
            "gpu_budget_mib": gpu_budget_mib,
            "host_budget_mib": host_budget_mib,
            "source": "caller",
        }
        if gpu_budget_mib is not None and host_budget_mib is not None:
            return result

        try:
            from ..utils.system_monitor import system_monitor

            if gpu_budget_mib is None:
                gpu = system_monitor.get_gpu_resource("usage")
                if getattr(gpu, "available", False) and gpu.gpus:
                    # free_gb is None on backends that cannot read per-device usage
                    # (Intel/DXGI and the generic GPU path); skip those instead of
                    # summing None, and leave the budget unset if none are readable.
                    readable = [g.free_gb for g in gpu.gpus if g.free_gb is not None]
                    if readable:
                        result["gpu_budget_mib"] = round(sum(readable) * 1024, 2)
                        result["gpu_count"] = len(readable)
                        result["source"] = "system_monitor"
            if host_budget_mib is None:
                dram = getattr(system_monitor.get_cpu_resource("usage"), "dram", None)
                free_gb = getattr(dram, "free_gb", None) if dram is not None else None
                if free_gb is not None:
                    result["host_budget_mib"] = round(free_gb * 1024, 2)
                    result["source"] = "system_monitor"
        except Exception as e:
            logger.warning(f"[GgufEstimator] Could not read hardware budgets: {e}")
        return result

    def recommend(  # noqa: PLR0913 - one knob per lever the search may pull
        self,
        model_path: str,
        *,
        n_ctx: int = 0,
        n_ctx_min: int = 4096,
        n_ctx_max: int = 0,
        gpu_budget_mib: float | None = None,
        host_budget_mib: float | None = None,
        margin_mib: float = DEFAULT_MARGIN_MIB,
        allow_kv_quant: bool = True,
        allow_ctx_reduction: bool = True,
        target_utilization: float = 0.9,
        kv_cache_types: list[str] | None = None,
        verify: bool = True,
        **estimate_kwargs: Any,  # noqa: ANN401 - forwarded to estimate()
    ) -> dict[str, Any]:
        """
        Find llama-server settings that use a GPU budget as fully as possible.

        The budget is a ceiling: the result never exceeds ``gpu_budget_mib - margin_mib``,
        but the search also tries to land *close* to it rather than merely under it.

        Three steps:

        1. Pick a working context. When ``n_ctx`` is pinned it is honoured, and only reduced
           (by binary search, not by halving) if it cannot fit and ``allow_ctx_reduction``
           is set. When ``n_ctx`` is 0 the search starts from ``n_ctx_min``.
        2. Maximise offload at that context - ``--n-cpu-moe`` for MoE models, because that
           keeps attention and the KV cache on the GPU, otherwise ``-ngl``. Layers are
           prioritised over context because offload dominates throughput.
        3. When ``n_ctx`` was left at 0, grow the context back up into whatever budget is
           still unused, capped at ``n_ctx_max`` (default: the model's trained length).

        ``target_utilization`` does not constrain the search; it only decides whether the
        result is reported as leaving unexplained headroom, together with the reason.

        ``kv_cache_types`` overrides the KV-cache ladder the search walks (first entry that
        fits wins). Pass a single type to pin it, which is how ``plan()`` produces one
        candidate per quantization.

        Returns the chosen settings, the binding constraint, utilisation, the resulting
        breakdown, and a ready-to-paste ``llama_server_args`` string.
        """
        try:
            info = self.load_info(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e), "model_path": model_path}

        budgets = self.resolve_budgets(gpu_budget_mib, host_budget_mib)
        gpu_budget = budgets["gpu_budget_mib"]
        if gpu_budget is None:
            return {
                "error": "No GPU memory budget available; pass gpu_budget_mib explicitly",
                "model_path": info.path,
                "budgets_mib": budgets,
            }
        host_budget = budgets["host_budget_mib"]
        slots = info.n_block + 1
        target = gpu_budget - margin_mib

        def gpu_total(**kwargs: Any) -> tuple[float, dict[str, Any]]:  # noqa: ANN401
            est = self.estimate(
                model_path, include_per_layer=False, **{**estimate_kwargs, **kwargs}
            )
            return est["memory_breakdown_mib"]["gpu_total"]["total"], est

        pinned = n_ctx > 0
        ctx_floor = min(n_ctx, n_ctx_min) if pinned else n_ctx_min
        ctx_ceiling = n_ctx_max or info.n_ctx_train or n_ctx_min
        if kv_cache_types:
            kv_candidates = list(kv_cache_types)
            unknown = [kv for kv in kv_candidates if kv not in KV_CACHE_TYPES]
            if unknown:
                return {
                    "error": (
                        f"Unsupported KV cache type(s) {unknown}; llama.cpp accepts "
                        f"{sorted(KV_CACHE_TYPES)}"
                    ),
                    "model_path": info.path,
                }
        else:
            kv_candidates = ["f16", "q8_0"] if allow_kv_quant else ["f16"]

        def solve(budget_target: float) -> dict[str, Any] | None:
            """Best configuration whose analytic GPU total stays under ``budget_target``."""
            for kv in kv_candidates:
                kv_args = {"cache_type_k": kv, "cache_type_v": kv}

                # Step 1: the context to size the offload search against.
                start_ctx = n_ctx if pinned else n_ctx_min
                if gpu_total(n_gpu_layers=0, n_ctx=start_ctx, **kv_args)[0] > budget_target:
                    if not (pinned and allow_ctx_reduction):
                        continue
                    reduced = self._search_max_ctx(
                        gpu_total, {"n_gpu_layers": 0}, kv_args, ctx_floor, n_ctx, budget_target
                    )
                    if reduced is None:
                        continue
                    start_ctx = reduced

                # Step 2: maximise offload at that context.
                common = {"n_ctx": start_ctx, **kv_args}
                offload: dict[str, Any] | None = None
                constraint = ""
                if info.is_moe:
                    ncmoe = self._search_min_cpu_moe(gpu_total, info.n_block, budget_target, common)
                    if ncmoe is not None:
                        offload = {"n_gpu_layers": -1, "n_cpu_moe": ncmoe}
                        constraint = "n_cpu_moe" if ncmoe else "full offload"
                if offload is None:
                    ngl = self._search_max_ngl(gpu_total, slots, budget_target, common)
                    if ngl is None:
                        continue
                    offload = {"n_gpu_layers": ngl, "n_cpu_moe": 0}
                    constraint = "n_gpu_layers" if ngl < slots else "full offload"

                # Step 3: spend the leftover budget on context, unless the caller pinned it.
                chosen_ctx = start_ctx
                if not pinned and ctx_ceiling > start_ctx:
                    grown = self._search_max_ctx(
                        gpu_total, offload, kv_args, start_ctx, ctx_ceiling, budget_target
                    )
                    if grown is not None and grown > chosen_ctx:
                        chosen_ctx = grown
                        if chosen_ctx >= ctx_ceiling:
                            constraint = "n_ctx ceiling"
                        elif constraint == "full offload":
                            constraint = "n_ctx"

                used, est = gpu_total(n_ctx=chosen_ctx, **offload, **kv_args)
                return {
                    **offload,
                    "n_ctx": chosen_ctx,
                    "kv_quant": kv,
                    "gpu_mib": used,
                    "estimate": est,
                    "constraint": constraint,
                }
            return None

        best = solve(target)

        # Close the loop: the analytic compute buffer is approximate, and when the goal is to
        # land near the cap an under-estimate would overshoot it. Ask llama.cpp for the real
        # figure and, if it exceeds the cap, re-solve against a target tightened by the
        # difference. The correction is nearly constant, so this converges in a round or two.
        correction_mib = 0.0
        exact: dict[str, Any] | None = None
        if verify and best is not None:
            for _ in range(_VERIFY_CORRECTION_ROUNDS):
                exact = self.probe_exact(
                    model_path,
                    n_gpu_layers=best["n_gpu_layers"],
                    n_ctx=best["n_ctx"],
                    cache_type_k=best["kv_quant"],
                    cache_type_v=best["kv_quant"],
                    n_cpu_moe=best["n_cpu_moe"],
                    n_batch=estimate_kwargs.get("n_batch", 2048),
                    n_ubatch=estimate_kwargs.get("n_ubatch", 512),
                    flash_attn=estimate_kwargs.get("flash_attn", True),
                    n_parallel=estimate_kwargs.get("n_parallel", 1),
                )
                if exact is None:
                    break
                exact_total = exact["memory_breakdown_mib"]["gpu_total"]["total"]
                if exact_total <= target:
                    break
                correction_mib += exact_total - best["gpu_mib"]
                retried = solve(target - correction_mib)
                if retried is None:
                    break
                best = retried

        if best is None:
            return {
                "error": (
                    f"No configuration fits {gpu_budget:.0f} MiB with a {margin_mib:.0f} MiB "
                    f"margin, even at -ngl 0 and n_ctx {ctx_floor}"
                ),
                "model_path": info.path,
                "model_info": self._model_info_dict(info),
                "budgets_mib": budgets,
            }

        estimate = best.pop("estimate")
        host_mib = estimate["memory_breakdown_mib"]["host"]["total"]
        args = self._format_server_args(best, flash_attn=estimate_kwargs.get("flash_attn", True))

        # Utilisation is reported against the measured figure when one is available, since
        # that is what the GPU will actually hold.
        exact_total: float | None = None
        if exact is not None:
            exact_total = exact["memory_breakdown_mib"]["gpu_total"]["total"]
        allocated = exact_total if exact_total is not None else best["gpu_mib"]

        usable = max(target, 0.0)
        utilization = (allocated / usable) if usable > 0 else 0.0
        headroom = round(usable - allocated, 2)

        result: dict[str, Any] = {
            "source": "analytic",
            "model_path": info.path,
            "model_info": self._model_info_dict(info),
            "budgets_mib": budgets,
            "margin_mib": margin_mib,
            "recommended": best,
            "utilization": {
                "gpu_budget_mib": gpu_budget,
                "usable_budget_mib": round(usable, 2),
                "allocated_mib": round(allocated, 2),
                "allocated_source": "llama_cpp" if exact_total is not None else "analytic",
                "analytic_mib": best["gpu_mib"],
                "headroom_mib": headroom,
                "utilization_pct": round(utilization * 100, 1),
                "within_budget": allocated <= usable,
                "meets_target": utilization >= target_utilization,
                "target_pct": round(target_utilization * 100, 1),
                "correction_mib": round(correction_mib, 2),
            },
            "memory_breakdown_mib": estimate["memory_breakdown_mib"],
            "placement": estimate["placement"],
            "host_fits": host_budget is None or host_mib <= host_budget,
            "llama_server_args": args,
            "notes": estimate["notes"],
        }

        if utilization < target_utilization:
            result["notes"].append(
                f"Allocates {allocated:.0f} of {usable:.0f} MiB usable VRAM "
                f"({utilization:.0%}), leaving {headroom:.0f} MiB unused. "
                f"{self._headroom_reason(best, info, pinned, ctx_ceiling)}"
            )
        if correction_mib:
            result["notes"].append(
                f"llama.cpp reported {correction_mib:.0f} MiB more than the analytic estimate, "
                f"so the search was re-run against a target tightened by that amount."
            )
        if exact_total is not None and allocated > usable:
            result["notes"].append(
                f"WARNING: llama.cpp would allocate {allocated:.0f} MiB, which eats into the "
                f"{margin_mib:.0f} MiB margin ({usable:.0f} MiB usable). The correction loop "
                f"could not converge; lower n_ctx or n_ubatch by hand, or raise margin_mib."
            )
        if info.is_recurrent:
            result["notes"].append(
                "This architecture allocates recurrent/SSM state that the analytic model does "
                "not cover; trust the verified figure over the analytic one."
            )

        if host_budget is not None and host_mib > host_budget:
            result["notes"].append(
                f"Host RAM is short: needs {host_mib:.0f} MiB but only {host_budget:.0f} MiB "
                f"is free. llama.cpp mmaps weights, so this may still run from page cache."
            )

        if exact is not None:
            result["verification"] = {
                "source": "llama_cpp",
                "memory_breakdown_mib": exact["memory_breakdown_mib"],
                "gpu_delta_mib": round(exact_total - best["gpu_mib"], 2),
            }
        elif verify:
            result["notes"].append(
                "llama-fit-params is unavailable, so the figures are analytic only and the "
                "compute buffer carries a margin of error; leave extra headroom."
            )
        return result

    def plan(  # noqa: PLR0913 - one knob per lever the search may pull
        self,
        model_path: str,
        *,
        gpu_budgets_mib: list[float] | None = None,
        kv_cache_types: list[str] | None = None,
        host_budget_mib: float | None = None,
        margin_mib: float = DEFAULT_MARGIN_MIB,
        n_ctx: int = 0,
        n_ctx_min: int = 4096,
        n_ctx_max: int = 0,
        target_utilization: float = 0.9,
        verify: bool = True,
        **estimate_kwargs: Any,  # noqa: ANN401 - forwarded to estimate()
    ) -> dict[str, Any]:
        """
        Build a menu of configurations: one per GPU budget per KV-cache quantization.

        ``recommend()`` answers "what should I run" with a single configuration, walking the
        KV ladder only until something fits. That hides the trade-off, because a quantized KV
        cache is not merely a fallback - on a fixed budget it buys context or offloaded
        layers. Here every requested quantization is solved against every requested budget,
        so the caller can see what each one costs and choose.

        Candidates within a budget are ranked by how much of the model lands on the GPU (the
        dominant term for throughput), bucketed to ``_OFFLOAD_TIE_PCT`` so a marginal layer
        does not outrank a far longer context, and then by context length. A candidate is
        flagged ``dominated`` when a less lossy quantization already reached the same layers
        and context, meaning its extra precision loss buys nothing.

        Budgets that cannot fit the model at all are reported with a reason rather than
        dropped, so a scan over a model library shows why a model is missing.
        """
        try:
            info = self.load_info(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e), "model_path": model_path}

        kv_types = list(kv_cache_types) if kv_cache_types else list(DEFAULT_KV_VARIANTS)
        unknown = [kv for kv in kv_types if kv not in KV_CACHE_TYPES]
        if unknown:
            return {
                "error": (
                    f"Unsupported KV cache type(s) {unknown}; llama.cpp accepts "
                    f"{sorted(KV_CACHE_TYPES)}"
                ),
                "model_path": info.path,
            }

        # Only ask the hardware for what the caller did not state.
        budgets = self.resolve_budgets(
            gpu_budgets_mib[0] if gpu_budgets_mib else None, host_budget_mib
        )
        if gpu_budgets_mib:
            gpu_budgets = sorted({float(b) for b in gpu_budgets_mib})
        elif budgets["gpu_budget_mib"] is not None:
            gpu_budgets = [float(budgets["gpu_budget_mib"])]
        else:
            return {
                "error": "No GPU memory budget available; pass gpu_budgets_mib explicitly",
                "model_path": info.path,
                "budgets_mib": budgets,
            }

        notes: list[str] = []
        if len(gpu_budgets) * len(kv_types) > MAX_PLAN_CELLS:
            return {
                "error": (
                    f"{len(gpu_budgets)} budgets x {len(kv_types)} KV types exceeds the "
                    f"{MAX_PLAN_CELLS}-cell limit; each cell runs a full search"
                ),
                "model_path": info.path,
            }

        model_mib = _to_mib(info.total_bytes)
        plans: list[dict[str, Any]] = []
        for budget in gpu_budgets:
            candidates: list[dict[str, Any]] = []
            for kv in kv_types:
                result = self.recommend(
                    model_path,
                    n_ctx=n_ctx,
                    n_ctx_min=n_ctx_min,
                    n_ctx_max=n_ctx_max,
                    gpu_budget_mib=budget,
                    host_budget_mib=host_budget_mib,
                    margin_mib=margin_mib,
                    target_utilization=target_utilization,
                    kv_cache_types=[kv],
                    verify=verify,
                    **estimate_kwargs,
                )
                candidates.append(self._plan_candidate(kv, result, info, model_mib))

            fitting = [c for c in candidates if c["fits"]]
            # Most of the model on the GPU first, then the longest context: offload dominates
            # tokens/s, and context only matters once the weights are placed. Near-equal
            # offload is bucketed so a marginal layer does not outrank a much longer context.
            best_offload = max((c["gpu_weight_pct"] for c in fitting), default=0.0)

            def rank(candidate: dict[str, Any], top: float = best_offload) -> tuple[int, int]:
                tier = int((top - candidate["gpu_weight_pct"]) // _OFFLOAD_TIE_PCT)
                return tier, -candidate["n_ctx"]

            fitting.sort(key=rank)
            for index, candidate in enumerate(fitting):
                better = fitting[:index]
                candidate["dominated"] = any(
                    kv_types.index(o["kv_quant"]) < kv_types.index(candidate["kv_quant"])
                    and o["gpu_weight_pct"] >= candidate["gpu_weight_pct"]
                    and o["n_ctx"] >= candidate["n_ctx"]
                    for o in better
                )
            ordered = fitting + [c for c in candidates if not c["fits"]]
            usable = max(budget - margin_mib, 0.0)
            plans.append(
                {
                    "gpu_budget_mib": budget,
                    "usable_budget_mib": round(usable, 2),
                    "candidates": ordered,
                    "recommended": next(
                        (c["llama_server_args"] for c in ordered if c["fits"]), None
                    ),
                    "fits": bool(fitting),
                }
            )

        unusable = [p["gpu_budget_mib"] for p in plans if not p["fits"]]
        if unusable:
            notes.append(
                f"No configuration fits at {unusable} MiB even at -ngl 0 with the smallest "
                f"context ({n_ctx_min}); the model's non-offloadable footprint alone exceeds "
                f"the usable budget."
            )

        return {
            "source": "analytic",
            "model_path": info.path,
            "model_info": self._model_info_dict(info),
            "margin_mib": margin_mib,
            "host_budget_mib": budgets["host_budget_mib"],
            "kv_cache_types": kv_types,
            "verified": verify and self._resolve_fit_binary() is not None,
            "plans": plans,
            "notes": notes,
        }

    @staticmethod
    def _plan_candidate(
        kv: str, result: dict[str, Any], info: GgufModelInfo, model_mib: float
    ) -> dict[str, Any]:
        """Flatten one recommend() result into a plan row."""
        if "error" in result:
            return {"kv_quant": kv, "fits": False, "reason": result["error"], "n_ctx": 0}

        best = result["recommended"]
        usage = result["utilization"]
        breakdown = result["memory_breakdown_mib"]
        gpu_weights = breakdown["gpu_total"]["model"]
        return {
            "kv_quant": kv,
            "fits": True,
            "n_gpu_layers": best["n_gpu_layers"],
            "n_cpu_moe": best["n_cpu_moe"],
            "n_ctx": best["n_ctx"],
            "llama_server_args": result["llama_server_args"],
            "gpu_mib": usage["allocated_mib"],
            "gpu_source": usage["allocated_source"],
            "host_mib": breakdown["host"]["total"],
            "utilization_pct": usage["utilization_pct"],
            "within_budget": usage["within_budget"],
            "gpu_weight_mib": gpu_weights,
            "gpu_weight_pct": round(100.0 * gpu_weights / model_mib, 1) if model_mib else 0.0,
            "blocks_on_gpu": result["placement"]["blocks_on_gpu"],
            "n_block": info.n_block,
            "constraint": best["constraint"],
            "host_fits": result["host_fits"],
            "warnings": [n for n in result["notes"] if "WARNING" in n],
            "dominated": False,
        }

    @staticmethod
    def _headroom_reason(
        best: dict[str, Any], info: GgufModelInfo, pinned: bool, ctx_ceiling: int
    ) -> str:
        """Explain why budget is left over, so the caller knows which knob would spend it."""
        if pinned:
            return (
                f"n_ctx is pinned at {best['n_ctx']}; leave n_ctx unset (0) to let the "
                f"search grow the context into the spare VRAM."
            )
        if best["n_ctx"] >= ctx_ceiling:
            return (
                f"The context is already at the ceiling of {ctx_ceiling} "
                f"({'the model trained length' if ctx_ceiling == info.n_ctx_train else 'n_ctx_max'}); "
                f"raise n_ctx_max to spend the rest, at the cost of rope scaling beyond the "
                f"trained length."
            )
        return (
            "The remainder is smaller than the next whole layer or 256-cell context step, "
            "so nothing else fits."
        )

    @staticmethod
    def _search_max_ctx(
        gpu_total: Any,  # noqa: ANN401 - local closure
        offload: dict[str, Any],
        kv_args: dict[str, Any],
        low_ctx: int,
        high_ctx: int,
        target: float,
    ) -> int | None:
        """
        Largest ``n_ctx`` in ``[low_ctx, high_ctx]`` that stays under ``target``, or None.

        Searched in whole 256-cell steps because llama.cpp pads the KV cache to 256, so
        anything finer cannot change the allocation. A binary search rather than the
        halving ladder it replaces: halving would settle for 20480 when 38912 also fits,
        leaving gigabytes of the budget unused.
        """
        step = N_CTX_PAD
        low, high = max(low_ctx // step, 1), high_ctx // step
        if low > high:
            return None
        if gpu_total(n_ctx=low * step, **offload, **kv_args)[0] > target:
            return None

        best = low
        while low <= high:
            mid = (low + high) // 2
            if gpu_total(n_ctx=mid * step, **offload, **kv_args)[0] <= target:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        return best * step

    @staticmethod
    def _search_max_ngl(
        gpu_total: Any,  # noqa: ANN401 - local closure
        slots: int,
        target: float,
        common: dict[str, Any],
    ) -> int | None:
        """Largest ``-ngl`` in ``[0, slots]`` that stays under ``target``, or None."""
        if gpu_total(n_gpu_layers=0, **common)[0] > target:
            return None
        low, high, best = 0, slots, 0
        while low <= high:
            mid = (low + high) // 2
            if gpu_total(n_gpu_layers=mid, **common)[0] <= target:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        return best

    @staticmethod
    def _search_min_cpu_moe(
        gpu_total: Any,  # noqa: ANN401 - local closure
        n_block: int,
        target: float,
        common: dict[str, Any],
    ) -> int | None:
        """Smallest ``--n-cpu-moe`` that fits at full offload, or None if even all-CPU fails."""
        if gpu_total(n_gpu_layers=-1, n_cpu_moe=n_block, **common)[0] > target:
            return None
        low, high, best = 0, n_block, n_block
        while low <= high:
            mid = (low + high) // 2
            if gpu_total(n_gpu_layers=-1, n_cpu_moe=mid, **common)[0] <= target:
                best, high = mid, mid - 1
            else:
                low = mid + 1
        return best

    @staticmethod
    def _format_server_args(best: dict[str, Any], *, flash_attn: bool = True) -> str:
        """
        Render the recommendation as llama-server CLI arguments.

        The result must stay directly pasteable, so caveats belong in ``notes`` rather than
        as a trailing comment here. ``-fa`` is always stated explicitly instead of left at
        llama.cpp's 'auto', so the arguments match the assumption the estimate was made
        under; a silent fallback to the non-flash path would inflate the compute buffer well
        beyond what was budgeted.
        """
        parts = [f"-ngl {best['n_gpu_layers']}", f"-c {best['n_ctx']}"]
        if best["n_cpu_moe"]:
            parts.append(f"-ncmoe {best['n_cpu_moe']}")
        if best["kv_quant"] != "f16":
            parts += [f"-ctk {best['kv_quant']}", f"-ctv {best['kv_quant']}"]
        parts.append("-fa on" if flash_attn else "-fa off")
        return " ".join(parts)


# Global instance
gguf_memory_estimator = GgufMemoryEstimator()
