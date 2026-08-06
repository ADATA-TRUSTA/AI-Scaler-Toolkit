"""
GGUF file inspector.

Reads the GGUF header (metadata KV pairs + tensor table) and exposes the exact per-tensor
byte size together with the hyper-parameters llama.cpp needs to size its KV cache.

Unlike a HuggingFace ``config.json``, a GGUF file carries the real tensor table, so the
weight footprint is *measured* rather than estimated. This module is pure I/O + parsing;
all placement policy lives in :mod:`service.inference.gguf_estimator`.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Tensor-name pattern that llama.cpp's --cpu-moe / --n-cpu-moe override matches.
# Mirrors LLM_FFN_EXPS_REGEX in llama.cpp common/common.h.
MOE_EXPS_REGEX = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")

_BLOCK_REGEX = re.compile(r"^blk\.(\d+)\.(.+)$")

# Filename suffix of a sharded GGUF, e.g. model-00001-of-00004.gguf.
_SHARD_REGEX = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

# Slot index used for the input layer (token embeddings). llama.cpp always keeps it on the
# CPU, see llama_model_base::load_tensors in src/llama-model.cpp.
INPUT_SLOT = -1

# The output projection. Its absence is what marks a model as tied-embedding.
OUTPUT_WEIGHT = "output.weight"

# Per-architecture default SWA period, used when the GGUF omits
# {arch}.attention.sliding_window_pattern. Values mirror src/models/*.cpp in llama.cpp.
# {arch}.attention.sliding_window_pattern. Keys are the GGUF architecture strings from
# LLM_ARCH_NAMES in src/llama-arch.cpp, which do not always match the source file name -
# gpt-oss is implemented in models/openai-moe.cpp.
_SWA_PERIOD_DEFAULTS: dict[str, int] = {
    "afmoe": 4,
    "cohere2": 4,
    "exaone4": 4,
    "exaone-moe": 4,
    "gemma2": 2,
    "gemma3": 6,
    "gemma3n": 5,
    "gemma-embedding": 6,
    "gpt-oss": 2,
    "llama4": 4,
    "modern-bert": 3,
    "olmo2": 4,
    "phi3": 1,
    "plamo3": 8,
    "smallthinker": 4,
}

# Architectures that pass dense_first=true, putting the full-attention layer at the start of
# each group instead of the end.
_SWA_DENSE_FIRST = {"smallthinker", "modern-bert"}

# Architectures whose trailing multi-token-prediction layers hold no KV cache; they set
# n_layer_kv_from_start = n_layer - nextn_predict_layers in src/models/*.cpp.
_NEXTN_NO_KV_ARCHS = {"bailingmoe2", "glm4", "glm4moe", "glm-dsa", "mimo2"}

# Hybrid architectures that interleave linear (gated delta net) layers with full attention:
# recurrent_layer_arr[il] = (il + 1) % full_attention_interval != 0. Only the full-attention
# layers hold a KV cache; the rest hold a fixed-size recurrent state instead, so assuming a
# KV cache everywhere over-estimates Qwen3.5 by roughly 3x.
_HYBRID_INTERVAL_ARCHS = {"qwen35", "qwen35moe", "qwen3next"}
_DEFAULT_FULL_ATTENTION_INTERVAL = 4

# Tensor-name markers for layers that carry recurrent state rather than a KV cache.
_RECURRENT_TENSOR_MARKERS = (".ssm_", ".time_mix_", ".shortconv")

# The recurrent state is always allocated as f32 (see llama_model::create_memory).
_RECURRENT_STATE_DTYPE_BYTES = 4

# gemma3n hard-codes the number of layers that own a KV cache (src/models/gemma3n.cpp).
_GEMMA3N_KV_LAYERS = 20

# ggml types the vendored llama.cpp writes but the pinned gguf package predates. Without
# these, GGUFReader raises "<n> is not a valid GGMLQuantizationType" on the whole file and
# nothing about the model can be read - gpt-oss is shipped as MXFP4, so this is not exotic.
# Values are (block size in elements, bytes per block) taken from ggml/src/ggml-common.h.
_EXTRA_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    39: ("MXFP4", 32, 17),  # uint8 scale + 32 packed 4-bit values
    40: ("NVFP4", 64, 36),  # 4 UE4M3 sub-block scales + 64 packed 4-bit values
    41: ("Q1_0", 128, 18),  # ggml_half delta + 128 1-bit quants
}


@dataclass(frozen=True)
class GgufTensor:
    """A single tensor from the GGUF tensor table."""

    name: str
    slot: int
    role: str
    n_bytes: int
    # True for the copy of token_embd that a tied-embedding model allocates as its output
    # projection. It is not in the file, but llama.cpp really does allocate it.
    duplicated: bool = False


@dataclass
class GgufModelInfo:
    """Everything the memory estimator needs to know about a GGUF file."""

    path: str
    arch: str
    n_block: int
    n_embd: int
    n_head: int
    n_head_kv: list[int]
    key_length: int
    value_length: int
    n_vocab: int
    n_ctx_train: int
    n_expert: int
    n_expert_used: int
    n_swa: int
    swa_layers: list[bool]
    file_type: int | None
    tensors: list[GgufTensor]
    shard_paths: list[str] = field(default_factory=list)
    is_recurrent: bool = False
    warnings: list[str] = field(default_factory=list)
    # Key/value head dimensions for sliding-window layers. 0 means "same as the full-
    # attention layers", which is what llama.cpp defaults to.
    key_length_swa: int = 0
    value_length_swa: int = 0
    # Layers from this index on share an earlier layer's KV cache and allocate none of their
    # own (llama_hparams::has_kv). -1 means every layer has its own cache.
    n_layer_kv_from_start: int = -1
    tied_embedding: bool = False
    # Per-layer linear/recurrent attention flags, and the row widths of the two state tensors
    # (llama_hparams::n_embd_r / n_embd_s). A recurrent layer holds state of a fixed size
    # instead of a KV cache that grows with the context.
    recurrent_layers: list[bool] = field(default_factory=list)
    n_embd_r: int = 0
    n_embd_s: int = 0

    @property
    def total_bytes(self) -> int:
        """Total bytes llama.cpp allocates for weights, including any tied-output copy."""
        return sum(t.n_bytes for t in self.tensors)

    @property
    def file_bytes(self) -> int:
        """Total bytes of the tensors actually stored in the file."""
        return sum(t.n_bytes for t in self.tensors if not t.duplicated)

    @property
    def is_moe(self) -> bool:
        """Whether the model has routed experts."""
        return self.n_expert > 0

    @property
    def output_weight_bytes(self) -> int:
        """
        Size of the output projection matrix.

        llama.cpp stages this tensor in VRAM even when the output layer itself is not
        offloaded, so it shows up in the GPU compute buffer. Models with tied embeddings
        have no ``output.weight`` of their own and get the copy appended by the reader.
        """
        output = sum(t.n_bytes for t in self.tensors if t.name.startswith(OUTPUT_WEIGHT))
        if output:
            return output
        return sum(t.n_bytes for t in self.tensors if t.role == "input")

    def n_head_kv_at(self, il: int) -> int:
        """Number of KV heads for layer ``il``."""
        if il < len(self.n_head_kv):
            return self.n_head_kv[il]
        return self.n_head_kv[-1] if self.n_head_kv else 0

    def is_swa(self, il: int) -> bool:
        """Whether layer ``il`` uses sliding-window attention."""
        return il < len(self.swa_layers) and self.swa_layers[il]

    def kv_dims_at(self, il: int) -> tuple[int, int]:
        """
        Key and value head dimensions for layer ``il``.

        Mirrors llama_hparams::n_embd_head_k/v, which switch to the ``*_swa`` dimensions on
        sliding-window layers. Gemma 4 halves them there, so ignoring this over-estimates
        its KV cache by a factor of two on 25 layers out of 30.
        """
        if self.is_swa(il):
            return (
                self.key_length_swa or self.key_length,
                self.value_length_swa or self.value_length,
            )
        return self.key_length, self.value_length

    def has_kv(self, il: int) -> bool:
        """Whether layer ``il`` allocates its own KV cache (llama_hparams::has_kv)."""
        if self.is_recurrent_layer(il):
            return False
        if self.n_layer_kv_from_start < 0:
            return True
        return il < self.n_layer_kv_from_start

    def is_recurrent_layer(self, il: int) -> bool:
        """Whether layer ``il`` uses linear attention and so holds recurrent state."""
        return il < len(self.recurrent_layers) and self.recurrent_layers[il]

    def recurrent_state_bytes(self, n_seq_max: int = 1) -> int:
        """
        Bytes of recurrent state one linear-attention layer allocates.

        ``(n_embd_r + n_embd_s) x rs_size`` f32 values, where ``rs_size`` is
        ``max(1, n_seq_max)``. Independent of the context length, which is the whole point of
        linear attention.
        """
        rows = self.n_embd_r + self.n_embd_s
        return rows * max(n_seq_max, 1) * _RECURRENT_STATE_DTYPE_BYTES

    def layer_bytes(self) -> list[dict[str, int]]:
        """Per-block byte totals broken down by tensor role."""
        roles = ("attn", "ffn", "moe_exps", "shexp", "other")
        out: list[dict[str, int]] = [dict.fromkeys(roles, 0) for _ in range(self.n_block)]
        for t in self.tensors:
            if 0 <= t.slot < self.n_block:
                out[t.slot][t.role] = out[t.slot].get(t.role, 0) + t.n_bytes
        for entry in out:
            entry["total"] = sum(entry[r] for r in roles)
        return out


def classify_tensor(name: str, n_block: int) -> tuple[int, str]:
    """
    Map a GGUF tensor name onto llama.cpp's (slot, role).

    Slots follow llama.cpp's layout: ``INPUT_SLOT`` for the token embeddings, ``0..n_block-1``
    for the repeating blocks, and ``n_block`` for the output layer.
    """
    match = _BLOCK_REGEX.match(name)
    if match is None:
        if name.startswith("token_embd"):
            return INPUT_SLOT, "input"
        return n_block, "output"

    slot = int(match.group(1))
    rest = match.group(2)
    if MOE_EXPS_REGEX.search(name):
        role = "moe_exps"
    elif "shexp" in rest:
        role = "shexp"
    elif rest.startswith("attn"):
        role = "attn"
    elif rest.startswith("ffn"):
        role = "ffn"
    else:
        role = "other"
    return slot, role


def resolve_shards(path: str) -> list[str]:
    """
    Expand a sharded GGUF path into the full list of shard files.

    Returns ``[path]`` for a single-file model, or for a shard whose siblings are missing.
    """
    match = _SHARD_REGEX.match(os.path.basename(path))
    if match is None:
        return [path]

    directory = os.path.dirname(os.path.abspath(path))
    stem = match.group("stem")
    total = int(match.group("total"))
    shards = [
        os.path.join(directory, f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)
    ]
    missing = [p for p in shards if not os.path.isfile(p)]
    if missing:
        logger.warning(
            f"[GgufInspect] Sharded model {path} is missing {len(missing)} shard(s); "
            f"reading the given file only"
        )
        return [path]
    return shards


def register_extra_ggml_types(gguf: Any) -> list[str]:  # noqa: ANN401 - the gguf module
    """
    Teach the installed ``gguf`` package about ggml types newer than it is.

    ``GGUFReader`` resolves every tensor's dtype through ``GGMLQuantizationType(value)``, so
    one unknown type makes the whole file unreadable rather than just that tensor. Rather
    than reimplement the tensor table, the missing members are grafted onto the enum with
    their block geometry, which is what ``ReaderTensor.n_bytes`` needs to come out exact.

    Idempotent, and a no-op once the pinned package catches up.

    Returns:
        The names of the types that had to be added, for logging.
    """
    quant_type = gguf.constants.GGMLQuantizationType
    sizes = gguf.constants.GGML_QUANT_SIZES
    added: list[str] = []
    # Grafting members onto an enum means touching its private maps, so a change in the
    # package must degrade to "cannot read this file" rather than break every file.
    try:
        for value, (name, block, type_size) in _EXTRA_GGML_TYPES.items():
            if value in quant_type._value2member_map_:
                continue
            member = int.__new__(quant_type, value)  # IntEnum member, so int is the base
            member._name_ = name
            member._value_ = value
            quant_type._value2member_map_[value] = member
            quant_type._member_map_[name] = member
            sizes[member] = (block, type_size)
            added.append(name)
    except Exception as e:
        logger.warning(f"[GgufInspect] Could not register newer ggml types: {e}")
        return added
    if added:
        logger.debug(f"[GgufInspect] Registered ggml types missing from gguf: {added}")
    return added


def _field_value(reader: Any, key: str, default: Any = None) -> Any:  # noqa: ANN401 - GGUFReader
    """Read a metadata KV pair, returning ``default`` when absent or unreadable."""
    field_obj = reader.fields.get(key)
    if field_obj is None:
        return default
    try:
        value = field_obj.contents()
    except Exception:
        logger.debug(f"[GgufInspect] Could not decode GGUF key {key}")
        return default
    return default if value is None else value


def _as_int_list(value: Any, length: int) -> list[int] | None:  # noqa: ANN401 - GGUF scalar/array
    """Normalize a GGUF scalar-or-array hyper-parameter into a per-layer list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [int(v) for v in value]
        if not items:
            return None
        return items + [items[-1]] * (length - len(items)) if len(items) < length else items
    return [int(value)] * length


def _build_swa_layers(arch: str, n_swa: int, pattern: Any, n_block: int) -> list[bool]:  # noqa: ANN401
    """
    Reproduce llama_hparams::set_swa_pattern for the given architecture.

    Transcribed from src/llama-hparams.cpp: the default is
    ``swa_layers[il] = il % period < period - 1``, which makes the *last* layer of each
    group the dense one, and the ``dense_first`` architectures use
    ``swa_layers[il] = il % period != 0`` instead.

    Both phases give the same layer *count*, so getting them the wrong way round only shows
    up under partial offload, where it charges the wrong layers' KV cache to the GPU.
    """
    if n_swa <= 0:
        return [False] * n_block

    per_layer = _as_int_list(pattern, n_block)
    if per_layer is not None and isinstance(pattern, (list, tuple)):
        # Some architectures (e.g. mimo2) store the per-layer flags directly.
        return [bool(v) for v in per_layer[:n_block]]

    period = per_layer[0] if per_layer else _SWA_PERIOD_DEFAULTS.get(arch, 0)
    if period <= 0:
        return [False] * n_block
    if arch in _SWA_DENSE_FIRST:
        return [(il % period) != 0 for il in range(n_block)]
    return [(il % period) < (period - 1) for il in range(n_block)]


def _build_recurrent_layers(
    arch: str,
    hp: Any,  # noqa: ANN401 - metadata accessor closure
    n_block: int,
    has_recurrent_tensors: bool,
) -> list[bool]:
    """
    Mark which layers use linear attention instead of a KV cache.

    The hybrid Qwen families interleave them on a fixed period; anything else that carries
    recurrent tensors (Mamba, RWKV) is recurrent throughout.
    """
    if arch in _HYBRID_INTERVAL_ARCHS:
        interval = int(
            hp("full_attention_interval", _DEFAULT_FULL_ATTENTION_INTERVAL)
            or _DEFAULT_FULL_ATTENTION_INTERVAL
        )
        return [((il + 1) % interval) != 0 for il in range(n_block)]
    if has_recurrent_tensors:
        return [True] * n_block
    return [False] * n_block


def _recurrent_state_dims(hp: Any) -> tuple[int, int]:  # noqa: ANN401 - metadata accessor
    """
    Row widths of the recurrent conv and SSM state tensors.

    Transcribed from llama_hparams::n_embd_r / n_embd_s for the Mamba-style layout, which is
    what the ``{arch}.ssm.*`` keys describe. RWKV and LFM2 use different formulas and report
    (0, 0) here, which the caller turns into a warning rather than a silent zero.
    """
    d_conv = int(hp("ssm.conv_kernel", 0) or 0)
    d_inner = int(hp("ssm.inner_size", 0) or 0)
    d_state = int(hp("ssm.state_size", 0) or 0)
    n_group = int(hp("ssm.group_count", 0) or 0)
    n_embd_r = max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state)
    n_embd_s = d_state * d_inner
    return n_embd_r, n_embd_s


def _resolve_kv_layer_limit(arch: str, hp: Any, n_block: int) -> int:  # noqa: ANN401 - closure
    """
    Resolve ``llama_hparams::n_layer_kv_from_start``: layers past it allocate no KV cache.

    Three sources, all read from the vendored llama.cpp: Gemma 4 subtracts
    ``attention.shared_kv_layers``, the GLM/Bailing/MiMo family subtracts its
    ``nextn_predict_layers`` multi-token-prediction layers, and gemma3n hard-codes 20.
    """
    shared_kv = int(hp("attention.shared_kv_layers", 0) or 0)
    if shared_kv > 0:
        return max(n_block - shared_kv, 0)
    if arch in _NEXTN_NO_KV_ARCHS:
        nextn = int(hp("nextn_predict_layers", 0) or 0)
        if nextn > 0:
            return max(n_block - nextn, 0)
    if arch == "gemma3n":
        return min(_GEMMA3N_KV_LAYERS, n_block)
    return -1


def read_gguf_model_info(path: str) -> GgufModelInfo:
    """
    Parse a GGUF file (or shard set) into a :class:`GgufModelInfo`.

    Only the header is read; tensor data is never touched, so this is fast even for
    multi-gigabyte files.

    Raises:
        FileNotFoundError: the path does not exist.
        RuntimeError: the ``gguf`` package is unavailable or the file is not valid GGUF.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GGUF file not found: {path}")

    try:
        import gguf
    except Exception as e:  # pragma: no cover - gguf is a pinned core dependency
        raise RuntimeError("The 'gguf' package is required to inspect GGUF files") from e

    register_extra_ggml_types(gguf)
    shards = resolve_shards(path)
    warnings: list[str] = []

    try:
        reader = gguf.GGUFReader(shards[0])
    except Exception as e:
        raise RuntimeError(f"Failed to read GGUF header from {shards[0]}: {e}") from e

    arch = _field_value(reader, "general.architecture", "") or ""

    def hp(suffix: str, default: Any = None) -> Any:  # noqa: ANN401 - passthrough of GGUF value
        return _field_value(reader, f"{arch}.{suffix}", default)

    n_block = int(hp("block_count", 0) or 0)
    n_embd = int(hp("embedding_length", 0) or 0)
    n_head = int(hp("attention.head_count", 0) or 0)

    head_kv = _as_int_list(hp("attention.head_count_kv"), n_block)
    if head_kv is None:
        head_kv = [n_head] * n_block

    head_dim = (n_embd // n_head) if n_head else 0
    key_length = int(hp("attention.key_length", head_dim) or head_dim)
    value_length = int(hp("attention.value_length", head_dim) or head_dim)
    key_length_swa = int(hp("attention.key_length_swa", 0) or 0)
    value_length_swa = int(hp("attention.value_length_swa", 0) or 0)

    n_swa = int(hp("attention.sliding_window", 0) or 0)
    swa_layers = _build_swa_layers(arch, n_swa, hp("attention.sliding_window_pattern"), n_block)
    if n_swa > 0 and not any(swa_layers):
        warnings.append(
            f"Architecture '{arch}' declares a sliding window of {n_swa} but no SWA pattern "
            f"could be determined; the KV cache estimate assumes full attention on every layer "
            f"(this over-estimates)."
        )

    n_layer_kv_from_start = _resolve_kv_layer_limit(arch, hp, n_block)

    tensors: list[GgufTensor] = []
    for shard in shards:
        shard_reader = reader if shard == shards[0] else gguf.GGUFReader(shard)
        for t in shard_reader.tensors:
            slot, role = classify_tensor(t.name, n_block)
            tensors.append(GgufTensor(name=t.name, slot=slot, role=role, n_bytes=int(t.n_bytes)))

    # A model without output.weight ties its output projection to the token embeddings, and
    # llama.cpp creates that copy with TENSOR_DUPLICATED - a second allocation of the same
    # size, on the output layer's device (measured on gemma-4 at -ngl 0, 1, 2, 6, 13 and 99).
    # Leaving it out under-reports VRAM, which is the direction that makes a budget-filling
    # recommendation overshoot. Match the name exactly: output_norm and rope_freqs share the
    # output slot but are not the projection.
    tied_embedding = not any(t.name == OUTPUT_WEIGHT for t in tensors)
    embd_bytes = sum(t.n_bytes for t in tensors if t.role == "input")
    if tied_embedding and embd_bytes:
        tensors.append(
            GgufTensor(
                name=f"{OUTPUT_WEIGHT} (tied to token_embd)",
                slot=n_block,
                role="output",
                n_bytes=embd_bytes,
                duplicated=True,
            )
        )

    n_vocab = 0
    for t in reader.tensors:
        if t.name.startswith("token_embd") or t.name == "output.weight":
            n_vocab = int(t.shape[-1])
            break

    # Detect recurrent/SSM layers from the tensors, not the metadata. Some writers emit the
    # whole {arch}.ssm.* key block regardless of architecture, so key presence proves nothing;
    # an ssm_* tensor means state really is allocated.
    is_recurrent = any(
        t.name.startswith("blk.") and any(m in t.name for m in _RECURRENT_TENSOR_MARKERS)
        for t in tensors
    )
    recurrent_layers = _build_recurrent_layers(arch, hp, n_block, is_recurrent)
    n_embd_r, n_embd_s = _recurrent_state_dims(hp)
    if any(recurrent_layers) and not (n_embd_r or n_embd_s):
        warnings.append(
            f"Architecture '{arch}' allocates recurrent state whose size could not be derived "
            f"from the {arch}.ssm.* keys; that state is missing from the estimate. Use the "
            f"llama.cpp probe for an exact figure."
        )

    info = GgufModelInfo(
        path=os.path.abspath(path),
        arch=arch,
        n_block=n_block,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=head_kv,
        key_length=key_length,
        value_length=value_length,
        n_vocab=n_vocab,
        n_ctx_train=int(hp("context_length", 0) or 0),
        n_expert=int(hp("expert_count", 0) or 0),
        n_expert_used=int(hp("expert_used_count", 0) or 0),
        n_swa=n_swa,
        swa_layers=swa_layers,
        file_type=_field_value(reader, "general.file_type"),
        tensors=tensors,
        shard_paths=shards,
        is_recurrent=is_recurrent,
        warnings=warnings,
        key_length_swa=key_length_swa,
        value_length_swa=value_length_swa,
        n_layer_kv_from_start=n_layer_kv_from_start,
        tied_embedding=tied_embedding,
        recurrent_layers=recurrent_layers,
        n_embd_r=n_embd_r,
        n_embd_s=n_embd_s,
    )
    logger.info(
        f"[GgufInspect] {os.path.basename(path)}: arch={arch} blocks={n_block} "
        f"tensors={len(tensors)} size={info.file_bytes / (1024**3):.2f} GiB"
        + (f" (+{embd_bytes / (1024**3):.2f} GiB tied output copy)" if tied_embedding else "")
    )
    return info
