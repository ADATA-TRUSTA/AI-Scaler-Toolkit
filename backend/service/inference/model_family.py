"""
Model-family detection based on the model's own metadata rather than its name.

Repo and file names drift constantly (vendor renames, community re-quants, private
fine-tunes), so a substring check on the name silently stops firing exactly when a
model-specific workaround is still needed. The functions here read the architecture
the model declares about itself:

* GGUF  -> ``general.architecture`` from the file header
* HF    -> ``model_type`` / ``architectures`` from ``config.json``

Both are normalized to a comparable form (``qwen3_5_moe`` and ``qwen35moe`` both
become ``qwen35moe``), so one arch set covers both formats. Name matching stays as
a fallback for the cases where no metadata is reachable (remote-only ids, a path
that no longer exists, a corrupt header).
"""

import json
import os
import re
import struct
import threading
from functools import lru_cache
from typing import Any

from ..settings import configure_logging

logger = configure_logging(__name__)

# Architectures that share the Qwen3.5-generation MoE/dense behaviour: they need
# repetition_penalty pinned to 1.0, the official temperature/top_p/top_k, thinking
# defaulted OFF (the chat template prefills '<think>'), and no '/think' soft switch.
# Vendors ship later marketing versions (e.g. Qwen3.6) on this same architecture,
# which is precisely why this is keyed on arch and not on the name.
QWEN35_FAMILY_ARCHS = frozenset({"qwen35", "qwen35moe"})

# Fallback name fragments, matched against the raw lowercased name. Deliberately not
# the normalized form: stripping separators would make "Qwen3-5B" collide with "qwen35".
_QWEN35_NAME_FRAGMENTS = ("qwen3.5", "qwen3_5")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_arch(value: Any) -> str:  # noqa: ANN401 - accepts whatever metadata holds
    """Lowercase and strip separators so ``Qwen3_5-MoE`` and ``qwen35moe`` compare equal."""
    return _NON_ALNUM.sub("", str(value or "").strip().lower())


# Sizes of the GGUF scalar value types, by type id; 8 is a string, 9 an array.
_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


class _GGUFHeaderUnreadable(Exception):
    """The header is not one the quick reader understands; use gguf.GGUFReader."""


def _scan_gguf_string_key(path: str, wanted: str) -> str | None:
    """
    One string from a GGUF header's key/value section, read straight from the file.

    gguf.GGUFReader builds every field before answering, the tokenizer included: on a
    Qwen3.6-35B GGUF that is a quarter of a million vocabulary entries, each one a
    numpy view, for a key that is the first one written. On a machine short of memory
    one read took minutes, and every /v1/models poll started another, so the settings
    page never learned a model was loaded. This walks the keys in order, skips each
    value by its size, and stops at the one asked for -- normally at once.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise _GGUFHeaderUnreadable("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):  # 1 used 32-bit counts; anything else may be big-endian
            raise _GGUFHeaderUnreadable(f"GGUF version {version}")
        _tensors, kv_count = struct.unpack("<QQ", f.read(16))

        def read_string() -> bytes:
            (length,) = struct.unpack("<Q", f.read(8))
            data = f.read(length)
            if len(data) != length:
                raise _GGUFHeaderUnreadable("truncated string")
            return data

        def skip(value_type: int) -> None:
            if value_type in _GGUF_SCALAR_SIZES:
                f.seek(_GGUF_SCALAR_SIZES[value_type], 1)
            elif value_type == _GGUF_STRING:
                (length,) = struct.unpack("<Q", f.read(8))
                f.seek(length, 1)
            elif value_type == _GGUF_ARRAY:
                item_type, count = struct.unpack("<IQ", f.read(12))
                if item_type in _GGUF_SCALAR_SIZES:
                    f.seek(_GGUF_SCALAR_SIZES[item_type] * count, 1)
                else:
                    for _ in range(count):
                        skip(item_type)
            else:
                raise _GGUFHeaderUnreadable(f"value type {value_type}")

        for _ in range(kv_count):
            key = read_string().decode("utf-8", "replace")
            (value_type,) = struct.unpack("<I", f.read(4))
            if key == wanted:
                if value_type != _GGUF_STRING:
                    return None
                return read_string().decode("utf-8", "replace")
            skip(value_type)
    return None


def _read_gguf_arch(path: str) -> str | None:
    """Read ``general.architecture`` from a GGUF header (header only, no tensor data)."""
    try:
        from .gguf_inspect import resolve_shards

        first = resolve_shards(path)[0]
        try:
            return _scan_gguf_string_key(first, "general.architecture") or None
        except (_GGUFHeaderUnreadable, struct.error):
            pass  # an unusual header: let the full reader have it

        import gguf

        from .gguf_inspect import _field_value, register_extra_ggml_types

        register_extra_ggml_types(gguf)
        reader = gguf.GGUFReader(first)
        arch = _field_value(reader, "general.architecture", "") or ""
        return str(arch) or None
    except Exception as e:
        logger.debug(f"[ModelFamily] Could not read GGUF arch from {path}: {e}")
        return None


def _read_hf_arch(path: str) -> str | None:
    """Read ``model_type`` (or the first entry of ``architectures``) from config.json."""
    config_path = path if path.endswith("config.json") else os.path.join(path, "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.debug(f"[ModelFamily] Could not read HF config from {config_path}: {e}")
        return None

    model_type = config.get("model_type")
    if model_type:
        return str(model_type)

    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return None


# Keys an HF config.json uses for the window the model was trained for, most
# specific first. Older Llama forks and several VLMs set only one of them.
_HF_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len")


def _read_hf_context_length(path: str) -> int | None:
    """
    Read the trained context window from config.json.

    Multimodal configs keep the language-model fields under ``text_config``, so the
    top level of a VLM config carries no window at all; look there too.
    """
    config_path = path if path.endswith("config.json") else os.path.join(path, "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.debug(f"[ModelFamily] Could not read HF config from {config_path}: {e}")
        return None

    nested = config.get("text_config")
    for block in (config, nested if isinstance(nested, dict) else {}):
        for key in _HF_CONTEXT_KEYS:
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value > 0:
                return value
    return None


@lru_cache(maxsize=64)
def _resolve_cached_repo_file(ref: str, filename: str) -> str | None:
    """
    Map an HF repo id to a cached file path, offline only.

    Models are commonly loaded by repo id (``google/gemma-4-E4B-it``) rather than by
    directory, and the hub cache layout (``hub/models--org--name/snapshots/<sha>/``)
    is not something to hand-assemble. Never hits the network: an uncached repo
    simply yields None.
    """
    if "/" not in ref or os.path.exists(ref):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo_id=ref, filename=filename)
    except Exception as e:
        logger.debug(f"[ModelFamily] Cache lookup failed for {ref}/{filename}: {e}")
        return None
    # try_to_load_from_cache returns a str on a hit, or a sentinel object /
    # None when the file is absent or the repo is not cached.
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


@lru_cache(maxsize=64)
def _read_arch_cached(path: str, _mtime: float) -> str | None:
    """Resolve a model's declared architecture. ``_mtime`` only keys the cache."""
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return _read_gguf_arch(path)
    if os.path.isdir(path) or path.endswith("config.json"):
        return _read_hf_arch(path)
    return None


def read_model_arch(model_path: str | None) -> str | None:
    """
    Return the architecture a model declares for itself, or None when unreachable.

    Results are cached per (path, mtime) so a hot generation path does not reopen the
    file on every request; a re-quantized or replaced file invalidates itself.
    """
    path = _localize(model_path, "config.json")
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    # One reader at a time. lru_cache only stores a result once it exists, so callers
    # arriving during the first read each started their own: /v1/models is polled,
    # and on a slow read the polls piled up into a dozen threads parsing one file.
    with _ARCH_READ_LOCK:
        return _read_arch_cached(path, mtime)


_ARCH_READ_LOCK = threading.Lock()


@lru_cache(maxsize=64)
def _read_context_length_cached(path: str, _mtime: float) -> int | None:
    """Resolve a model's trained context window. ``_mtime`` only keys the cache."""
    if os.path.isdir(path) or path.endswith("config.json"):
        return _read_hf_context_length(path)
    # A GGUF states its own ceiling as n_ctx_train, which llama-server already
    # publishes over /v1/models; re-reading the file here would only duplicate it.
    return None


def read_model_context_length(model_path: str | None) -> int | None:
    """
    Return the context window a model declares it was trained for, or None.

    This is the model's own ceiling, not the window a given load serves. Cached per
    (path, mtime) like ``read_model_arch``, and reads the same config.json.
    """
    path = _localize(model_path, "config.json")
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return _read_context_length_cached(path, mtime)


@lru_cache(maxsize=32)
def _read_chat_template_cached(path: str, _mtime: float) -> str | None:
    """Read a model's chat template. ``_mtime`` only keys the cache."""
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        try:
            import gguf

            from .gguf_inspect import _field_value, register_extra_ggml_types, resolve_shards

            register_extra_ggml_types(gguf)
            reader = gguf.GGUFReader(resolve_shards(path)[0])
            return _field_value(reader, "tokenizer.chat_template", None) or None
        except Exception as e:
            logger.debug(f"[ModelFamily] Could not read GGUF chat template from {path}: {e}")
            return None

    # Accept either a model directory or a direct path to one of the two files
    # that can hold a template (the repo-id path resolves to the file itself).
    if os.path.isdir(path):
        jinja = os.path.join(path, "chat_template.jinja")
        tok_cfg = os.path.join(path, "tokenizer_config.json")
    elif path.endswith(".jinja"):
        jinja, tok_cfg = path, ""
    else:
        jinja, tok_cfg = "", path

    if jinja and os.path.isfile(jinja):
        try:
            with open(jinja, encoding="utf-8") as f:
                return f.read() or None
        except Exception as e:
            logger.debug(f"[ModelFamily] Could not read {jinja}: {e}")

    if not tok_cfg:
        return None
    try:
        with open(tok_cfg, encoding="utf-8") as f:
            template = json.load(f).get("chat_template")
    except Exception as e:
        logger.debug(f"[ModelFamily] Could not read chat template from {tok_cfg}: {e}")
        return None

    # Some exports use a list of named templates; the default one is what serving uses.
    if isinstance(template, list):
        for entry in template:
            if isinstance(entry, dict) and entry.get("name") == "default":
                return str(entry.get("template") or "") or None
        return None
    return str(template) if template else None


def read_chat_template(model_path: str | None) -> str | None:
    """
    Return the model's chat template text, or None when unreachable.

    The template is the most reliable statement of the tool-call syntax a model
    emits -- it is literally what a tool-call parser has to parse -- so callers
    use it to pick a parser instead of guessing from the model name.
    """
    # Newer exports ship a standalone .jinja; older ones embed it in tokenizer_config.
    path = _localize(model_path, "chat_template.jinja") or _localize(
        model_path, "tokenizer_config.json"
    )
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return _read_chat_template_cached(path, mtime)


def _localize(model_ref: str | None, cache_filename: str) -> str | None:
    """
    Return a usable local path for a model reference.

    Accepts a local file/directory as-is, and resolves an HF repo id to the cached
    ``cache_filename`` inside it. Returns None when nothing local can be found.
    """
    ref = str(model_ref or "").strip()
    if not ref:
        return None
    if os.path.exists(ref):
        return ref
    return _resolve_cached_repo_file(ref, cache_filename)


def is_qwen35_family(
    model_path: str | None = None,
    model_name: str | None = None,
) -> bool:
    """
    Detect the Qwen3.5-generation family from its declared architecture.

    Falls back to matching the name only when no architecture metadata is reachable,
    so behaviour never regresses for remote-only model ids.
    """
    arch = read_model_arch(model_path)
    if arch is not None:
        matched = normalize_arch(arch) in QWEN35_FAMILY_ARCHS
        logger.debug(f"[ModelFamily] arch={arch!r} qwen35_family={matched} path={model_path}")
        return matched

    # No metadata: fall back to the historical name check over both fields.
    for candidate in (model_name, model_path):
        lowered = str(candidate or "").strip().lower()
        if any(fragment in lowered for fragment in _QWEN35_NAME_FRAGMENTS):
            return True
    return False
