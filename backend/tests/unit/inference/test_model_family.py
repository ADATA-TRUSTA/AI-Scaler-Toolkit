"""Unit tests for architecture-based model-family detection.

The point of `model_family` is that vendor names drift while the declared
architecture does not, so these tests pin the cases where name matching and arch
matching disagree.
"""

import json

import pytest

from service.inference import model_family
from service.inference.model_family import (
    read_chat_template,
    is_qwen35_family,
    normalize_arch,
    read_model_arch,
)

pytestmark = pytest.mark.inference


def test_normalize_arch_unifies_gguf_and_hf_spellings():
    # GGUF says "qwen35moe", HF config.json says "qwen3_5_moe" -- one arch set
    # must cover both, so both normalize to the same token.
    assert normalize_arch("qwen35moe") == "qwen35moe"
    assert normalize_arch("qwen3_5_moe") == "qwen35moe"
    assert normalize_arch("Qwen3_5-MoE") == "qwen35moe"
    assert normalize_arch(None) == ""


def test_arch_wins_over_a_name_that_looks_like_a_different_version(monkeypatch):
    """The regression this module exists for: Qwen3.6 on the qwen35moe arch."""
    monkeypatch.setattr(model_family, "read_model_arch", lambda _p: "qwen35moe")
    assert (
        is_qwen35_family(
            model_path="/models/Qwen3.6-35B-A3B-Uncensored-Q4_K_M.gguf",
            model_name="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        )
        is True
    )


def test_arch_also_overrides_a_misleading_qwen35_name(monkeypatch):
    """A plain Qwen3 MoE must not pick up Qwen3.5 policy just from its name."""
    monkeypatch.setattr(model_family, "read_model_arch", lambda _p: "qwen3moe")
    assert is_qwen35_family(model_path="/models/qwen3.5-lookalike.gguf") is False


def test_falls_back_to_name_when_arch_unreadable(monkeypatch):
    monkeypatch.setattr(model_family, "read_model_arch", lambda _p: None)
    assert is_qwen35_family(model_name="Qwen/Qwen3.5-35B-A3B") is True
    assert is_qwen35_family(model_name="Qwen/Qwen3_5-35B-A3B") is True
    assert is_qwen35_family(model_name="Qwen/Qwen3-4B") is False


def test_name_fallback_does_not_collide_on_separator_stripping(monkeypatch):
    """"Qwen3-5B" must not be read as "qwen3.5" once separators are gone."""
    monkeypatch.setattr(model_family, "read_model_arch", lambda _p: None)
    assert is_qwen35_family(model_name="Qwen3-5B-Instruct") is False


def test_read_model_arch_reads_hf_config_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    assert read_model_arch(str(tmp_path)) == "qwen3_5_moe"
    assert normalize_arch(read_model_arch(str(tmp_path))) in model_family.QWEN35_FAMILY_ARCHS


def test_read_model_arch_falls_back_to_architectures_list(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]})
    )
    assert read_model_arch(str(tmp_path)) == "Qwen3_5MoeForConditionalGeneration"


def test_read_model_arch_returns_none_for_unreadable_targets(tmp_path):
    assert read_model_arch(None) is None
    assert read_model_arch("") is None
    assert read_model_arch("/nonexistent/path/model.gguf") is None
    # A directory with no config.json is not an error, just unknown.
    assert read_model_arch(str(tmp_path)) is None


def test_repo_id_resolves_through_the_hf_cache(tmp_path, monkeypatch):
    """Models are loaded by repo id, not by directory. A live run showed the
    detection silently doing nothing because only real paths were handled."""
    cached = tmp_path / "chat_template.jinja"
    cached.write_text("<|tool_call>call:f{}<tool_call|>")

    def _fake_cache_lookup(repo_id, filename):
        assert repo_id == "google/gemma-4-E4B-it"
        return str(cached) if filename == "chat_template.jinja" else None

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", _fake_cache_lookup, raising=False
    )
    model_family._resolve_cached_repo_file.cache_clear()

    assert read_chat_template("google/gemma-4-E4B-it") == "<|tool_call>call:f{}<tool_call|>"


def test_repo_id_lookup_never_touches_the_network(monkeypatch):
    """An uncached repo must degrade to None, not stall on a download."""

    def _boom(*a, **k):
        raise OSError("would have hit the network")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _boom, raising=False)
    model_family._resolve_cached_repo_file.cache_clear()
    assert read_chat_template("some-org/not-cached") is None
    assert read_model_arch("some-org/not-cached") is None


def test_read_chat_template_from_tokenizer_config(tmp_path):
    """Older exports keep the template inside tokenizer_config.json."""
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "<tool_call>{}</tool_call>"})
    )
    assert read_chat_template(str(tmp_path)) == "<tool_call>{}</tool_call>"


def test_read_model_arch_cache_invalidates_on_file_change(tmp_path):
    """The cache is keyed on mtime so a replaced/re-quantized file re-reads."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3moe"}))
    assert read_model_arch(str(tmp_path)) == "qwen3moe"

    import os

    cfg.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    stat = os.stat(tmp_path)
    os.utime(tmp_path, (stat.st_atime + 10, stat.st_mtime + 10))
    assert read_model_arch(str(tmp_path)) == "qwen3_5_moe"
