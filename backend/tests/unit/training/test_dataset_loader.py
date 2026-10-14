"""Unit tests for service.training.core.dataset_loader.load_training_dataset.

Covers the three supported local formats (text / prompt+completion / OpenAI
chat `messages`), the HuggingFace-hub passthrough branch, and the error paths:
malformed messages, unrecognised schema, missing file, and invalid JSON.

Local .jsonl files are parsed by the `datasets` library fully offline.
"""

import json

import pytest

import service.training.core.dataset_loader as dataset_loader
from service.training.core.dataset_loader import load_training_dataset


pytestmark = pytest.mark.finetune


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------------------
# Happy paths for the three local formats
# --------------------------------------------------------------------------

def test_loads_text_format(tmp_path):
    p = _write_jsonl(tmp_path / "text.jsonl", [{"text": "hello"}, {"text": "world"}])
    ds = load_training_dataset(p)
    assert len(ds) == 2
    assert ds[0]["text"] == "hello"


def test_loads_prompt_completion_format_without_combining(tmp_path):
    # NOTE: the docstring claims prompt+completion are combined into a single
    # `text` field, but the implementation only validates and passes them
    # through. This test pins the *actual* behaviour: columns stay separate.
    p = _write_jsonl(
        tmp_path / "pc.jsonl",
        [{"prompt": "Q1", "completion": "A1"}, {"prompt": "Q2", "completion": "A2"}],
    )
    ds = load_training_dataset(p)
    assert len(ds) == 2
    assert set(ds.column_names) >= {"prompt", "completion"}
    assert "text" not in ds.column_names
    assert ds[0]["prompt"] == "Q1" and ds[0]["completion"] == "A1"


def test_loads_openai_chat_messages_format(tmp_path):
    rows = [
        {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]}
    ]
    p = _write_jsonl(tmp_path / "chat.jsonl", rows)
    ds = load_training_dataset(p)
    assert len(ds) == 1
    msgs = ds[0]["messages"]
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hello"


# --------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------

def test_malformed_messages_raises_value_error(tmp_path):
    # `messages` is a list, but its entries lack role/content.
    p = _write_jsonl(tmp_path / "bad_msgs.jsonl", [{"messages": [{"foo": "bar"}]}])
    with pytest.raises(ValueError, match="must be a list"):
        load_training_dataset(p)


def test_unrecognised_schema_raises_value_error(tmp_path):
    p = _write_jsonl(tmp_path / "wrong.jsonl", [{"instruction": "x", "input": "y"}])
    with pytest.raises(ValueError, match="Found fields"):
        load_training_dataset(p)


def test_missing_file_raises_value_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.jsonl")
    with pytest.raises(ValueError, match="Dataset validation failed"):
        load_training_dataset(missing)


def test_invalid_json_content_raises_value_error(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text("this is not json at all\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_training_dataset(str(p))


# --------------------------------------------------------------------------
# HuggingFace-hub passthrough branch (non-file dataset name)
# --------------------------------------------------------------------------

def test_non_file_path_delegates_to_huggingface_hub(monkeypatch):
    calls = {}

    def fake_load(name, *args, **kwargs):
        calls["name"] = name
        return {"train": "SENTINEL_SPLIT"}

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fake_load)

    result = load_training_dataset("some-org/some-dataset")
    assert result == "SENTINEL_SPLIT"
    assert calls["name"] == "some-org/some-dataset"
