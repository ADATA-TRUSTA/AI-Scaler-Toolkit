"""
Chat settings parsing and prompt selection.

This is the code the batch and parallel examples share, so a mistake here shows
up in both. Type problems raise TypeError and value problems raise ValueError,
which the callers rely on to tell a malformed settings file from an empty one.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from ai_scaler_console.helpers.chat import (
    ChatSettings,
    build_messages,
    load_chat_settings,
    parse_chat_settings,
    require_prompts,
    select_prompt,
)


def test_defaults_are_applied_to_an_empty_settings_dict() -> None:
    settings = parse_chat_settings({})

    assert settings.model_name == "trusta-ast/trusta-ast-default"
    assert settings.system_prompt == "You are a helpful assistant."
    assert settings.stream is False
    assert settings.prompts == ()
    assert settings.continue_on_error is True
    assert settings.backend_url is None
    assert settings.api_key is None


def test_values_are_taken_from_the_settings_dict() -> None:
    settings = parse_chat_settings(
        {
            "model": "some/model",
            "system_prompt": "Be terse.",
            "stream": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "max_tokens": 42,
            "extra_body": {"top_k": 7},
            "prompts": ["a", "b"],
            "continue_on_error": False,
            "backend_url": "http://host:9000",
            "api_key": "k",
        }
    )

    assert settings.model_name == "some/model"
    assert settings.stream is True
    assert settings.max_tokens == 42
    assert settings.extra_body == {"top_k": 7}
    assert settings.prompts == ("a", "b")
    assert settings.continue_on_error is False
    assert settings.backend_url == "http://host:9000"
    assert settings.api_key == "k"


@pytest.mark.parametrize("blank", ["", None])
def test_blank_backend_url_and_api_key_become_none(blank: str | None) -> None:
    """None means "fall back to .env", which is what create_client expects."""
    settings = parse_chat_settings({"backend_url": blank, "api_key": blank})

    assert settings.backend_url is None
    assert settings.api_key is None


def test_non_list_prompts_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        parse_chat_settings({"prompts": "not a list"})


@pytest.mark.parametrize("prompts", [[""], ["  "], ["ok", ""], [1, 2]])
def test_prompts_must_all_be_non_empty_strings(prompts: list) -> None:
    with pytest.raises(ValueError):
        parse_chat_settings({"prompts": prompts})


def test_require_prompts_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError):
        require_prompts(ChatSettings())


def test_require_prompts_returns_the_prompts() -> None:
    assert require_prompts(ChatSettings(prompts=("a", "b"))) == ("a", "b")


def test_select_prompt_picks_by_index() -> None:
    settings = ChatSettings(prompts=("first", "second", "third"))

    assert select_prompt(settings, 0) == "first"
    assert select_prompt(settings, 2) == "third"


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_select_prompt_rejects_out_of_range(index: int) -> None:
    settings = ChatSettings(prompts=("a", "b", "c"))

    with pytest.raises(IndexError):
        select_prompt(settings, index)


def test_build_messages_shape() -> None:
    messages = build_messages("sys", "user")

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_settings_are_immutable() -> None:
    """Frozen so a worker cannot mutate settings shared with its caller."""
    settings = ChatSettings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model_name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "path",
    [
        "configs/chat/batch.json",
        "configs/chat/parallel_8.json",
        "configs/chat/parallel_16.json",
    ],
)
def test_shipped_prompt_configs_load(path: str) -> None:
    settings = load_chat_settings(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    assert len(settings.prompts) == len(raw["prompts"])
    assert all(p.strip() for p in settings.prompts)


def test_parallel_configs_have_the_prompt_counts_their_names_claim() -> None:
    assert len(load_chat_settings("configs/chat/parallel_8.json").prompts) == 8
    assert len(load_chat_settings("configs/chat/parallel_16.json").prompts) == 16
