"""Unit tests for service.inference.generate.gpt_parser.is_gpt_model.

Pure name-pattern detection for the OpenAI GPT-OSS family. Edge cases:
empty/None input, case-insensitivity, and avoiding false positives such as
unrelated "gpt2" model names.
"""

import pytest

from service.inference.generate.gpt_parser import is_gpt_model


pytestmark = pytest.mark.inference


@pytest.mark.parametrize(
    "name",
    [
        "openai/gpt-oss-20b",
        "gpt_oss",
        "GPT-OSS-120B",          # case-insensitive
        "openai/gpt-4o",         # matches the "openai/gpt" pattern
        "/models/local/gpt-oss-20b",
    ],
)
def test_is_gpt_model_true(name):
    assert is_gpt_model(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "meta-llama/Llama-3-8B",
        "some-gpt2-model",       # gpt2 is NOT a GPT-OSS model
        "mistralai/Mistral-7B",
        "chatgpt-clone",         # no matching pattern
    ],
)
def test_is_gpt_model_false(name):
    assert is_gpt_model(name) is False


def test_is_gpt_model_handles_empty_and_none():
    assert is_gpt_model("") is False
    assert is_gpt_model(None) is False
