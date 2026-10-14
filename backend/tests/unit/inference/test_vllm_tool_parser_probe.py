"""Regression guard for vllm_tool_parser_probe.py's test payload shape.

That module runs only under the isolated vllm_server venv's interpreter (see
its own docstring) and imports vllm/transformers lazily inside main(), so its
actual template-rendering behaviour cannot be exercised here -- there is no
guarantee the backend's own environment even has a matching vllm version, and
the whole point of the isolated venv is that it does not need to. What can
still be locked down from here, with no vllm/transformers import required, is
the shape of the payload the probe sends to every chat template.
"""

from types import SimpleNamespace

import pytest

import service.inference.engines.vllm_tool_parser_probe as probe

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]


def test_probe_assistant_turn_uses_empty_string_content_not_none():
    """
    Regression guard for a real failure found live: Qwen/Qwen3-VL-4B-Instruct's
    chat template (vLLM 0.27.1) unconditionally iterates a non-string
    message.content assuming it is a list of parts, so the more "textbook"
    OpenAI shape for a pure tool-call turn -- content: None -- raised
    "TypeError: 'NoneType' object is not iterable" while rendering. That made
    the probe fail closed for a model whose tool-call syntax it never even got
    to look at, which is indistinguishable from "no parser fits" to every
    caller: --enable-auto-tool-choice never gets set and the client sees vLLM's
    "auto tool choice requires ..." 400. "" satisfies `content is string` in
    every template exercised so far. Reverting this to None silently reopens
    that regression, so pin the shape here.
    """
    assistant_turn = probe._MESSAGES[-1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"] == ""


class _RequestReadingParser:
    """Stand-in for vLLM's minicpm5 parser: it types arguments from ``request.tools``."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_tool_calls(self, model_output, request):
        # The real parser dereferences request.tools unconditionally.
        known = {t["function"]["name"] for t in request.tools}
        name = "trusta_probe_fn"
        called = name in known and name in model_output
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments='{"city": "Taipei"}'))
        return SimpleNamespace(tools_called=called, tool_calls=[call] if called else [])


class _RequestIgnoringParser(_RequestReadingParser):
    def extract_tool_calls(self, model_output, request):
        call = SimpleNamespace(function=SimpleNamespace(name="trusta_probe_fn", arguments="{}"))
        return SimpleNamespace(tools_called=True, tool_calls=[call])


_SEGMENT = '<function name="trusta_probe_fn"><param name="city">Taipei</param></function>'


def test_parser_that_reads_the_request_matches_when_given_one():
    """
    A parser that reads ``request.tools`` must be offered a request carrying the
    probe tool. Found live on openbmb/MiniCPM5-1B with vLLM 0.28.0: the probe
    passed request=None, the minicpm5 parser raised AttributeError, the error was
    swallowed, and the model was reported as having no parser -- tool calling
    silently stayed off under the default vllm_tool_call_parser=auto.
    """
    request = SimpleNamespace(tools=probe._TOOLS)
    matches = probe._matching_parsers(
        {"minicpm5": _RequestReadingParser}, object(), _SEGMENT, request
    )
    assert matches == ["minicpm5"]


def test_parser_that_reads_the_request_cannot_match_without_one():
    # Pins why the request matters: this is exactly what the probe used to do.
    matches = probe._matching_parsers({"minicpm5": _RequestReadingParser}, object(), _SEGMENT, None)
    assert matches == []


def test_mangled_arguments_are_still_not_a_match():
    # Passing a request must not loosen the argument check.
    request = SimpleNamespace(tools=probe._TOOLS)
    matches = probe._matching_parsers(
        {"loose": _RequestIgnoringParser}, object(), _SEGMENT, request
    )
    assert matches == []
