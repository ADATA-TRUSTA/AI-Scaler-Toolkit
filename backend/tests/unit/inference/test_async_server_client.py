"""
Unit tests for the async HTTP client used by the server inference engines.

Covers the currently-used surface: payload mapping, multimodal normalization,
streaming tool_call reassembly, endpoint resolution, and the generate /
generate_stream behaviour (against a stubbed OpenAI client, no network).

No pytest-asyncio dependency: async bodies run via asyncio.run().
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from service.config_models import InferenceEngine
from service.inference.async_server_client import (
    AsyncServerClient,
    _accumulate_tool_call_deltas,
    _assemble_tool_calls,
    _extract_reasoning,
    _fallback_parse_tool_call,
    _looks_like_fallback_prefix,
    _refit_max_tokens,
    build_chat_payload,
    client_for_config,
    is_server_engine,
    normalize_message_roles,
    reorder_multimodal_content,
    resolve_server_endpoint,
)

pytestmark = pytest.mark.inference


# ----------------------------- pure functions -----------------------------
def test_build_chat_payload_mapping_and_clamping():
    payload = build_chat_payload(
        "m",
        [{"role": "user", "content": "hi"}],
        {
            "max_new_tokens": 128,
            "temperature": 5.0,   # clamps to 2.0
            "top_p": 2.0,         # clamps to 1.0
            "top_k": 40,
            "repetition_penalty": 1.2,
            "enable_thinking": True,
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto",
        },
        stream=True,
    )
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 2.0
    assert payload["top_p"] == 1.0
    assert payload["stream"] is True
    assert payload["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert payload["tool_choice"] == "auto"
    assert payload["extra_body"]["top_k"] == 40
    assert payload["extra_body"]["repetition_penalty"] == 1.2
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_build_chat_payload_defaults_no_extra_body():
    payload = build_chat_payload("m", [{"role": "user", "content": "hi"}], {}, stream=False)
    # No cap asked for, none sent: a substituted 512 truncated callers that
    # never named one, and a reasoning model can spend that entire budget
    # thinking and return an empty answer.
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert "extra_body" not in payload
    assert "tools" not in payload


def test_build_chat_payload_output_cap_clamps_and_never_invents(monkeypatch):
    """
    MAX_OUTPUT_TOKENS_CAP is a ceiling, not a default.

    The 512 this replaces was a default: substituted whenever a caller named no
    cap, which truncated every auxiliary call Hermes makes (titling, context
    compression, classification) at exactly 512 generated tokens -- a budget a
    reasoning model spends entirely inside its think block. A configured cap is
    different in kind: an operator asked for it, so it applies to everyone, but
    only ever downwards.
    """
    import service.inference.async_server_client as client_mod

    messages = [{"role": "user", "content": "hi"}]

    # Unset: still nothing sent, so the served context window is the only bound.
    monkeypatch.setattr(client_mod, "MAX_OUTPUT_TOKENS_CAP", None)
    payload = build_chat_payload("m", messages, {}, stream=False)
    assert "max_tokens" not in payload
    payload = build_chat_payload("m", messages, {"max_new_tokens": 4000}, stream=False)
    assert payload["max_tokens"] == 4000

    # Set: applied to a caller that named nothing, and clamped onto one that did.
    monkeypatch.setattr(client_mod, "MAX_OUTPUT_TOKENS_CAP", 2048)
    payload = build_chat_payload("m", messages, {}, stream=False)
    assert payload["max_tokens"] == 2048
    payload = build_chat_payload("m", messages, {"max_new_tokens": 4000}, stream=False)
    assert payload["max_tokens"] == 2048
    # A caller asking for less than the ceiling keeps its own smaller figure.
    payload = build_chat_payload("m", messages, {"max_new_tokens": 100}, stream=False)
    assert payload["max_tokens"] == 100


def test_output_token_cap_setting_treats_blank_and_nonpositive_as_unset(monkeypatch):
    """An empty or nonsense value must not become a cap -- unset is the safe read."""
    from service.settings import _read_output_token_cap

    for raw in ("", "   ", "0", "-1", "not-a-number"):
        monkeypatch.setenv("MAX_OUTPUT_TOKENS_CAP", raw)
        assert _read_output_token_cap() is None, raw

    monkeypatch.setenv("MAX_OUTPUT_TOKENS_CAP", "8192")
    assert _read_output_token_cap() == 8192

    monkeypatch.delenv("MAX_OUTPUT_TOKENS_CAP", raising=False)
    assert _read_output_token_cap() is None


def _tool_history():
    return [
        {"role": "user", "content": "weather in Taipei?"},
        # A pure tool-call turn: no text content, only tool_calls.
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "22C"},
    ]


def test_payload_preserves_tool_calling_structure():
    """
    Dropping tool fields on the way out made the model stop after a preamble
    instead of emitting the next tool call: the history it saw had tool results
    appearing from nowhere, and it imitated that pattern."""
    msgs = build_chat_payload("m", _tool_history(), {}, stream=False)["messages"]

    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    # null content must stay null, never the literal string "None"
    assert msgs[1]["content"] is None
    assert msgs[2]["tool_call_id"] == "call_1"
    assert msgs[2]["name"] == "get_weather"
    assert msgs[2]["content"] == "22C"


def test_payload_is_json_serializable_with_tool_calls():
    """The payload goes out over HTTP; tool fields must need no conversion."""
    payload = build_chat_payload("m", _tool_history(), {}, stream=False)
    assert json.loads(json.dumps(payload))["messages"][1]["content"] is None


def test_payload_passes_messages_through_untouched():
    """No allowlist rebuild: whatever the caller built must arrive intact."""
    msgs = [{"role": "user", "content": "hi", "some_future_field": 1}]
    assert build_chat_payload("m", msgs, {}, stream=False)["messages"] == msgs


def test_payload_preserves_multipart_content():
    """Multi-part content must never be str()-flattened, or images are lost."""
    multi = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    assert build_chat_payload("m", multi, {}, stream=False)["messages"] == multi


def test_reorder_does_not_mutate_the_callers_messages():
    """
    app.py reuses its passthrough list for session history, so reordering must
    copy. This was previously masked by a normalization pass that handed over
    fresh dicts; that pass is gone, so the guarantee has to live here."""
    original_content = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    caller_msgs = [{"role": "user", "content": original_content}]
    out = reorder_multimodal_content(caller_msgs)

    # the returned copy is reordered ...
    assert [p["type"] for p in out[0]["content"]] == ["image_url", "text"]
    # ... while the caller's own dict and list are untouched
    assert caller_msgs[0]["content"] is original_content
    assert [p["type"] for p in original_content] == ["text", "image_url"]


def test_reorder_returns_same_list_when_nothing_to_do():
    """No needless copying on the common text-only path."""
    msgs = [{"role": "user", "content": "hi"}]
    assert reorder_multimodal_content(msgs) is msgs


def test_reorder_multimodal_media_before_text():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    ]
    out = reorder_multimodal_content(msgs)
    assert [p["type"] for p in out[0]["content"]] == ["image_url", "text"]


def test_reorder_multimodal_noop_when_text_only():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    out = reorder_multimodal_content(msgs)
    # no media -> order untouched
    assert [p["text"] for p in out[0]["content"]] == ["a", "b"]


# ------------------------- role normalization -------------------------
# Regression coverage for a real pi-agent failure: it sends "developer" as its
# instructions turn (OpenAI's system-message-equivalent role for reasoning
# models), which a local chat template that only knows system/user/assistant/tool
# rejects with "Unexpected message role." -- reproduced live against
# QuantTrio/Qwen3.5-4B-AWQ's own template under vLLM 0.27.1.


def test_normalize_message_roles_remaps_developer_to_system():
    msgs = [{"role": "developer", "content": "instructions"}, {"role": "user", "content": "hi"}]
    out = normalize_message_roles(msgs)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "instructions"
    assert out[1]["role"] == "user"


def test_normalize_message_roles_leaves_known_roles_untouched():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t", "tool_call_id": "c1"},
    ]
    assert normalize_message_roles(msgs) == msgs


def test_normalize_message_roles_does_not_mutate_the_callers_messages():
    """Same copy-on-write contract as reorder_multimodal_content: app.py reuses
    its passthrough list for session history."""
    caller_msgs = [{"role": "developer", "content": "instructions"}]
    out = normalize_message_roles(caller_msgs)
    assert out[0]["role"] == "system"
    assert caller_msgs[0]["role"] == "developer"  # caller's own dict untouched


def test_normalize_message_roles_returns_same_list_when_nothing_to_do():
    """No needless copying on the common path (no developer-role messages)."""
    msgs = [{"role": "user", "content": "hi"}]
    assert normalize_message_roles(msgs) is msgs


def test_build_chat_payload_remaps_developer_role_for_local_templates():
    payload = build_chat_payload(
        "m",
        [{"role": "developer", "content": "You are a coding agent."}, {"role": "user", "content": "hi"}],
        {},
        stream=False,
    )
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a coding agent."


def test_tool_call_fragment_reassembly():
    acc = {}
    _accumulate_tool_call_deltas(acc, [SimpleNamespace(index=0, id="c1", type="function", function=SimpleNamespace(name="get_weather", arguments=""))])
    _accumulate_tool_call_deltas(acc, [SimpleNamespace(index=0, id=None, type=None, function=SimpleNamespace(name=None, arguments='{"city"'))])
    _accumulate_tool_call_deltas(acc, [SimpleNamespace(index=0, id=None, type=None, function=SimpleNamespace(name=None, arguments=':"NY"}'))])
    assembled = _assemble_tool_calls(acc)
    assert assembled == [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NY"}'}}
    ]
    assert _assemble_tool_calls({}) is None


def _llama_config(**overrides):
    base = {
        "engine": InferenceEngine.LLAMA_SERVER,
        "llama_server_auto_start": True,
        "llama_server_host": "127.0.0.1",
        "llama_server_port": 5001,
        "llama_server_url": None,
        "llama_server_api_key": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_server_endpoint_and_is_server_engine():
    llama = resolve_server_endpoint(_llama_config())
    vllm = resolve_server_endpoint(SimpleNamespace(engine=InferenceEngine.VLLM))
    assert llama["base_url"].endswith("/v1")
    assert vllm["base_url"].endswith("/v1")
    assert is_server_engine(SimpleNamespace(engine=InferenceEngine.VLLM)) is True
    assert is_server_engine(SimpleNamespace(engine=InferenceEngine.TRANSFORMERS)) is False
    assert is_server_engine(None) is False


def test_resolve_server_endpoint_managed_uses_config_host_port():
    # Managed mode must follow the port the engine actually binds, not the
    # env default -- a load config on another port would otherwise be
    # unreachable from the main process.
    endpoint = resolve_server_endpoint(
        _llama_config(llama_server_host="127.0.0.1", llama_server_port=6123)
    )
    assert endpoint["base_url"] == "http://127.0.0.1:6123/v1"


def test_resolve_server_endpoint_external_uses_config_url():
    endpoint = resolve_server_endpoint(
        _llama_config(
            llama_server_auto_start=False,
            llama_server_url="http://10.0.0.9:9999/",
            llama_server_api_key="sekret",
        )
    )
    assert endpoint["base_url"] == "http://10.0.0.9:9999/v1"
    assert endpoint["api_key"] == "sekret"


def test_resolve_server_endpoint_url_wins_over_auto_start():
    # Mirrors LlamaServerEngine.load_model: an explicit llama_server_url wins
    # even in managed (auto_start) mode, so the main process targets the same
    # server the engine health-checks.
    endpoint = resolve_server_endpoint(
        _llama_config(llama_server_auto_start=True, llama_server_url="http://10.0.0.9:9999")
    )
    assert endpoint["base_url"] == "http://10.0.0.9:9999/v1"


def test_resolve_server_endpoint_external_without_url_uses_host_port():
    # External mode with no URL: the engine falls back to host:port, so the
    # client must too (not the env default).
    endpoint = resolve_server_endpoint(
        _llama_config(
            llama_server_auto_start=False,
            llama_server_host="10.0.0.7",
            llama_server_port=7777,
        )
    )
    assert endpoint["base_url"] == "http://10.0.0.7:7777/v1"


def test_client_for_config_sets_engine():
    # The engine drives payload naming in build_chat_payload (repeat_penalty
    # vs repetition_penalty), so it must be wired through.
    assert client_for_config(_llama_config()).engine == InferenceEngine.LLAMA_SERVER
    vllm_cfg = SimpleNamespace(
        engine=InferenceEngine.VLLM, model_name="m", model_path=None
    )
    assert client_for_config(vllm_cfg).engine == InferenceEngine.VLLM


def test_client_for_config_vllm_model_prefers_served_name(monkeypatch):
    import service.inference.async_server_client as asc

    cfg = SimpleNamespace(
        engine=InferenceEngine.VLLM,
        model_name="org/hf-name",
        model_path="/models/local",
    )
    # model_path wins over model_name: VllmEngine serves under
    # VLLM_SERVED_MODEL_NAME or model_path or model_name, and vLLM 404s on
    # any other model value.
    assert client_for_config(cfg)._model == "/models/local"

    monkeypatch.setattr(asc, "VLLM_SERVED_MODEL_NAME", "served-alias")
    assert client_for_config(cfg)._model == "served-alias"

    # llama-server keeps model_name (llama.cpp ignores the model field, and
    # Qwen detection relies on the name)
    llama_client = client_for_config(
        _llama_config(model_name="gguf-name", model_path="/x.gguf")
    )
    assert llama_client._model == "gguf-name"


# ----------------------------- stubbed client -----------------------------
class _Fn:
    def __init__(self, name=None, arguments=""):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, index=0, id=None, type="function", name=None, arguments=""):
        self.index = index
        self.id = id
        self.type = type
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _Msg:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, delta=None, message=None, finish_reason=None):
        self.delta = delta
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, p, c, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class _Event:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _Completion:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _Stream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def close(self):
        self.closed = True


class _Completions:
    def __init__(self, handler):
        self._handler = handler

    async def create(self, **kwargs):
        result = self._handler(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, handler):
        self.chat = SimpleNamespace(completions=_Completions(handler))
        self.models = SimpleNamespace(list=self._list)

    async def _list(self, timeout=None):
        return SimpleNamespace(data=[SimpleNamespace(id="stub-model")])

    async def close(self):
        pass


def _client_with(handler, model="m"):
    c = AsyncServerClient("http://x/v1", model=model)
    c.client = _FakeClient(handler)
    return c


def test_generate_non_stream_result_usage_tool_calls():
    def handler(kwargs):
        assert kwargs["stream"] is False
        return _Completion(
            choices=[_Choice(message=_Msg(content=None, tool_calls=[_ToolCall(id="c1", name="f", arguments="{}")]), finish_reason="tool_calls")],
            usage=_Usage(3, 4, 7),
        )

    c = _client_with(handler)
    res = asyncio.run(c.generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    assert res["gen_tokens"] == 4 and res["prompt_tokens"] == 3 and res["total_tokens"] == 7
    assert res["finish_reason"] == "tool_calls"
    assert res["tool_calls"][0]["function"]["name"] == "f"


def test_generate_stream_order_and_done_stats():
    def handler(kwargs):
        assert kwargs["stream"] is True
        return _Stream([
            _Event([_Choice(delta=_Delta(content="a"))]),
            _Event([_Choice(delta=_Delta(content="b"))]),
            _Event([_Choice(delta=_Delta())], usage=_Usage(2, 2, 4)),
        ])

    c = _client_with(handler)

    async def run():
        chunks, done = [], None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            if item.get("done"):
                done = item
            elif item.get("chunk"):
                chunks.append(item["chunk"])
        return chunks, done

    chunks, done = asyncio.run(run())
    assert chunks == ["a", "b"]
    assert done["gen_tokens"] == 2 and done["total_tokens"] == 4
    assert "tool_calls" not in done


def test_generate_stream_tool_calls_on_done():
    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(tool_calls=[_ToolCall(index=0, id="c1", name="get_weather", arguments="")]))]),
            _Event([_Choice(delta=_Delta(tool_calls=[_ToolCall(index=0, arguments='{"city"')]))]),
            _Event([_Choice(delta=_Delta(tool_calls=[_ToolCall(index=0, arguments=':"NY"}')]))]),
            _Event([_Choice(delta=_Delta(), finish_reason="tool_calls")], usage=_Usage(1, 5, 6)),
        ])

    c = _client_with(handler)

    async def run():
        done = None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            if item.get("done"):
                done = item
        return done

    done = asyncio.run(run())
    assert done["tool_calls"][0]["function"]["name"] == "get_weather"
    assert done["tool_calls"][0]["function"]["arguments"] == '{"city":"NY"}'
    assert done["finish_reason"] == "tool_calls"


# ==================== fallback tool-call recovery ====================
# Some models (observed from Qwen2.5-Coder under vLLM 0.27.1 with
# --enable-auto-tool-choice --tool-call-parser hermes) do not reliably emit the
# <tool_call> tag their own chat template documents, so the configured parser
# finds nothing and the tool call ends up as plain content text instead.


def test_fallback_parse_tool_call_recovers_tools_tag():
    content = '<tools>\n{"name": "get_weather", "arguments": {"city": "Taipei"}}\n</tools>'
    calls = _fallback_parse_tool_call(content)
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Taipei"}
    assert calls[0]["type"] == "function"


def test_fallback_parse_tool_call_recovers_fenced_response_tag():
    content = '```xml\n<response>\n    {"name": "get_weather", "arguments": {"city": "Taipei"}}\n</response>\n```'
    calls = _fallback_parse_tool_call(content)
    assert calls[0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "Sure, here is the weather in Taipei: sunny, 30C.",
        "<tools>not json</tools>",
        '<tools>{"arguments": {"city": "Taipei"}}</tools>',  # no "name"
        '<tools>{"name": 5, "arguments": {}}</tools>',  # name not a string
    ],
)
def test_fallback_parse_tool_call_returns_none_when_unrecoverable(content):
    assert _fallback_parse_tool_call(content) is None


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [
        ("", True),
        ("<to", True),
        ("<response", True),
        ("```", True),
        ("```xml\n<resp", True),
        ("Sure, here", False),
        ("<tools>{}</tools>trailing prose", True),  # still starts with the tag
    ],
)
def test_looks_like_fallback_prefix(buffer, expected):
    assert _looks_like_fallback_prefix(buffer) is expected


def test_fallback_parse_tool_call_recovers_bare_fenced_json_after_prose():
    """Reproduces a real pi-agent request: the model explains itself in prose,
    then puts the call in an untagged ```json fence -- no <tools>/<response>."""
    content = (
        "To list all files in the current workspace, you can use the bash "
        "function to execute a ls command. Here's how you can do it:\n\n"
        '```json\n{\n  "name": "bash",\n  "arguments": {\n    "command": "ls -la"\n  }\n}\n```'
        "\n\nThis command will list all files."
    )
    calls = _fallback_parse_tool_call(content)
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}


def test_fallback_parse_tool_call_rejects_names_the_request_never_declared():
    """A fenced JSON object with the right shape but an unknown name is not a
    tool call -- without this check the bare-fence pattern would also catch a
    JSON example the model was legitimately asked to write."""
    content = '```json\n{"name": "unrelated_thing", "arguments": {}}\n```'
    assert _fallback_parse_tool_call(content, frozenset({"bash", "read"})) is None
    assert _fallback_parse_tool_call(content) is not None  # no declared list: unchecked


def test_fallback_parse_tool_call_accepts_a_declared_name():
    content = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
    calls = _fallback_parse_tool_call(content, frozenset({"bash", "read"}))
    assert calls[0]["function"]["name"] == "bash"


def test_generate_non_stream_recovers_fallback_tool_call():
    def handler(kwargs):
        assert kwargs["tools"]  # only engaged because the request offered tools
        return _Completion(
            choices=[
                _Choice(
                    message=_Msg(
                        content='<tools>\n{"name": "get_weather", "arguments": {"city": "Taipei"}}\n</tools>'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=_Usage(3, 4, 7),
        )

    c = _client_with(handler)
    res = asyncio.run(
        c.generate(
            [{"role": "user", "content": "weather?"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        )
    )
    assert res["tool_calls"][0]["function"]["name"] == "get_weather"
    assert res["result"] == ""
    assert res["finish_reason"] == "tool_calls"


def test_generate_non_stream_recovers_prose_prefixed_bare_json_tool_call():
    """The exact shape seen from real pi-agent traffic: prose lead-in, then a
    bare ```json fence with no <tools>/<response> wrapper at all."""

    def handler(kwargs):
        assert kwargs["tools"]
        return _Completion(
            choices=[
                _Choice(
                    message=_Msg(
                        content=(
                            "To list all files in the current workspace, you can use "
                            "the bash function. Here's how:\n\n"
                            '```json\n{"name": "bash", "arguments": {"command": "ls -la"}}\n```'
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=_Usage(3, 4, 7),
        )

    c = _client_with(handler)
    res = asyncio.run(
        c.generate(
            [{"role": "user", "content": "list all files"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "bash"}}]},
        )
    )
    assert res["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(res["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    assert res["finish_reason"] == "tool_calls"
    # The prose lead-in is kept rather than discarded: it was already shown to
    # a streaming caller by the time the wrapper resolves, so the non-stream
    # path stays consistent with that instead of silently dropping it here.
    assert res["result"].startswith("To list all files")
    assert "```json" not in res["result"]


def test_generate_non_stream_does_not_fallback_on_unrelated_json_example():
    """A JSON code block the model was legitimately asked to write, whose "name"
    is not one of the request's declared tools, must not be swallowed."""

    def handler(kwargs):
        return _Completion(
            choices=[
                _Choice(
                    message=_Msg(content='Example config:\n```json\n{"name": "example"}\n```'),
                    finish_reason="stop",
                )
            ],
            usage=_Usage(1, 1, 2),
        )

    c = _client_with(handler)
    res = asyncio.run(
        c.generate(
            [{"role": "user", "content": "show me a config example"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "bash"}}]},
        )
    )
    assert "tool_calls" not in res
    assert "```json" in res["result"]


def test_generate_non_stream_does_not_fallback_without_tools_in_request():
    """The recovery must never trigger for a request that never offered tools."""

    def handler(kwargs):
        assert "tools" not in kwargs
        return _Completion(
            choices=[_Choice(message=_Msg(content="<tools>{}</tools>"), finish_reason="stop")],
            usage=_Usage(1, 1, 2),
        )

    c = _client_with(handler)
    res = asyncio.run(c.generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    assert "tool_calls" not in res
    assert res["result"] == "<tools>{}</tools>"


def test_generate_stream_reports_length_truncation():
    """A max_tokens cut-off must reach the caller as "length", not a silent "stop"."""

    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(content="half an ans"))]),
            _Event([_Choice(delta=_Delta(), finish_reason="length")], usage=_Usage(2, 512, 514)),
        ])

    c = _client_with(handler)

    async def run():
        done = None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            if item.get("done"):
                done = item
        return done

    done = asyncio.run(run())
    assert done["finish_reason"] == "length"
    assert "tool_calls" not in done


def test_generate_stream_reports_plain_stop():
    """A normal EOS stop is reported explicitly rather than left to a caller default."""

    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(content="done"))]),
            _Event([_Choice(delta=_Delta(), finish_reason="stop")], usage=_Usage(2, 3, 5)),
        ])

    c = _client_with(handler)

    async def run():
        done = None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            if item.get("done"):
                done = item
        return done

    assert asyncio.run(run())["finish_reason"] == "stop"


def test_generate_stream_keeps_server_finish_reason_over_tool_call_guess():
    """The server's own reason wins; the tool_calls fallback only fills a blank."""

    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(tool_calls=[_ToolCall(index=0, id="c1", name="f", arguments="{}")]))]),
            _Event([_Choice(delta=_Delta(), finish_reason="length")], usage=_Usage(1, 512, 513)),
        ])

    c = _client_with(handler)

    async def run():
        done = None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            if item.get("done"):
                done = item
        return done

    done = asyncio.run(run())
    # Truncated mid tool-call: the arguments are unusable, so the caller must see
    # "length" instead of a "tool_calls" reason that implies a complete call.
    assert done["finish_reason"] == "length"
    assert done["tool_calls"][0]["function"]["name"] == "f"


def test_generate_stream_recovers_fallback_tool_call_and_withholds_content():
    """The wrapper text must never reach the caller as a content chunk first."""
    wrapped = '<tools>\n{"name": "get_weather", "arguments": {"city": "Taipei"}}\n</tools>'

    def handler(kwargs):
        assert kwargs["tools"]
        # Split across several small deltas, the way real token streaming would.
        return _Stream(
            [_Event([_Choice(delta=_Delta(content=wrapped[i : i + 4]))]) for i in range(0, len(wrapped), 4)]
            + [_Event([_Choice(delta=_Delta(), finish_reason="stop")], usage=_Usage(1, 5, 6))]
        )

    c = _client_with(handler)

    async def run():
        chunks = []
        done = None
        async for item in c.generate_stream(
            [{"role": "user", "content": "weather?"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        ):
            if item.get("done"):
                done = item
            elif item.get("chunk"):
                chunks.append(item["chunk"])
        return chunks, done

    chunks, done = asyncio.run(run())
    assert chunks == []  # nothing leaked before the tool call was recognised
    assert done["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"city": "Taipei"}
    assert done["finish_reason"] == "tool_calls"


def test_generate_stream_flushes_buffer_when_not_a_wrapped_tool_call():
    """A real answer that merely starts like a fence must not lose any text."""

    def handler(kwargs):
        return _Stream(
            [
                _Event([_Choice(delta=_Delta(content="```python\n"))]),
                _Event([_Choice(delta=_Delta(content="def add(a, b):\n"))]),
                _Event([_Choice(delta=_Delta(content="    return a + b\n```"))]),
                _Event([_Choice(delta=_Delta(), finish_reason="stop")], usage=_Usage(1, 5, 6)),
            ]
        )

    c = _client_with(handler)

    async def run():
        chunks = []
        async for item in c.generate_stream(
            [{"role": "user", "content": "write add()"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        ):
            if item.get("chunk"):
                chunks.append(item["chunk"])
        return chunks

    chunks = asyncio.run(run())
    assert "".join(chunks) == "```python\ndef add(a, b):\n    return a + b\n```"


def test_generate_stream_forwards_prose_prefix_live_then_recovers_bare_json():
    """The real pi-agent bug: prose lead-in must stream live (not be withheld),
    and the trailing bare ```json fence must still resolve into tool_calls."""
    prefix = "To list all files, you can use the bash function. Here's how:\n\n"
    wrapped = '```json\n{"name": "bash", "arguments": {"command": "ls -la"}}\n```'
    full = prefix + wrapped

    def handler(kwargs):
        assert kwargs["tools"]
        return _Stream(
            [_Event([_Choice(delta=_Delta(content=full[i : i + 5]))]) for i in range(0, len(full), 5)]
            + [_Event([_Choice(delta=_Delta(), finish_reason="stop")], usage=_Usage(1, 5, 6))]
        )

    c = _client_with(handler)

    async def run():
        chunks = []
        done = None
        async for item in c.generate_stream(
            [{"role": "user", "content": "list files"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "bash"}}]},
        ):
            if item.get("done"):
                done = item
            elif item.get("chunk"):
                chunks.append(item["chunk"])
        return chunks, done

    chunks, done = asyncio.run(run())
    # The prose was streamed live; the fenced wrapper was not.
    assert "".join(chunks) == prefix
    assert done["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    assert done["finish_reason"] == "tool_calls"


def test_generate_stream_resumes_scanning_after_a_false_marker():
    """An earlier fenced block that turns out not to be a real tool call must
    not permanently disable recovery for a real one later in the same stream.

    The decoy is a non-JSON fence (a python snippet): it fails the
    still-plausible check quickly and gets flushed, which is what actually
    exercises the watching -> scanning fallback (a JSON-shaped decoy would
    instead stay withheld until the real call is found within the same span,
    which is also correct but does not exercise this path)."""
    decoy = "```python\nprint('hi')\n```"
    real = ' Now the actual call: ```json\n{"name": "get_weather", "arguments": {"city": "Taipei"}}\n```'
    full = f"Config example: {decoy}{real}"

    def handler(kwargs):
        return _Stream(
            [_Event([_Choice(delta=_Delta(content=full[i : i + 6]))]) for i in range(0, len(full), 6)]
            + [_Event([_Choice(delta=_Delta(), finish_reason="stop")], usage=_Usage(1, 5, 6))]
        )

    c = _client_with(handler)

    async def run():
        chunks = []
        done = None
        async for item in c.generate_stream(
            [{"role": "user", "content": "weather?"}],
            {"total_timeout": 5, "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        ):
            if item.get("done"):
                done = item
            elif item.get("chunk"):
                chunks.append(item["chunk"])
        return chunks, done

    chunks, done = asyncio.run(run())
    assert done["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"city": "Taipei"}
    assert done["finish_reason"] == "tool_calls"
    # Nothing from the decoy block or the lead-in text was lost along the way.
    joined = "".join(chunks)
    assert "Config example" in joined
    assert "print('hi')" in joined


def test_generate_stream_stop_event_aborts():
    def handler(kwargs):
        return _Stream([_Event([_Choice(delta=_Delta(content=str(i)))]) for i in range(10)])

    c = _client_with(handler)

    async def run():
        stop = asyncio.Event()
        got = []
        stopped = None
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}, stop_event=stop):
            if item.get("done"):
                stopped = item.get("stopped")
            elif item.get("chunk"):
                got.append(item["chunk"])
                if len(got) == 3:
                    stop.set()
        return got, stopped

    got, stopped = asyncio.run(run())
    assert len(got) < 10
    assert stopped is True


def test_generate_error_becomes_runtime_error():
    def handler(kwargs):
        return ValueError("upstream boom")

    c = _client_with(handler)
    err = None
    try:
        asyncio.run(c.generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    except RuntimeError as e:
        err = str(e)
    assert err and "boom" in err


def test_generate_stream_error_becomes_runtime_error():
    def handler(kwargs):
        return OSError("connection reset")

    c = _client_with(handler)

    async def run():
        async for _ in c.generate_stream([{"role": "user", "content": "hi"}], {"total_timeout": 5}):
            pass

    err = None
    try:
        asyncio.run(run())
    except RuntimeError as e:
        err = str(e)
    assert err and "connection reset" in err


# ------------------- reasoning / engine-specific payload -------------------
def test_build_chat_payload_llama_uses_repeat_penalty():
    p = build_chat_payload(
        "llama-x", [{"role": "user", "content": "hi"}],
        {"repetition_penalty": 1.2}, stream=False, engine=InferenceEngine.LLAMA_SERVER,
    )
    assert p["extra_body"]["repeat_penalty"] == 1.2
    assert "repetition_penalty" not in p["extra_body"]
    # llama always sends repeat_penalty, defaulting to 1.1 when unset
    p2 = build_chat_payload(
        "llama-x", [{"role": "user", "content": "hi"}], {}, stream=False,
        engine=InferenceEngine.LLAMA_SERVER,
    )
    assert p2["extra_body"]["repeat_penalty"] == 1.1


def test_build_chat_payload_vllm_uses_repetition_penalty():
    p = build_chat_payload(
        "vllm-x", [{"role": "user", "content": "hi"}],
        {"repetition_penalty": 1.2}, stream=False, engine=InferenceEngine.VLLM,
    )
    assert p["extra_body"]["repetition_penalty"] == 1.2
    assert "repeat_penalty" not in p["extra_body"]


def test_apply_thinking_tag_qwen():
    p = build_chat_payload(
        "Qwen/Qwen2.5-0.5B-Instruct", [{"role": "user", "content": "hi"}],
        {"enable_thinking": False}, stream=False, engine=InferenceEngine.VLLM,
    )
    assert p["messages"][-1]["content"].endswith("/no_think")
    p2 = build_chat_payload(
        "Qwen/Qwen2.5-0.5B-Instruct", [{"role": "user", "content": "hi"}],
        {"enable_thinking": True}, stream=False, engine=InferenceEngine.VLLM,
    )
    assert p2["messages"][-1]["content"].endswith("/think")


def test_reasoning_effort_reaches_vllm_as_a_request_field():
    # vLLM hands the request field to the template and to Harmony (gpt-oss).
    p = build_chat_payload(
        "openai/gpt-oss-20b", [{"role": "user", "content": "hi"}],
        {"enable_thinking": True, "reasoning_effort": "high"}, stream=False,
        engine=InferenceEngine.VLLM,
    )
    assert p["extra_body"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in p["extra_body"]["chat_template_kwargs"]


def test_reasoning_effort_reaches_llama_server_as_a_template_variable():
    # llama-server has no reasoning_effort field; gpt-oss's template reads the variable.
    p = build_chat_payload(
        "gpt-oss-20b", [{"role": "user", "content": "hi"}],
        {"enable_thinking": True, "reasoning_effort": "low"}, stream=False,
        engine=InferenceEngine.LLAMA_SERVER,
    )
    assert p["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in p["extra_body"]


@pytest.mark.parametrize("engine", [InferenceEngine.VLLM, InferenceEngine.LLAMA_SERVER])
def test_no_reasoning_effort_sends_no_level(engine):
    p = build_chat_payload(
        "m", [{"role": "user", "content": "hi"}], {"enable_thinking": False}, stream=False,
        engine=engine,
    )
    assert "reasoning_effort" not in p["extra_body"]
    assert "reasoning_effort" not in p["extra_body"]["chat_template_kwargs"]


def test_apply_thinking_tag_skipped_for_qwen35():
    # Qwen3.5 has no /think soft switch; only chat_template_kwargs toggles
    # thinking. The literal tag would leak into the prompt and confuse the
    # model (verified against Qwen3.5-35B-A3B on llama-server).
    for flag in (True, False):
        p = build_chat_payload(
            "unsloth/Qwen3.5-35B-GGUF", [{"role": "user", "content": "hi"}],
            {"enable_thinking": flag}, stream=False, engine=InferenceEngine.LLAMA_SERVER,
        )
        assert p["messages"][-1]["content"] == "hi"
        assert p["extra_body"]["chat_template_kwargs"]["enable_thinking"] is flag


def test_apply_thinking_tag_noop_non_qwen_or_unset():
    # non-Qwen model: untouched
    p = build_chat_payload(
        "llama-x", [{"role": "user", "content": "hi"}],
        {"enable_thinking": False}, stream=False, engine=InferenceEngine.LLAMA_SERVER,
    )
    assert p["messages"][-1]["content"] == "hi"
    # Qwen but enable_thinking unset: untouched
    p2 = build_chat_payload(
        "Qwen/Qwen2.5", [{"role": "user", "content": "hi"}], {}, stream=False,
        engine=InferenceEngine.VLLM,
    )
    assert p2["messages"][-1]["content"] == "hi"


def test_generate_stream_reasoning_on_its_own_key():
    def handler(kwargs):
        assert kwargs["stream"] is True
        return _Stream([
            _Event([_Choice(delta=_Delta(reasoning_content="think A"))]),
            _Event([_Choice(delta=_Delta(reasoning_content=" think B"))]),
            _Event([_Choice(delta=_Delta(content="answer"))]),
            _Event([_Choice(delta=_Delta())], usage=_Usage(3, 2, 5)),
        ])

    c = _client_with(handler)

    async def run():
        content, reasoning = [], []
        async for item in c.generate_stream(
            [{"role": "user", "content": "hi"}], {"enable_thinking": True, "total_timeout": 5}
        ):
            if item.get("chunk"):
                content.append(item["chunk"])
            if item.get("reasoning"):
                reasoning.append(item["reasoning"])
        return "".join(content), "".join(reasoning)

    content, reasoning = asyncio.run(run())
    assert reasoning == "think A think B"
    assert content == "answer"
    # No <think> markup is ever synthesised into the answer.
    assert "<think>" not in content and "</think>" not in content


def test_generate_non_stream_reasoning_on_its_own_key():
    def handler(kwargs):
        return _Completion(
            choices=[_Choice(message=_Msg(content="answer", reasoning_content="because reasons"), finish_reason="stop")],
            usage=_Usage(3, 2, 5),
        )

    c = _client_with(handler)
    res = asyncio.run(c.generate([{"role": "user", "content": "hi"}], {"enable_thinking": True, "total_timeout": 5}))
    assert res["reasoning_content"] == "because reasons"
    assert res["result"] == "answer"
    assert "<think>" not in res["result"]


def _collect(c, params):
    async def run():
        content, reasoning = [], []
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], params):
            if item.get("chunk"):
                content.append(item["chunk"])
            if item.get("reasoning"):
                reasoning.append(item["reasoning"])
        return "".join(content), "".join(reasoning)

    return asyncio.run(run())


def test_generate_stream_drops_separator_split_across_deltas():
    # vLLM splits at </think> and leaves the template's "\n\n" on the answer;
    # it can arrive one newline per delta, so the strip has to span chunks.
    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(reasoning_content="think"))]),
            _Event([_Choice(delta=_Delta(content="\n"))]),
            _Event([_Choice(delta=_Delta(content="\n"))]),
            _Event([_Choice(delta=_Delta(content="answer"))]),
            _Event([_Choice(delta=_Delta())], usage=_Usage(3, 2, 5)),
        ])

    content, reasoning = _collect(_client_with(handler), {"enable_thinking": True, "total_timeout": 5})
    assert reasoning == "think"
    assert content == "answer"


def test_generate_stream_keeps_newlines_inside_the_answer():
    # Only the separator goes; paragraph breaks the model wrote are content.
    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(reasoning_content="think"))]),
            _Event([_Choice(delta=_Delta(content="\n\npara one\n\npara two"))]),
            _Event([_Choice(delta=_Delta())], usage=_Usage(3, 2, 5)),
        ])

    content, _ = _collect(_client_with(handler), {"enable_thinking": True, "total_timeout": 5})
    assert content == "para one\n\npara two"


def test_generate_stream_leaves_content_alone_without_reasoning():
    # No reasoning block means no separator: a leading newline is the model's own.
    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=_Delta(content="\n\nanswer"))]),
            _Event([_Choice(delta=_Delta())], usage=_Usage(3, 2, 5)),
        ])

    content, reasoning = _collect(_client_with(handler), {"total_timeout": 5})
    assert reasoning == ""
    assert content == "\n\nanswer"


def test_generate_non_stream_drops_separator_after_reasoning():
    def handler(kwargs):
        return _Completion(
            choices=[_Choice(message=_Msg(content="\n\nanswer", reasoning_content="why"), finish_reason="stop")],
            usage=_Usage(3, 2, 5),
        )

    res = asyncio.run(_client_with(handler).generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    assert res["reasoning_content"] == "why"
    assert res["result"] == "answer"


def test_generate_non_stream_leaves_content_alone_without_reasoning():
    def handler(kwargs):
        return _Completion(
            choices=[_Choice(message=_Msg(content="\n\nanswer"), finish_reason="stop")],
            usage=_Usage(3, 2, 5),
        )

    res = asyncio.run(_client_with(handler).generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    assert "reasoning_content" not in res
    assert res["result"] == "\n\nanswer"


def test_extract_reasoning_field_and_location_variants():
    # attribute reasoning_content (llama-server / gpt-oss style)
    assert _extract_reasoning(SimpleNamespace(reasoning_content="rc", model_extra=None)) == "rc"
    # attribute reasoning (some servers)
    assert _extract_reasoning(SimpleNamespace(reasoning="r2", model_extra=None)) == "r2"
    # only in model_extra under 'reasoning' (vLLM reasoning parser via OpenAI SDK)
    d = SimpleNamespace(reasoning_content=None, model_extra={"reasoning": "from_extra"})
    assert _extract_reasoning(d) == "from_extra"
    # model_extra under 'reasoning_content'
    d2 = SimpleNamespace(reasoning_content=None, model_extra={"reasoning_content": "rc_extra"})
    assert _extract_reasoning(d2) == "rc_extra"
    # nothing -> None
    assert _extract_reasoning(SimpleNamespace(content="hi", model_extra={})) is None
    assert _extract_reasoning(None) is None


def test_generate_stream_reasoning_via_model_extra():
    # simulate vLLM: reasoning arrives only in delta.model_extra['reasoning']
    def handler(kwargs):
        return _Stream([
            _Event([_Choice(delta=SimpleNamespace(content=None, tool_calls=None, model_extra={"reasoning": "step 1"}))]),
            _Event([_Choice(delta=SimpleNamespace(content=None, tool_calls=None, model_extra={"reasoning": " step 2"}))]),
            _Event([_Choice(delta=SimpleNamespace(content="final", tool_calls=None, model_extra={}))]),
            _Event([_Choice(delta=SimpleNamespace(content=None, tool_calls=None, model_extra={}))], usage=_Usage(2, 3, 5)),
        ])

    c = _client_with(handler)

    async def run():
        content, reasoning = [], []
        async for item in c.generate_stream([{"role": "user", "content": "hi"}], {"enable_thinking": True, "total_timeout": 5}):
            if item.get("chunk"):
                content.append(item["chunk"])
            if item.get("reasoning"):
                reasoning.append(item["reasoning"])
        return "".join(content), "".join(reasoning)

    content, reasoning = asyncio.run(run())
    assert reasoning == "step 1 step 2"
    assert content == "final"

def test_generate_non_stream_qwen_default_thinking_kept():
    # Qwen3.5 thinks by default (template prefill), so reasoning arrives with
    # a normal answer even when enable_thinking was never set — it must not be
    # dropped.
    def handler(kwargs):
        return _Completion(
            choices=[_Choice(message=_Msg(content="391", reasoning_content="17*23 math"), finish_reason="stop")],
            usage=_Usage(3, 2, 5),
        )

    c = _client_with(handler, model="Qwen3.5-35B-A3B")
    res = asyncio.run(c.generate([{"role": "user", "content": "hi"}], {"total_timeout": 5}))
    assert res["reasoning_content"] == "17*23 math"
    assert res["result"] == "391"


# ------------------------ context overflow refit ------------------------
# What llama.cpp actually returns; the two numbers are the whole point.
_OVERFLOW = (
    "Error code: 400 - {'error': {'code': 400, 'message': 'request (65545 tokens) "
    "exceeds the available context size (65536 tokens), try increasing the context "
    "size', 'type': 'exceed_context_size_error'}}"
)


def test_refit_max_tokens_leaves_room_for_the_prompt():
    # 65545 in with 65536 asked names a 9-token prompt. A client that sizes
    # max_tokens from the advertised window overflows by exactly that much.
    assert _refit_max_tokens({"max_tokens": 65536}, Exception(_OVERFLOW)) == 65536 - 9 - 64


def test_refit_max_tokens_declines_when_the_prompt_is_what_does_not_fit():
    # 65800 in with 300 asked is a 65,500-token prompt against a 65,536 window:
    # no output cap rescues that, and trimming to a two-token answer would hide it.
    assert _refit_max_tokens({"max_tokens": 300}, Exception(_OVERFLOW.replace("65545", "65800"))) is None


def test_refit_max_tokens_ignores_unrelated_failures():
    assert _refit_max_tokens({"max_tokens": 512}, Exception("connection refused")) is None


def test_refit_max_tokens_ignores_a_request_without_a_cap():
    assert _refit_max_tokens({}, Exception(_OVERFLOW)) is None


def test_generate_stream_retries_once_with_a_cap_that_fits():
    seen = []

    def handler(kwargs):
        seen.append(kwargs.get("max_tokens"))
        if len(seen) == 1:
            raise RuntimeError(_OVERFLOW)
        return _Stream([_Event([_Choice(delta=_Delta(content="ok"))])])

    c = _client_with(handler)

    async def run():
        out = []
        async for item in c.generate_stream(
            [{"role": "user", "content": "hi"}],
            {"total_timeout": 5, "max_new_tokens": 65536},
        ):
            if item.get("chunk"):
                out.append(item["chunk"])
        return out

    assert asyncio.run(run()) == ["ok"]
    assert seen == [65536, 65536 - 9 - 64]


def test_generate_retries_once_then_gives_up():
    seen = []

    def handler(kwargs):
        seen.append(kwargs.get("max_tokens"))
        raise RuntimeError(_OVERFLOW)

    c = _client_with(handler)
    with pytest.raises(RuntimeError):
        asyncio.run(
            c.generate(
                [{"role": "user", "content": "hi"}],
                {"total_timeout": 5, "max_new_tokens": 65536},
            )
        )
    # Refitted once, then surfaced -- never a loop.
    assert seen == [65536, 65536 - 9 - 64]


def test_build_chat_payload_forwards_a_cap_that_was_asked_for():
    payload = build_chat_payload(
        "m", [{"role": "user", "content": "hi"}], {"max_new_tokens": 4096}, stream=False
    )
    assert payload["max_tokens"] == 4096


def test_build_chat_payload_ignores_a_meaningless_cap():
    # None arrives whenever an OpenAI client omits max_tokens; 0 and negatives
    # cannot be honoured. Sending any of them would be worse than sending none.
    for bad in (None, 0, -1):
        payload = build_chat_payload(
            "m", [{"role": "user", "content": "hi"}], {"max_new_tokens": bad}, stream=False
        )
        assert "max_tokens" not in payload, bad


# --------------------------------------------------------------------------
# total_timeout comes from the setting, not a literal
#
# DEFAULT_GENERATION_TIMEOUT existed but was read nowhere: every generation path
# carried its own `300`, so the env var changed nothing and a long prefill could
# only be rescued by passing total_timeout on each request. A 256K prefill was
# measured at ~320s, i.e. guaranteed to be cut off.
# --------------------------------------------------------------------------


def _timeout_handed_to_create(params, monkeypatch, default):
    """The `timeout` the client passes to create() for a streaming request.

    That value is the whole mechanism: it becomes the httpx read timeout, and on
    a streaming request the wait for the first byte *is* the prefill. A 256K
    prefill measured at ~320s against a 300s read timeout is cut off before the
    first token, which is the failure this fixes.
    """
    import service.inference.async_server_client as mod

    monkeypatch.setattr(mod, "DEFAULT_GENERATION_TIMEOUT", default)
    seen = {}

    def handler(kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _Stream([_Event([_Choice(delta=_Delta())], usage=_Usage(1, 1, 2))])

    c = _client_with(handler)

    async def run():
        async for _ in c.generate_stream([{"role": "user", "content": "hi"}], params):
            pass

    asyncio.run(run())
    return seen["timeout"]


def test_stream_timeout_comes_from_the_setting_when_the_caller_says_nothing(monkeypatch):
    assert _timeout_handed_to_create({}, monkeypatch, 1800) == 1800


def test_an_explicit_total_timeout_still_wins_over_the_setting(monkeypatch):
    assert _timeout_handed_to_create({"total_timeout": 45}, monkeypatch, 1800) == 45


def test_no_generation_path_still_hardcodes_300():
    """
    Guard against the literal creeping back in.

    The bug was not the number, it was that the number lived in seven places and
    the setting lived in none of them.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "service"
    offenders = []
    for path in root.rglob("*.py"):
        if "llama.cpp" in str(path):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"total_timeout[^\n]*\b300\b", line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")

    assert not offenders, "total_timeout must default to DEFAULT_GENERATION_TIMEOUT:\n" + "\n".join(
        offenders
    )
