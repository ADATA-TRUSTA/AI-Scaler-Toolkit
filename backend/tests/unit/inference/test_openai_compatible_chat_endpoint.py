import json
import io
from types import SimpleNamespace

from fastapi.testclient import TestClient

from service.app import app


# As this machine: see test_remote_access.py.
client = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


import pytest

pytestmark = pytest.mark.inference


def _use_server_engine(monkeypatch):
    """Declare a server engine so the endpoint routes through the async HTTP path.

    Tool calling only exists there: the transformers worker drops `tools` before
    the chat template, so the endpoint now rejects tooling payloads on that path.
    """
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(engine="llama_server", model_path=None, model_name="Qwen/Qwen3-4B"),
        raising=False,
    )


def test_openai_chat_completions_non_stream(monkeypatch):
    captured = {}

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {
            "result": "你好，我是回覆",
            "prompt_tokens": 12,
            "gen_tokens": 8,
            "total_tokens": 20,
        }

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是助理"},
            {"role": "user", "content": "第一題"},
            {"role": "assistant", "content": "第一題回答"},
            {"role": "user", "content": "第二題"},
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200

    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "你好" in data["choices"][0]["message"]["content"]
    assert data["usage"]["total_tokens"] == 20

    # The converted prompt must preserve the multi-turn conversation.
    prompt = captured["prompt"]
    assert isinstance(prompt, list)
    assert prompt[-1]["role"] == "user"
    assert prompt[-1]["content"] == "第二題"


def test_openai_chat_completions_stream(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate_stream(**kwargs):
        yield {"chunk": "哈", "done": False}
        yield {"chunk": "囉", "done": False}
        yield {"chunk": "", "done": True}

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_generate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "messages": [
            {"role": "user", "content": "打個招呼"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    # Must contain at least one content chunk plus [DONE].
    payload_lines = [line for line in sse_lines if line.strip() != "data: [DONE]"]
    payloads = [json.loads(line.split("data:", 1)[1].strip()) for line in payload_lines]
    delta_texts = [
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
        for p in payloads
    ]
    assert any(t in {"哈", "囉"} for t in delta_texts)
    assert any(line.strip() == "data: [DONE]" for line in sse_lines)

    # Check that one of the payloads matches the OpenAI chunk structure.
    first_payload = json.loads(payload_lines[0].split("data:", 1)[1].strip())
    assert first_payload["object"] == "chat.completion.chunk"


def test_openai_chat_completions_uses_session_history(monkeypatch):
    captured = {}

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr(
        "service.app.session_manager.get_history",
        lambda _session_id: [
            {"role": "system", "content": "既有系統提示"},
            {"role": "user", "content": "先前問題"},
            {"role": "assistant", "content": "先前回答"},
        ],
    )

    def _fake_generate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {"result": "ok"}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "user": "session-1",
        "messages": [{"role": "user", "content": "新問題"}],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    prompt = captured["prompt"]
    assert prompt[0]["role"] == "system"
    assert "既有系統提示" in prompt[0]["content"]
    assert prompt[1] == {"role": "user", "content": "先前問題"}
    assert prompt[2] == {"role": "assistant", "content": "先前回答"}
    assert prompt[-1] == {"role": "user", "content": "新問題"}


def test_openai_chat_completions_stream_propagates_internal_error(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate_stream(**kwargs):
        yield {"error": "backend stream failed", "done": True}

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_generate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "messages": [
            {"role": "user", "content": "打個招呼"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.strip() != "data: [DONE]"
    ]

    assert any(payload.get("error", {}).get("message") == "backend stream failed" for payload in payloads)
    assert not any(
        ((payload.get("choices") or [{}])[0].get("finish_reason") == "stop")
        for payload in payloads
    )


def test_tools_rejected_on_transformers_engine(monkeypatch):
    """The transformers worker drops `tools` before the chat template, so the model
    never sees them. Refuse loudly instead of silently answering in prose."""
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: False)

    def _must_not_run(**kwargs):
        raise AssertionError("generation must not be reached")

    monkeypatch.setattr("service.app.model_manager.generate", _must_not_run)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "messages": [{"role": "user", "content": "台北天氣?"}],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert "transformers" in resp.json()["detail"]


def test_tool_role_messages_also_rejected_on_transformers_engine(monkeypatch):
    """A tool result in the history is just as unsupported as a tools list."""
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: False)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [
            {"role": "user", "content": "台北天氣?"},
            {"role": "tool", "tool_call_id": "c1", "content": "22C"},
        ],
    }

    assert client.post("/v1/chat/completions", json=body).status_code == 400


def test_plain_chat_still_works_on_transformers_engine(monkeypatch):
    """The guard must only fire on tooling payloads -- chat is unaffected."""
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr("service.app.model_manager.uses_async_http", lambda: False)
    monkeypatch.setattr(
        "service.app.model_manager.generate", lambda **kwargs: {"result": "你好"}
    )

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [{"role": "user", "content": "你好"}],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "你好" in resp.json()["choices"][0]["message"]["content"]


def test_openai_chat_completions_stream_reports_length_finish_reason(monkeypatch):
    """A truncated stream must not be reported to the client as a clean "stop"."""
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate_stream(**kwargs):
        yield {"chunk": "half an ans", "done": False}
        yield {"chunk": "", "done": True, "finish_reason": "length", "gen_tokens": 512}

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_generate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "messages": [{"role": "user", "content": "寫一篇長文"}],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.strip() != "data: [DONE]"
    ]
    reasons = [(p.get("choices") or [{}])[0].get("finish_reason") for p in payloads]
    assert "length" in reasons
    assert "stop" not in reasons


def test_openai_chat_completions_non_stream_reports_length_finish_reason(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr(
        "service.app.model_manager.generate",
        lambda **kwargs: {"result": "half an ans", "finish_reason": "length"},
    )

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [{"role": "user", "content": "寫一篇長文"}],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_openai_chat_completions_accepts_chat_template_kwargs_and_presence_penalty(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["enable_thinking"] = kwargs.get("enable_thinking")
        captured["repetition_penalty"] = kwargs.get("repetition_penalty")
        return {
            "result": "你好",
            "prompt_tokens": 10,
            "gen_tokens": 2,
            "total_tokens": 12,
        }

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "user", "content": "你好"},
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["enable_thinking"] is False
    assert captured["repetition_penalty"] == 1.5


def test_openai_chat_completions_accepts_max_completion_tokens_alias(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["max_new_tokens"] = kwargs.get("max_new_tokens")
        return {
            "result": "你好",
            "prompt_tokens": 10,
            "gen_tokens": 2,
            "total_tokens": 12,
        }

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "max_completion_tokens": 321,
        "messages": [
            {"role": "user", "content": "你好"},
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["max_new_tokens"] == 321


def test_openai_chat_completions_tool_calls_non_stream(monkeypatch):
    captured = {}

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    _use_server_engine(monkeypatch)

    async def _fake_agenerate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        captured["tools"] = kwargs.get("tools")
        captured["tool_choice"] = kwargs.get("tool_choice")
        return {
            "result": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "prompt_tokens": 14,
            "gen_tokens": 6,
            "total_tokens": 20,
        }

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查詢天氣",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200

    data = resp.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert captured["tools"][0]["function"]["name"] == "get_weather"
    assert captured["tool_choice"] == "auto"
    assert captured["prompt"][0]["role"] == "user"


def test_openai_chat_completions_tool_calls_text_parts_preserved(monkeypatch):
    captured = {}

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    _use_server_engine(monkeypatch)

    async def _fake_agenerate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        captured["tool_choice"] = kwargs.get("tool_choice")
        return {
            "result": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        }

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "tool_choice": {"type": "auto"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "幫我查"},
                    {"type": "text", "text": "台北天氣"},
                ],
            },
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["prompt"][0]["content"] == [
        {"type": "text", "text": "幫我查"},
        {"type": "text", "text": "台北天氣"},
    ]
    assert captured["tool_choice"] == "auto"


def test_openai_chat_completions_tool_calls_stream(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    # Was engine="transformers": that asserted tool streaming over the worker
    # path, which never actually produced tool_calls and is now rejected up front.
    _use_server_engine(monkeypatch)

    captured = {"generate_called": 0, "stream_called": 0}

    async def _fake_agenerate(**kwargs):
        captured["generate_called"] += 1
        return {
            "result": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }
            ],
            "done": True,
            "finish_reason": "tool_calls",
        }

    async def _fake_agenerate_stream(**kwargs):
        captured["stream_called"] += 1
        yield {
            "chunk": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }
            ],
            "done": True,
            "finish_reason": "tool_calls",
        }

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)
    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_agenerate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payload_lines = [line for line in sse_lines if line.strip() != "data: [DONE]"]
    payloads = [json.loads(line.split("data:", 1)[1].strip()) for line in payload_lines]
    tool_deltas = [
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
        for p in payloads
    ]
    assert captured["generate_called"] == 0
    assert captured["stream_called"] == 1
    assert any(delta for delta in tool_deltas)
    assert any(
        ((p.get("choices") or [{}])[0].get("finish_reason") == "tool_calls")
        for p in payloads
    )
    assert any(line.strip() == "data: [DONE]" for line in sse_lines)


def test_openai_chat_completions_stream_includes_usage_chunk(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate_stream(**kwargs):
        yield {"chunk": "哈", "done": False}
        yield {
            "chunk": "",
            "done": True,
            "prompt_tokens": 11,
            "gen_tokens": 7,
            "total_tokens": 18,
        }

    monkeypatch.setattr("service.app.model_manager.generate_stream", _fake_generate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "user", "content": "打個招呼"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.strip() != "data: [DONE]"
    ]

    usage_chunks = [payload for payload in payloads if payload.get("usage")]
    assert usage_chunks
    assert usage_chunks[-1]["usage"]["prompt_tokens"] == 11
    assert usage_chunks[-1]["usage"]["completion_tokens"] == 7
    assert usage_chunks[-1]["usage"]["total_tokens"] == 18
    assert usage_chunks[-1]["choices"] == []


def test_streamed_tool_calls_satisfy_the_openai_chunk_schema(monkeypatch):
    """Every streamed tool_call carries `index`, which ChoiceDeltaToolCall requires.

    The calls are accumulated into message shape (id/type/function) and sent in one
    closing chunk; without `index` a strict client drops them. Validate against the
    SDK's own model rather than a hand-written shape, and use two parallel calls so
    the indices have to differ.
    """
    from openai.types.chat import ChatCompletionChunk

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    _use_server_engine(monkeypatch)

    calls = [
        {"id": "call_a", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'}},
        {"id": "call_b", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}},
    ]

    async def _must_not_run(**kwargs):
        raise AssertionError("a streamed request must not fall back to non-stream")

    async def _fake_agenerate_stream(**kwargs):
        yield {"chunk": "", "tool_calls": calls, "done": True, "finish_reason": "tool_calls"}

    monkeypatch.setattr("service.app.model_manager.agenerate", _must_not_run)
    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_agenerate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [{"type": "function",
                   "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "messages": [{"role": "user", "content": "台北和東京天氣?"}],
    }
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        payloads = [
            json.loads(line.split("data:", 1)[1])
            for line in resp.iter_lines()
            if line.startswith("data:") and line.strip() != "data: [DONE]"
        ]

    chunks = [ChatCompletionChunk.model_validate(p) for p in payloads]
    streamed = [tc for c in chunks for ch in c.choices for tc in (ch.delta.tool_calls or [])]
    assert [tc.index for tc in streamed] == [0, 1]
    assert [tc.id for tc in streamed] == ["call_a", "call_b"]
    assert [tc.function.arguments for tc in streamed] == ['{"city":"Taipei"}', '{"city":"Tokyo"}']


def test_openai_chat_completions_tool_calls_stream_propagates_internal_error(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    # Was engine="transformers": that asserted tool streaming over the worker
    # path, which never actually produced tool_calls and is now rejected up front.
    _use_server_engine(monkeypatch)

    async def _fake_agenerate_stream(**kwargs):
        yield {"error": "tool stream failed", "done": True}

    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_agenerate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.strip() != "data: [DONE]"
    ]

    assert any(payload.get("error", {}).get("message") == "tool stream failed" for payload in payloads)
    assert not any(line.strip() == "data: [DONE]" for line in sse_lines)


def test_openai_chat_completions_tool_calls_stream_uses_stream_backend_for_llama_server(monkeypatch):
    # llama_server is a server engine: the endpoint must route through the
    # async HTTP backend (agenerate_stream), never the sync worker path.
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(engine="llama_server", model_path=None, model_name="Qwen/Qwen3-4B"),
        raising=False,
    )

    captured = {"generate_called": 0, "stream_called": 0}

    async def _fake_agenerate(**kwargs):
        captured["generate_called"] += 1
        return {
            "result": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
                }
            ],
            "finish_reason": "tool_calls",
            "prompt_tokens": 10,
            "gen_tokens": 5,
            "total_tokens": 15,
        }

    async def _fake_agenerate_stream(**kwargs):
        captured["stream_called"] += 1
        yield {
            "chunk": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
                }
            ],
            "done": True,
            "finish_reason": "tool_calls",
            "prompt_tokens": 10,
            "gen_tokens": 5,
            "total_tokens": 15,
        }

    def _poison_worker_path(**kwargs):
        raise AssertionError("sync worker path must not be used for llama_server")

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)
    monkeypatch.setattr(
        "service.app.model_manager.agenerate_stream", _fake_agenerate_stream
    )
    monkeypatch.setattr("service.app.model_manager.generate", _poison_worker_path)
    monkeypatch.setattr(
        "service.app.model_manager.generate_stream", _poison_worker_path
    )

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payload_lines = [line for line in sse_lines if line.strip() != "data: [DONE]"]
    payloads = [json.loads(line.split("data:", 1)[1].strip()) for line in payload_lines]

    assert captured["generate_called"] == 0
    assert captured["stream_called"] == 1
    assert any(
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
        for p in payloads
    )
    assert any(line.strip() == "data: [DONE]" for line in sse_lines)


def test_openai_chat_completions_tool_calls_stream_runtime_error_returns_sse_error(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    # Was engine="transformers": that asserted tool streaming over the worker
    # path, which never actually produced tool_calls and is now rejected up front.
    _use_server_engine(monkeypatch)

    async def _fake_agenerate_stream(**kwargs):
        raise RuntimeError("tool streaming backend failed")
        yield

    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_agenerate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for line in sse_lines
        if line.strip() != "data: [DONE]"
    ]

    assert any(
        payload.get("error", {}).get("message") == "tool streaming backend failed"
        for payload in payloads
    )
    assert not any(line.strip() == "data: [DONE]" for line in sse_lines)


def test_openai_chat_completions_tool_calls_stream_known_mismatch_falls_back_to_non_stream(monkeypatch):
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    # Was engine="transformers": that asserted tool streaming over the worker
    # path, which never actually produced tool_calls and is now rejected up front.
    _use_server_engine(monkeypatch)

    captured = {"generate_called": 0, "stream_called": 0}

    async def _fake_agenerate(**kwargs):
        captured["generate_called"] += 1
        return {
            "result": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        }

    async def _fake_agenerate_stream(**kwargs):
        captured["stream_called"] += 1
        raise RuntimeError("Invalid diff: now finding less tool calls!")
        yield

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)
    monkeypatch.setattr("service.app.model_manager.agenerate_stream", _fake_agenerate_stream)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "幫我查台北天氣"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        sse_lines = [line for line in resp.iter_lines() if line.startswith("data:")]

    payload_lines = [line for line in sse_lines if line.strip() != "data: [DONE]"]
    payloads = [json.loads(line.split("data:", 1)[1].strip()) for line in payload_lines]

    assert captured["generate_called"] == 1
    assert captured["stream_called"] == 1
    assert any(
        ((p.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
        for p in payloads
    )
    assert any(
        ((p.get("choices") or [{}])[0].get("finish_reason") == "tool_calls")
        for p in payloads
    )
    assert any(line.strip() == "data: [DONE]" for line in sse_lines)


def test_openai_chat_completions_accepts_multipart_image_upload(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {"result": "已收到圖片"}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    payload = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [
            {"role": "user", "content": "請描述這張圖片"},
        ],
    }

    resp = client.post(
        "/v1/chat/completions",
        data={"payload": json.dumps(payload, ensure_ascii=False)},
        files={"image": ("example.png", io.BytesIO(b"fake-image-bytes"), "image/png")},
    )

    assert resp.status_code == 200
    content = captured["prompt"][-1]["content"]
    assert content[0] == {"type": "text", "text": "請描述這張圖片"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_chat_completions_logs_sanitized_payload(monkeypatch):
    """INFO carries the shape of a request, never its words; DEBUG the sanitized payload."""
    log_messages = []
    debug_messages = []
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        return {"result": "ok"}

    def _collect(into):
        def _log(message, *args, **kwargs):
            into.append(message % args if args else message)

        return _log

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)
    monkeypatch.setattr("service.app.logger.info", _collect(log_messages))
    monkeypatch.setattr("service.app.logger.debug", _collect(debug_messages))
    monkeypatch.setattr("service.app.logger.isEnabledFor", lambda _level: True)

    raw_data_url = "data:image/png;base64," + ("A" * 256)
    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "請描述圖片"},
                    {"type": "image_url", "image_url": {"url": raw_data_url}},
                ],
            }
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200

    combined_logs = "\n".join(log_messages)
    assert "Received /v1/chat/completions request" in combined_logs
    assert "請描述圖片" not in combined_logs
    assert raw_data_url not in combined_logs

    debug_logs = "\n".join(debug_messages)
    assert "請描述圖片" in debug_logs
    assert "...(len=" in debug_logs
    assert raw_data_url not in debug_logs


def test_openai_chat_completions_accepts_json_attachments(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {"result": "已收到附件圖片"}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "unsloth/Qwen3.5-35B-GGUF",
        "stream": False,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "請介紹這張圖片"}]},
        ],
        "attachments": [
            {
                "name": "example.png",
                "mimeType": "image/png",
                "encoding": "base64",
                "content": "ZmFrZS1pbWFnZS1ieXRlcw==",
            }
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["prompt"][-1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="},
    }


def test_openai_chat_completions_accepts_input_image_content(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {"result": "ok"}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)

    body = {
        "model": "unsloth/Qwen3.5-35B-GGUF",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "content": "請介紹這張圖片"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                ],
            }
        ],
    }

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["prompt"][-1]["content"] == [
        {"type": "text", "text": "請介紹這張圖片"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]


def test_inference_chat_endpoint_is_deprecated():
    resp = client.post("/inference/chat", json={"message": "hello", "stream": False})
    assert resp.status_code == 410
    payload = resp.json()
    assert payload["detail"]["error"] == "deprecated_endpoint"


def test_qwen35_generation_options_honor_explicit_enable_thinking(monkeypatch):
    # Explicit request values must pass through; only an unspecified value
    # falls back to the stability default (off). Regression for the old
    # behaviour that force-disabled thinking unconditionally.
    from service.app import _resolve_model_aware_generation_options

    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(
            model_path="Qwen3.5-35B-A3B-Q4_K/Qwen3.5-35B-A3B-Q4_K_M.gguf",
            model_name="unsloth/Qwen3.5-35B-GGUF",
        ),
    )

    base = dict(temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0)
    on = _resolve_model_aware_generation_options(**base, enable_thinking=True)
    off = _resolve_model_aware_generation_options(**base, enable_thinking=False)
    unset = _resolve_model_aware_generation_options(**base, enable_thinking=None)
    assert on["enable_thinking"] is True
    assert off["enable_thinking"] is False
    assert unset["enable_thinking"] is False


def test_non_qwen35_generation_options_untouched(monkeypatch):
    from service.app import _resolve_model_aware_generation_options

    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(model_path=None, model_name="Qwen/Qwen3-4B"),
    )

    resolved = _resolve_model_aware_generation_options(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        enable_thinking=None,
    )
    assert resolved["enable_thinking"] is None
    assert resolved["temperature"] == 0.7
    assert resolved["repetition_penalty"] == 1.1


def _post_capturing_enable_thinking(monkeypatch, extra_body):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured["enable_thinking"] = kwargs.get("enable_thinking")
        return {"result": "ok", "prompt_tokens": 1, "gen_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)
    body = {"stream": False, "messages": [{"role": "user", "content": "hi"}], **extra_body}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    return captured["enable_thinking"]


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", False),
        ("minimal", False),
        (" NONE ", False),
        ("low", True),
        ("high", True),
        ("max", True),
        # hermes' top rung, past OpenAI's vocabulary; and any level yet to come.
        ("ultra", True),
        ("some-future-level", True),
        ("", None),
    ],
)
def test_reasoning_effort_maps_onto_enable_thinking(monkeypatch, effort, expected):
    """
    hermes (and the OpenAI SDK) express thinking as reasoning_effort, which the
    service used to drop, so a MiniCPM5 template was left with enable_thinking
    undefined and the model decided on its own whether to emit <think>. Only the
    two "off" levels turn it off; anything else asks for thinking.
    """
    got = _post_capturing_enable_thinking(monkeypatch, {"reasoning_effort": effort})
    assert got is expected


@pytest.mark.parametrize(
    ("effort", "forwarded"),
    [
        ("none", None),
        ("minimal", None),
        ("", None),
        (None, None),
        ("low", "low"),
        ("Medium", "medium"),
        ("high", "high"),
        # Past the graded core every model accepts: ask for the most it offers.
        ("xhigh", "high"),
        ("max", "high"),
        ("ultra", "high"),
        ("some-future-level", "high"),
    ],
)
def test_forwarded_reasoning_effort_stays_inside_the_core(effort, forwarded):
    """
    vLLM renders gpt-oss through Harmony, which takes only low/medium/high and fails
    on anything else -- "none" included -- so only that core is ever forwarded.
    """
    from service.app import _forwarded_reasoning_effort

    assert _forwarded_reasoning_effort(effort) == forwarded


def test_reasoning_effort_level_reaches_the_backend(monkeypatch):
    captured = {}
    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {"result": "ok", "prompt_tokens": 1, "gen_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr("service.app.model_manager.generate", _fake_generate)
    body = {
        "stream": False,
        "reasoning_effort": "ultra",
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert captured["enable_thinking"] is True
    assert captured["reasoning_effort"] == "high"


def test_explicit_thinking_controls_beat_reasoning_effort(monkeypatch):
    assert (
        _post_capturing_enable_thinking(
            monkeypatch, {"reasoning_effort": "high", "enable_thinking": False}
        )
        is False
    )
    assert (
        _post_capturing_enable_thinking(
            monkeypatch,
            {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}},
        )
        is True
    )


def _vllm_config(monkeypatch, reasoning_parser, model_name="openbmb/MiniCPM5-1B"):
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(
            engine="vllm",
            model_path=None,
            model_name=model_name,
            vllm_reasoning_parser=reasoning_parser,
        ),
    )


def _resolve_thinking(enable_thinking, for_tool_call=False):
    from service.app import _resolve_model_aware_generation_options

    return _resolve_model_aware_generation_options(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        enable_thinking=enable_thinking,
        for_tool_call=for_tool_call,
    )["enable_thinking"]


@pytest.mark.parametrize("for_tool_call", [False, True])
def test_vllm_reasoning_parser_pins_thinking_off_when_unspecified(monkeypatch, for_tool_call):
    """
    With a reasoning parser on and the template left undefined, MiniCPM5 often
    skipped <think>, and vLLM's qwen3 parser then filed the whole reply -- tool
    call included -- as reasoning: 10/10 replies after a tool result were empty.
    Pinned off rather than on so thinking stays opt-in: a MiniCPM5-1B LoRA lost
    knowledge with thinking on (18/40 probes vs 27/40 off).
    """
    _vllm_config(monkeypatch, "qwen3")
    assert _resolve_thinking(None, for_tool_call) is False


def test_vllm_reasoning_parser_keeps_an_explicit_choice(monkeypatch):
    _vllm_config(monkeypatch, "qwen3")
    assert _resolve_thinking(True) is True
    assert _resolve_thinking(False) is False


@pytest.mark.parametrize("parser", ["off", "none"])
def test_vllm_without_reasoning_parser_stays_opt_in(monkeypatch, parser):
    _vllm_config(monkeypatch, parser)
    assert _resolve_thinking(None) is None


def test_vllm_unset_parser_on_a_model_without_a_default_stays_opt_in(monkeypatch):
    _vllm_config(monkeypatch, None, model_name="Qwen/Qwen2.5-0.5B-Instruct")
    assert _resolve_thinking(None) is None


def test_vllm_minicpm5_default_parser_pins_thinking_like_an_explicit_one(monkeypatch):
    """Unset on MiniCPM5 runs qwen3 (its checked default), so the chat path must treat
    the parser as on: the same pin as an explicit qwen3, or replies come back empty."""
    _vllm_config(monkeypatch, None)
    assert _resolve_thinking(None) is False
    assert _resolve_thinking(True) is True


def test_llama_server_thinking_stays_opt_in(monkeypatch):
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(
            engine="llama_server", model_path=None, model_name="openbmb/MiniCPM5-1B-GGUF"
        ),
    )
    assert _resolve_thinking(None) is None


def _qwen35_config(monkeypatch):
    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(
            model_path="Qwen3.5-35B-A3B-Q4_K/Qwen3.5-35B-A3B-Q4_K_M.gguf",
            model_name="unsloth/Qwen3.5-35B-GGUF",
        ),
    )


def test_qwen35_tool_request_thinks_and_keeps_caller_sampling(monkeypatch):
    # A tool call is structured output, not prose. Measured on Qwen3.5-35B-A3B
    # against the same request: thinking off produced a markdown code block and
    # zero tool_calls, thinking on produced the call. Forcing temperature to 1.0
    # on top of that makes emitting a well-formed call a dice roll, so a tool
    # request keeps whatever sampling the caller asked for.
    from service.app import _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        enable_thinking=None,
        for_tool_call=True,
        max_tokens=32860,
    )
    assert resolved["enable_thinking"] is True
    assert resolved["temperature"] == 0.0
    assert resolved["top_p"] == 0.9
    assert resolved["top_k"] == 50


def test_qwen35_tool_request_skips_thinking_on_a_tight_budget(monkeypatch):
    # Thinking is what burns the budget: at max_tokens=512 the model finished
    # inside <think> and never answered. Below the threshold, keep the old
    # stability default so a small-budget caller still gets content back.
    from service.app import _TOOL_CALL_THINKING_MIN_TOKENS, _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        enable_thinking=None,
        for_tool_call=True,
        max_tokens=_TOOL_CALL_THINKING_MIN_TOKENS - 1,
    )
    assert resolved["enable_thinking"] is False


def test_qwen35_tool_request_without_a_cap_is_free_to_think(monkeypatch):
    from service.app import _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        enable_thinking=None,
        for_tool_call=True,
        max_tokens=None,
    )
    assert resolved["enable_thinking"] is True


def test_qwen35_tool_request_honours_an_explicit_thinking_choice(monkeypatch):
    # The budget default only fills in a value the caller never gave.
    from service.app import _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    base = dict(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        for_tool_call=True,
        max_tokens=32860,
    )
    assert (
        _resolve_model_aware_generation_options(**base, enable_thinking=False)["enable_thinking"]
        is False
    )
    assert (
        _resolve_model_aware_generation_options(**base, enable_thinking=True)["enable_thinking"]
        is True
    )


def test_qwen35_tool_request_still_clamps_repetition_penalty(monkeypatch):
    # The sampling exemption is about reliability of structured output. The
    # repetition_penalty clamp guards against mixed-language gibberish, which is
    # a correctness problem for tool arguments too -- it stays unconditional.
    from service.app import _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        enable_thinking=None,
        for_tool_call=True,
        max_tokens=32860,
    )
    assert resolved["repetition_penalty"] == 1.0


def test_qwen35_non_tool_request_keeps_the_official_sampling_defaults(monkeypatch):
    # The prose path is unchanged: official Qwen3.5 sampling, thinking off.
    from service.app import _resolve_model_aware_generation_options

    _qwen35_config(monkeypatch)
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        enable_thinking=None,
        max_tokens=32860,
    )
    assert resolved["enable_thinking"] is False
    assert resolved["temperature"] == 1.0
    assert resolved["top_p"] == 0.95
    assert resolved["top_k"] == 20


def test_non_qwen35_tool_request_is_left_alone(monkeypatch):
    # for_tool_call must not leak Qwen3.5 policy onto other families.
    from service.app import _resolve_model_aware_generation_options

    monkeypatch.setattr(
        "service.app.model_manager.config",
        SimpleNamespace(model_path=None, model_name="google/gemma-4-26B-A4B-it"),
    )
    resolved = _resolve_model_aware_generation_options(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        enable_thinking=None,
        for_tool_call=True,
        max_tokens=32860,
    )
    assert resolved["enable_thinking"] is None
    assert resolved["temperature"] == 0.0
    assert resolved["repetition_penalty"] == 1.1


def test_generation_kwargs_bind_to_every_model_manager_entry_point(monkeypatch):
    """Whatever the endpoint sends must be accepted by both dispatch targets.

    _backend_generate hands one kwargs dict to model_manager.agenerate (server
    engines) or model_manager.generate (transformers worker) depending on the
    loaded engine, so a kwarg added for one path breaks the other with a bare
    TypeError at request time. The fakes elsewhere in this file take **kwargs
    and so cannot catch that; bind the captured kwargs to the real signatures.
    """
    import inspect

    from service.model_manager import ModelManager

    captured = {}

    monkeypatch.setattr("service.app.model_manager.is_loaded", lambda: True)
    _use_server_engine(monkeypatch)

    async def _fake_agenerate(**kwargs):
        captured.update(kwargs)
        return {"result": "ok", "prompt_tokens": 1, "gen_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr("service.app.model_manager.agenerate", _fake_agenerate)

    body = {
        "model": "Qwen/Qwen3-4B",
        "stream": False,
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "messages": [{"role": "user", "content": "台北天氣?"}],
    }
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert captured, "endpoint never reached generation"

    for target in (ModelManager.agenerate, ModelManager.generate, ModelManager.agenerate_stream):
        # bind_partial with a placeholder self: raises TypeError on an unknown kwarg,
        # which is exactly the failure this guards against.
        inspect.signature(target).bind_partial(None, **captured)
