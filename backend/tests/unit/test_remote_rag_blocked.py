"""Regression: a remote inference key must not reach RAG through use_rag."""

from service import app as app_module
from service.app import _drop_rag_for_remote
from service.config_models import OpenAIChatCompletionRequest


def _request():
    return OpenAIChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "use_rag": True}
    )


def test_remote_client_loses_use_rag():
    req = _request()
    _drop_rag_for_remote(req, "192.168.1.20")
    assert req.use_rag is False
    assert app_module._resolve_rag_context_text(req) is None


def test_local_client_keeps_use_rag():
    req = _request()
    _drop_rag_for_remote(req, "127.0.0.1")
    assert req.use_rag is True
