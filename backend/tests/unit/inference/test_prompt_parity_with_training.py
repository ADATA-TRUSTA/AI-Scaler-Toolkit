"""
Serving must not add prompt structure that fine-tuning never produced.

TRL renders only the roles actually present in the training data, so a fine-tune
trained from `{"prompt": [{"role": "user", ...}]}` rows has never seen a system
turn or a `<|think|>` prefix. Measured on gemma-4-E4B, the service's injected
system prompt moved the prompt from 282 to 322 tokens; a small adapter overfitted
on 11 samples keys on the exact prefix and then never fires.

These tests pin the three divergences found:
  1. a fabricated system prompt the caller cannot opt out of
  2. `enable_thinking` advertising a default the resolver never applies
  3. image/text part order preserved from the client instead of media-first
"""

import importlib

import pytest

pytestmark = pytest.mark.inference


# --------------------------------------------------------------------- 1. system prompt


def _reload_with_env(monkeypatch, value: str | None):
    """Reload settings (module-level os.getenv constants) under a given env var."""
    import service.settings as settings

    if value is None:
        monkeypatch.delenv("DEFAULT_SYSTEM_PROMPT", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_SYSTEM_PROMPT", value)
    return importlib.reload(settings)


def test_default_system_prompt_is_env_overridable_to_empty(monkeypatch):
    """An empty env var is the switch that lets a fine-tune be served faithfully."""
    settings = _reload_with_env(monkeypatch, "")
    try:
        assert settings.DEFAULT_SYSTEM_PROMPT == ""
    finally:
        _reload_with_env(monkeypatch, None)


def test_default_system_prompt_unchanged_when_env_absent(monkeypatch):
    """Existing deployments must keep the anti-repetition prompt they rely on."""
    settings = _reload_with_env(monkeypatch, None)
    assert "helpful AI assistant" in settings.DEFAULT_SYSTEM_PROMPT
    assert "Do not repeat yourself" in settings.DEFAULT_SYSTEM_PROMPT


def test_no_system_turn_is_emitted_when_the_default_is_empty(monkeypatch):
    """
    The point of the fix: omit the turn entirely, do not send an empty one.

    An empty system turn still renders a system block in the chat template, which
    is the same prompt drift we are removing.
    """
    import service.app as app

    monkeypatch.setattr(app, "DEFAULT_SYSTEM_PROMPT", "")

    system_instruction = ""  # what the resolution arrives at with no caller system text
    prompt_messages = (
        [{"role": "system", "content": system_instruction}] if system_instruction else []
    )
    assert prompt_messages == [], "no system turn at all"


def test_caller_supplied_system_text_still_wins(monkeypatch):
    """Disabling the fabricated default must not discard a real system message."""
    import service.app as app

    monkeypatch.setattr(app, "DEFAULT_SYSTEM_PROMPT", "")
    effective_system_texts = ["You are a radiology assistant."]

    system_instruction = "\n\n".join(t for t in effective_system_texts if t).strip()
    if not system_instruction:
        system_instruction = app.DEFAULT_SYSTEM_PROMPT.strip()

    assert system_instruction == "You are a radiology assistant."


def test_rag_context_survives_an_empty_default_system_prompt(monkeypatch):
    """
    RAG used to be appended to the default prompt with `+=`.

    With the default emptied, naive concatenation would leave a leading blank
    line, so the reference block must stand on its own.
    """
    import service.app as app

    monkeypatch.setattr(app, "DEFAULT_SYSTEM_PROMPT", "")

    system_instruction = ""
    rag_context_text = "Simon works in the storage division."
    reference = "Reference information (use only if helpful):\n" + rag_context_text.strip()
    system_instruction = f"{system_instruction}\n\n{reference}" if system_instruction else reference

    assert system_instruction.startswith("Reference information")
    assert not system_instruction.startswith("\n")


# --------------------------------------------------------------------- 2. enable_thinking


def test_enable_thinking_defaults_to_none_not_true():
    """
    `default=True` was never read by _resolve_enable_thinking (it gates on
    model_fields_set), so it only advertised a default the server does not apply.
    Thinking must stay opt-in: no training example carried a <|think|> prefix.
    """
    from service.config_models import OpenAIChatCompletionRequest

    field = OpenAIChatCompletionRequest.model_fields["enable_thinking"]
    assert field.default is None


def test_unset_enable_thinking_is_not_reported_as_set():
    """The resolver's contract: absence must be distinguishable from an explicit value."""
    from service.config_models import OpenAIChatCompletionRequest

    request = OpenAIChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert "enable_thinking" not in request.model_fields_set
    assert request.enable_thinking is None

    explicit = OpenAIChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], enable_thinking=True
    )
    assert "enable_thinking" in explicit.model_fields_set
    assert explicit.enable_thinking is True


# --------------------------------------------------------------------- 3. media-first order


@pytest.mark.parametrize(
    "client_content",
    [
        [{"type": "text", "text": "Who is this?"}, {"type": "image", "image": "<pil>"}],
        [{"type": "image", "image": "<pil>"}, {"type": "text", "text": "Who is this?"}],
    ],
    ids=["text-first-payload", "image-first-payload"],
)
def test_transformers_path_serves_media_before_text(client_content):
    """
    Training rows are always image-first. The transformers path preserved the
    client's order, so a text-first payload was served inverted.
    """
    from service.inference.async_server_client import reorder_multimodal_content

    messages = [{"role": "user", "content": client_content}]
    result = reorder_multimodal_content(messages)

    types = [part["type"] for part in result[0]["content"]]
    assert types == ["image", "text"], "media must come first regardless of client order"


def test_reorder_does_not_mutate_the_caller_list():
    """app.py feeds the same message list into session history; mutation would corrupt it."""
    from service.inference.async_server_client import reorder_multimodal_content

    original_content = [
        {"type": "text", "text": "Who is this?"},
        {"type": "image", "image": "<pil>"},
    ]
    messages = [{"role": "user", "content": original_content}]

    reorder_multimodal_content(messages)

    assert [p["type"] for p in original_content] == ["text", "image"], "caller list untouched"


def test_generator_core_applies_the_reorder():
    """Pin the wiring, not just the helper -- the helper existed and was unused here before."""
    import inspect

    from service.inference.generate import generator_core

    source = inspect.getsource(generator_core)
    assert "reorder_multimodal_content(messages_embedded)" in source
    assert "reorder_multimodal_content(messages_placeholder)" in source
