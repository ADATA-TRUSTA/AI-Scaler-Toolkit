"""
GPT Model Harmony Parser - Handles OpenAI GPT-OSS model response parsing
Uses openai-harmony library to parse structured responses with channels (analysis, commentary, final).
"""

import logging

# Import TIKTOKEN_CACHE_DIR from settings (must be imported early)
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Try to import openai-harmony
try:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        load_harmony_encoding,
    )

    HARMONY_AVAILABLE = True
    logger.info("✅ openai-harmony library available")
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("⚠️  openai-harmony not available. Install with: pip install openai-harmony")


def is_gpt_model(model_name: str) -> bool:
    """
    Detect whether the model belongs to the OpenAI GPT-OSS family.

    Args:
        model_name: Model name or path

    Returns:
        True if it is a GPT-OSS model
    """
    if not model_name:
        return False

    model_name_lower = model_name.lower()

    # Check GPT-OSS model name patterns
    gpt_patterns = [
        "gpt-oss",
        "openai/gpt",
        "gpt_oss",
    ]

    return any(pattern in model_name_lower for pattern in gpt_patterns)


class TokenIDStreamer:
    """
    Streamer that returns token IDs instead of decoded text for GPT StreamableParser.

    Note: This streamer is designed to work with generation threads that may fail.
    It will timeout and raise StopIteration if no tokens are received for too long,
    allowing the main loop to check for generation exceptions.
    """

    def __init__(self, skip_prompt: bool = True, timeout: float = 30.0) -> None:
        self.token_queue = []
        self.stop = False
        self.skip_prompt = skip_prompt
        self.prompt_skipped = False
        self.timeout = timeout  # Timeout in seconds for waiting for next token
        self.last_token_time = None
        self.exception = None  # Store exception from generation thread

    def put(self, value: Any) -> None:  # noqa: ANN401 - token-ids tensor or int from generate()
        """Enqueue generated token ids; a value of None signals end of stream."""
        if value is None:
            self.stop = True
            return

        # Skip the first batch (prompt) if skip_prompt is True
        if self.skip_prompt and not self.prompt_skipped:
            self.prompt_skipped = True
            return

        # value is token_ids tensor, could be [batch, seq] or [seq]
        if hasattr(value, "tolist"):
            ids = value.tolist()
            # Handle batch dimension
            if isinstance(ids, list):
                if isinstance(ids[0], list):
                    # [[token_ids...]]
                    for token_id in ids[0]:
                        self.token_queue.append(token_id)
                else:
                    # [token_ids...]
                    for token_id in ids:
                        self.token_queue.append(token_id)
        else:
            # Direct integer
            self.token_queue.append(value)

        # Update last token time whenever we receive tokens
        self.last_token_time = time.time()

    def end(self) -> None:
        """Signal the end of the token stream."""
        self.stop = True

    def set_exception(self, exception: Exception) -> None:
        """Allow generation thread to signal an exception."""
        self.exception = exception
        self.stop = True

    def __iter__(self) -> "TokenIDStreamer":
        return self

    def __next__(self) -> int:
        # Initialize last_token_time on first call
        if self.last_token_time is None:
            self.last_token_time = time.time()

        # Wait for token or stop signal
        while not self.token_queue:
            # Check if exception was set by generation thread
            if self.exception is not None:
                logger.error(f"[TokenIDStreamer] Generation exception detected: {self.exception}")
                raise self.exception

            # Check for stop signal
            if self.stop:
                raise StopIteration

            # Check for timeout (no tokens received for too long)
            elapsed = time.time() - self.last_token_time
            if elapsed > self.timeout:
                logger.error(
                    f"[TokenIDStreamer] Timeout after {elapsed:.1f}s waiting for next token"
                )
                raise TimeoutError(
                    f"TokenIDStreamer timeout after {elapsed:.1f}s - generation thread may have failed"
                )

            time.sleep(0.001)

        # Update last token time when we successfully return a token
        self.last_token_time = time.time()
        return self.token_queue.pop(0)


def is_harmony_available() -> bool:
    """Check whether the openai-harmony library is available."""
    return HARMONY_AVAILABLE


class GPTResponseParser:
    """GPT response parser - handles structured responses in Harmony format."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize the GPT response parser.

        Args:
            model_name: Model name (used for detection)
        """
        self.model_name = model_name
        self.is_gpt = is_gpt_model(model_name)

        if self.is_gpt and HARMONY_AVAILABLE:
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            logger.info(f"🚀 GPT Harmony Parser initialized for model: {model_name}")
        else:
            self.enc = None
            if self.is_gpt:
                logger.warning(f"⚠️  Model {model_name} is GPT but harmony library not available")

    def should_parse(self) -> bool:
        """Whether Harmony parsing should be used."""
        return self.is_gpt and self.enc is not None

    def render_conversation_for_completion(self, messages: list[dict[str, str]]) -> list[int]:
        """
        Render a conversation to tokens using the Harmony format.

        Args:
            messages: List of conversation messages, e.g. [{"role": "system", "content": "..."}, ...]

        Returns:
            List of token IDs
        """
        if not self.should_parse():
            raise ValueError("Harmony parser not available or not a GPT model")

        # should_parse() guarantees the encoding is loaded.
        assert self.enc is not None  # noqa: S101

        try:
            # Convert to Harmony Message format
            harmony_messages = []
            for msg in messages:
                role_str = msg.get("role", "user")
                content = msg.get("content", "")

                # Map the role
                if role_str == "system":
                    role = Role.SYSTEM
                elif role_str == "assistant":
                    role = Role.ASSISTANT
                else:
                    role = Role.USER

                harmony_messages.append(Message.from_role_and_content(role, content))

            # Build the conversation
            convo = Conversation.from_messages(harmony_messages)

            # Render to tokens
            input_tokens = self.enc.render_conversation_for_completion(convo, Role.ASSISTANT)

            logger.debug(f"Rendered {len(messages)} messages to {len(input_tokens)} tokens")
            return input_tokens

        except Exception as e:
            logger.exception(f"Failed to render conversation: {e}")
            raise

    def parse_generated_tokens(self, token_ids: list[int], strict: bool = False) -> dict[str, Any]:
        """
        Parse generated tokens into a structured response, wrapping analysis/commentary in <think></think> tags.

        Args:
            token_ids: Generated token IDs
            strict: Whether to parse strictly

        Returns:
            Structured response dict:
            {
                "formatted_text": str,  # Full formatted text (thinking wrapped in <think></think>)
                "parsed": bool          # Whether parsing succeeded
            }
        """
        if not self.should_parse():
            return {"formatted_text": "", "parsed": False}

        # should_parse() guarantees the encoding is loaded.
        assert self.enc is not None  # noqa: S101

        try:
            # Parse generated tokens with the official API
            parsed_messages = self.enc.parse_messages_from_completion_tokens(
                token_ids, role=Role.ASSISTANT, strict=strict
            )

            formatted_parts = []

            # Iterate over the parsed messages
            for i, msg in enumerate(parsed_messages):
                try:
                    if hasattr(msg, "content"):
                        # openai_harmony content items are duck-typed; attributes such as
                        # text/channel/analysis/final are probed via hasattr below.
                        contents: list[Any] = (
                            msg.content if isinstance(msg.content, list) else [msg.content]
                        )

                        for content in contents:
                            # Extract text and channel info
                            text = None
                            channel = None

                            if hasattr(content, "text"):
                                text = content.text
                            if hasattr(content, "channel"):
                                channel = content.channel

                            # Format according to the channel
                            if text:
                                # Thinking channels (analysis / commentary) wrapped in <think></think>
                                if channel in ["analysis", "commentary"]:
                                    formatted_parts.append(f"<think>\n{text}\n</think>")
                                # Actual reply (final channel) emitted as-is
                                elif channel == "final":
                                    formatted_parts.append(text)
                                # First message without an explicit channel is treated as thinking
                                elif not channel and i == 0:
                                    formatted_parts.append(f"<think>\n{text}\n</think>")
                                # Otherwise emit as-is
                                else:
                                    formatted_parts.append(text)

                            # Handle other possible attributes
                            if hasattr(content, "analysis") and content.analysis:
                                formatted_parts.append(f"<think>\n{content.analysis}\n</think>")

                            if hasattr(content, "final") and content.final:
                                formatted_parts.append(content.final)

                except Exception as e:
                    logger.debug(f"Error parsing message {i + 1}: {e}")
                    continue

            # Combine the results
            formatted_text = "\n\n".join(formatted_parts)

            return {"formatted_text": formatted_text, "parsed": True}

        except Exception as e:
            logger.warning(f"Failed to parse GPT response: {e}")
            return {"formatted_text": "", "parsed": False}


def create_gpt_parser(model_name: str) -> GPTResponseParser | None:
    """
    Create a GPT parser instance (factory function).

    Args:
        model_name: Model name

    Returns:
        A GPTResponseParser instance, or None if not a GPT model or the library is unavailable
    """
    if not is_gpt_model(model_name):
        return None

    if not HARMONY_AVAILABLE:
        logger.warning(f"GPT model detected ({model_name}) but openai-harmony not available")
        return None

    return GPTResponseParser(model_name)


def create_stream_parser(model_name: str) -> Any:  # noqa: ANN401 - openai_harmony StreamableParser or None
    """
    Create a GPT streaming parser (using the official StreamableParser).

    Args:
        model_name: Model name

    Returns:
        A StreamableParser instance, or None if not a GPT model or the library is unavailable
    """
    if not is_gpt_model(model_name):
        return None

    if not HARMONY_AVAILABLE:
        logger.warning(f"GPT model detected ({model_name}) but openai-harmony not available")
        return None

    try:
        from openai_harmony import StreamableParser

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stream_parser = StreamableParser(enc, role=Role.ASSISTANT)
        logger.info(f"🚀 GPT StreamableParser initialized for model: {model_name}")
        return stream_parser
    except Exception as e:
        logger.exception(f"Failed to create StreamableParser: {e}")
        return None
