"""
Example of calling AST backend `/v1/chat/completions` with the OpenAI Python SDK.

This matches the `runChatCompletions` payload in Trusta-AST-Frontend,
and is the standard no-UI integration flow.

Workflow:
     1) Load a model first with `load_model_example.py` (any engine:
         transformers / llama_server / vllm).
     2) Run this file to send chat requests to `<backend_url>/v1/chat/completions`.
     3) Optionally run `unload_model_example.py` to release resources.

OpenAI standard parameters:
    model, messages, temperature, top_p, max_tokens, stream, stream_options,
    presence_penalty, user, ...

AST backend extra fields (passed via OpenAI SDK `extra_body`):
    - repetition_penalty
    - top_k
    - total_timeout
    - session_id, reset_history
    - enable_thinking
    - chat_template_kwargs
    - use_rag, rag_top_k, rag_query, rag_include_sources
    - request_id
    - tools, tool_choice
"""

from openai import OpenAI
import httpx

from helpers.default_backend import load_default_backend_url

# ---- Backend settings ----
BACKEND_URL = load_default_backend_url()
BASE_URL = BACKEND_URL.rstrip("/") + "/v1"
API_KEY = "your-api-key"  # Any string is accepted when backend auth is disabled
MODEL_NAME = (
    "trusta-ast/trusta-ast-default"  # Fixed binding name on OpenAI-compatible endpoint
)

# Skip SSL verification (dev backend uses a self-signed cert)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False),
)


def chat_stream(user_message: str) -> None:
    """Streaming mode: print output token by token."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=512,
        stream=True,
        stream_options={"include_usage": True},
        # Put AST backend-specific fields in extra_body
        extra_body={
            "top_k": 50,
            "repetition_penalty": 1.1,
            "total_timeout": 300,
            "enable_thinking": False,
            # Optional fields for session / RAG / tracing (remove if not needed)
            # "session_id": "your-session-uuid",
            # "reset_history": False,
            # "use_rag": False,
            # "rag_top_k": 3,
            # "rag_query": None,
            # "rag_include_sources": True,
            # "request_id": "your-trace-id",
        },
    )

    print(f"Prompt: {user_message}")
    print("Assistant: ", end="", flush=True)
    final_usage = None
    for chunk in response:
        # Content chunk
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        # Final chunk includes usage when stream_options.include_usage=True
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage
    print()

    if final_usage:
        print(f"\n[Stats] usage: {final_usage}")


def chat_completion(user_message: str) -> None:
    """Non-stream mode: fetch the full response in one call."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=800,
        stream=False,
        extra_body={
            "top_k": 50,
            "repetition_penalty": 1.1,
            "total_timeout": 300,
            "enable_thinking": False,
        },
    )

    print(f"Prompt: {user_message}")
    print(f"Assistant: {response.choices[0].message.content}")
    print(f"\n[Stats] usage: {response.usage}")


if __name__ == "__main__":
    # Example 1: streaming
    chat_stream("Hello, please introduce yourself.")

    # Example 2: non-streaming (uncomment to use)
    # chat_completion("Hello, please introduce yourself.")
