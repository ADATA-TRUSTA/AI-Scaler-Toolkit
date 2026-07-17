"""
AST Console AI - Complete Workflow Example (load -> chat -> unload)

This is a complete end-to-end example showing the full cycle from model load,
chat requests, and model unload.
You can run this script directly instead of manually running multiple commands.

Workflow:
    1) Load model from the specified inference config
    2) Wait for model load to complete
    3) Run multiple chat prompts from chat settings
    4) Unload model and release resources

Usage:
    # Use workflow config file (recommended)
    uv run complete-workflow-example.py --workflow-config app_settings/workflow-config-default.json

    # Use default settings (transformers)
    uv run complete-workflow-example.py

    # Use a specific inference config
    uv run complete-workflow-example.py --config infer_model_configs/inference-config-vllm.json

    # Specify both inference config and chat settings
    uv run complete-workflow-example.py --config infer_model_configs/inference-config-vllm.json \
        --chat-settings app_settings/app_settings_parallel_terminal_batch_chat_8.json

    # Override backend URL and timeout
    uv run complete-workflow-example.py --config infer_model_configs/inference-config-transformers.json \
        --backend-url http://your-backend-host --timeout 900
"""

import argparse
import json
import sys

from openai import OpenAI
import httpx

from ai_client import AIClient
from helpers.config_loader import load_and_create_config
from helpers.default_backend import load_default_backend_url
from helpers.model_loader import prepare_and_load_model, unload_model


def main() -> None:
    """Main entry point for the complete workflow."""
    try:
        default_backend_url = load_default_backend_url()
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Failed to load default backend URL: {e}")
        sys.exit(1)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Complete workflow: load model -> chat -> unload model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workflow-config",
        type=str,
        default=None,
        help="Path to workflow config JSON (e.g. app_settings/workflow-config-default.json). "
        "If specified, other parameters can override individual settings.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config JSON (default: from workflow config or inference-config-transformers.json)",
    )
    parser.add_argument(
        "--chat-settings",
        type=str,
        default=None,
        help="Path to chat settings JSON (default: from workflow config)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help=f"Backend base URL (default: from workflow config or {default_backend_url})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds (default: from workflow config or 600)",
    )
    args = parser.parse_args()

    # Load workflow config if provided
    workflow_config = {}
    if args.workflow_config:
        try:
            with open(args.workflow_config, "r", encoding="utf-8") as f:
                workflow_config = json.load(f)
                print(f"[INFO] Loaded workflow config from: {args.workflow_config}")
        except FileNotFoundError:
            print(f"[ERROR] Workflow config file not found: {args.workflow_config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse workflow config: {e}")
            sys.exit(1)

    # Resolve effective parameters: CLI > workflow config > defaults
    model_config = args.config or workflow_config.get(
        "inference_config", "infer_model_configs/inference-config-transformers.json"
    )
    chat_settings_path = args.chat_settings or workflow_config.get("chat_settings")
    backend_url = args.backend_url or workflow_config.get(
        "backend_url", default_backend_url
    )
    timeout = args.timeout or workflow_config.get("timeout", 600)
    verbose = workflow_config.get("verbose", False)

    print("\n" + "=" * 70)
    print("  AST Console AI - Complete Workflow (Load -> Chat -> Unload)")
    print("=" * 70)
    print(f"  Backend URL:   {backend_url}")
    print(f"  Model config:  {model_config}")
    if chat_settings_path:
        print(f"  Chat settings: {chat_settings_path}")
    print(f"  Timeout:       {timeout}s")
    print("=" * 70 + "\n")

    # ========== Phase 1: Load model ==========
    try:
        print("[PHASE 1] Loading Model")
        print("-" * 70)

        # Load inference config
        try:
            infer_config = load_and_create_config(model_config)
        except FileNotFoundError:
            print(f"[ERROR] Config file not found: {model_config}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to parse config: {e}")
            sys.exit(1)

        print(f"  Engine:        {infer_config.engine}")
        print(f"  Model:         {infer_config.model_name}")
        if infer_config.model_path:
            print(f"  Model path:    {infer_config.model_path}")
        print()

        # Create API client and load model
        client = AIClient(
            base_url=backend_url,
            timeout=timeout,
            log_requests=verbose,
        )

        prepare_and_load_model(client, infer_config, infer_config.model_name)
        print("\nModel loaded successfully.\n")

    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        sys.exit(1)

    # ========== Phase 2: Run chat ==========
    try:
        print("[PHASE 2] Chat Conversations")
        print("-" * 70 + "\n")

        # Load chat settings
        chat_config = {}
        if chat_settings_path:
            try:
                with open(chat_settings_path, "r", encoding="utf-8") as f:
                    chat_config = json.load(f)
                    print(f"[INFO] Loaded chat settings from: {chat_settings_path}\n")
            except FileNotFoundError:
                print(f"[ERROR] Chat settings file not found: {chat_settings_path}")
                # Continue with defaults
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse chat settings: {e}")
                # Continue with defaults

        # Resolve chat settings with defaults
        base_url = chat_config.get("backend_url", backend_url).rstrip("/") + "/v1"
        api_key = chat_config.get("api_key", "your-api-key")
        model_name = chat_config.get("model", "trusta-ast/trusta-ast-default")
        system_prompt = chat_config.get("system_prompt", "You are a helpful assistant.")
        temperature = chat_config.get("temperature", 0.7)
        top_p = chat_config.get("top_p", 0.9)
        max_tokens = chat_config.get("max_tokens", 512)
        stream = chat_config.get("stream", False)
        continue_on_error = chat_config.get("continue_on_error", True)
        extra_body = chat_config.get(
            "extra_body",
            {
                "top_k": 50,
                "repetition_penalty": 1.1,
                "total_timeout": 300,
            },
        )

        # Default prompts if no prompts in settings
        prompts = chat_config.get(
            "prompts",
            [
                "Hello, please introduce yourself.",
                "Please explain what machine learning is.",
                "Recommend 3 learning resources for machine learning.",
            ],
        )

        # OpenAI-compatible client
        chat_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(verify=False),
        )

        # Multi-turn chat loop
        for i, user_message in enumerate(prompts, 1):
            print(f"[Question {i}]")
            print(f"User: {user_message}")
            print("Assistant: ", end="", flush=True)

            try:
                response = chat_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=stream,
                    extra_body=extra_body,
                )

                # Handle stream/non-stream output
                if stream:
                    final_usage = None
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            print(chunk.choices[0].delta.content, end="", flush=True)
                        if getattr(chunk, "usage", None):
                            final_usage = chunk.usage
                    print()
                    if final_usage:
                        print(
                            f"[Stats] Tokens - Input: {final_usage.prompt_tokens}, "
                            f"Output: {final_usage.completion_tokens}, "
                            f"Total: {final_usage.total_tokens}"
                        )
                else:
                    assistant_message = response.choices[0].message.content
                    print(assistant_message)
                    if response.usage:
                        print(
                            f"[Stats] Tokens - Input: {response.usage.prompt_tokens}, "
                            f"Output: {response.usage.completion_tokens}, "
                            f"Total: {response.usage.total_tokens}"
                        )

            except Exception as e:
                print(f"\n[ERROR] Chat request failed: {e}")
                if not continue_on_error:
                    raise
                # Continue with next prompt

            print()

    except Exception as e:
        print(f"\n[ERROR] Chat phase failed: {e}")
        # Try to unload model even if chat fails

    # ========== Phase 3: Unload model ==========
    try:
        print("\n[PHASE 3] Unloading Model")
        print("-" * 70)

        unload_model(client)
        print("\nModel unloaded successfully.\n")

    except Exception as e:
        print(f"\n[ERROR] Model unloading failed: {e}")
        sys.exit(1)

    # Completed
    print("=" * 70)
    print("  Complete workflow finished successfully.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
