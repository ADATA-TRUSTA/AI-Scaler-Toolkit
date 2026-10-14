"""
Batch chat example for AST backend OpenAI-compatible endpoint.

Usage:
    uv run examples/chat/batch.py
    uv run examples/chat/batch.py --settings configs/chat/batch.json

Workflow:
    1) Load a model first (for example with examples/model/load.py).
    2) Run this script to send a batch of prompts to /v1/chat/completions.
    3) Unload model when done.
"""

from __future__ import annotations

import argparse

import httpx2
from openai import APIError

from ai_scaler_console.helpers.chat import (
    load_chat_settings,
    require_prompts,
    run_prompt,
)
from ai_scaler_console.openai_client import create_client
from ai_scaler_console.settings import BACKEND_URL

DEFAULT_SETTINGS_PATH = "configs/chat/batch.json"


def main() -> None:
    """Run every prompt in the settings file, one after another."""
    parser = argparse.ArgumentParser(
        description="Batch run chat prompts via OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=DEFAULT_SETTINGS_PATH,
        help=f"Settings JSON path (default: {DEFAULT_SETTINGS_PATH})",
    )
    args = parser.parse_args()

    settings = load_chat_settings(args.settings)
    prompts = require_prompts(settings)
    backend_url = settings.backend_url or BACKEND_URL

    client = create_client(api_key=settings.api_key, backend_url=backend_url)

    print("=" * 70)
    print("Batch Chat")
    print("=" * 70)
    print(f"Backend URL: {backend_url}")
    print(f"Model: {settings.model_name}")
    print(f"Stream: {settings.stream}")
    print(f"Prompt count: {len(prompts)}")

    success_count = 0
    failure_count = 0

    for idx, prompt in enumerate(prompts, start=1):
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(prompts)}]")
        print("-" * 70)
        try:
            run_prompt(client, settings, prompt)
            success_count += 1
        except (APIError, httpx2.HTTPError, ValueError) as e:
            failure_count += 1
            print(f"[ERROR] Prompt failed: {e}")
            if not settings.continue_on_error:
                print("Stopping batch because continue_on_error is false.")
                break

    print("\n" + "=" * 70)
    print("Batch Summary")
    print("=" * 70)
    print(f"Success: {success_count}")
    print(f"Failure: {failure_count}")


if __name__ == "__main__":
    main()
