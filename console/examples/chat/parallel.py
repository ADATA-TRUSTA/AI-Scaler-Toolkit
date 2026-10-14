"""
Parallel chat example for the AST backend OpenAI-compatible endpoint.

Opens one terminal UI with a grid of panes, one pane per prompt, and runs each
chat in its own worker process so you can watch N independent clients stream
concurrently. Works on Windows, Linux, and macOS -- including over SSH.

Usage:
    uv run examples/chat/parallel.py
    uv run examples/chat/parallel.py --settings configs/chat/parallel_16.json
    uv run examples/chat/parallel.py --columns 4

Workflow:
    1) Load a model first (with examples/model/load.py).
    2) Run this script to open a grid of N panes for N prompts.
    3) Each worker sends one chat request and streams into its own pane.
       Press q to quit.

One process per prompt is deliberate: it gives genuinely independent clients and
real process isolation, which an asyncio loop in a single process would not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx2
from openai import APIError

from ai_scaler_console.helpers.chat import (
    load_chat_settings,
    require_prompts,
    run_prompt,
    select_prompt,
)
from ai_scaler_console.openai_client import create_client

DEFAULT_SETTINGS_PATH = "configs/chat/parallel_8.json"
SCRIPT_PATH = Path(__file__).resolve()


def run_worker(settings_path: str, prompt_index: int) -> None:
    """Worker mode: run exactly one prompt in this process."""
    settings = load_chat_settings(settings_path)
    prompt = select_prompt(settings, prompt_index)
    prompt_count = len(settings.prompts)

    client = create_client(api_key=settings.api_key, backend_url=settings.backend_url)

    print("=" * 70)
    print(f"Worker {prompt_index + 1}/{prompt_count}")
    print("=" * 70)
    print(f"Model: {settings.model_name}")
    print(f"Stream: {settings.stream}")
    print()

    try:
        run_prompt(client, settings, prompt)
    except (APIError, httpx2.HTTPError, ValueError) as e:
        print(f"[ERROR] Prompt failed: {e}")
        raise


def run_controller(settings_path: str, columns: int | None) -> None:
    """Controller mode: validate settings, then open the pane grid."""
    settings = load_chat_settings(settings_path)
    prompts = require_prompts(settings)

    # Imported here so worker processes never pay to load a TUI framework.
    from ai_scaler_console.helpers.parallel_ui import ParallelChatApp

    ParallelChatApp(
        worker_script=SCRIPT_PATH,
        settings_path=settings_path,
        prompts=prompts,
        model_name=settings.model_name,
        columns=columns,
    ).run()


def main() -> None:
    """Dispatch to worker mode or open the pane grid."""
    parser = argparse.ArgumentParser(
        description="Run parallel chat workers in a terminal grid, one pane per prompt.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=DEFAULT_SETTINGS_PATH,
        help=f"Settings JSON path (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Number of grid columns (default: roughly square for the prompt count)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag used by spawned workers.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=-1,
        help="Internal worker prompt index.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.prompt_index < 0:
            raise ValueError("--prompt-index is required in worker mode")
        run_worker(args.settings, args.prompt_index)
        return

    if args.columns is not None and args.columns < 1:
        print("[ERROR] --columns must be at least 1")
        sys.exit(1)

    run_controller(args.settings, args.columns)


if __name__ == "__main__":
    main()
