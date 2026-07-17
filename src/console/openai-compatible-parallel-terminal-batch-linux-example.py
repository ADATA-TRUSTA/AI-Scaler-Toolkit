"""
Parallel terminal batch chat example for AST backend OpenAI-compatible endpoint.

Linux-specific launcher: opens one Linux terminal window per prompt and runs
each chat synchronously inside its own worker process.

Usage:
    uv run openai-compatible-parallel-terminal-batch-linux-example.py
    uv run openai-compatible-parallel-terminal-batch-linux-example.py --settings app_settings/app_settings_parallel_terminal_batch_chat.json

Workflow:
    1) Load a model first (for example with load_model_example.py).
    2) Run this script to spawn N terminal windows for N prompts.
    3) Each worker sends one synchronous chat request and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import httpx
from openai import APIError, OpenAI

from helpers.default_backend import load_default_backend_url

DEFAULT_SETTINGS_PATH = "app_settings/app_settings_parallel_terminal_batch_chat.json"


def load_settings(settings_path: str) -> dict[str, Any]:
    """Load settings from JSON file."""
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_single_prompt(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    stream: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict[str, Any],
) -> None:
    """Run one synchronous prompt and print the response."""
    print(f"Prompt: {user_prompt}")

    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=cast(Any, build_messages(system_prompt, user_prompt)),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )

        final_usage = None
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
            if getattr(chunk, "usage", None):
                final_usage = chunk.usage
        print()
        if final_usage:
            print(f"[Stats] usage: {final_usage}")
        return

    response = client.chat.completions.create(
        model=model_name,
        messages=cast(Any, build_messages(system_prompt, user_prompt)),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        extra_body=extra_body,
    )

    print(f"Assistant: {response.choices[0].message.content}")
    print(f"[Stats] usage: {response.usage}")


def run_worker(settings_path: str, prompt_index: int) -> None:
    """Worker mode: execute one prompt in this process."""
    settings = load_settings(settings_path)
    default_backend_url = load_default_backend_url()

    backend_url = settings.get("backend_url", default_backend_url)
    api_key = settings.get("api_key", "your-api-key")
    model_name = settings.get("model", "trusta-ast/trusta-ast-default")
    system_prompt = settings.get("system_prompt", "You are a helpful assistant.")
    stream = bool(settings.get("stream", False))
    temperature = float(settings.get("temperature", 0.5))
    top_p = float(settings.get("top_p", 0.9))
    max_tokens = int(settings.get("max_tokens", 800))
    extra_body = dict(settings.get("extra_body", {}))
    prompts = settings.get("prompts", [])

    if not isinstance(prompts, list) or not prompts:
        raise ValueError("settings.prompts must be a non-empty list of strings")
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(
            f"prompt index {prompt_index} is out of range for {len(prompts)} prompts"
        )

    prompt = prompts[prompt_index]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"settings.prompts[{prompt_index}] must be a non-empty string")

    base_url = backend_url.rstrip("/") + "/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False),
    )

    print("=" * 70)
    print(f"Worker {prompt_index + 1}/{len(prompts)}")
    print("=" * 70)
    print(f"Backend URL: {backend_url}")
    print(f"Model: {model_name}")
    print(f"Stream: {stream}")
    print()

    try:
        run_single_prompt(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    except (APIError, httpx.HTTPError, ValueError) as e:
        print(f"[ERROR] Prompt failed: {e}")
        raise
    finally:
        print()
        print("=" * 70)
        input("Press Enter to close this terminal...")
        print("=" * 70)


def get_terminal_launcher() -> tuple[list[str], str]:
    """Return the terminal launcher command prefix and argument style."""
    candidates = [
        ("x-terminal-emulator", "-e"),
        ("gnome-terminal", "--"),
        ("konsole", "-e"),
        ("xterm", "-e"),
        ("xfce4-terminal", "-x"),
        ("mate-terminal", "--"),
        ("lxterminal", "-e"),
    ]

    for executable, mode in candidates:
        if shutil.which(executable):
            return [executable], mode

    raise RuntimeError(
        "No supported terminal emulator found. Install one of: x-terminal-emulator, "
        "gnome-terminal, konsole, xterm, xfce4-terminal, mate-terminal, lxterminal."
    )


def spawn_worker_terminals(
    settings_path: str, prompt_count: int
) -> list[subprocess.Popen[Any]]:
    """Spawn one Linux terminal window per prompt."""
    if os.name != "posix":
        raise RuntimeError("This script is intended for Linux/Unix environments.")

    launcher_prefix, mode = get_terminal_launcher()
    script_path = Path(__file__).resolve()
    processes: list[subprocess.Popen[Any]] = []

    for prompt_index in range(prompt_count):
        worker_command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--settings",
            settings_path,
            "--prompt-index",
            str(prompt_index),
        ]

        if mode == "-x":
            command = launcher_prefix + [mode] + worker_command
        else:
            command = launcher_prefix + [mode] + worker_command

        process = subprocess.Popen(command)
        processes.append(process)

    return processes


def run_controller(settings_path: str) -> None:
    """Controller mode: validate settings and spawn N worker terminals."""
    settings = load_settings(settings_path)
    prompts = settings.get("prompts", [])

    if not isinstance(prompts, list) or not prompts:
        raise ValueError("settings.prompts must be a non-empty list of strings")
    if not all(isinstance(item, str) and item.strip() for item in prompts):
        raise ValueError("every item in settings.prompts must be a non-empty string")

    print("=" * 70)
    print("Parallel Terminal Batch Chat")
    print("=" * 70)
    print(f"Settings: {settings_path}")
    print(f"Prompt count: {len(prompts)}")
    print("Each prompt will open in its own terminal window.")

    processes = spawn_worker_terminals(settings_path, len(prompts))

    for process in processes:
        return_code = process.wait()
        if return_code != 0:
            print(f"[WARN] A worker exited with code {return_code}")

    print("\nAll worker terminals have finished.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open N Linux terminal windows and run one synchronous chat per window.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=DEFAULT_SETTINGS_PATH,
        help=f"Settings JSON path (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag used by spawned worker windows.",
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

    run_controller(args.settings)


if __name__ == "__main__":
    main()
