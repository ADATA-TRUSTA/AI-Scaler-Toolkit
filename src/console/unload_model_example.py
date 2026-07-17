"""
AST Console AI - Unload Model Example (no-UI workflow)

Call backend `/inference/unload_model` to release resources of the currently loaded model.

Usage:
    uv run unload_model_example.py
    uv run unload_model_example.py --backend-url http://your-backend-host
"""

import argparse
import json
import sys

from ai_client import AIClient
from helpers.default_backend import load_default_backend_url
from helpers.model_loader import unload_model


def main() -> None:
    try:
        default_backend_url = load_default_backend_url()
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Failed to load default backend URL: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Unload the currently loaded model on the AST backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=default_backend_url,
        help=f"Backend base URL (default: {default_backend_url})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    client = AIClient(
        base_url=args.backend_url,
        timeout=args.timeout,
        log_requests=True,
    )

    try:
        unload_model(client)
    except Exception as e:
        print(f"\n[ERROR] Unload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
