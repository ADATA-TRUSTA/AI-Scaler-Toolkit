"""
AST Console AI - Unload Model Example (no-UI workflow).

Call backend `/inference/unload_model` to release resources of the currently loaded model.

Usage:
    uv run examples/model/unload.py
    uv run examples/model/unload.py --backend-url http://your-backend-host
"""

import argparse
import sys

from ai_scaler_console import AIClient
from ai_scaler_console.helpers.model_loader import unload_model
from ai_scaler_console.settings import BACKEND_URL


def main() -> None:
    """Unload the model currently loaded on the backend."""
    parser = argparse.ArgumentParser(
        description="Unload the currently loaded model on the AST backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=BACKEND_URL,
        help=f"Backend base URL (default from .env: {BACKEND_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    with AIClient(
        base_url=args.backend_url,
        timeout=args.timeout,
        log_requests=True,
    ) as client:
        try:
            unload_model(client)
        except Exception as e:
            print(f"\n[ERROR] Unload failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
