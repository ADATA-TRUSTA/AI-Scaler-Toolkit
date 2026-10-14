"""
AST Console AI - Get Model List Example.

Fetch the unified model list from the Trusta AST backend's `/config/models`
endpoint. This mirrors the frontend's `getModelList()` call.

Usage:
    uv run examples/model/list.py
    uv run examples/model/list.py --backend-url http://your-backend-host
    uv run examples/model/list.py --json
"""

from __future__ import annotations

import argparse
import json
import sys

from ai_scaler_console import AIClient
from ai_scaler_console.config_models import ListedModel, ModelListResponse
from ai_scaler_console.exceptions import AIClientError
from ai_scaler_console.settings import BACKEND_URL


def print_model_group(title: str, models: list[ListedModel]) -> None:
    """Render one model group in a compact table-like format."""
    print(f"\n[{title}] ({len(models)})")
    if not models:
        print("  (empty)")
        return

    for index, model in enumerate(models, start=1):
        context_text = (
            str(model.max_context_length) if model.max_context_length is not None else "-"
        )
        method_text = model.method or "-"
        # A fine-tuned entry may carry no label / base model name; fall back to the
        # path so the row is still identifiable.
        print(f"  {index}. {model.label or model.model_path or '(unlabelled)'}")
        print(f"     model_name: {model.model_name or '-'}")
        print(f"     model_path: {model.model_path or '-'}")
        print(
            f"     size: {model.size or '-'} | max_context_length: {context_text} "
            f"| method: {method_text}"
        )


def print_summary(model_list: ModelListResponse) -> None:
    """Print a readable summary for interactive terminal use."""
    print("\n" + "=" * 60)
    print(" Model List")
    print("=" * 60)
    print_model_group("Base Models", model_list.base_models)
    print_model_group("Finetuned Models", model_list.finetuned_models)
    print_model_group("Llama GGUF Models", model_list.llama_gguf_models)

    if model_list.last_finetuned:
        print("\n[Last Finetuned]")
        for key, value in model_list.last_finetuned.items():
            print(f"  {key}: {value}")


def main() -> None:
    """Fetch and display the backend's unified model list."""
    parser = argparse.ArgumentParser(
        description="Fetch the backend unified model list from /config/models.",
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response instead of the formatted summary",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" Get Model List")
    print("=" * 60)
    print(f"  Backend URL: {args.backend_url}")

    with AIClient(
        base_url=args.backend_url,
        timeout=args.timeout,
        log_requests=True,
    ) as client:
        try:
            model_list = client.get_model_list()
        except AIClientError as exc:
            print(f"\n[ERROR] Failed to fetch model list: {exc}")
            sys.exit(1)

    if args.json:
        print(json.dumps(model_list.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

    print_summary(model_list)


if __name__ == "__main__":
    main()
