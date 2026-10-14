"""
AST Console AI - Load Model Example (no-UI workflow).

Use the backend's `inference/load_model` endpoint and load a model from a config
file under `configs/inference/*.json`. After loading is complete, use
`examples/chat/basic.py` / `examples/chat/image.py` for chat.

Usage:
    # transformers (full precision / quantization)
    uv run examples/model/load.py --config configs/inference/transformers.json

    # transformers + GPU/CPU offload
    uv run examples/model/load.py --config configs/inference/transformers_offload.json

    # llama.cpp (GGUF)
    uv run examples/model/load.py --config configs/inference/llama.json

    # vLLM
    uv run examples/model/load.py --config configs/inference/vllm.json

    # Override the backend URL from .env
    uv run examples/model/load.py --config configs/inference/vllm.json \
        --backend-url http://your-backend-host
"""

import argparse
import sys

from ai_scaler_console import AIClient
from ai_scaler_console.helpers.config_loader import load_and_create_config
from ai_scaler_console.helpers.model_loader import prepare_and_load_model
from ai_scaler_console.settings import BACKEND_URL


def main() -> None:
    """Load a model on the backend from an inference config JSON."""
    parser = argparse.ArgumentParser(
        description="Load a model on the AST backend using a configs/inference JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a model config JSON under configs/inference/ "
        "(e.g. configs/inference/transformers.json)",
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
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" Load Model")
    print("=" * 60)
    print(f"  Backend URL: {args.backend_url}")
    print(f"  Config file: {args.config}")
    print()

    # 1) Build InferenceConfig from JSON (engine-agnostic; pydantic validates fields)
    try:
        infer_config = load_and_create_config(args.config)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}")
        sys.exit(1)

    print(f"  Engine: {infer_config.engine}")
    print(f"  Model: {infer_config.model_name}")
    if infer_config.model_path:
        print(f"  Model path: {infer_config.model_path}")
    print()

    # 2) Send to backend `/inference/load_model` and wait for it to be ready
    with AIClient(
        base_url=args.backend_url,
        timeout=args.timeout,
        log_requests=True,
    ) as client:
        try:
            prepare_and_load_model(client, infer_config, infer_config.model_name)
        except Exception as e:
            print(f"\n[ERROR] Model loading failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" Model Loaded")
    print("=" * 60)
    print("Next: chat via the OpenAI-compatible endpoint.")
    print("  - Text:  uv run examples/chat/basic.py")
    print("  - Image: uv run examples/chat/image.py")


if __name__ == "__main__":
    main()
