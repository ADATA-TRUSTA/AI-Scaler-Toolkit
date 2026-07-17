"""
AST Console AI - Load Model Example (no-UI workflow)

Use Trusta-AST-Frontend's `inference/load_model` endpoint and load a model
from a config file under `infer_model_configs/*.json`.
After loading is complete, use `openai-compatible-example.py` /
`openai-compatible-image-example.py` for chat.

Usage:
    # transformers (full precision / quantization)
    uv run load_model_example.py --config infer_model_configs/inference-config-transformers.json

    # transformers + GPU/CPU offload
    uv run load_model_example.py --config infer_model_configs/inference-config-transformers-offload.json

    # llama.cpp (GGUF)
    uv run load_model_example.py --config infer_model_configs/inference-config-llama.json

    # vLLM
    uv run load_model_example.py --config infer_model_configs/inference-config-vllm.json

    # Override backend URL
    uv run load_model_example.py --config infer_model_configs/inference-config-vllm.json \
        --backend-url http://your-backend-host
"""

import argparse
import json
import sys

from ai_client import AIClient
from helpers.config_loader import load_and_create_config
from helpers.default_backend import load_default_backend_url
from helpers.model_loader import prepare_and_load_model


def main() -> None:
    try:
        default_backend_url = load_default_backend_url()
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Failed to load default backend URL: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Load a model on the AST backend using infer_model_configs JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a model config JSON under infer_model_configs/ "
        "(e.g. infer_model_configs/inference-config-transformers.json)",
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
    client = AIClient(
        base_url=args.backend_url,
        timeout=args.timeout,
        log_requests=True,
    )

    try:
        prepare_and_load_model(client, infer_config, infer_config.model_name)
    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" Model Loaded")
    print("=" * 60)
    print("Next: chat via the OpenAI-compatible endpoint.")
    print("  - Text:  uv run openai-compatible-example.py")
    print("  - Image: uv run openai-compatible-image-example.py")


if __name__ == "__main__":
    main()
