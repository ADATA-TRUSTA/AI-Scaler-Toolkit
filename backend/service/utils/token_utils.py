"""Hugging Face token loading helpers."""

import logging
import os

logger = logging.getLogger(__name__)


def load_hf_token() -> str | None:
    """
    Load Hugging Face token.

    Priority:
    1. env HF_HUB_TOKEN
    2. env HF_TOKEN
    3. token stored by `hf auth login` (huggingface_hub cache / HF_TOKEN_PATH)
    """
    for env_name in ("HF_HUB_TOKEN", "HF_TOKEN"):
        token = os.getenv(env_name)
        if token:
            logger.info(f"Loaded HF token from env {env_name}")
            return token

    # Fall back to the token saved by `hf auth login`, so gated models work
    # without also exporting HF_TOKEN. get_token() reads env vars too, but we
    # check those first to preserve the documented precedence.
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            logger.info("Loaded HF token from huggingface_hub login cache")
            return token
    except Exception as e:
        logger.debug(f"huggingface_hub token lookup failed: {e}")

    logger.warning("No HF token found; private models may fail to load")
    return None
