import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def load_hf_token() -> Optional[str]:
    """
    Load Hugging Face token from environment variables.

    Priority:
    1. env HF_HUB_TOKEN
    2. env HF_TOKEN
    """
    for env_name in ("HF_HUB_TOKEN", "HF_TOKEN"):
        token = os.getenv(env_name)
        if token:
            logger.info(f"Loaded HF token from env {env_name}")
            return token

    logger.warning("No HF token found; private models may fail to load")
    return None
