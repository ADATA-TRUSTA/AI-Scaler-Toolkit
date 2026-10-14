"""
Headless Python client for the AI Scaler (Trusta AST) backend.

Exposes the HTTP client plus the request/response models used by the
``examples/`` scripts: model load/unload, OpenAI-compatible chat, and
fine-tune training.
"""

from ai_scaler_console.client import AIClient
from ai_scaler_console.exceptions import (
    AIClientError,
    ModelLoadFailedError,
    TrainingFailedError,
)

__all__ = [
    "AIClient",
    "AIClientError",
    "ModelLoadFailedError",
    "TrainingFailedError",
]
