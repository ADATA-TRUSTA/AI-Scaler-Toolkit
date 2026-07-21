"""
Service package for LLM inference and training
"""

# NOTE:
# Do NOT import `service.app` at package import time.
# Otherwise `python -m service.app` may warn:
#   "service.app found in sys.modules ... prior to execution"
# because importing package `service` would import `service.app` first.

__all__ = []


def __getattr__(name: str):
    """Lazy exports for optional convenience imports."""
    if name == "app":
        from .app import app as _app

        return _app
    if name == "model_manager":
        from .model_manager import model_manager as _model_manager

        return _model_manager
    if name == "training_manager":
        from .training_manager import training_manager as _training_manager

        return _training_manager

    # Re-export config models lazily
    if name in {
        "InferenceConfig",
        "ChatRequest",
        "TrainingConfig",
        "ModelStatus",
        "TrainingStatus",
        "QuantizationType",
        "TrainingMethod",
    }:
        from . import config_models as _cm

        return getattr(_cm, name)

    raise AttributeError(name)
