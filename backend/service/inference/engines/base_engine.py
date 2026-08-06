"""Abstract base class shared by all inference engine implementations."""

from abc import ABC, abstractmethod
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from typing import Any

from ...config_models import InferenceConfig


class BaseEngine(ABC):
    """Common lifecycle contract for worker-side inference engines."""

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ) -> None:
        self.status_queue = status_queue
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.stop_generation_flag = stop_generation_flag
        self.config: InferenceConfig | None = None

    @abstractmethod
    def load_model(self, config: InferenceConfig) -> None:
        """Load the model based on configuration."""
        pass

    def generate(self, request: dict[str, Any]) -> None:
        """
        Handle a non-stream generation request.

        Default: unsupported. In-process engines (e.g. transformers) override
        this. Server engines (llama-server / vLLM) are served directly over
        async HTTP from the main process and never generate via the worker, so
        they intentionally do not implement it — reaching here is a misroute.
        """
        self.data_queue.put(
            {
                "type": "error",
                "request_id": request.get("request_id"),
                "error": "This engine does not support worker-side generate() "
                "(server engines are served via direct async HTTP).",
            }
        )

    def generate_stream(self, request: dict[str, Any]) -> None:
        """Handle a stream generation request. Default: unsupported (see generate)."""
        self.data_queue.put(
            {
                "type": "error",
                "request_id": request.get("request_id"),
                "error": "This engine does not support worker-side generate_stream() "
                "(server engines are served via direct async HTTP).",
            }
        )

    @abstractmethod
    def unload(self) -> None:
        """Unload model and tokenizer."""
        pass

    def apply_chat_template(self, request: dict[str, Any]) -> None:  # noqa: B027 - optional hook, no-op by default, subclasses may override
        """Apply chat template."""
        # Default implementation or override in subclasses
        pass

    def cleanup_generation_memory(self) -> None:  # noqa: B027 - optional hook, no-op by default, subclasses may override
        """Clean up generation memory."""
        pass
