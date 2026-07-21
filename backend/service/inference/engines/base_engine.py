from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from ...config_models import InferenceConfig

class BaseEngine(ABC):
    def __init__(self, status_queue: Queue, data_queue: Queue, stop_event: EventClass, stop_generation_flag: EventClass):
        self.status_queue = status_queue
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.stop_generation_flag = stop_generation_flag
        self.config: Optional[InferenceConfig] = None

    @abstractmethod
    def load_model(self, config: InferenceConfig):
        """Load the model based on configuration."""
        pass

    @abstractmethod
    def generate(self, request: Dict[str, Any]):
        """Handle generation request."""
        pass

    @abstractmethod
    def generate_stream(self, request: Dict[str, Any]):
        """Handle stream generation request."""
        pass

    @abstractmethod
    def unload(self):
        """Unload model and tokenizer."""
        pass

    def apply_chat_template(self, request: Dict[str, Any]):
        """Apply chat template."""
        # Default implementation or override in subclasses
        pass
    
    def cleanup_generation_memory(self):
        """Clean up generation memory."""
        pass
