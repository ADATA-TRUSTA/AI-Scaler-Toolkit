"""
Training core: the building blocks the training worker assembles a run from.

The exported names are resolved on first access (PEP 562). The API process imports the
light submodules (job_logger, training_events) to report on jobs; importing every export
here made that pull in torch, transformers, peft and datasets at service startup.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataset_loader import load_training_dataset, split_train_eval
    from .job_logger import (
        JobLogWriter,
        build_resource_snapshot,
        events_file,
        is_terminal_event,
        job_log_dir,
        read_events,
        read_meta,
    )
    from .mem_probe import MemoryAggregator, log_mem, probe_snapshot, start_memory_sampler
    from .model_loader import ModelLoader, dataset_has_images, select_processing_class
    from .model_saver import save_training_results
    from .pipeline import TrainingComponents, build_training_components
    from .strategies import CausalLMStrategy, SFTStrategy, StrategyFactory, TrainingStrategy
    from .training_events import PHASE_LABELS, TERMINAL_PHASES, EventType, Phase, phase_to_status

__all__ = [
    "ModelLoader",
    "dataset_has_images",
    "select_processing_class",
    "build_training_components",
    "TrainingComponents",
    "load_training_dataset",
    "split_train_eval",
    "StrategyFactory",
    "TrainingStrategy",
    "SFTStrategy",
    "CausalLMStrategy",
    "save_training_results",
    "log_mem",
    "start_memory_sampler",
    "probe_snapshot",
    "MemoryAggregator",
    "Phase",
    "EventType",
    "PHASE_LABELS",
    "TERMINAL_PHASES",
    "phase_to_status",
    "JobLogWriter",
    "build_resource_snapshot",
    "read_events",
    "read_meta",
    "job_log_dir",
    "events_file",
    "is_terminal_event",
]


_EXPORT_MODULES = {
    "load_training_dataset": ".dataset_loader",
    "split_train_eval": ".dataset_loader",
    "JobLogWriter": ".job_logger",
    "build_resource_snapshot": ".job_logger",
    "events_file": ".job_logger",
    "is_terminal_event": ".job_logger",
    "job_log_dir": ".job_logger",
    "read_events": ".job_logger",
    "read_meta": ".job_logger",
    "MemoryAggregator": ".mem_probe",
    "log_mem": ".mem_probe",
    "probe_snapshot": ".mem_probe",
    "start_memory_sampler": ".mem_probe",
    "ModelLoader": ".model_loader",
    "dataset_has_images": ".model_loader",
    "select_processing_class": ".model_loader",
    "save_training_results": ".model_saver",
    "TrainingComponents": ".pipeline",
    "build_training_components": ".pipeline",
    "CausalLMStrategy": ".strategies",
    "SFTStrategy": ".strategies",
    "StrategyFactory": ".strategies",
    "TrainingStrategy": ".strategies",
    "PHASE_LABELS": ".training_events",
    "TERMINAL_PHASES": ".training_events",
    "EventType": ".training_events",
    "Phase": ".training_events",
    "phase_to_status": ".training_events",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - whatever the exported object is
    """Import an exported name from its submodule on first access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
