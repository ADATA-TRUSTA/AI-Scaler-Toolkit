"""
Inference module - Contains inference-related components.

Import from the submodules (`service.inference.memory_estimator` and so on). This file
deliberately imports nothing: every `service.inference.*` import runs it, and the
re-exports it used to carry dragged torch, transformers and peft into the API process
at startup although only the transformers worker uses them.
"""
