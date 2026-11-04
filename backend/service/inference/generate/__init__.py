"""
Generate module - Contains text generation and parsing components.

Import from the submodules (`generator_core`, `generator_worker`, `gpt_parser`,
`llama_server_runner`). This file deliberately imports nothing: llama_server_engine needs
only llama_server_runner, and re-exporting generator_core here made every llama.cpp,
vLLM and Strata path import torch and transformers.
"""
