2026/04/22
Current isolated `vllm_server` environment is pinned by `pyproject.toml` and `uv.lock`.
The current pinned vLLM version is `0.20.1`.
If Gemma-4-E2B-it requires a future vLLM change, update `pyproject.toml` and regenerate `uv.lock` instead of manually upgrading the environment in place.
Gemma-4 needs `transformers >= 5.5.0`.
