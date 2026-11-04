# Tests

## Layout & tiers

| Tier | Path | Runs where | External needs |
|------|------|-----------|----------------|
| **unit** | `tests/unit/` | anywhere; CI on every PR | none |
| **smoke** | `tests/smoke/` | GPU machine; CI `gpu` job (manual) | CUDA + small model (auto-downloaded); the training modules also need DeepSpeed and large local checkpoints (never downloaded) |
| **e2e** | `tests/e2e/` | GPU machine; CI `gpu` job (manual) | CUDA + an engine (llama-server binary, vLLM, hermes CLI) + a model, small ones auto-downloaded |
| **xpu** | `tests/xpu/` | Intel GPU machine | Intel XPU + torch `+xpu` build; llama layer needs a Vulkan device |
| **stress** | `tests/stress_tests/` | manual only | running backend |

smoke/e2e/xpu are pytest modules that **auto-skip** (with an explicit reason) when
their resources are missing, so `pytest tests` is always safe to run. On a machine
that does have them, it also starts the long training runs (`-m effectiveness` trains
for ~14–53 min per model); deselect with `-m "not effectiveness"`.

## Tags

Every test module carries one of three markers (registered in `pytest.ini`).
`unit/test_platform_support.py` and `unit/test_linux_only_gating.py` carry both
`inference` and `finetune`, because they cover the gate for both Linux-only features:

- `inference` — engines, async HTTP client, dispatcher, OpenAI endpoint, concurrency
- `finetune` — training pipeline, log/SSE streaming, datasets, strategies
- `models` — model management: download, conversion, registry/status, memory estimation

```bash
pytest tests -m inference        # only inference tests
pytest tests -m finetune         # only fine-tune tests
pytest tests -m models           # only model-management tests
```

`finetune` is applied by path, in `tests/conftest.py`, to everything under
`tests/unit/training/` plus the four training modules in `tests/smoke/`. It used
to be hand-written per module and had drifted onto 11 of the 23 training files,
which left `-m "not finetune"` quietly still selecting the other 12.

## Platform gating

Fine-tuning and the vLLM engine are Linux-only — both are refused with a 501 off
Linux, and `deepspeed` / `vllm` are declared `sys_platform == 'linux'` in
`pyproject.toml`, so they are not even installed on Windows. Their tests carry a
fourth marker and are skipped automatically when `platform.system() != "Linux"`:

- `linux_only` — the whole `finetune` set, plus the five vLLM engine modules
  (`test_vllm_server_args`, `test_vllm_error_classifier`, `test_vllm_nvidia_repair`,
  `test_vllm_tool_parser_probe`, `smoke/test_vllm_server_smoke`)

```bash
pytest tests -m "not linux_only"   # what a Windows host actually runs
```

Two things stay ungated on purpose. `unit/test_platform_support.py` fakes the OS
with monkeypatch and asserts those 501s, so it is precisely the file that must
keep running on Windows. And the vLLM config/payload assertions inside the shared
inference modules (`test_api_compatibility.py`, `test_async_server_client.py`, …)
build dicts and CLI args without importing vllm, over code shared with the
engines Windows does run. See the comment block in `tests/conftest.py`.

## Running

```bash
PY=.venv/bin/python

$PY -m pytest tests/unit -q                              # CI-safe unit suite
$PY -m pytest tests/smoke -q                             # real-GPU smoke (auto-download model)
$PY -m pytest tests/e2e -q                               # real engines: llama-server, vLLM, hermes
$PY -m pytest tests/xpu -q -s                            # Intel GPU, layer by layer
bash tests/run_tests.sh                                  # unit, smoke, e2e + local result reports
bash tests/run_tests.sh unit                             # a single tier via the runner
```

## CI (`.github/workflows/test.yml`)

Both jobs run on a self-hosted runner.

| Job | Trigger | What it runs |
|-----|---------|--------------|
| `unit` | every PR into `dev` / `main` | `ruff check`, `ruff format --check`, `tests/unit` with `CUDA_VISIBLE_DEVICES=""` |
| `gpu` | `workflow_dispatch` only; `suite: quick` or `full` | `tests/unit` with a GPU, the vLLM server smoke, single-GPU inference smoke/e2e; with `full` also the single-GPU training smoke and `multigpu` |

The `gpu` job runs its steps in this order, each as its own pytest process:

```bash
CUDA_VISIBLE_DEVICES=0   pytest tests/unit                                   # bf16, model-class, llama-fit-params cases
CUDA_VISIBLE_DEVICES=0   pytest tests/smoke/test_vllm_server_smoke.py       # against a `vllm serve` the job starts
CUDA_VISIBLE_DEVICES=0   pytest tests/smoke tests/e2e -m "not multigpu and not effectiveness and not offload"
CUDA_VISIBLE_DEVICES=0   pytest tests/smoke -m "(effectiveness or offload) and not multigpu"   # full
CUDA_VISIBLE_DEVICES=0,1 pytest tests/smoke tests/e2e -m multigpu                            # full
```

Why it is split this way:

- **One GPU per step.** Single-GPU steps must use GPU0. The single-GPU training worker sets
  `CUDA_VISIBLE_DEVICES="0"` itself, so pinning a step to GPU1 still trains on GPU0. The
  multi-GPU path drops the variable for `deepspeed --num_gpus N`.
- **Training apart from inference.** The in-process ZeRO-3 training tests leave transformers'
  global DeepSpeed config set. A transformers worker forked after them then refuses to load with
  a `device_map`. Run in one process, the transformers smoke errors; run alone, it passes.
- **Its own JIT cache.** `TORCH_EXTENSIONS_DIR` points to a CI-only directory, so the DeepSpeed
  ops it builds (`cpu_adam`, `async_io`) do not overwrite another venv's builds.

The job refuses to start while a GPU it needs holds more than 2 GB (GPU0 for `quick`, both GPUs
for `full`). It also needs things that are not in git:

| Repository variable | Default use |
|---------------------|-------------|
| `TESTS_MODEL_CACHE` | HF_HOME-style cache with the large checkpoints |
| `TESTS_IMAGE_DATASET` | a local image fine-tuning set, linked in as `dataset/mixed_finetune_demo` |
| `LLAMA_CPP_DIR` | a built llama.cpp, linked in as `service/utils/llama.cpp` (for `llama-fit-params`) |
| `LLAMA_SERVER_BINARY` | llama-server for the e2e suite |
| `VLLM_SMOKE_MODEL` | model for the vLLM server smoke (default `Qwen/Qwen2.5-0.5B-Instruct`) |

The hermes graded test is pointed at the gpt-oss MXFP4 GGUF via `HERMES_E2E_GRADED_GGUF`.
Without that, it takes the first `*.gguf` in `ggml-org/gpt-oss-20b-GGUF`, which is an eagle3
draft model.

**Read the skip list, not just the pass count.** A missing resource is a skip, so a run can be
green while having tested little. With huggingface-hub ≥ 1.31, `local_snapshot()` also treats a
partially downloaded repo as missing (for example a GGUF repo holding one quantisation, or a
snapshot fetched with `ignore_patterns` unless the same patterns are passed back). JUnit XML is
uploaded as an artifact: `unit-junit` for 14 days, `gpu-junit-<suite>` for 30.

## Where output goes

**Nothing a run produces belongs next to the source.** Every artefact lands under
`tests/results/`, which is gitignored in full, so `git status` only ever shows real
code changes:

```
tests/results/
├── <timestamp>/          # tests/run_tests.sh: JUnit XML, full logs, summary.md
├── e2e_logs/             # backend + llama-server logs from the e2e suite
├── captures/             # scripts/capture_*.py: config_sent.json, events_all.jsonl,
│                         #   sse_sample.txt, meta.json, driver.log per run
├── benchmarks/           # tests/e2e/benchmark_*.py result JSON
└── logs/
    └── stress_tests/     # tests/stress_tests/*.py run logs
```

Also gitignored, for the same reason: the model cache `tests/.cache/` and the demo
datasets under `dataset/*_finetune_demo/` that
`scripts/generate_*_finetune_demo.py` regenerate on demand. **Results and models
stay local, never committed.**

If a script or test needs a new output location, put it under `tests/results/` and
anchor it at the repo root rather than the cwd:

```python
RESULTS = Path(__file__).resolve().parent.parent / "tests" / "results" / "captures"
```

### Test models (auto-provisioned)

Real-run tests download small models by themselves into `tests/.cache/hf/`
(via `tests/model_fixtures.py`) — no dependence on machine-specific paths or a
pre-populated global HF cache. First run fetches ~531MB (GGUF) + ~1GB
(transformers snapshot) + ~500MB (SmolVLM, for the multimodal tests); afterwards
runs are fully offline. Checkpoints larger than that (the offload, multi-GPU and
effectiveness training runs, the vLLM KV-offload e2e) are never downloaded: they
are used from a local hub cache if present, and skipped otherwise.

| Env var | Effect |
|---------|--------|
| `LLAMA_GGUF` | absolute path to a GGUF file (skips download) |
| `SMOKE_MODEL` | HF repo id for the transformers smoke model |
| `VLM_MODEL` | HF repo id for the multimodal test model (default SmolVLM-256M-Instruct) |
| `LLAMA_SERVER_BINARY` | llama binary (default: `service/utils/llama-bin/llama-server[.exe]`, what setup_env installs) |
| `TESTS_MODEL_CACHE` | alternative cache dir (default `tests/.cache/hf`) |
| `TESTS_NO_DOWNLOAD=1` | never hit the network; cache-hit or skip |

Run smoke and e2e **sequentially** on a single-GPU machine (each suite unloads
its model on teardown). Nothing targets ports 8000/5001 — a developer's live
backend is never touched: the e2e suite spawns its own backend on a free port.

## Test inventory

Counts are collected items (`pytest --co`), so each parametrized case counts once.
**Gate** marks modules skipped off Linux. "some cases" in **Needs** means the module
runs without it and only those cases skip.
Each unit section is exactly what its `-m` selects, so the two files tagged both
`inference` and `finetune` appear in both sections.

### unit — inference (`-m inference`) — 45 files, 976 tests

| File | Tests | Gate | Covers | Needs |
|------|------:|------|--------|-------|
| `unit/test_env_example_matches_settings.py` | 3 |  | `.env.example` documents the defaults `settings.py` applies | none |
| `unit/test_install_strata.py` | 8 | linux_only | `scripts/linux/install_strata.sh`: package check, setup args, `.env`, re-run behaviour | bash + sha256sum |
| `unit/test_install_strata_windows.py` | 8 |  | `scripts/windows/install_strata.ps1`: the same for the zip (skipped off Windows) | Windows PowerShell |
| `unit/test_linux_only_gating.py` | 16 |  | the suite's own Linux-only gate still points at real files and markers | none |
| `unit/test_llama_paths_agree.py` | 3 |  | llama install location and pinned version agree across `settings.py` and both setup scripts | none |
| `unit/test_platform_support.py` | 23 |  | 501 refusals for training/vLLM off Linux (OS faked), WSL counts as Linux, `/health` reports support | none |
| `unit/test_rag_chroma_optional.py` | 3 |  | RAG without the opt-in `chroma` extra installed | none |
| `unit/test_remote_access.py` | 121 |  | who may reach the service: host/origin checks, CORS and preflight, API-key lockout, llama/vLLM args that cannot move off loopback | none |
| `unit/inference/test_api_compatibility.py` | 53 |  | health/load/status/unload endpoint contracts, `/v1/models` and `/v1/models/{id}` (context window, capabilities, model details, served id), OpenAPI examples load as configs | none |
| `unit/inference/test_async_error_reporting.py` | 11 |  | server-engine failures reach `/inference/error_details` classified (real OOM vs a word containing "oom", dropped socket recoverable) | none |
| `unit/inference/test_async_server_client.py` | 95 |  | AsyncServerClient: payload mapping/clamping, multimodal reorder, tool_call reassembly, stream stop/error, per-engine endpoint & repetition-key resolution | none |
| `unit/inference/test_dev_parity_regressions.py` | 13 |  | behaviour the old queue path had and the direct async HTTP path must keep: reasoning on its own key, repeat-penalty names, prompt cache, tok/s split, client lifecycle | none |
| `unit/inference/test_gpt_parser.py` | 10 |  | gpt-oss harmony output parsing | none |
| `unit/inference/test_image_generation_support.py` | 4 |  | image loading & prompt tokenization | none |
| `unit/inference/test_inference_config_engine_split.py` | 13 |  | per-engine grouping of InferenceConfig, `extra_args` → typed field sync | none |
| `unit/inference/test_inference_dispatcher.py` | 18 |  | data_queue dispatcher demux: per-request FIFO routing, no cross-talk (concurrent consumers), drops, lifecycle, generate/generate_stream error & timeout paths | none |
| `unit/inference/test_llama_server_engine_startup.py` | 19 |  | managed-server CLI args (mmproj), HF_HOME-relative model path resolution | none |
| `unit/inference/test_llama_server_runtime_error_handling.py` | 8 |  | llama-server crash/OOM payloads, stream error propagation through the dispatcher, managed-server shutdown | none |
| `unit/inference/test_memory_estimator_kv_cache.py` | 10 |  | KV cache sizing | none |
| `unit/inference/test_memory_estimator_offline.py` | 10 |  | estimator behaviour when the Hub cannot be reached | none |
| `unit/inference/test_memory_estimator_params.py` | 8 |  | how the estimator arrives at a parameter count, checked on real text and multimodal configs | Qwen2.5-0.5B (auto-download) + SmolVLM-256M (auto-download) |
| `unit/inference/test_memory_estimator_remote_code.py` | 5 |  | whose `trust_remote_code` the estimator is willing to run | none |
| `unit/inference/test_memory_estimator_runtime.py` | 19 |  | activation (logits-dominated) and device/host overhead terms, against measured peaks | none |
| `unit/inference/test_model_class_resolution.py` | 6 |  | load class picked from the checkpoint config, not the repo/directory name (SmolVLM servable, renamed fine-tunes) | some cases: checkpoint in a local hub cache |
| `unit/inference/test_model_family.py` | 19 |  | architecture-based model-family detection (HF configs and GGUF metadata) | some cases: `gguf` package |
| `unit/inference/test_model_manager_async.py` | 14 |  | ModelManager async path: uses_async_http gate, client cache/invalidation, stop-event registry, dual-path stop_generation, agenerate(_stream) delegation | none |
| `unit/inference/test_openai_compatible_chat_endpoint.py` | 65 |  | OpenAI protocol surface: non-stream/stream, sessions, tool calls (stream + non-stream), usage chunk, finish_reason, aliases, image/JSON attachments, reasoning_effort / enable_thinking mapping per engine | none |
| `unit/inference/test_openai_endpoint_backend_routing.py` | 8 |  | /v1/chat/completions routing (async HTTP vs worker), 500/507 mapping, SSE error events, client-cancel stops backend, 6-way concurrent overlap | none |
| `unit/inference/test_peft_offload_skip_keys.py` | 7 |  | PEFT re-dispatch keeps accelerate's device-placement skip list | `accelerate` |
| `unit/inference/test_prompt_parity_with_training.py` | 11 |  | serving adds no prompt structure training never produced: empty default system prompt, `enable_thinking` left unset, media-before-text order | none |
| `unit/inference/test_stop_generation_concurrency.py` | 5 |  | targeted vs global stop on the async server path, single and concurrent streams | none |
| `unit/inference/test_strata_engine.py` | 26 |  | Strata engine: setup checks, run config, the model list's strata_supported mark, first-load install and setup, lifecycle, sleep/wake | none |
| `unit/inference/test_transformers_engine_device_allocation.py` | 5 |  | device_map / max_memory allocation | none |
| `unit/inference/test_vllm_admin_guard.py` | 38 |  | guard in front of vLLM's admin endpoints: local sleep/wake allowed, the rest of the admin API absent, web pages refused | none |
| `unit/inference/test_vllm_error_classifier.py` | 53 | linux_only | vLLM startup stderr classification | none |
| `unit/inference/test_vllm_kv_disk_tier.py` | 28 |  | size cap for vLLM's fs KV tier: usage accounting, LRU eviction, janitor schedule, `/inference/kv_cache/disk/cleanup` | none |
| `unit/inference/test_vllm_nvidia_repair.py` | 6 | linux_only | repairing the NVIDIA runtime without breaking the environment | none |
| `unit/inference/test_vllm_offload_config.py` | 54 |  | expert-weight offload and DRAM/NVMe KV offload: config fields, CLI flags, preflight | some cases: `vllm` |
| `unit/inference/test_vllm_server_args.py` | 85 | linux_only | vLLM serve CLI argument construction | some cases: `vllm` |
| `unit/inference/test_vllm_sleep_mode.py` | 15 | linux_only | vLLM sleep mode: flag, manager sleep/wake state, auto-wake, API | none |
| `unit/inference/test_vllm_tool_parser_probe.py` | 4 | linux_only | payload shape of `vllm_tool_parser_probe.py` | none |
| `unit/utils/test_integrated_gpu_detection.py` | 29 |  | integrated vs discrete GPU detection (synthetic DRM tree, DXGI) | none |
| `unit/utils/test_llama_backend_probe.py` | 9 |  | device/backend parsing in `scripts/llama_backend.py` | none |
| `unit/utils/test_memory_modules.py` | 6 |  | DRAM module listing (udev on Linux, Windows), without root | none |
| `unit/utils/test_system_resources_endpoint.py` | 2 |  | `/system/resources` path guard does not block the event loop | none |

### unit — fine-tune (`-m finetune`) — 27 files, 336 tests

| File | Tests | Gate | Covers | Needs |
|------|------:|------|--------|-------|
| `unit/test_linux_only_gating.py` | 16 |  | the suite's own Linux-only gate still points at real files and markers | none |
| `unit/test_oneccl_library_path.py` | 2 |  | `<venv>/lib` on LD_LIBRARY_PATH when it holds oneCCL's plugin (XPU training / vLLM) | non-Windows |
| `unit/test_platform_support.py` | 23 |  | 501 refusals for training/vLLM off Linux (OS faked), WSL counts as Linux, `/health` reports support | none |
| `unit/training/test_causal_lm_labels.py` | 4 | linux_only | prompt/completion label masking | none |
| `unit/training/test_dataset_loader.py` | 8 | linux_only | dataset loading/validation | none |
| `unit/training/test_dataset_loader_images.py` | 11 | linux_only | image datasets: path resolution, lazy decode, multimodal format validation | none |
| `unit/training/test_deepspeed_aio_patch.py` | 2 | linux_only | the installed DeepSpeed closes the descriptors its async NVMe path opens (`patch_deepspeed_aio.py` applied) | `deepspeed` |
| `unit/training/test_deepspeed_selection.py` | 17 | linux_only | asking for DeepSpeed and silently not getting it is impossible (incl. refusal on Intel XPU) | none |
| `unit/training/test_entrypoint_parity.py` | 19 | linux_only | single- and multi-GPU entry points run the same setup sequence | none |
| `unit/training/test_eval_split.py` | 10 | linux_only | held-out test split: dataset splitting, metric tagging, log routing | some cases: bf16 GPU |
| `unit/training/test_full_finetune_scope.py` | 9 | linux_only | full fine-tune keeps the output head trainable; freeze scope on multimodal models | SmolVLM-256M (auto-download) |
| `unit/training/test_generation_check.py` | 16 | linux_only | post-training generation check (verdict by answers, not loss) | none |
| `unit/training/test_gradient_checkpointing_grads.py` | 4 | linux_only | LoRA adapters inside checkpointed blocks receive gradients | Qwen2.5-0.5B (auto-download) |
| `unit/training/test_label_supervision_guard.py` | 9 | linux_only | truncation that leaves no supervised tokens is refused | none |
| `unit/training/test_memory_aggregation.py` | 11 | linux_only | runs record the memory they actually used | none |
| `unit/training/test_multigpu_lifecycle.py` | 10 | linux_only | multi-GPU failure/cancellation keeps the error and frees GPUs (launcher mocked) | none |
| `unit/training/test_multigpu_status_reporting.py` | 12 | linux_only | multi-GPU `__STATUS__` reporting paths (launcher mocked) | none |
| `unit/training/test_multimodal_gradient_flow.py` | 4 | linux_only | on a real VLM: text-only scope trains only the language tower, bridge scope reaches the projector, image content reaches the loss | SmolVLM-256M (auto-download) |
| `unit/training/test_multimodal_model_loading.py` | 63 | linux_only | multimodal checkpoints load via AutoModelForImageTextToText, LoRA kept off the vision tower, full FT freezes vision | SmolVLM-256M (auto-download); some cases: bf16 GPU + Qwen2.5-0.5B (auto-download) |
| `unit/training/test_qlora_guards.py` | 8 | linux_only | QLoRA combinations that cannot work are refused | none |
| `unit/training/test_strategy_factory.py` | 5 | linux_only | full/lora/qlora strategy selection | none |
| `unit/training/test_training_api.py` | 7 | linux_only | /training/* endpoints (incl. num_gpu alias) | none |
| `unit/training/test_training_log_sse.py` | 35 | linux_only | job log writer/reader, SSE stream (backlog, tail, keepalive, resume, disconnect) | none |
| `unit/training/test_training_status_errors.py` | 4 | linux_only | a failed run stays visible through /training/status and error_details | none |
| `unit/training/test_upgrade_regression_diff.py` | 12 | linux_only | diff logic of `scripts/upgrade_regression.py` | none |
| `unit/training/test_zero3_buffer_restore.py` | 9 | linux_only | ZeRO-3 loader restores buffers of parameter-less modules (gemma-4 `layer_scalar`) | none |
| `unit/training/test_zero3_utils.py` | 6 | linux_only | `drain_zero3_inflight` before a manual GatheredParameters save | none |

### unit — models (`-m models`) — 6 files, 229 tests

| File | Tests | Gate | Covers | Needs |
|------|------:|------|--------|-------|
| `unit/conversion/test_conversion_manager.py` | 16 |  | GGUF/LoRA conversion: work dir, streaming merge, multimodal fallback, mmproj source (base vs merged), tokenizer copy, UTF-8 child output | none |
| `unit/download/test_download_manager.py` | 10 |  | download worker (split GGUF + mmproj, failures), size reporting, cancellation and cleanup | none |
| `unit/inference/test_gguf_estimator.py` | 178 |  | GGUF / llama.cpp memory estimator; some cases checked against `llama-fit-params` | some cases: `gguf`, `llama-fit-params` on a GPU backend, test GGUF |
| `unit/inference/test_memory_estimator.py` | 11 |  | model memory estimation (model size extraction) | none |
| `unit/inference/test_runtime_smoke.py` | 4 |  | ModelManager singleton/status/config | none |
| `unit/models/test_model_deletion.py` | 10 |  | model deletion state: `deleting` visible to all, partial failure, protected paths, restart mid-delete | none |

### smoke — real model or real training run, auto-skip

| File | Tests | Tag | Covers | Needs |
|------|------:|-----|--------|-------|
| `smoke/test_transformers_worker_smoke.py` | 3 | inference | real worker subprocess: load/generate/stream/stop + concurrent streams without cross-talk | CUDA + Qwen2.5-0.5B auto-download (~1GB) |
| `smoke/test_vllm_server_smoke.py` | 1 | inference, linux_only | AsyncServerClient against a live vLLM server | live server at `VLLM_BASE` |
| `smoke/test_offload_training_matrix.py` | 5 | finetune, offload | real DeepSpeed ZeRO-3 offload runs, a few steps each: parameters really partitioned (`resident == 0`), loss finite and non-zero, GPU peak under a ceiling. Cases: `smolvlm-full-image-cpu`, `qwen3-4b-lora-text-cpu`, `gemma4-e4b-lora-image-nvme`, `qwen35-35b-lora-text-cpu`, `qwen35-35b-lora-image-cpu` | CUDA + `deepspeed`; large checkpoints in a local hub cache (15–67 GB, never downloaded) |
| `smoke/test_multigpu_training_matrix.py` | 3 | finetune, multigpu | real `deepspeed --num_gpus 2` through the production launcher: two ranks requested, every rank exits 0, rank 0 saves the adapter, the `__STATUS__` channel delivers. Cases: `smolvlm-lora-text-2gpu`, `smolvlm-lora-image-2gpu`, `qwen35-35b-lora-image-2gpu` | ≥2 NVIDIA GPUs + `deepspeed`; large checkpoints local only |
| `smoke/test_image_training_effectiveness.py` | 6 | finetune, effectiveness | single-GPU image fine-tune under ZeRO-3 offload really teaches held-out identities: the base model answers 0/3 (control, same engine with the adapter off), the tuned one answers, the loss moved. Cases: `gemma4-e4b` (~14 min), `qwen35-35b` (~53 min) | CUDA + `deepspeed`; gemma-4-E4B-it / Qwen3.5-35B-A3B in a local hub cache; `dataset/mixed_finetune_demo/` |
| `smoke/test_multigpu_image_effectiveness.py` | 4 | finetune, effectiveness, multigpu | the same check on the multi-GPU path, with the generation check running inside the ranks. Cases: `gemma4-e4b-2gpu`, `qwen35-35b-2gpu` | ≥2 NVIDIA GPUs, plus the above |

All four training modules are `linux_only` through the `finetune` path rule. Select
them by marker: `-m offload`, `-m multigpu`, `-m effectiveness`.

### e2e — isolated backend + real engine, auto-skip

| File | Tests | Tag | Covers | Needs |
|------|------:|-----|--------|-------|
| `e2e/test_llama_server_e2e.py` | 5 | inference | isolated backend subprocess + managed llama-server: status, 4-way concurrent non-stream, 4-way concurrent stream, per-request mid-stream stop (other stream survives), unload kills server | CUDA + llama-server binary + auto-download GGUF (~531MB) |
| `e2e/test_hermes_reasoning_e2e.py` | 4 | inference | hermes CLI → recording proxy → backend → engine, once per `--reasoning` level, on `vllm` and `llama_server`: `none`/`minimal` give no reasoning, every other level gives `reasoning_content`; no `<think>` or raw tool XML leaks into the answer; a graded model (gpt-oss-20b) follows the level | `hermes` (`HERMES_BIN`) + CUDA; MiniCPM5-1B (HF + GGUF) and gpt-oss-20b in a local hub cache; the `vllm` cases also need Linux + vllm. Overrides: `HERMES_E2E_MODEL`, `HERMES_E2E_GGUF_REPO`, `HERMES_E2E_GRADED_MODEL`, `HERMES_E2E_GRADED_GGUF` |
| `e2e/test_vllm_moe_nvme_kv_offload_e2e.py` | 1 | inference, kv_offload, linux_only | a MoE larger than the GPU loads with only its experts offloaded; prefix KV is written to the NVMe tier and, after unload + reload, served back from disk (fs-tier hits, real block-device reads, faster than recompute) | Linux + CUDA + vllm, ≥13 GiB free GPU memory, ~44 GiB available host RAM, Qwen3.5-35B-A3B-GPTQ-8bit in a local hub cache (41 GB, never downloaded), KV dir on a real block device with ≥20 GiB free. Overrides: `KV_OFFLOAD_E2E_MODEL`, `KV_OFFLOAD_E2E_DIR` |

### xpu — Intel GPU, auto-skip

Layered: if torch cannot use the device, the transformers and llama layers above it
cannot either, so read failures bottom-up.

| File | Tests | Tag | Covers | Needs |
|------|------:|-----|--------|-------|
| `xpu/test_torch_xpu.py` | 7 | inference | layer 1: torch XPU sees the device without oneAPI `setvars`, matmul / half precision match CPU, host↔device round trip, memory and cleanup API | Intel XPU + torch `+xpu` build |
| `xpu/test_transformers_xpu.py` | 3 | inference | layer 2: a real HF model's weights, forward pass and generate run on the XPU | Intel XPU + Qwen2.5-0.5B auto-download |
| `xpu/test_llama_xpu.py` | 2 | inference | layer 3: llama.cpp offloads to the Intel GPU via Vulkan and completes | llama binary + an Intel Vulkan device + test GGUF auto-download |

### manual

Not pytest modules; run them by hand against a running backend.

| File | Covers | Needs |
|------|--------|-------|
| `e2e/benchmark_llama_server_prefill.py` | llama-server load time, TTFT and TPS: cold / warm / changed-tail, across unload + reload | running backend; `GPT_OSS_GGUF_PATH` and/or `QWEN_GGUF_PATH` |
| `e2e/benchmark_offload_vram.py` | measured VRAM reduction and throughput of offload against the full-GPU requirement; writes `tests/results/benchmarks/offload_vram.json` | running backend + GPU; `QWEN3_14B_MODEL_DIR` |
| `stress_tests/stress_test_inference.py` | repeated load/chat/unload cycles | running backend |
| `stress_tests/stress_test_finetune.py` | repeated fine-tune cycles | running backend |

## Conventions

- Files are named `test_*.py` and live in the tier/domain tree above.
- Every module sets `pytestmark = pytest.mark.<tag>` (one of inference/finetune/models).
  Under `tests/unit/training/` the `finetune` tag comes from the path instead, so a
  new file there is tagged — and platform-gated — without anyone remembering to.
- Real-run modules must auto-skip with an explicit reason instead of failing
  when GPU/binary/model/network are unavailable.
- Streaming endpoints: `TestClient`/`ASGITransport` buffer SSE — drive
  `resp.body_iterator` directly for cancellation/live-injection cases (see
  `test_openai_endpoint_backend_routing.py` and `test_training_log_sse.py`).
