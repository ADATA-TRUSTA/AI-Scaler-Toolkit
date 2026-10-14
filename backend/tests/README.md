# Tests

## Layout & tiers

| Tier | Path | Runs where | External needs |
|------|------|-----------|----------------|
| **unit** | `tests/unit/` | anywhere / CI | none |
| **smoke** | `tests/smoke/` | GPU machine | CUDA + small model (auto-downloaded) |
| **e2e** | `tests/e2e/` | GPU machine | CUDA + llama-server binary + small GGUF (auto-downloaded) |
| **stress** | `tests/stress_tests/` | manual only | running backend |
| **hardware** | `tests/DeepNVMe/` | manual only | NVMe device + DeepSpeed async_io |

smoke/e2e are pytest modules that **auto-skip** (with an explicit reason) when
their resources are missing, so `pytest tests` is always safe to run.

## Tags

Every test module carries one of three markers (registered in `pytest.ini`):

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
$PY -m pytest tests/e2e -q                               # real llama-server e2e + concurrency
bash tests/run_tests.sh                                  # all tiers + local result reports
bash tests/run_tests.sh unit                             # a single tier via the runner
```

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
stay local, never uploaded to GitLab.**

If a script or test needs a new output location, put it under `tests/results/` and
anchor it at the repo root rather than the cwd:

```python
RESULTS = Path(__file__).resolve().parent.parent / "tests" / "results" / "captures"
```

### Test models (auto-provisioned)

Real-run tests download small models by themselves into `tests/.cache/hf/`
(via `tests/model_fixtures.py`) — no dependence on machine-specific paths or a
pre-populated global HF cache. First run fetches ~531MB (GGUF) + ~1GB
(transformers snapshot); afterwards runs are fully offline.

| Env var | Effect |
|---------|--------|
| `LLAMA_GGUF` | absolute path to a GGUF file (skips download) |
| `SMOKE_MODEL` | HF repo id for the transformers smoke model |
| `LLAMA_SERVER_BINARY` | llama binary (default: `service/utils/llama-bin/llama-server[.exe]`, what setup_env installs) |
| `TESTS_MODEL_CACHE` | alternative cache dir (default `tests/.cache/hf`) |
| `TESTS_NO_DOWNLOAD=1` | never hit the network; cache-hit or skip |

Run smoke and e2e **sequentially** on a single-GPU machine (each suite unloads
its model on teardown). Nothing targets ports 8000/5001 — a developer's live
backend is never touched: the e2e suite spawns its own backend on a free port.

## Test inventory

### unit — inference (`-m inference`)

| File | Tests | Covers | Needs |
|------|------:|--------|-------|
| `unit/inference/test_async_server_client.py` | 17 | AsyncServerClient: payload mapping/clamping, multimodal reorder, tool_call reassembly, stream stop/error, per-engine endpoint & repetition-key resolution | none |
| `unit/inference/test_inference_dispatcher.py` | 13 | data_queue dispatcher demux: per-request FIFO routing, no cross-talk (concurrent consumers), drops, lifecycle, generate/generate_stream error & timeout paths | none |
| `unit/inference/test_model_manager_async.py` | 11 | ModelManager async path: uses_async_http gate, client cache/invalidation, stop-event registry, dual-path stop_generation, agenerate(_stream) delegation | none |
| `unit/inference/test_openai_endpoint_backend_routing.py` | 8 | /v1/chat/completions routing (async HTTP vs worker), 500/507 mapping, SSE error events, client-cancel stops backend, 6-way concurrent overlap | none |
| `unit/inference/test_openai_compatible_chat_endpoint.py` | 19 | OpenAI protocol surface: non-stream/stream, sessions, tool calls, aliases | none |
| `unit/inference/test_vllm_error_classifier.py` | 28 | vLLM startup stderr classification | none |
| `unit/inference/test_llama_server_runtime_error_handling.py` | 8 | llama-server crash/OOM payloads, stream error propagation through the dispatcher, managed-server shutdown | none |
| `unit/inference/test_llama_server_engine_startup.py` | 3 | managed-server CLI args (mmproj), HF_HOME-relative model path resolution | none |
| `unit/inference/test_vllm_server_args.py` | 2 | vLLM serve CLI args (incl. --max-num-batched-tokens) | none |
| `unit/inference/test_transformers_engine_device_allocation.py` | 5 | device_map / max_memory allocation | none |
| `unit/inference/test_image_generation_support.py` | 4 | image loading & prompt tokenization | none |
| `unit/inference/test_gpt_parser.py` | 10 | gpt-oss harmony output parsing | none |
| `unit/inference/test_api_compatibility.py` | 6 | health/load/status/unload endpoint contracts | none |

### unit — fine-tune (`-m finetune`)

| File | Tests | Covers | Needs |
|------|------:|--------|-------|
| `unit/training/test_training_log_sse.py` | 35 | job log writer/reader, SSE stream (backlog, tail, keepalive, resume, disconnect) | none |
| `unit/training/test_dataset_loader.py` | 8 | dataset loading/validation | none |
| `unit/training/test_training_api.py` | 7 | /training/* endpoints (incl. num_gpu alias) | none |
| `unit/training/test_strategy_factory.py` | 5 | full/lora/qlora strategy selection | none |
| `unit/training/test_causal_lm_labels.py` | 4 | prompt/completion label masking | none |

### unit — models (`-m models`)

| File | Tests | Covers | Needs |
|------|------:|--------|-------|
| `unit/inference/test_memory_estimator.py` | 10 | model memory estimation | none |
| `unit/conversion/test_conversion_manager.py` | 5 | GGUF/LoRA conversion manager | none |
| `unit/inference/test_runtime_smoke.py` | 4 | ModelManager singleton/status/config | none |
| `unit/download/test_download_manager.py` | 2 | model download manager | none |

### smoke — real model, auto-skip

| File | Tests | Tag | Covers | Needs |
|------|------:|-----|--------|-------|
| `smoke/test_transformers_worker_smoke.py` | 3 | inference | real worker subprocess: load/generate/stream/stop + concurrent streams without cross-talk | CUDA + auto-download (~1GB) |
| `smoke/test_vllm_server_smoke.py` | 1 | inference | AsyncServerClient against a live vLLM server | live server at `VLLM_BASE` |

### e2e — real llama-server, auto-skip

| File | Tests | Tag | Covers | Needs |
|------|------:|-----|--------|-------|
| `e2e/test_llama_server_e2e.py` | 5 | inference | isolated backend subprocess + managed llama-server: status, 4-way concurrent non-stream, 4-way concurrent stream, per-request mid-stream stop (other stream survives), unload kills server | CUDA + llama-server binary + auto-download GGUF (~531MB) |
| `e2e/benchmark_llama_server_prefill.py` | — | manual | prefill throughput benchmark script | live llama-server |

### manual / hardware

| File | Tag | Covers | Needs |
|------|-----|--------|-------|
| `stress_tests/stress_test_inference.py` | — | repeated load/chat/unload cycles | running backend |
| `stress_tests/stress_test_finetune.py` | — | repeated fine-tune cycles | running backend |
| `DeepNVMe/test_deepnvme_aio.py` (3) | finetune | DeepSpeed async_io pinned-buffer I/O | NVMe + async_io |
| `DeepNVMe/test_zero_inference_nvme.py` (3) | inference | ZeRO-Inference NVMe offload | NVMe + deepspeed |

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
