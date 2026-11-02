---
title: Troubleshooting by symptom
summary: What to check first for a given symptom; the cases where the symptom points the wrong way come first.
updatedAt: 2026-10-29
---

# Troubleshooting by symptom

**The entry point is the symptom, not the component.** Causes are explained on their own pages;
this one only says "see X, check Y". For install and everyday problems see
[`docs/faq.en.md`](../../../docs/faq.en.md).

## Symptoms that point the wrong way

| Symptom | Looks like | Actually |
|---|---|---|
| "Training failed", but `output_dir` has weights | Training broke | **The final GGUF conversion failed.** The weights are saved; do not retrain — convert with `POST /config/models/convert`. See [GGUF conversion](../models/gguf-conversion.md) |
| vLLM says the model is not found, but the path is right | Wrong model path | **A GPU compiler is missing**: `intel-ocloc` on Intel (`No such file or directory: 'ocloc'`), `nvcc` on NVIDIA |
| Changed a setting and reloaded; nothing changed | The setting was not saved | **`already_loaded` compares only six fields.** Unload, then load |
| The fine-tuned model answers worse than before | Fine-tuning failed | **Thinking is on**, and the training data had no think block. Retry with `enable_thinking: false` |
| The agent returns an empty string, like a failed merge | The merge broke | **LoRA scaling became 1.0**, so the adapter barely acts |

## Another machine cannot connect

**Read the status code and message first:**

| Response | Meaning | What to do |
|---|---|---|
| `403` …*available on the computer running AI Scaler Toolkit only* | This path is local only; the web interface and `/docs` always are | Open it locally, or check you are calling `/v1/` |
| `403` …*the inference API (/v1/) only* | The path is in the management API and `SERVICE_REMOTE_ACCESS` is the default `inference` | Set `all` to open it, **after reading the risk note** |
| `403` …*not enabled on this service* | `SERVICE_API_KEY` is empty | Enable remote access in Settings and set a key |
| `401` …*A valid API key is required* | Wrong key or wrong format | `Authorization: Bearer <key>`; bearer only |
| `403` …*Web pages from other sites cannot use this service* | A cross-site page, or `Origin` is `null` | Open it from a page this machine serves |
| `403` …*must be addressed to localhost or an IP address* | `Host` is neither localhost nor an IP | Connect by `localhost` or IP |
| `429`, everything refused after a few tries | 10 wrong keys in a minute → 5-minute lockout | Wait 5 minutes, or restart the service |
| Through a reverse proxy, every API call from the web interface is `403` | The proxy sends `X-Forwarded-For` | See [Behind a reverse proxy](reverse-proxy.md) |

Without a key `/health` only says the service is alive, so another machine cannot see the engine list.

## Loading a model

| Symptom | Cause |
|---|---|
| `409 A model is already loaded` | One at a time; unload first |
| `409 A model is already being loaded` | The previous load is still running |
| `501` with a sentence | Unsupported here (vLLM off Linux, vLLM not installed); the sentence is the reason |
| `ValueError: When passing device_map as a string...` | `device_map` was `none`; use `cuda:0` or `auto` |
| Loads, but runs on the CPU | `device_map` omitted lands on the CPU |
| A multimodal GGUF has no vision | mmproj is missing, **and nothing says so** |
| A MoE larger than the GPU will not load | vLLM: `vllm_cpu_offload_params=["experts"]`; llama.cpp: `n_cpu_moe`; or `recommend` from [Memory estimation](../memory/estimation.md) |
| Loads fine, but every generation fails to connect | A llama-server port problem; check `llama_server_url` is not set to point elsewhere |

## vLLM

| Symptom | Check |
|---|---|
| No vLLM in the menu | Not Linux, or not installed; see `GET /health`'s `supported_engines`. On Intel install with `TRUSTA_SETUP_VLLM=1` |
| On an Intel iGPU, the first load works and the second fails | The kernel's TTM page pool kept up to half of RAM, which vLLM counts as used |
| `is less than desired gpu memory utilization` | Lower `vllm_gpu_memory_utilization`; a bigger card is not the fix |
| Empty replies after a tool result | The reasoning parser took the whole reply as thinking; with a parser thinking defaults to off, so check the request does not force it on |
| Port in use after a restart | A leftover `vllm serve`. The service sweeps them at startup; ones you started by hand are not |
| `/dev/shm` keeps filling | Offload regions left by a killed server |
| The KV offload directory keeps growing | vLLM never deletes from the disk tier; set `vllm_kv_offloading_disk_max_gb` or call cleanup |
| An NVMe tier but no hits | The DRAM tier cannot hold the prefix to reuse (DRAM is the gate) |

## Sleep / wake

| Symptom | Cause |
|---|---|
| Sleep returns `409` | The engine does not support it; only vLLM (load with `vllm_enable_sleep_mode`) and Strata |
| vLLM sleep mode refused before loading | Not allowed when `VLLM_SERVER_HOST` is not loopback |
| **Wake returns `502`** | Another program took the GPU memory while it slept. The most common wake failure |
| The conversation is gone after sleep (vLLM) | Expected: vLLM's sleep drops the KV. Use Strata to keep it |
| Sleep takes a long time to return | The default `mode: "wait"` waits for running generations; use `abort` to stop at once |

## Strata

| Symptom | Cause |
|---|---|
| No `strata` in the menu | `STRATA_HOME` is unset, or has no `serve/server.py` (Strata installs on NVIDIA hosts only) |
| `STRATA_ENGINE_BINARY is empty` | Required; must point to a deep-sleep build |
| The desktop session logs out while loading a 50 GB model | systemd-oomd. Make sure `STRATA_UNBUFFERED_LOAD` and `STRATA_NO_ARENA_THP` are not set to 0 |
| Decode much slower than expected | `strata_vram_limit_gb` is too low; leave it unset |
| Load times out | `STRATA_HEALTH_TIMEOUT` defaults to 900 s |

## Fine-tuning

| Symptom | Cause |
|---|---|
| Killed after about 61 minutes with `cudaErrorLaunchTimeout` | The GPU also drives the display. Keep `save_steps` small, or move the display to another card |
| Out of GPU memory with plenty of DRAM left | Multimodal full fine-tuning does not fit in 16 GB and CPU offload cannot help; use LoRA |
| NVMe offload uses tens of GB of DRAM | Not a leak; adjust `DEEPSPEED_NVME_PINNED_BUDGET_GB` |
| OOM right at the start of training | Is an inference model still loaded? Training does not exclude inference |
| Inference works on Intel but training says `ze_data was not initialized` | oneCCL needs `libze_loader.so` (the `libze-dev` package) |

## Downloads

| Symptom | Cause |
|---|---|
| Cancelled, but it keeps running for a long time | With `HF_XET_ENABLED=1` a transfer cannot be interrupted midway |
| A small model downloaded many GB | A GGUF repo without `filename` downloads every quantization |
| A mistyped model id did not fail immediately | The endpoint returns a `task_id` at once; the failure comes when the task reaches the Hub |
| The task list is empty after a restart | Download tasks live in memory only |

## Chat

| Symptom | Cause |
|---|---|
| The thinking disappeared | It is in the `reasoning_content` field, not `<think>` tags |
| No thinking from the transformers engine | Expected; that path has no reasoning parser |
| Long context drops before the first response | A proxy or client timeout is too short; the wait for the first byte is the whole prefill |
| The answer stops inside the thinking | `max_tokens` is too small |
| A fine-tuned adapter seems to do nothing | The default system prompt the service injects changes the prompt; set `DEFAULT_SYSTEM_PROMPT` to an empty string |
| Qwen3.5 ignores `repetition_penalty` | On purpose: values above 1.0 become 1.0. See [Chat](../api/chat.md) |

## Check upstream before calling it broken

When a third-party package misbehaves, **check its issues, changelog and compatibility table
first**. vLLM 0.27.1 failing on gemma-4 and the GGUF converter's `global_head_dim` KeyError were
both known upstream issues.

## Related

- [Pitfalls found by running it](known-pitfalls.md)
- [Error model](../api/errors.md)
- [`docs/faq.en.md`](../../../docs/faq.en.md)
