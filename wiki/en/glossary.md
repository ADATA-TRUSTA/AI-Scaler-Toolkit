---
title: Glossary
summary: Terms that mean something different here than in general use, or the same word meaning different things in different places.
updatedAt: 2026-10-29
---

# Glossary

**Only two kinds of terms are listed**: those whose meaning here differs from general use, and
**the same word meaning different things in different places**. General LLM terms are not.

## Same word, different meanings

### `engine`

| Where | Meaning |
|---|---|
| The `engine` field of a load request | One of `transformers` / `llama_server` / `vllm` / `strata` |
| Engine classes | The four do different jobs: **only transformers generates inside a worker**; the other three only start and stop a server |
| Text in the interface | Shows `llama_cpp`, a name for people; the API value is `llama_server` |

### `offload`

| Context | What it means |
|---|---|
| `device_map` (transformers) | Where weights go (GPU/CPU/disk) |
| `vllm_cpu_offload_params` | Only the **MoE expert weights** go to the CPU |
| `vllm_kv_offloading_size` | The **KV cache** goes to DRAM, not weights |
| DeepSpeed ZeRO-3 offload | Parameters and optimizer state **during training** |

**When you see offload, first ask "offload what?"** See [Offload](memory/offload.md).

### `quantization`

| Engine | Values | How it is set |
|---|---|---|
| transformers | `int4` / `int8` / `none` (bitsandbytes) | The `quantization` field |
| llama.cpp, Strata | In the GGUF (`Q4_K_M` etc.) | Which GGUF file you pick |
| vLLM | AWQ / GPTQ / FP8… | The `vllm_quantization` field |

### `status`

| Where | Values |
|---|---|
| Download tasks | `pending` / `running` / `cancelling` / `completed` / `failed` / `cancelled` |
| Registered models | `ready` / `deleting` / `delete_failed` |
| Training | `idle` / `initializing` / `running` / `completed` / `stopped` / `error` |

`cancelling` is not `cancelled`: the cancel was received but the background work has not stopped yet.

### `job` and `task`

No difference in meaning, just inconsistent naming: downloads use `task_id`; GGUF conversion and
training use `job_id`.

### Context length

| Field | Meaning |
|---|---|
| `max_context_length` | The limit recorded in the model list |
| `n_ctx` | The window llama-server is started with |
| **`served_context_length`** | **The window the engine actually gives this load** |
| `n_ctx_train` | The model's own training limit (reported by llama.cpp) |

Put the last two side by side to see whether a limit comes from the engine or the model.

## The project's own terms

| Term | Meaning |
|---|---|
| **Worker process** | A child process started with `spawn`, one for inference and one for training. Unloading a model ends it, which is how VRAM is reclaimed |
| **Server engine** | `llama_server`, `vllm`, `strata`: the three that serve OpenAI-compatible HTTP. Generation does not go through a worker |
| **Phase** | A fine-grained training stage (`loading_model`, `training`, `saving`…); `status` is the coarse one |
| **Slot** | A llama-server concurrency slot (`-np`). For text-only models, a slot's KV can be saved and restored |
| **mmproj** | The vision projector file of a multimodal GGUF. **Without it vision simply does not work, and nothing says so** |
| **Sleep / wake** | Release the GPU without unloading the model. vLLM drops the KV; Strata keeps the conversation |
| **Run config** | Strata's model description (`strata-<model>.json`), written by Strata's setup |

## Related

- [Overview](overview.md)
- [Choosing an inference engine](engines/index.md)
- [Offload](memory/offload.md)
