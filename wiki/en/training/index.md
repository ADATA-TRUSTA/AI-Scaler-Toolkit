---
title: Fine-tuning overview
summary: The fine-tuning flow, two strategies, two orderings that must not be broken, the phase lifecycle, and how to read the live event stream (SSE).
updatedAt: 2026-10-29
---

# Fine-tuning overview

> Fine-tuning is **Linux only** (WSL2 counts). Elsewhere `POST /training/start` returns `501` with the reason.

## The one-minute version

```
POST /training/start
   │
   ▼
spawn the training worker
   │
   ├─ num_gpus == 1 → train in-process
   └─ num_gpus  > 1 → start a deepspeed child process
          │
          ▼
   shared setup: tokenizer → dataset (eval split) → model
          ▼
   trainer → train → save → merge LoRA → convert to GGUF (Q4_K_M)
```

**One job at a time.** And **training does not exclude inference**: unload the inference model
before training, or a 16 GB card will usually run out of memory.

## Methods and strategies

| Method | Notes |
|---|---|
| full | All parameters |
| LoRA | Adapter only |
| QLoRA | Quantized base + LoRA |

| Strategy (`use_sft_trainer`) | Uses | Who preprocesses data |
|---|---|---|
| `true` (default) | TRL's `SFTTrainer` | TRL (including image collation) |
| `false` | HF `Trainer` | The backend, **always with the tokenizer** |

## Main fields

| Group | Fields |
|---|---|
| Basics | `model_name`, `method`, `dataset_path`, `output_dir` |
| LoRA | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` |
| Multimodal | `multimodal_scope`, `max_image_pixels` |
| Hyperparameters | `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `warmup_steps` |
| Saving | `save_steps`, `save_total_limit`, `save_tokenizer` |
| Dataset field names | `text_field`, `prompt_field`, `completion_field`, `messages_field` |
| Evaluation | `eval_split_ratio`, `eval_steps`, `eval_split_seed` |
| DeepSpeed | `use_deepspeed`, `deepspeed_profile`, `deepspeed_config`, `offload_folder` |
| Other | `packing`, `gradient_checkpointing`, `use_flash_attention`, `num_gpus`, `max_seq_length` |

Full example: `GET /examples/training`.

### Keep `save_steps` small

Not for save frequency, but for **what you lose when a long run is interrupted**. The API **has
no resume**; you restart from the last checkpoint. Measured: on a GPU that also drives the display,
a ZeRO-3 run was killed after about 61 minutes by the driver watchdog (`cudaErrorLaunchTimeout`).
See [Pitfalls found by running it](../operations/known-pitfalls.md).

### The eval split happens before anything touches the data

With `eval_split_ratio` set, the dataset is **split as soon as it is loaded**, so evaluation sees
samples training never touched.

## Two orderings that must not be broken

Both fail silently, which is why single-GPU and multi-GPU share one setup sequence.

| Ordering | What happens if broken |
|---|---|
| **The DeepSpeed configuration is created before the model loads** | ZeRO-3 **silently gives up** sharding at load time; on a 16 GB card that is the difference between OOM and training |
| **Image training mode is enabled before the model loads** | The multimodal checkpoint loads as a text-only model, **without its vision tower** |

## Saving LoRA bypasses DeepSpeed

With DeepSpeed + LoRA both save modes break: `gather=True` runs out of memory, `gather=False` hits
`FileExists`. When DeepSpeed is used and the model is a PEFT model, only the adapter is saved.

## Progress: phase and status

| phase (fine) | status (coarse) |
|---|---|
| `pending`, `queued` | `starting` |
| `resolving_deepspeed`, `launching_workers`, `loading_tokenizer`, `loading_dataset`, `loading_model`, `preparing_trainer` | `initializing` |
| `training`, `saving`, `merging_lora`, `converting_gguf` | `running` |
| `completed` | `completed` |
| `stopped`, `cancelled` | `stopped` |
| `error` | `error` |

Terminal phases: `completed`, `stopped`, `error`, `cancelled`. `pending`, `queued` and `cancelled`
are reserved for a queue that is not implemented yet. With several GPUs, the child process only
runs up to `training`; saving, merging and conversion are done by the parent after the child exits.

## The event stream

Each job has one structured event stream, written to two files:

```
{LOG_DIR}/training/{job_id}/events.jsonl   # for machines, one JSON per line
{LOG_DIR}/training/{job_id}/training.log   # for people
```

| Field | Always | Notes |
|---|---|---|
| `ts` | yes | Unix time (float) |
| `seq` | yes | Monotonic; **what resuming uses** |
| `job_id` | yes | |
| `type` | yes | `stage` (phase change) / `resource` (GPU, DRAM, SSD snapshot) / `metric` (loss, lr…) / `error` / `info` |
| `phase`, `msg`, `data` | no | |

### Two ways to read it

| Endpoint | Use |
|---|---|
| `GET /training/{job_id}/log?since=<seq>` | Catch up on history |
| `GET /training/{job_id}/log/stream` | **SSE** live stream |

- To resume after a disconnect, send the last `seq` you received in the `Last-Event-ID` header
  (or `?since=`); the header wins.
- A `stage` event with a terminal phase means the job is over.
- Behind a reverse proxy, **turn off the proxy's buffering**, or events pile up and arrive at once.
  See [Behind a reverse proxy](../operations/reverse-proxy.md).
- The event files are on disk, so **history survives a service restart**, and a job started by any
  uvicorn worker can be read by any other.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/training/start` | Start; returns `job_id` |
| GET | `/training/status` | Coarse status and progress |
| POST | `/training/stop` | Stop |
| GET | `/training/{job_id}/log`, `/log/stream` | See above |
| GET | `/training/error_details` | Error details with the full traceback |
| GET | `/training/status/{session_id}/history` | Loss, learning rate and other history |
| POST | `/training/force_cleanup_gpu` | Force-clean training GPU processes and memory |

## Related

- [Multimodal fine-tuning](multimodal.md)
- [DeepSpeed ZeRO-3 offload](deepspeed-offload.md)
- [GGUF conversion](../models/gguf-conversion.md)
- [LoRA training a 35B MoE on a 16 GB GPU](../benchmarks/deepspeed-lora-16gb.md)
