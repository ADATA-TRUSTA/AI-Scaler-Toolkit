# Maintenance & FAQ

[← Back to README](../README.md)

## Maintenance Updates

Each time you pull a new version from GitHub, it is recommended to resync dependencies.

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash backend/scripts/linux/setup_env.sh
```

If switching to CUDA:

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
```

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel xpu
```

If switching to CUDA:

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel cuda
```

---

## FAQ

### Q1. The startup script says `.venv` cannot be found

This means the Python environment has not been created yet. Run:

- Linux: `TRUSTA_ACCEL=xpu bash backend/scripts/linux/setup_env.sh`
- Windows: `.\backend\scripts\windows\setup_env.ps1 -Accel xpu`

### Q2. After moving to a new machine, the model cache still points to the old path

Check `.env` first:

```dotenv
HF_HOME=<your cache path>
TIKTOKEN_RS_CACHE_DIR=<your project root>
```

### Q3. The API is running, but the home page does not show the frontend

First verify that the frontend files exist in the project:

- `frontend/dist/index.html`
- `frontend/dist/assets/`

If files already exist under `dist`, simply reopen `http://127.0.0.1:8000/`.

### Q4. I only need the backend API, not the frontend

Use these routes directly:

- `/docs`
- `/health`
- `/v1/chat/completions`

### Q5. `setup_env.ps1` fails on Windows

Verify that:

- Python 3.12+ is installed
- `uv` is installed
- If using XPU, Intel oneAPI is installed

If needed, first update the PowerShell execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q6. Building the environment with `-Accel cuda` fails on Windows

Make sure CUDA Toolkit is installed and the driver version is compatible. If you are not using an NVIDIA GPU, use:

```powershell
.\backend\scripts\windows\setup_env.ps1 -Accel xpu
```

### Q7. What should I do if fine-tuning with DeepSpeed hits offload / buffer / swapper errors?

The built-in DeepSpeed profiles are located at:

- `backend/service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `backend/service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `backend/service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `backend/service/configs/deepspeed/zero3_offload_disk_disk.json`

If the error mentions `buffer`, `swapper`, NVMe offload, aio, or offload queue problems, the most common first step is to increase `zero_optimization.offload_param.buffer_count`.

For example:

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 6
}
```

If it is still unstable, increase it gradually to `8` instead of jumping too far at once.

If `offload_optimizer.device` is also `nvme`, you can also adjust `zero_optimization.offload_optimizer.buffer_count`.

> In practice, when you see errors like “buffer 0 / swapper 1”, adjusting `offload_param.buffer_count` first is usually the most effective change.

#### Common DeepSpeed Parameters to Tune

| Parameter | Location | When to Adjust | Typical Effect / Trade-off |
|------|------|----------|------------------|
| `buffer_count` | `zero_optimization.offload_param.buffer_count` | Buffer, swapper, or queue errors during NVMe/CPU offload | Increases available buffers and often improves offload stability, but uses more memory / RAM |
| `buffer_count` | `zero_optimization.offload_optimizer.buffer_count` | Similar errors when optimizer state is offloaded to NVMe | Similar to above, but affects optimizer state |
| `buffer_size` | `zero_optimization.offload_param.buffer_size` | Disk I/O is too fragmented, throughput is unstable, or offload happens too often | Larger values reduce chunking overhead, but increase single-buffer usage |
| `max_in_cpu` | `zero_optimization.offload_param.max_in_cpu` | NVMe is too slow and you want to keep more parameters in RAM | Higher values reduce disk reads/writes, but increase system RAM usage |
| `pin_memory` | `zero_optimization.offload_param.pin_memory` / `zero_optimization.offload_optimizer.pin_memory` | RAM is sufficient and you want better CPU↔GPU transfer | May improve transfer efficiency, but can increase RAM pressure; disable if memory is tight |
| `stage3_prefetch_bucket_size` | `zero_optimization.stage3_prefetch_bucket_size` | ZeRO-3 prefetch is too large and causes VRAM / RAM pressure, or I/O instability | Smaller is more conservative and stable; larger may improve throughput |
| `reduce_bucket_size` | `zero_optimization.reduce_bucket_size` | Communication or aggregation uses too much VRAM | Smaller values reduce peak memory pressure, but may affect speed |
| `stage3_param_persistence_threshold` | `zero_optimization.stage3_param_persistence_threshold` | Frequent parameter movement causes unstable performance | Adjusts persistence behavior and should be tested against model size |
| `sub_group_size` | `zero_optimization.sub_group_size` | Memory or performance is poor with very large model partitioning | Smaller is safer; larger may improve efficiency |
| `contiguous_gradients` | `zero_optimization.contiguous_gradients` | Gradient fragmentation or memory allocation issues | Usually keep `true`; toggle only for specific compatibility issues |
| `overlap_comm` | `zero_optimization.overlap_comm` | Communication overlap causes instability, memory spikes, or abnormal performance | Disabling may improve stability, but can reduce speed |

#### Training Parameters Often Tuned Together

If changing DeepSpeed config alone is not enough, it is also common to reduce training pressure:

| Parameter | Location | Suggested Direction |
|------|------|----------|
| `per_device_train_batch_size` | Training request body | Reduce first when VRAM is insufficient |
| `gradient_accumulation_steps` | Training request body or DeepSpeed config | Increase to recover effective batch size after lowering batch size |
| `max_seq_length` | Training request body | Reduce first if the sequence is too long, for example `4096 -> 2048` |
| `gradient_checkpointing` | Training request body | Keep enabled when memory is insufficient |

#### Recommended Tuning Order

1. Confirm which profile you are using first (CPU offload or NVMe offload).
2. If the error includes `buffer` / `swapper`, adjust `offload_param.buffer_count` first.
3. If NVMe offload is still unstable, adjust `offload_optimizer.buffer_count`, `buffer_size`, and `max_in_cpu`.
4. If VRAM / RAM is insufficient, reduce `per_device_train_batch_size` and `max_seq_length`.
5. For throughput or stability issues, fine-tune `stage3_prefetch_bucket_size`, `reduce_bucket_size`, and `overlap_comm`.

If you use NVMe offload, also verify:

- The disk at `nvme_path` has enough free space
- Prefer SSD / NVMe instead of a slow HDD
- Avoid sharing the same disk with other heavy I/O workloads
