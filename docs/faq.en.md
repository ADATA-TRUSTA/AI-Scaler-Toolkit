# Maintenance & FAQ

[← Back to README](../README.md)

## Keeping it up to date

**The application**: Settings → Updates → **Check now**, then **Update now**. See [Installation → Updating](installation.en.md#updating).

**A source checkout**: after `git pull`, run `setup_env` again so the environment matches the new dependencies. It installs only what changed:

```bash
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh      # or TRUSTA_ACCEL=xpu
```

```powershell
.\backend\scripts\windows\setup_env.ps1 -Accel cuda            # or -Accel xpu
```

---

## FAQ

### Q1. The first launch takes a long time

It downloads and installs about 6-10 GB: the Python packages for your GPU and the llama.cpp runtime (on Linux, llama.cpp is also compiled). 5-15 minutes is normal, longer on a slow connection. The window shows each step. If you close it or the PC restarts, start the application again: it says "Resuming setup", and finished parts go past in seconds.

### Q2. Setup failed

The window shows the error. The full output is in `provision.log` in the logs folder (Windows `%LOCALAPPDATA%\AIScalerToolkit\logs\`, Linux `~/.local/share/ai-scaler-toolkit/logs/`). Most failures are the network: start the application again and it continues. If it keeps failing at the same step, use Settings → Environment → **Reinstall the environment**.

### Q3. On Linux, double-clicking the AppImage does nothing

Make it executable: `chmod +x AIScalerToolkit-*-x86_64.AppImage`, or tick "Allow executing file as program" in its properties. If it then says `fusermount: not found`, install FUSE 3: `sudo apt install fuse3`.

### Q4. On Linux, why does it ask for my password?

Only on the first launch, once, to install the system packages it needs: WebKitGTK for its window, `build-essential` and `cmake` to compile llama.cpp for your CPU, and with an Intel GPU its runtime packages. It asks through your desktop's own password dialog. On a machine with no desktop, install them yourself and use `--headless` (see [Installation](installation.en.md#no-desktop-linux-servers)).

### Q5. My Intel GPU is not used on Linux

Use Ubuntu 24.04 or newer: 22.04's archive has no Level Zero. A recent Intel GPU (Lunar Lake, Panther Lake, Arc B-series) also needs Intel's graphics package source, because the driver in Ubuntu's own archive is too old to see it. Add it, then start the application again:

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
```

### Q6. My NVIDIA GPU is not used, or loading fails with a CUDA error

The NVIDIA build needs a card with compute capability 7.5 or newer (RTX 20 / GTX 16 series and later) and driver 580 or newer. Check with `nvidia-smi`: the top line shows the driver version and `CUDA Version: 13.x`. Update the driver from NVIDIA, then start the application again.

### Q7. vLLM or fine-tuning is missing

Both run on Linux only. vLLM is installed with an NVIDIA GPU; on an Intel GPU it is not installed by default. Fine-tuning works on Linux with NVIDIA or Intel. On Windows the engines are Transformers and llama.cpp. With an NVIDIA GPU you can still use them on Windows by running the Linux version inside WSL2 (an Ubuntu distribution), for example the AppImage with `--headless` (see [Installation](installation.en.md#no-desktop-linux-servers)).

### Q8. The service will not start: the port is in use

Another program is using port `8571`. Change it in Settings → Ports → Application port, or quit the other program. Apps you connected to this one need the new port too.

### Q9. "Config AST Endpoint" is greyed out, or does not change Hermes

It needs a loaded model: load one first, then press it again. If Hermes is already set to another provider that you chose, it leaves that alone on purpose and shows the commands to switch by hand. The result is also written to `launcher.log`.

### Q10. Another computer cannot reach this one

Turn on **Let another computer use this machine's model** on the Hermes card in Settings. The card shows the address and the API key to use; the other computer must send the key (`Authorization: Bearer <key>`), or it gets `401`. A `403` means it asked for something other than the API, such as the web interface, which stays on this computer. If it cannot connect at all, a firewall is usually the reason: the card offers to add a rule (Windows asks for administrator approval). Anyone with the key can use the model, so do this only on a network you trust.

### Q11. Why does my laptop list eight 4 GB memory modules?

That is how its memory is built. Laptops with memory soldered to the board (LPDDR5) report each memory channel as its own device, so 32 GB shows as 8 × 4 GB. Windows reports the same.

### Q12. Where are my models, and are they kept when I uninstall?

Downloaded models are in `~/.cache/huggingface` (Windows: `%USERPROFILE%\.cache\huggingface`), the standard Hugging Face cache, shared with other Hugging Face tools. Uninstalling on Windows asks whether to remove them; on Linux they stay until you delete them. After installing again the files are still there, but the model list starts over: register them again from the Models page (they are found in the cache, so nothing is downloaded twice). The starter model is registered again on its own.

### Q13. I only need the API, not the window

The API is at `http://127.0.0.1:8571`, documented at `/docs`. See [Usage → Connecting other applications](usage.en.md#connecting-other-applications). On a Linux server, run the AppImage with `--headless`.

### Q14. I have an NVIDIA and an Intel GPU: which one is used?

NVIDIA. Setup looks for an NVIDIA GPU first and installs the CUDA build when it finds one, so a PC with an NVIDIA card and an Intel iGPU (or Arc card) uses the NVIDIA card. To use the Intel GPU instead, set the environment variable `TRUSTA_ACCEL=xpu` for your account, start the application, and use Settings → Environment → **Reinstall the environment**. ComfyUI is chosen separately: pick its card in Settings → Components.

### Q15. What should I do if fine-tuning with DeepSpeed hits offload / buffer / swapper errors?

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
