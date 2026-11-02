# MODELS.md — Which models to download and where to put them

English | [繁體中文](MODELS.zh-TW.md)

This pack **does not include model weights** (170.6 GB in total, plus the optional 0.63 GB Turbo LoRA), only the workflows, engine and skills.
Below is the **complete list** actually referenced by the 12 workflows, generated directly from `engine/workflows/*.json`,
so the file names are authoritative — **file names must match exactly**; ComfyUI finds files under `models/` by string match.

To verify (run this once downloads finish; it tells you what's missing and where it goes):

```
python tools/doctor.py --models
```

---

## You don't need to download everything

Sized per feature; install only what you'll use:

| Feature you want | Files | Size | Notes |
|---|---|---|---|
| **Images only** | 14 | **72.0 GB** | All imggen models |
| └ `qwen21` series only (image editing, outfit swap, multi-image compositing, moving people) | 3 | 16.1 GB | **Qwen research license, non-commercial**, see below |
| 　└ Speed-up: Viggle Turbo (optional, 2.5–3x faster) | 1 + custom node | 0.63 GB | Same as above, non-commercial; if not installed, automatically falls back to the slower base |
| └ `zimage` only (smallest, default) | 3 | 11.3 GB | Fast, cartoon/illustration |
| └ `realvis` only | 1 | 6.5 GB | Photorealistic, smallest single file |
| └ `flux` only (clean commercial license) | 4 | 16.9 GB | Apache |
| └ `qwen` only (best prompt adherence, can render Chinese text) | 3 | 21.1 GB | |
| **Video only** | 13 | **98.6 GB** | |
| └ `i2v` only (Wan 2.2 14B, default, silent) | 6 | 28.9 GB | |
| └ `r2v` only (MiniMax-H3, **the only one with audio**, can lock a character) | 5 | 59.1 GB | Includes fl2va for chaining |
| └ `--tier 5b` (Wan TI2V-5B fp16) | 3 | 16.9 GB | Rarely used; memory protection usually refuses to run it |
| **Everything** | 27 | **170.6 GB** | Excludes the optional Turbo LoRA |

> **Minimum usable set**: `zimage` (11.3 GB) + `i2v` (28.9 GB) = **40.2 GB**,
> enough for the main path "text → still image → make it move" (`t2v` is disabled in this setup, see the vidgen SKILL.md).
>
> Some files are shared, so adding up the rows above exceeds the totals (e.g. `z_image_*` serves both zimage and img2img,
> `umt5_xxl` serves both 14B and 5B).

---

## Complete list

`models/` means ComfyUI's model root:

- Windows portable: `<COMFY_HOME>\ComfyUI\models\`
- Linux git clone: `<COMFY_HOME>/models/`

**The subdirectory must be right.** ComfyUI looks for files in the subdirectory that matches the loader node type;
put a file in the wrong directory and "the file is clearly there but ComfyUI says it can't find it" — the most common and hardest-to-diagnose install error.

| File | Subdirectory under `models/` | Size | Needed by |
|---|---|---|---|
| `RealVisXL_V5.0_fp16.safetensors` | `checkpoints/` | 6.46 GB | imggen realvis |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 19.53 GB | vidgen r2v chaining |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 19.53 GB | vidgen r2v (H3, with audio) |
| `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models/` | 9.31 GB | vidgen --tier 5b |
| `qwen_image_2.1_int8_convrot.safetensors` | `diffusion_models/` | 6.76 GB | imggen qwen21 / qwen21_edit / qwen21_move |
| `z_image_turbo_int8_convrot.safetensors` | `diffusion_models/` | 5.78 GB | imggen zimage / img2img |
| `Wan22_Lightning_I2V_high_noise.safetensors` | `loras/` | 1.14 GB | vidgen i2v (14B, default) |
| `Wan22_Lightning_I2V_low_noise.safetensors` | `loras/` | 1.14 GB | vidgen i2v (14B, default) |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | `loras/` | 0.63 GB | **Optional**: turbo for the qwen21 series (added by gen.py, not in the workflows) |
| `clip_l.safetensors` | `text_encoders/` | 0.23 GB | imggen flux |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | 14.61 GB | vidgen r2v (H3) |
| `qwen3vl_8b_int8_convrot.safetensors` | `text_encoders/` | 8.71 GB | imggen qwen21 series |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | 8.74 GB | imggen qwen |
| `qwen_3_4b_fp8_mixed.safetensors` | `text_encoders/` | 5.25 GB | imggen zimage / img2img |
| `t5xxl_fp8_e4m3fn.safetensors` | `text_encoders/` | 4.56 GB | imggen flux |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 6.27 GB | vidgen i2v (14B), --tier 5b |
| `Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf` | `unet/` | 10.05 GB | vidgen i2v (14B, default) |
| `Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf` | `unet/` | 10.05 GB | vidgen i2v (14B, default) |
| `flux1-schnell-Q8_0.gguf` | `unet/` | 11.82 GB | imggen flux |
| `qwen-image-Q4_K_M.gguf` | `unet/` | 12.17 GB | imggen qwen |
| `ae.safetensors` | `vae/` | 0.31 GB | imggen flux |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | 0.56 GB | vidgen r2v (H3) |
| `minimax_h3_video_vae_fp16.safetensors` | `vae/` | 4.85 GB | vidgen r2v (H3) |
| `qwen_image_2.1_vae_bf16.safetensors` | `vae/` | 0.63 GB | imggen qwen21 series |
| `qwen_image_vae.safetensors` | `vae/` | 0.24 GB | imggen qwen |
| `wan2.2_vae.safetensors` | `vae/` | 1.31 GB | vidgen --tier 5b |
| `wan_2.1_vae.safetensors` | `vae/` | 0.24 GB | vidgen i2v (14B, default) |
| `z_image_ae.safetensors` | `vae/` | 0.31 GB | imggen zimage / img2img |

> **Note `unet/` vs `diffusion_models/`**: GGUF goes in `unet/` (`UnetLoaderGGUF` node),
> safetensors goes in `diffusion_models/` (`UNETLoader` node). These two directories have changed over ComfyUI's history;
> `doctor.py` only requires that a file "can be found under `models/`", but it tells you the expected subdirectory. Following the table is safest.

---

## How to download

### Recommended: let ComfyUI fetch them (least error-prone)

Open the ComfyUI web UI → load a workflow from `engine/workflows/` → **missing files trigger a download prompt**;
it knows the correct source and target directory, so you don't have to place files by hand. This is the easiest way and the least likely to put files in the wrong place.

### Manual download

I can only confirm the sources for **MiniMax-H3** and **Qwen-Image-2.1** (there is a download record to check):
all 5 H3 files come from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3),
and that repo's folder structure **maps exactly** onto ComfyUI's `models/` subdirectories, so you can download straight into place:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Comfy-Org/MiniMax-H3",
    allow_patterns=[
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ],
    local_dir="<COMFY_HOME>/ComfyUI/models",   # on Linux this is <COMFY_HOME>/models
)
```

**Qwen-Image-2.1** (the `qwen21` series): its 3 files come from [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1),
and its structure likewise maps onto the `models/` subdirectories:

```python
snapshot_download(
    repo_id="Comfy-Org/Qwen-Image-2.1",
    allow_patterns=[
        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ],
    local_dir="<COMFY_HOME>/ComfyUI/models",   # on Linux this is <COMFY_HOME>/models
)
```

The `*_pe_i2i` / `*_pe_t2i` files in the same repo (prompt enhancer, 9 B) are **not needed**:
the official ComfyUI image-editing template doesn't use them, and in testing, wiring one in actually stuffs the whole `<think>` reasoning block into the prompt.

(Why these H3 quantizations were chosen:
the diffusion model comes in 5 precisions and `pruned_int8_convrot` is the official template default;
the text encoder uses `nvfp4_awq` because sm_120 runs NVFP4 natively. Full precision 123.6 GB → this set 42.5 GB.)

**I have no download record for the other 19 files, so no URLs are listed here** — a wrong guess would send you chasing a pile of 404s.
Search Hugging Face directly by the **full file name**; the names are distinctive enough that the search will find the right repo.
The file names in the table above can be copied and pasted directly.

> Reminder: `flux1-schnell` and the old `qwen` (Qwen-Image) are Apache-licensed (commercial use OK).
> **Qwen-Image-2.1 (the `qwen21` series) and Viggle Turbo are under the Qwen RESEARCH LICENSE — non-commercial use only**;
> commercial use requires a separate license from Tongyi (model-business@notice.qwencloud.com).
> Other models have varying licenses; check the license terms of the source repo yourself before commercial use.

---

## Besides models, you also need one custom node

**`ComfyUI-GGUF`** — without it the `UnetLoaderGGUF` node doesn't exist, and
`imggen qwen`, `imggen flux` and `vidgen i2v` (14B, i.e. **the default video path**) all fail to work.

```
cd <COMFY_HOME>/ComfyUI/custom_nodes      # Linux: <COMFY_HOME>/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
```

`doctor.py` reports its absence as **FAIL**, not a warning.

### Optional: Viggle Turbo (speed-up for the `qwen21` series)

`gen.py` **uses turbo by default** for `qwen21` / `qwen21_edit` / `qwen21_move` (6 steps, 2.5–3x faster, no visible quality difference in testing).
It needs two files, both from [`Viggle/Qwen-Image-2.1-viggle-turbo`](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo):

| File (path in the repo) | Where to put it |
|---|---|
| `comfyui/viggle_turbo.py` | `<ComfyUI>/custom_nodes/viggle_turbo.py` |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | `models/loras/` |

Restart ComfyUI once they're in place. **This pack deliberately does not ship these two files**: they are derivatives under the Qwen research license (non-commercial),
so you download them and accept the license yourself. If either one is missing, `gen.py` automatically falls back to the base sampler
and records the reason in the `turbo` field of its output JSON; `doctor.py` also reports it as a warning.

> `v0.2.1` is used because the whole pipeline was validated with it. The repo already has a newer `v0.3`; switching versions means changing `TURBO_LORA` in `gen.py` and re-validating.

The MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `VAEDecodeAudio`)
are **built into ComfyUI core** and need no extra install, but they require a recent enough version
(the reference machine runs **ComfyUI 0.31.1**; older versions don't have these nodes).
