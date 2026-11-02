# MODELS.md — 要下載哪些模型、放到哪裡

[English](MODELS.md) | 繁體中文

這包**不含模型權重**（共 170.6 GB，另有選配的 Turbo LoRA 0.63 GB），只含 workflow、引擎與 skill。
下面是這 12 個 workflow 實際引用的**完整清單**，由 `engine/workflows/*.json` 直接產出，
所以檔名是權威的 —— **檔名必須完全一致**，ComfyUI 是按字串比對去 `models/` 底下找檔案的。

驗證方式（下載完就跑這個，它會告訴你缺什麼、該放哪）：

```
python tools/doctor.py --models
```

---

## 你不需要全部下載

按功能分開算，只裝你要用的：

| 你想要的功能 | 檔數 | 大小 | 說明 |
|---|---|---|---|
| **只要生圖** | 14 | **72.0 GB** | imggen 全部模型 |
| └ 只要 `qwen21` 系列（改圖、換裝、多圖合成、移動人物） | 3 | 16.1 GB | **Qwen 研究授權，非商用**，見下方 |
| 　└ 加速：Viggle Turbo（選配，快 2.5–3 倍） | 1 + 自訂節點 | 0.63 GB | 同上，非商用；不裝會自動改用較慢的 base |
| └ 只要 `zimage`（最省，預設） | 3 | 11.3 GB | 快、卡通/插畫 |
| └ 只要 `realvis` | 1 | 6.5 GB | 寫實照片，單檔最省 |
| └ 只要 `flux`（商用授權乾淨） | 4 | 16.9 GB | Apache |
| └ 只要 `qwen`（prompt 遵循最好、能畫中文字） | 3 | 21.1 GB | |
| **只要生影片** | 13 | **98.6 GB** | |
| └ 只要 `i2v`（Wan 2.2 14B，預設、無聲） | 6 | 28.9 GB | |
| └ 只要 `r2v`（MiniMax-H3，**唯一有聲音**、可鎖角色） | 5 | 59.1 GB | 含接力用的 fl2va |
| └ `--tier 5b`（Wan TI2V-5B fp16） | 3 | 16.9 GB | 很少用，記憶體保護通常會拒跑 |
| **全部** | 27 | **170.6 GB** | 不含選配的 Turbo LoRA |

> **最低可用組合**：`zimage`(11.3 GB) + `i2v`(28.9 GB) = **40.2 GB**，
> 就能走完「文字 → 生靜圖 → 讓它動」這條主線（`t2v` 在這套設定裡是被停用的，見 vidgen SKILL.md）。
>
> 有些檔案是共用的，所以上表逐項相加會超過總數（例如 `z_image_*` 同時給 zimage 和 img2img 用，
> `umt5_xxl` 同時給 14B 和 5B 用）。

---

## 完整清單

`models/` 指的是 ComfyUI 的模型根目錄：

- Windows portable：`<COMFY_HOME>\ComfyUI\models\`
- Linux git clone：`<COMFY_HOME>/models/`

**子目錄一定要放對。** ComfyUI 是依 loader 節點的種類去對應的子目錄找檔案的，
放錯目錄的結果是「檔案明明在，ComfyUI 卻說找不到」——這是最常見也最難查的安裝錯誤。

| 檔案 | 放進 `models/` 的哪個子目錄 | 大小 | 哪個功能需要 |
|---|---|---|---|
| `RealVisXL_V5.0_fp16.safetensors` | `checkpoints/` | 6.46 GB | imggen realvis |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 19.53 GB | vidgen r2v 接力 |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 19.53 GB | vidgen r2v（H3，有聲） |
| `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models/` | 9.31 GB | vidgen --tier 5b |
| `qwen_image_2.1_int8_convrot.safetensors` | `diffusion_models/` | 6.76 GB | imggen qwen21 / qwen21_edit / qwen21_move |
| `z_image_turbo_int8_convrot.safetensors` | `diffusion_models/` | 5.78 GB | imggen zimage / img2img |
| `Wan22_Lightning_I2V_high_noise.safetensors` | `loras/` | 1.14 GB | vidgen i2v（14B，預設） |
| `Wan22_Lightning_I2V_low_noise.safetensors` | `loras/` | 1.14 GB | vidgen i2v（14B，預設） |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | `loras/` | 0.63 GB | **選配**：qwen21 系列的 turbo（由 gen.py 加入，不在 workflow 裡） |
| `clip_l.safetensors` | `text_encoders/` | 0.23 GB | imggen flux |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | 14.61 GB | vidgen r2v（H3） |
| `qwen3vl_8b_int8_convrot.safetensors` | `text_encoders/` | 8.71 GB | imggen qwen21 系列 |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | 8.74 GB | imggen qwen |
| `qwen_3_4b_fp8_mixed.safetensors` | `text_encoders/` | 5.25 GB | imggen zimage / img2img |
| `t5xxl_fp8_e4m3fn.safetensors` | `text_encoders/` | 4.56 GB | imggen flux |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 6.27 GB | vidgen i2v（14B）、--tier 5b |
| `Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf` | `unet/` | 10.05 GB | vidgen i2v（14B，預設） |
| `Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf` | `unet/` | 10.05 GB | vidgen i2v（14B，預設） |
| `flux1-schnell-Q8_0.gguf` | `unet/` | 11.82 GB | imggen flux |
| `qwen-image-Q4_K_M.gguf` | `unet/` | 12.17 GB | imggen qwen |
| `ae.safetensors` | `vae/` | 0.31 GB | imggen flux |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | 0.56 GB | vidgen r2v（H3） |
| `minimax_h3_video_vae_fp16.safetensors` | `vae/` | 4.85 GB | vidgen r2v（H3） |
| `qwen_image_2.1_vae_bf16.safetensors` | `vae/` | 0.63 GB | imggen qwen21 系列 |
| `qwen_image_vae.safetensors` | `vae/` | 0.24 GB | imggen qwen |
| `wan2.2_vae.safetensors` | `vae/` | 1.31 GB | vidgen --tier 5b |
| `wan_2.1_vae.safetensors` | `vae/` | 0.24 GB | vidgen i2v（14B，預設） |
| `z_image_ae.safetensors` | `vae/` | 0.31 GB | imggen zimage / img2img |

> **注意 `unet/` vs `diffusion_models/`**：GGUF 走 `unet/`（`UnetLoaderGGUF` 節點），
> safetensors 走 `diffusion_models/`（`UNETLoader` 節點）。這兩個目錄在 ComfyUI 歷史上有過變動，
> `doctor.py` 只要求「在 `models/` 底下找得到」，但會提示你預期的子目錄，照表放最保險。

---

## 怎麼下載

### 建議做法：讓 ComfyUI 自己抓（最不會出錯）

開啟 ComfyUI 網頁 → 載入 `engine/workflows/` 裡的 workflow → **缺檔會跳出下載提示**，
它知道正確的來源與目標目錄，不用你手動歸位。這是最省事也最不容易放錯位置的方式。

### 手動下載

我只能確認 **MiniMax-H3** 與 **Qwen-Image-2.1** 的來源（有下載紀錄可查）：
全部 5 個 H3 檔案都來自 [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)，
而且該 repo 的資料夾結構**剛好對應** ComfyUI 的 `models/` 子目錄，可直接下載到位：

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
    local_dir="<COMFY_HOME>/ComfyUI/models",   # Linux 是 <COMFY_HOME>/models
)
```

**Qwen-Image-2.1**（`qwen21` 系列）的 3 個檔案來自 [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)，
結構同樣對應 `models/` 子目錄：

```python
snapshot_download(
    repo_id="Comfy-Org/Qwen-Image-2.1",
    allow_patterns=[
        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ],
    local_dir="<COMFY_HOME>/ComfyUI/models",   # Linux 是 <COMFY_HOME>/models
)
```

同一個 repo 裡的 `*_pe_i2i` / `*_pe_t2i`（prompt enhancer，9 B）**不需要**：
官方的 ComfyUI 改圖範本不用它，實測接上它反而會把整段 `<think>` 推理塞進 prompt。

（H3 的量化版本選擇理由：
擴散模型有 5 種精度，`pruned_int8_convrot` 是官方模板預設；
文字編碼器選 `nvfp4_awq` 是因為 sm_120 原生跑 NVFP4。全精度 123.6 GB → 這組 42.5 GB。）

**其餘 19 個檔案我沒有下載紀錄可依據，所以不在這裡列 URL** —— 猜錯會讓你追一堆 404。
請直接用**完整檔名**去 Hugging Face 搜尋，檔名夠獨特，搜尋一定找得到正確的 repo。
上面那張表的檔名可以直接複製貼上。

> 提醒：`flux1-schnell` 與舊版 `qwen`（Qwen-Image）是 Apache 授權（可商用）。
> **Qwen-Image-2.1（`qwen21` 系列）與 Viggle Turbo 是 Qwen RESEARCH LICENSE，只能非商用**，
> 商用要另向通義申請授權（model-business@notice.qwencloud.com）。
> 其他模型的授權各不相同，商用前請自行確認來源 repo 的授權條款。

---

## 除了模型，還需要一個自訂節點

**`ComfyUI-GGUF`** —— 沒有它，`UnetLoaderGGUF` 節點不存在，
`imggen qwen`、`imggen flux`、`vidgen i2v`（14B，也就是**預設的影片路徑**）三者全部無法運作。

```
cd <COMFY_HOME>/ComfyUI/custom_nodes      # Linux: <COMFY_HOME>/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
```

`doctor.py` 會把它缺席列為 **FAIL**，不是警告。

### 選配：Viggle Turbo（`qwen21` 系列加速）

`gen.py` 對 `qwen21` / `qwen21_edit` / `qwen21_move` **預設用 turbo**（6 步，快 2.5–3 倍，畫質實測看不出差別），
需要兩個檔案，都來自 [`Viggle/Qwen-Image-2.1-viggle-turbo`](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)：

| 檔案（repo 內路徑） | 放到哪裡 |
|---|---|
| `comfyui/viggle_turbo.py` | `<ComfyUI>/custom_nodes/viggle_turbo.py` |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | `models/loras/` |

放好後重啟 ComfyUI。**這包刻意不附這兩個檔案**：它們是 Qwen 研究授權（非商用）的衍生物，
由使用者自行下載、自行接受授權。缺任一個時 `gen.py` 會自動改用 base 取樣器，
並在輸出 JSON 的 `turbo` 欄位寫明原因；`doctor.py` 也會列為警告。

> 用 `v0.2.1` 是因為整套流程是用它驗證的。repo 裡已有較新的 `v0.3`，換版要改 `gen.py` 的 `TURBO_LORA` 並重新驗證。

MiniMax-H3 的節點（`MiniMaxH3ReferenceToVideo`、`MiniMaxH3SigmaShift`、`VAEDecodeAudio`）
是 **ComfyUI 核心內建**，不需要額外安裝，但需要夠新的版本
（參考機為 **ComfyUI 0.31.1**；太舊的版本沒有這些節點）。
