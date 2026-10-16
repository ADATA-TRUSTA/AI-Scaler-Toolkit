# 維護與常見問題

[← 回到 README](../README.zh-TW.md)

## 保持最新

**應用程式**：AST 的 Settings → Updates → **Check now**，再按 **Update now**。見[安裝 → 更新](installation.md#更新)。

**原始碼**：`git pull` 之後再跑一次 `setup_env`，讓環境跟上新的相依套件。它只會裝有變動的部分：

```bash
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh      # Intel 用 TRUSTA_ACCEL=xpu
```

```powershell
.\backend\scripts\windows\setup_env.ps1 -Accel cuda            # Intel 用 -Accel xpu
```

---

## 常見問題

### Q1. 第一次啟動要等很久

它要下載並安裝約 6-10 GB：對應 GPU 的 Python 套件與 llama.cpp 執行環境（Linux 上還要編譯 llama.cpp）。5-15 分鐘是正常的，網路慢會更久。視窗會顯示每個步驟。如果中途關掉視窗或重開機，再開一次應用程式即可：它會顯示「Resuming setup」，已經完成的部分幾秒就跳過。

### Q2. 設定失敗

視窗會顯示錯誤。完整的輸出在 log 資料夾的 `provision.log`（Windows `%LOCALAPPDATA%\AIScalerToolkit\logs\`，Linux `~/.local/share/ai-scaler-toolkit/logs/`）。大部分失敗是網路問題：再開一次應用程式就會繼續。如果每次都卡在同一步，用 Settings → Environment → **Reinstall the environment**。

### Q3. Linux 上雙擊 AppImage 沒有反應

把它設成可執行：`chmod +x AIScalerToolkit-*-x86_64.AppImage`，或在檔案內容勾選「允許以程式執行檔案」。如果接著出現 `fusermount: not found`，安裝 FUSE 3：`sudo apt install fuse3`。

### Q4. Linux 上為什麼要輸入密碼？

只有第一次啟動會問一次，用來安裝需要的系統套件：畫視窗用的 WebKitGTK、針對你的 CPU 編譯 llama.cpp 用的 `build-essential` 與 `cmake`，有 Intel GPU 時還有它的執行環境套件。它用的是桌面系統自己的密碼視窗。沒有桌面的機器，請自己先裝好，再用 `--headless`（見[安裝](installation.md#沒有桌面linux-伺服器)）。

### Q5. Linux 上沒有用到 Intel GPU

請用 Ubuntu 24.04 以上：22.04 的套件庫沒有 Level Zero。較新的 Intel GPU（Lunar Lake、Panther Lake、Arc B 系列）還需要 Intel 的顯示套件來源，因為 Ubuntu 套件庫裡的驅動太舊、看不到它。加入後再開一次應用程式：

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
```

### Q6. 沒有用到 NVIDIA GPU，或載入時出現 CUDA 錯誤

NVIDIA 版需要 compute capability 7.5 以上的顯卡（RTX 20／GTX 16 系列以後）與 580 以上的驅動。用 `nvidia-smi` 確認：最上面一行會顯示驅動版本與 `CUDA Version: 13.x`。到 NVIDIA 官網更新驅動後，再開一次應用程式。

### Q7. 找不到 vLLM 或微調

兩者都只在 Linux 上。有 NVIDIA GPU 時會裝 vLLM；Intel GPU 預設不裝。微調在 Linux 上的 NVIDIA 或 Intel 都可以。Windows 上的引擎是 Transformers 與 llama.cpp。有 NVIDIA 顯卡的話，還是可以在 Windows 上用：在 WSL2（Ubuntu 發行版）裡執行 Linux 版，例如用 `--headless` 執行 AppImage（見[安裝](installation.md#沒有桌面linux-伺服器)）。

### Q8. 服務起不來：port 被佔用

別的程式正在用 `8571`。到 Settings → Ports → Application port 改掉，或關掉那個程式。已經連到這裡的應用程式也要改成新的 port。

### Q9. 「Config AST Endpoint」是灰的，或按了 Hermes 沒變

它需要已經載入的模型：先載入一個，再按一次。如果 Hermes 已經設成你自己選的其他 provider，它會刻意不動，並顯示手動切換的指令。結果也會寫進 `launcher.log`。

### Q10. 別台電腦連不到這台

在 Settings → Network 開啟 **Let another computer use this machine's model**，那裡會顯示要用的位址和 API key；另一台電腦要帶上 key（`Authorization: Bearer <key>`），否則會得到 `401`。得到 `403` 表示它要的不是 API，例如網頁介面，那只有這台電腦能開。完全連不上的話，通常是防火牆：那裡的 **Cannot connect?** 會提供新增規則的按鈕（Windows 會要求系統管理員同意）。拿到 key 的人都能使用模型，只在你信任的網路上開啟。

### Q11. 為什麼筆電顯示 8 條 4 GB 的記憶體？

這是它記憶體的構造。記憶體焊在主機板上的筆電（LPDDR5），每個記憶體通道會各自回報成一個裝置，所以 32 GB 會顯示成 8 × 4 GB。Windows 也是這樣回報。

### Q12. 模型放在哪裡？解除安裝會刪掉嗎？

下載的模型在 `~/.cache/huggingface`（Windows：`%USERPROFILE%\.cache\huggingface`），是 Hugging Face 的標準快取位置，和其他 Hugging Face 工具共用。Windows 解除安裝時會問要不要刪；Linux 上要自己刪才會刪。重新安裝後檔案還在，但模型清單會重新開始：從 Models 頁面重新登錄即可（會從快取找到，不用重新下載）。入門模型會自己重新登錄。

### Q13. 我只需要 API，不需要視窗

API 在 `http://127.0.0.1:8571`，說明在 `/docs`。見[使用 → 連接其他應用程式](usage.md#連接其他應用程式)。Linux 伺服器上可以用 `--headless` 執行 AppImage。

### Q14. 同時有 NVIDIA 和 Intel 顯卡，用的是哪一張？

NVIDIA。設定時會先找 NVIDIA 顯卡，找到就裝 CUDA 版，所以有 NVIDIA 顯卡加 Intel 內顯（或 Arc 顯卡）的電腦會用 NVIDIA。要改用 Intel：替你的帳號設定環境變數 `TRUSTA_ACCEL=xpu`，打開應用程式，再用 Settings → Environment → **Reinstall the environment**。ComfyUI 另外選：在 Settings → Components 選它要用的顯示卡。

### Q15. 用 DeepSpeed 做 fine-tune 時出現 offload / buffer / swapper 類錯誤怎麼辦？

本專案內建的 DeepSpeed profile 位於：

- `backend/service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `backend/service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `backend/service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `backend/service/configs/deepspeed/zero3_offload_disk_disk.json`

若錯誤訊息提到 `buffer`、`swapper`、NVMe offload、aio / offload queue 類問題，最常見的第一步是調高 `zero_optimization.offload_param.buffer_count`。

例如：

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 6
}
```

若仍不穩，再逐步調到 `8`，不要一次改太大。

如果 `offload_optimizer.device` 也是 `nvme`，也可以一併調整 `zero_optimization.offload_optimizer.buffer_count`。

> 實務上若看到類似「buffer 0 / swapper 1」錯誤，通常先從 `offload_param.buffer_count` 開始調最有效。

#### DeepSpeed 常見可調參數

| 參數 | 位置 | 何時調整 | 常見效果 / 取捨 |
|------|------|----------|------------------|
| `buffer_count` | `zero_optimization.offload_param.buffer_count` | NVMe/CPU offload 過程出現 buffer、swapper、queue 類錯誤 | 增加可用緩衝數，通常可改善 offload 穩定性，但會多吃記憶體 / 主記憶體 |
| `buffer_count` | `zero_optimization.offload_optimizer.buffer_count` | optimizer offload 到 NVMe 時出現類似錯誤 | 與上面類似，但作用在 optimizer state |
| `buffer_size` | `zero_optimization.offload_param.buffer_size` | 磁碟 I/O 過碎、吞吐不穩、頻繁 offload | 調大可減少切片次數，但會增加單次緩衝占用 |
| `max_in_cpu` | `zero_optimization.offload_param.max_in_cpu` | NVMe 太慢、希望多保留一部分參數在 RAM | 提高可減少磁碟讀寫，但會增加系統 RAM 使用 |
| `pin_memory` | `zero_optimization.offload_param.pin_memory` / `zero_optimization.offload_optimizer.pin_memory` | CPU RAM 夠、希望改善 CPU↔GPU 傳輸 | 可能提升傳輸效率，但也可能增加 RAM 壓力；記憶體吃緊時可關掉 |
| `stage3_prefetch_bucket_size` | `zero_optimization.stage3_prefetch_bucket_size` | ZeRO-3 prefetch 過大導致顯存 / RAM 壓力，或 I/O 不順 | 調小較保守、較穩；調大可能提升吞吐 |
| `reduce_bucket_size` | `zero_optimization.reduce_bucket_size` | 通訊或聚合過程吃太多顯存 | 調小可降瞬時記憶體壓力，但可能影響速度 |
| `stage3_param_persistence_threshold` | `zero_optimization.stage3_param_persistence_threshold` | 參數頻繁搬移造成效能不穩 | 調整可改變常駐策略，需配合模型大小實測 |
| `sub_group_size` | `zero_optimization.sub_group_size` | 超大模型分組處理時記憶體或效能不理想 | 調小較保守，調大可能提升效率 |
| `contiguous_gradients` | `zero_optimization.contiguous_gradients` | 梯度碎片化或記憶體配置問題 | 通常維持 `true`；若遇特殊相容性問題可測試切換 |
| `overlap_comm` | `zero_optimization.overlap_comm` | 通訊重疊造成不穩、記憶體尖峰、效能異常 | 關閉可提升穩定性，但可能變慢 |

#### 不只 DeepSpeed config，也常一起調整的訓練參數

若單改 DeepSpeed 仍不穩，通常也要一起降低訓練壓力：

| 參數 | 位置 | 建議方向 |
|------|------|----------|
| `per_device_train_batch_size` | 訓練請求 body | 顯存不足時先往下調 |
| `gradient_accumulation_steps` | 訓練請求 body 或 DeepSpeed config | batch size 降低後，可往上補回有效 batch |
| `max_seq_length` | 訓練請求 body | 序列太長時先降，例如 `4096 -> 2048` |
| `gradient_checkpointing` | 訓練請求 body | 記憶體不足時建議保持啟用 |

#### 建議調整順序

1. 先確認使用的是哪個 profile（CPU offload / NVMe offload）。
2. 若錯誤含 `buffer` / `swapper`，先調 `offload_param.buffer_count`。
3. 若 NVMe offload 仍不穩，再調 `offload_optimizer.buffer_count`、`buffer_size`、`max_in_cpu`。
4. 若是顯存 / RAM 不足，再降低 `per_device_train_batch_size`、`max_seq_length`。
5. 若是吞吐或穩定性問題，再微調 `stage3_prefetch_bucket_size`、`reduce_bucket_size`、`overlap_comm`。

若使用 NVMe offload，另外也建議確認：

- `nvme_path` 所在磁碟空間足夠
- 盡量使用 SSD / NVMe，不要放在慢速 HDD
- 避免與其他高 I/O 工作共用同一顆磁碟
