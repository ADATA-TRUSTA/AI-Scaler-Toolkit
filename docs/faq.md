# 維護與常見問題

[← 回到 README](../README.zh-TW.md)

## 維護更新

每次從 GitHub 拉新版本後，建議重新同步依賴。

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash scripts/linux/setup_env.sh
```

若改用 CUDA：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash scripts/linux/setup_env.sh
```

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\scripts\windows\setup_env.ps1 -Accel xpu
```

若改用 CUDA：

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\scripts\windows\setup_env.ps1 -Accel cuda
```

---

## 常見問題

### Q1. 執行啟動腳本時出現找不到 `.venv`

代表尚未建立 Python 環境，請先執行：

- Linux：`TRUSTA_ACCEL=xpu bash scripts/linux/setup_env.sh`
- Windows：`.\scripts\windows\setup_env.ps1 -Accel xpu`

### Q2. 新機器下載後，模型快取還指到舊路徑

請優先檢查 `.env`：

```dotenv
HF_HOME=<你的快取路徑>
TIKTOKEN_RS_CACHE_DIR=<你的專案根目錄>
```

### Q3. API 已經啟動，但首頁沒有前端畫面

請先確認專案內的前端資料夾已有檔案：

- `src/frontend/dist/index.html`
- `src/frontend/dist/assets/`

若 `dist` 內已有檔案，直接重新開啟 `http://127.0.0.1:8000/` 即可。

### Q4. 只需要後端 API，不需要前端

直接使用以下路徑即可：

- `/docs`
- `/health`
- `/v1/chat/completions`

### Q5. Windows 上 `setup_env.ps1` 執行失敗

先確認：

- 已安裝 Python 3.12+
- 已安裝 `uv`
- 若使用 XPU，已安裝 Intel oneAPI

必要時可先調整 PowerShell 執行原則：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q6. Windows 上 `-Accel cuda` 建環境失敗

請確認已安裝 CUDA Toolkit，且驅動版本相容。若不是 NVIDIA GPU，請改用：

```powershell
.\scripts\windows\setup_env.ps1 -Accel xpu
```

### Q7. 用 DeepSpeed 做 fine-tune 時出現 offload / buffer / swapper 類錯誤怎麼辦？

本專案內建的 DeepSpeed profile 位於：

- `src/service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `src/service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `src/service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `src/service/configs/deepspeed/zero3_offload_disk_disk.json`

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
