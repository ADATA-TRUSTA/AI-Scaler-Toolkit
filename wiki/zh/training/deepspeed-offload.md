---
title: DeepSpeed ZeRO-3 offload
summary: 五個 offload profile、NVMe 路徑與 pinned DRAM 上限、存檔的兩個陷阱，以及 DeepSpeed 為什麼釘在 0.19.2。
updatedAt: 2026-10-29
---

# DeepSpeed ZeRO-3 offload

四種 offload 的對照見 [Offload](../memory/offload.md)；本頁只講訓練這一種。

## 五個 profile

`backend/service/configs/deepspeed/`，命名是 `<參數放哪>_<optimizer 放哪>`：

| profile | 參數 | optimizer |
|---|---|---|
| `zero3_offload_cpu_cpu` | CPU | CPU |
| `zero3_offload_cpu_cpu_tight` | CPU | CPU（buffer 更小、更省） |
| `zero3_offload_cpu_disk` | CPU | NVMe |
| `zero3_offload_disk_cpu` | NVMe | CPU |
| `zero3_offload_disk_disk` | NVMe | NVMe |

用 `deepspeed_profile` 選，或用 `deepspeed_config` 給自己的檔案。

## NVMe 放哪：`AUTO`

NVMe profile 裡的路徑寫的是 `AUTO`，執行時由 `DEEPSPEED_NVME_DIR` 決定
（預設是專案底下的 `.deepspeed_offload`）。每個 job 的 `offload_folder` 優先。
建議指到一顆快的 NVMe。

## pinned DRAM 的上限

`DEEPSPEED_NVME_PINNED_BUDGET_GB`，預設 24。

swap buffer 每一個都要容納**最大的單一參數**，而 DeepSpeed 不管 `pin_memory` 設什麼都會 pin 它們；
沒有上限時，依模型算出來可能是數十 GB。後端透過**減少 buffer 數量**守住上限（buffer 不能比最大參數小）。

> **NVMe offload 吃掉大量 DRAM 不是洩漏。** 實測確認那是真的 RSS（不是 page cache）：
> swap buffer 加 fp32 staging。要調的是這個預算。

## 存檔的兩個陷阱（後端已處理）

| 陷阱 | 處理 |
|---|---|
| DeepSpeed + LoRA：`gather=True` OOM、`gather=False` `FileExists` | 偵測到 PEFT 時只存 adapter |
| `Cannot partition a param in flight`：ZeRO-3 預取下一批參數時在訓練迴圈中途存檔 | 存檔前先把預取中的參數排空 |

## 版本釘選：不要順手升級

| 套件 | 版本 | 為什麼 |
|---|---|---|
| `deepspeed` | **0.19.2** | 0.19.3 起，LoRA + gradient checkpointing + ZeRO-3 offload 下**凍結權重不會從 GPU 釋放**。Qwen3.5-35B-A3B 在 16 GB 卡上從 peak 3.08 GiB 變成第一步 OOM。見 [實測](../benchmarks/deepspeed-lora-16gb.md) |
| `transformers`、`trl` | 刻意不追新 | 新版改了 layer 名稱，依名稱的解析會**靜默**找不到 |

## 長時間訓練

| 風險 | 對策 |
|---|---|
| 顯卡同時驅動螢幕時，長時間 kernel 會被驅動中止（實測約 61 分鐘，`cudaErrorLaunchTimeout`） | `save_steps` 設小；或把螢幕接到另一張卡或內顯 |
| API 沒有 resume | 只能從最後一個 checkpoint 重新開始 |

## 相關

- [Offload](../memory/offload.md)
- [微調總覽](index.md)
- [16 GB 顯卡上用 LoRA 訓練 35B MoE](../benchmarks/deepspeed-lora-16gb.md)
