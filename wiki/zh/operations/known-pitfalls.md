---
title: 跑了才發現的坑
summary: 讀程式碼看不出來、跑起來才知道的事；每一條都標明是實測還是事故紀錄。
updatedAt: 2026-10-29
---

# 跑了才發現的坑

> **收錄標準：讀程式碼看不出來的事。** 每一條標明驗證程度：
> **實測**＝有量測數字；**事故**＝真實發生過、記錄了原因，但沒有保留量測數字。

## 順序錯了不會報錯

最危險的一類：沒有任何錯誤訊息。後端已經把這幾條固定下來，改程式碼時要知道。

| 坑 | 後果 |
|---|---|
| 設定 `spawn` 之前就 import 了碰 CUDA 的模組 | `fork` 複製已初始化的 CUDA context（未定義行為） |
| 在讀設定之前 import transformers | `HF_HOME` 還沒生效，模型快取落到別的地方 |
| 先載入模型才建立 DeepSpeed 設定 | ZeRO-3 在載入期的分片靜默放棄 |
| 先載入模型才開圖片訓練模式 | 多模態模型沒有 vision tower |

## 硬體與長時間執行

| 坑 | 細節 | 驗證 |
|---|---|---|
| **顯卡同時驅動螢幕會中止長時間訓練** | RTX 5060 Ti 同時驅動桌面，ZeRO-3 訓練約 61 分鐘被 `cudaErrorLaunchTimeout` 中止。API 沒有 resume | 實測 |
| **省略 `device_map` 落在 CPU** | 不是 GPU | 實測 |
| **`device_map="none"` 拋 `ValueError`** | 雖然 `GET /config/offload_types` 列出了它 | 實測 |
| **多模態 full fine-tune 在 16 GB 放不下** | backward 時 OOM；DRAM 夠也沒用 | 實測 |
| **載入 50 GiB 的 Strata 模型讓桌面被登出** | page cache 讓 memory pressure 停在 oomd 的 50% 門檻之上（69.8%）。預設已改成 unbuffered，尖峰 0.2% | 實測 |
| **NVMe offload 的 DRAM 用量是真的** | swap buffer + fp32 staging，不是 page cache | 實測 |

## 量測陷阱

| 陷阱 | 說明 |
|---|---|
| **在 AST 之外量 llama-server，decode 慢約 16%** | AST 的環境剛好用到 PyTorch 附的 libgomp。單獨量時設 `OMP_WAIT_POLICY=active`；做 A/B 時兩邊要一致 |
| **ZFS 上的「冷讀」其實是 page cache** | 一次讀到 22 GB/s 的量測其實是 ARC 快取。冷讀要用 `mincore()` 驗證 |
| **`posix_fadvise(DONTNEED)` 趕不走已 mmap 的頁** | 只有真實的記憶體壓力能回收 |
| **在 Linux 上把 `/dev/shm` 的 KV 層算兩次** | 同一行程映射兩次；一份 vLLM 的 DRAM 用量曾被高估為 47.6 GB，實際約 30 GB |

## 設定存在、讀取不存在

| 設定 | 問題 | 現況 |
|---|---|---|
| `HF_HUB_DISABLE_XET` 在設定檔裡設 | `huggingface_hub` 只在 import 時讀一次 | 下載模組直接改 hub 的常數 |
| `DEFAULT_MAX_NEW_TOKENS` | 沒有任何地方讀它 | 改成 `MAX_OUTPUT_TOKENS_CAP`，預設不設 |

## 相依版本不能順手升級

| 套件 | 版本 | 為什麼 |
|---|---|---|
| `deepspeed` | 0.19.2 | 0.19.3 起 LoRA 下凍結權重不釋放。見 [實測](../benchmarks/deepspeed-lora-16gb.md) |
| `transformers`、`trl` | 刻意不追新 | 新版改 layer 名稱，依名稱的解析會靜默失效 |
| CUDA 編譯工具鏈 | 同一個 CUDA minor 一起釘 | 只釘 `nvcc` 不夠：`cicc` 與 headers 來自另外兩個套件、沒有版本下界，組錯會得到 `ptxas fatal: Unsupported .version`。**只裝了驅動、沒有系統 CUDA toolkit 的機器**（大多數）才會中，而且是下載好幾 GB 之後才失敗 |

## 部署

| 坑 | 細節 |
|---|---|
| 代理的逾時要跟上後端 | 後端的生成逾時是 1800 秒，而串流時等第一個 byte 就是 prefill |
| SSE 必須關代理緩衝 | 否則訓練 log 不即時 |
| 沒有 `.env` 時 `.env.example` 就是設定 | 服務找 `.env`，找不到就讀 `.env.example` |

## 相關

- [從症狀反查](troubleshooting.md)
- [放在反向代理後面](reverse-proxy.md)
