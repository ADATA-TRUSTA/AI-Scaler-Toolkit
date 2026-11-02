---
title: transformers 引擎
summary: 唯一在 worker 內生成、也唯一不需要任何轉換的引擎；適合微調後立刻驗證。附 device_map 與量化的注意事項。
updatedAt: 2026-10-29
---

# transformers 引擎

## 它的價值：不需要轉換

| 引擎 | 微調產物要怎麼處理 |
|---|---|
| **transformers** | **LoRA／QLoRA adapter 給路徑就能載**（base 與 adapter 分開給） |
| llama.cpp | 先 merge，再轉 GGUF |
| vLLM | 先 merge |

所以它最適合**微調完立刻驗證**。要確認微調的效果，用這個引擎載 f16 的 adapter 最乾淨：
不經過量化與轉換兩層損失。

代價是併發：一次一個 forward pass，沒有 batching。

## `device_map`：唯一的放置控制

由 `accelerate` 執行。

| 值 | 結果 |
|---|---|
| `cuda:0` | 全部放 GPU |
| `auto` | 交給 accelerate 分配 GPU／CPU |
| `cpu` | 全部放 CPU |
| **省略不送** | **落在 `cpu`**，不是 GPU |
| `none` | ⚠️ 會被原樣傳給 `from_pretrained` 而失敗（`ValueError`） |

> ⚠️ `GET /config/offload_types` 會列出 `none`（「Keep all in GPU」），但在 0.3.0 送出這個值會載入失敗。
> **要全部放 GPU 請用 `cuda:0`**；要自動分配用 `auto`。

### 量化時才會「聰明推導」

三個條件同時成立才會推導放置：**需要量化**（`int4`／`int8`）、`device_map == "auto"`、**有 `max_memory`**。
推導時先在 `meta` device 上建一次模型量結構，再依量化後縮小的預算分配。
不量化時直接用 accelerate 的 `"auto"`。

### Intel GPU 要先載到 CPU 再搬

`accelerate` 不認得 `"xpu:N"`，所以目標是 XPU 時會先載到 CPU，載完再搬過去。

## 載入後回報的放置資訊

| 欄位 | 內容 |
|---|---|
| `device` | 主要裝置 |
| `device_map_summary` | 例如 `cuda:0:30, cpu:10` |
| `layer_lines` | 抽樣的層放置 |
| `memory_usage` | GPU 用量 |

## 沒有 `reasoning_content`

server engine 的上游伺服器會把思考過程分到 `reasoning_content`；
這條路徑沒有伺服器端的 reasoning parser，所以**不會出現這個欄位**。這是路徑差異，不是 bug。

## 記憶體清理

| 方法 | 做什麼 |
|---|---|
| `POST /inference/cleanup_generation_memory` | 清生成期間的暫存 |
| 卸載 | **結束整個 worker 行程**——VRAM 一定收得回來 |
| `POST /inference/force_cleanup_gpu` | worker 異常後的強制復原 |

## 什麼時候選它

| 情境 | 選它？ |
|---|---|
| 剛微調完，想立刻試 adapter | ✅ 最好的用途 |
| 單人、低併發、HF 模型 | ✅ |
| Windows | ✅ |
| 要服務多個使用者 | ❌ 沒有 batching |
| 模型比顯卡大 | ❌ 用 llama.cpp、vLLM 的專家 offload 或 Strata |

## 相關

- [推論引擎怎麼選](index.md)
- [Offload](../memory/offload.md)
