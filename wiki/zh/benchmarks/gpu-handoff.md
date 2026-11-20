---
title: 把 GPU 讓給其他程式：三種切換方式
summary: 255.5K token 的對話下，Strata sleep/wake 從拿回 GPU 到答出第一句約 1.8 秒，Strata 卸載加 slot 還原約 25.9 秒，llama.cpp 卸載加 slot 還原約 32.7 秒；三者都完整接回對話，差別在讓出期間 sleep 仍佔約 63 GB RAM。
updatedAt: 2026-11-20
measuredAt: 2026-10-20
---

# 把 GPU 讓給其他程式：三種切換方式

情境：同一張顯卡要輪流給 LLM 與影像生成（例如 ComfyUI）用，而 LLM 的對話很長、不想重算。

## 條件

| 項目 | 值 |
|---|---|
| 硬體 | RTX 5060 Ti 16 GB、125 GB DRAM、NVMe SSD、Linux |
| 模型 | Qwen3.8-Flash-Next 衍生模型的 IQ3_XXS 社群量化版（512 個專家，專家權重 49.8 GiB）+ mmproj F16 |
| context | 262,144，KV 8-bit；對話開頭帶一張 1024×1024 的圖，共 255.5K token |
| Strata | deep sleep build、GPU 開滿、MTP 開、unbuffered 載入 |
| llama.cpp | build 11348，`-ncmoe 48`，不開 MTP |

這一輪**沒有**在讓出與拿回之間真的執行影像生成。

## 切換結果

| 階段 | Strata sleep/wake | Strata 卸載 + slot 還原 | llama.cpp 卸載 + slot 還原 |
|---|---|---|---|
| 讓出 GPU | 1.96 s | 4.3 s | 4.7 s |
| 拿回 GPU | wake 0.71 s | 載入 22.1 s + 還原 3.48 s | 載入 30.4 s + 還原 0.93 s |
| **拿回到答出第一句** | **約 1.8 s** | 約 25.9 s | 約 32.7 s |
| 讓出期間 LLM 佔的 RAM | 63.1 GB（含 5.0 GB pinned 備份） | 0.25 GB | 0 |
| SSD | 不需要 | slot 檔 3.9 GB | slot 檔 4.1 GB |

**怎麼選**：DRAM 充足（建議 80 GB 以上）用 sleep/wake；64 GB 以下用卸載 + slot 還原。

## 醒著時的推論

| | Strata | llama.cpp |
|---|---|---|
| 讀完 255.5K（分段加總） | 177.8 s | 約 589 s |
| 255K 時的 decode | 58.1 tok/s | 11.7–11.9 tok/s |

256K 時 llama.cpp 的 KV 佔掉 GPU，只剩約 1 層專家放得下。

> llama.cpp 的 slot 還原有使用限制：還原後必須接著原本的對話往後加。
> 重問同一句要倒回 recurrent 狀態，而 slot 檔沒有 checkpoint，只能整段重算（約 10 分鐘）。

## 其他驗證

- 三種切換之後，圖片問題都答對——255K 對話最前面的那張圖仍記得。這是功能驗證，不是視覺品質評測。
- OCR 與區塊定位：乾淨、小字、翻拍、表格幾乎全對，IoU 平均約 0.9；
  `min_tokens 1024` 之後 8px 的字也全對，代價是每張圖多 0.4–0.7 秒。
- 影像生成這一側：同長度的影片，可用 VRAM 從約 15.5 GB 降到約 8 GB 只慢 3–6%；
  VRAM 決定的是能算多長（720p 單段 81 幀 → 65 幀），不是算多快。

## Strata 的 SSD 備份 sleep（2026-11）

上面的結果是 16 GB 的卡。當用掉的 GPU 記憶體是 16 到 46 GB，而 sleep 的備份寫到 SSD（`--sleep-backup file`）、wake 時讀回來，會怎樣？

| 項目 | 值 |
|---|---|
| 硬體 | 一張 48 GB 的 GPU、20 核工作站等級 CPU、503 GiB DRAM；備份放在 PCIe 3.0 x4 的 NVMe SSD，模型放在 PCIe 5.0 x4 的；Ubuntu |
| 引擎 | Strata 0.1.52（v0.2.7 加 sleep 的 patch）；直接起，沒有經 AST |
| 模型 | Qwen3.8-Flash-Next Q8_0、開 MTP、256K context、int8 KV、vision 關 |
| 方法 | 用 `--vram-reserve-mib` 調整用掉的 GPU 記憶體；每次啟動：sleep 後直接 wake（「hot」），再兩次 sleep、強制把備份寫到磁碟並清出 RAM、wake（「ssd」）。每次 wake 後同一段對話接著問：30 次全部答對 |

**預設的備份裡有什麼。** 專家快取**不在**裡面：sleep 時丟掉，wake 時從專家的 RAM 副本或模型檔重填。其他 1 MiB 以上的都在
（KV、prompt cache、MTP、緩衝區、非專家的權重）。45.8 GB 的卡裡，專家快取 36.75 GiB，備份 7.07 GiB。
`STRATA_SLEEP_KEEP_CACHE=1` 會連專家也備份。

**預設（專家快取從模型檔重填）：**

| 用掉的 GPU 記憶體 | 備份 | 重填的專家 | sleep＋flush | wake | 其中重填 |
|---|---|---|---|---|---|
| 16 GB | 7.07 GiB | 7.6 GiB | 7.3 s | 6.1 s | 2.8 s |
| 32 GB | 7.07 GiB | 23.2 GiB | 7.3～7.4 s | 11.8～12.0 s | 8.5 s |
| 46 GB | 7.07 GiB | 36.7 GiB | 7.1～8.0 s | 16.6～17.1 s | 13.3～13.6 s |

**連專家也備份（`STRATA_SLEEP_KEEP_CACHE=1`），備份放 Gen3 的碟：**

| 用掉的 GPU 記憶體 | 備份 | sleep（HTTP） | ＋flush | wake（從 SSD） | wake（頁還在 RAM） |
|---|---|---|---|---|---|
| 16.0 GB | 14.7 GiB | 10.1 s | 3.6～4.8 s | 5.9 s | 2.9 s |
| 24.0 GB | 22.5 GiB | 15.2～16.4 s | 5.7～7.4 s | 8.4～8.5 s | 4.3 s |
| 32.0 GB | 30.3 GiB | 20.6～21.0 s | 8.0～8.8 s | 10.8～11.2 s | 6.1 s |
| 40.0 GB | 38.1 GiB | 27.3～27.6 s | 8.4～8.6 s | 13.4 s | 12.3 s（只有一次，原因沒查） |
| 45.8 GB | 43.8 GiB | 34.0～34.7 s | 4.9～8.6 s | 15.3～15.9 s | 8.3 s |

- **wake 隨用掉的記憶體大致線性增加，兩種方式都是每 GiB 約 0.34～0.36 s。** 16 GB 的卡約 6 s 醒來，用滿的 48 GB 卡約 15.5～17 s。
- **兩種方式的差別在 sleep。** 預設只寫 7 GiB，約 7 s。連專家一起備份要寫整張卡：46 GB 約 40 s。把 VRAM 拷進映射檔只有約 1.3 GB/s。
- **換更快的碟幫助有限。** 換成 PCIe 5.0 的碟（16 GB 的情況），flush 是 1.6 s（原本 3.6～4.8 s），但 wake 只快 11%（5.24 s 對 5.9 s）：
  不論哪種碟，讀回來都約 3 GB/s，瓶頸在讀回的路徑，不在磁碟。

沒有涵蓋：經 AST、開 vision、模型檔還在 page cache 時（`STRATA_UNBUFFERED_LOAD=0`）、或 v0.2.11 的引擎（sleep 的程式碼相同）。

## 穩定性

載入 50 GB 專家權重時，systemd-oomd（memory pressure 69.8% > 50% 持續 20 秒）把桌面登出；
改成 unbuffered 並關掉 arena THP 之後壓力降到 0.2%，載入速度從 6.40 提升到 9.1–10.0 GiB/s。
這已經是 AST 的預設值，見 [Strata](../engines/strata.md)。

## 相關

- [Strata](../engines/strata.md)
- [llama.cpp 引擎](../engines/llama-server.md)
- [vLLM 的 sleep／wake](../engines/vllm.md)
