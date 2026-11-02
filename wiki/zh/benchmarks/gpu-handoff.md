---
title: 把 GPU 讓給其他程式：三種切換方式
summary: 255.5K token 的對話下，Strata sleep/wake 從拿回 GPU 到答出第一句約 1.8 秒，Strata 卸載加 slot 還原約 25.9 秒，llama.cpp 卸載加 slot 還原約 32.7 秒；三者都完整接回對話，差別在讓出期間 sleep 仍佔約 63 GB RAM。
updatedAt: 2026-10-29
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

## 穩定性

載入 50 GB 專家權重時，systemd-oomd（memory pressure 69.8% > 50% 持續 20 秒）把桌面登出；
改成 unbuffered 並關掉 arena THP 之後壓力降到 0.2%，載入速度從 6.40 提升到 9.1–10.0 GiB/s。
這已經是 AST 的預設值，見 [Strata](../engines/strata.md)。

## 相關

- [Strata](../engines/strata.md)
- [llama.cpp 引擎](../engines/llama-server.md)
- [vLLM 的 sleep／wake](../engines/vllm.md)
