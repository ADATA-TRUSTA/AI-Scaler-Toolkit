---
title: Strata 與 vLLM 在一張 48 GB 卡上，專家放 DRAM
summary: 同一台機器、同樣的 prompt、173 GB 的 MoE、專家放 DRAM：四人各讀 4K，Strata 23 秒完成、vLLM 150 秒；四個 32K 是 54 秒對 504 秒；有 KV cache 的 4K 追問是 2.3 秒對 16～19 秒。
updatedAt: 2026-11-20
measuredAt: 2026-11-19 to 2026-11-20
---

# Strata 與 vLLM 在一張 48 GB 卡上，專家放 DRAM

情境：一個遠大於 GPU 的 MoE，用一張卡服務多人，專家與 PLE 表（每個 token 要查的大型 embedding 表）都放在 DRAM。

## 條件

| 項目 | 值 |
|---|---|
| GPU | 一張 48 GB VRAM，PCIe Gen4 x16 |
| CPU、RAM | 20 核工作站等級 CPU（有 AMX）、503 GiB（與第二張 GPU 上載入的另一個模型共用） |
| 模型 | Qwen3.8-Flash-Next。Strata：**Q8_0** GGUF。vLLM：官方 **FP8**（173 GB）。兩者都是 8 bit，但不是同一份權重；vLLM 0.32 沒有 GGUF loader |
| Strata | `--resident-budget-gib 125`：專家在 RAM 有 95 GiB 的副本、PLE 表鎖在 RAM、GPU 上有一份熱專家的快取（36.5 GiB）；開 MTP。「AMX」＝CPU 分擔一部分專家的運算，「upstream」＝沒有 |
| vLLM | 0.32.0；`--cpu-offload-gb 120 --cpu-offload-params experts`（110 GiB 的專家放 DRAM，GPU 經 UVA 讀）；PLE 表放 pinned host memory（`cpu_offload`）；GPU 上是 11.9 GiB 權重與 29.5 GiB KV；vision 關 |
| 請求 | 多人測試 `max_tokens` 256，兩邊用同樣的 prompt。「batch」＝四個請求一起送；「queued」＝一個接一個 |

## 單人

| | Strata（AMX 關／開） | vLLM |
|---|---|---|
| 4K prompt 的首字 | 4.1 s／2.0 s（1,012／2,148 tok/s） | 13.6 s（302 tok/s） |
| 8K prompt 的首字 | 沒量 | 24.0 s（315 tok/s） |
| decode | 沒有單獨量 | 約 9.5 tok/s（16 個 token） |

## 四人各 4K（首字／完成，3 輪平均）

| | 第 1 個 | 第 2 個 | 第 3 個 | 第 4 個 |
|---|---|---|---|---|
| Strata，batch，AMX | 2.3／23.1 s | 6.3／23.1 s | 10.1／23.2 s | 13.9／23.1 s |
| Strata，batch，upstream | 4.1／26.0 s | 8.3／25.9 s | 12.5／26.1 s | 16.7／26.2 s |
| Strata，queued，AMX | 2.1／6.1 s | 8.6／12.3 s | 14.3／18.2 s | 20.2／24.0 s |
| vLLM | 16.7／149.1 s | 30.4／150.0 s | 44.0／150.6 s | 54.6／150.9 s |

## 四人各 32K

| | 第 1 個 | 第 2 個 | 第 3 個 | 第 4 個 |
|---|---|---|---|---|
| Strata，batch（2 輪） | 9.3／21.9 s | 22.6／39.9 s | 35.9／49.1 s | 49.3／53.6 s |
| vLLM（1 輪） | 107.7／491.9 s | 212.3／498.0 s | 316.7／502.2 s | 418.2／504.4 s |

32K 不走 AMX（見 [用 AMX 讀 prompt](strata-amx-prefill.md)），所以這裡兩個 Strata build 相同。vLLM 讀一個 32K 的 prompt 約 105 s（約 310 tok/s）。

## 有 KV cache 的多輪對話

第一輪送一份文件，之後每輪加新內容。首字時間；Strata 是 3 段對話 × 3 次追問的中位數，vLLM 是 1 段對話 × 3 次追問。

| 歷史 + 新增 | Strata，AMX | Strata，upstream | vLLM |
|---|---|---|---|
| 第一輪，32K 文件 | 9.2 s | 9.3 s | 104.1 s |
| 第一輪，64K 文件 | 17.0 s | 16.9 s | 213.0 s |
| 32K + 4K 追問 | 2.34 s | 4.12 s | 19.3 s |
| 64K + 4K 追問 | 2.33 s | 4.40 s | 16.5 s |
| 32K + 8K 追問 | 3.58 s | 4.43 s | 34.9 s |
| 64K + 8K 追問 | 3.51 s | 4.47 s | 31.3 s |

兩邊都重用了歷史：Strata 重用的 token 每次都等於前面的歷史；vLLM 的 cached token 以 block 為單位，所以會多讀幾百到幾千個 token。
有 cache 時差距約 5～9 倍；還沒有 cache 的第一輪約 11～13 倍。

## 為什麼（沒有 profile）

vLLM 的 offload 讓 GPU 每次 forward 都經 PCIe 從 DRAM 讀專家，GPU 上沒有專家快取。Strata 在 GPU 上留 36.5 GiB 的熱專家，其餘大部分由 CPU 算。
這是依設定推論的；這次比較沒有 profile 任何一邊，也沒有試 vLLM 的其他 offload 設定（prefetch、其他 batch 大小）。

## 執行時的狀況

vLLM 載入時被 systemd-oomd（memory pressure 超過 50% 達 20 秒）連同啟動它的桌面工作階段一起殺掉兩次：它是從工作階段自己的 scope 裡啟動的。
改用 `systemd-run --user --scope` 啟動，並先把另一個模型卸載後，165 秒載入完成、正常運作。

## 限制

1. 權重不同（FP8 對 Q8_0）；沒有比較輸出內容。
2. vLLM 只試了一組設定（4 個 sequence、vision 關；單人測試的 model length 是 32K，長文與多輪是 98,304）。它的 32K 與多輪各只有一輪。
3. RAM 與另一個已載入的模型共用；沒有單獨量任何一個引擎。
4. Strata 的 decode 速度沒有單獨量，所以它的完成時間混著 prefill 與 decode。

## 相關

- [用 AMX 讀 prompt](strata-amx-prefill.md)
- [Strata](../engines/strata.md)
- [vLLM](../engines/vllm.md)
