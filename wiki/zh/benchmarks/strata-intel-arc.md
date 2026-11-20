---
title: Strata 在 Intel Arc 內顯上
summary: 58 GB 的 Gyro-S（Q1、TQ1_0）在 30～32 GB 共用記憶體的筆電上跑得動：Windows 開 MTP 時 decode 約 11 tok/s、prefill 約 175 tok/s，Linux 不開 MTP 時 decode 6～8 tok/s；64K～256K 的對話可以存檔、重啟後還原。
updatedAt: 2026-11-20
measuredAt: 2026-11-03 to 2026-11-20
---

# Strata 在 Intel Arc 內顯上

情境：一台只有內建 Arc 顯示晶片、沒有獨立 VRAM 的輕薄筆電，要跑遠大於它記憶體的模型。
Intel 套件**沒有 deep sleep**（見 [Strata](../engines/strata.md)），所以這裡量的是速度、長對話，以及對話跨重啟存檔。

## 條件

| 項目 | 值 |
|---|---|
| 硬體 | 筆電 CPU，16 核心（4 效能核 + 12 效率核）；Intel Arc 內顯（Xe3），用系統記憶體、沒有自己的 VRAM |
| 記憶體 | Linux：30 GB 共用。Windows：32 GB |
| 作業系統 | Linux：Ubuntu 24.04。Windows：11 |
| 模型 | Qwen3.8 Flash-Next **Gyro-S TQ1_0**（最低精度；檔案 58.5 GB，權重約 27.6 GB） |
| 引擎 | Strata Intel（SYCL）套件；Linux 的量測用的是已有三個修正的引擎，這三個修正現在都在 strata-sleep 的套件裡 |
| Linux 設定 | 除非另外說明：8K context、`--mmap-experts --resident-budget-gib 12`、不開 MTP |
| Windows 設定 | 32K context、開 MTP（`--mtp`）、專家的串流由引擎自己決定 |

每個回答另外有自動檢查（關鍵字、亂碼、重複迴圈）。

## 速度

**Windows**（串流請求，每個 prompt 的內容都不同，不會命中 prefix cache；prefill ＝ prompt tokens ÷ 第一個輸出的時間，decode ＝ 第一個輸出之後的速率）：

| prompt | 第一個輸出 | prefill | decode（200 tokens） |
|---|---|---|---|
| 約 560 tokens | 8.4～8.7 s | 64～66 tok/s | 11.0、12.6 tok/s |
| 約 2,800 tokens | 20.5～21.0 s | 133～136 tok/s | 11.9、11.3 tok/s |
| 約 8,700 tokens | 54.1～54.2 s | 160 tok/s | 11.6、9.3 tok/s |

- **穩態 prefill 約 175 tok/s**，用 2,800 與 8,700 tokens 兩組的差算出來。短 prompt 看起來慢，是因為每個請求在第一個輸出之前有 5～8 秒固定的時間。
- **decode 平均約 11.3 tok/s**（9.3～12.6）；專家快取命中率 77～93%。

**Linux**（8K context、不開 MTP、修正之後；短 prompt 約 25 tokens，長回答約 470 tokens，冷 ＝ 第一次讀、沒有任何快取）：

| 設定 | 短 prompt decode | 長回答 decode | 冷的 1,903 tokens prompt |
|---|---|---|---|
| DRAM 層 12 GiB | 5.5～5.7 tok/s | 7.0 tok/s | 121 tok/s |
| DRAM 層 6 GiB，更多專家交給 GPU 算 | 5.9～6.0 tok/s | 7.6 tok/s | 120 tok/s |
| 開 MTP | 4.8～5.2 tok/s | 6.2～7.4 tok/s | 122～123 tok/s |

- 第一列更早的一次，同一個 prompt 讀到 184 tok/s，原因沒有找到。
- **MTP 在這裡幫不上忙。** 每個 token 在每一層從 512 個專家選 10 個，一起驗證的 token 選的大多不同，CPU 要算的量大約變 3 倍。
  （上面的 Windows 是開 MTP 跑的，比較快；兩邊是不同的機器、記憶體大小與設定，不是同條件的比較。）
- 瓶頸是 **SSD 層**：30 GB 裝不下 23 GiB 的專家，沒進 DRAM 層的每次都從 GGUF 讀。

## 長對話的存檔與還原（Linux、開 MTP）

兩個事實藏在 context 的 30% 與 70% 處；填滿後：問第一個、存檔、整個 server 重啟、還原、問第二個。

| context | prompt | 讀取 | 存檔 | 重啟後還原 | 第二個事實 |
|---|---|---|---|---|---|
| 64K | 58,338 tokens | 162 tok/s | 1.88 GB，0.6 s | 1.7 s | 對，追問約 13 s |
| 128K | 120,778 tokens | 155 tok/s | 3.64 GB，1.0 s | 3.1 s | 對，約 15 s |
| 256K | 245,658 tokens（33 分鐘） | 124 tok/s | 7.16 GB，1.6 s | 6.0 s | 對，約 13 s |

- 還原之後整段對話沿用快取，只讀新的 tokens。
- 還原會把檔案讀進 RAM 並保留一些空間：每個長度都要把 DRAM 層縮小才放得下，256K 還要把保留的下限降低。
- 不還原、整段重讀同一個對話要 360 s（64K）、783 s（128K）、1,982 s（256K）。
- 問「某個數字第一次出現在哪一天」（要在上千筆裡找），三個長度都答錯。這是模型的能力問題，不是快取。

## Agent 實跑（Linux、64K、不開 MTP）

| 項目 | 結果 |
|---|---|
| 三個小型寫程式任務（從零寫、修 bug、加功能），各三輪，用任務自己的單元測試判定 | 9／9 通過；中位數 166 s；引擎 0 次重啟 |
| 一個 agent session 連續 15 輪做一個小專案，共四個 session，第 8 輪後把 server 重啟 | 58／60 輪通過；重啟後還原每次 0.6～0.8 s；94% 的 tokens 沿用快取；17 小時內沒有計畫外的引擎重啟 |

沒過的兩輪，一個是驗收要求一個固定的錯誤字串，一個是超過 60 分鐘上限、但當時的功能已經是對的：都不是引擎的問題。

## 限制

1. 只有一個模型、一種量化，而且是最低精度。Q1 的回答品質是模型的上限。
2. 沒寫範圍的數字都是一次執行；Windows 的 decode 和舊套件比慢了約 5～8%，只比了一次，可能是雜訊。
3. Windows 與 Linux 是不同的機器與設定，不要把其中一邊當成另一邊的加速。
4. 記憶體和桌面共用：Linux 上，圖形記憶體的頁面池被填滿（到 RAM 的一半）後，下一次載入的專家快取可能小很多，重開機才會清掉。
5. Agent 實跑是直接連 Strata，沒有經過 AST。

## 相關

- [Strata](../engines/strata.md)
- [三種切換方式](gpu-handoff.md) — NVIDIA deep sleep 的數字
