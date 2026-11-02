---
title: Gen5 NVMe 對本地 AI 流程的影響
summary: 兩組實測：只換系統碟（Gen3 → Gen5）重跑影像與影片生成，效益從 −43% 到 +2%，取決於有多少時間在等磁碟；以及 Linux 上預熱 page cache 讓 MoE 模型 decode 快 2 倍。
updatedAt: 2026-10-29
measuredAt: 2026-09-21
---

# Gen5 NVMe 對本地 AI 流程的影響

## 一、只換系統碟：Gen3 → Gen5

### 條件

| 項目 | 值 |
|---|---|
| 硬體 | RTX 5080 Laptop 16 GB、32 GB RAM、Windows |
| 系統碟 | PCIe Gen3 x4 → **ADATA T7P5 7.68 TB Gen5 x4** |
| 方法 | 整碟 clone，OS、驅動、Python 環境、模型檔逐位元相同；seed、prompt、起始圖相同 |
| 驗證 | 24 張圖的 sha256 與 Gen3 那次完全相同——換碟只改了速度 |

### 結果

| 工作 | Gen3 | Gen5 | 變化 |
|---|---|---|---|
| Z-Image（8 步）熱啟 | 10.6 s | 6.1 s | **−42.5%** |
| FLUX.1-schnell（4 步）熱啟 | 48.5 s | 41.1 s | −15.3% |
| RealVisXL 熱啟（不碰磁碟，對照組） | 8.6 s | 8.5 s | −1.2% |
| Qwen-Image（20 步）熱啟 | 154.1 s | 157.6 s | +2.3% |
| Wan 14B Lightning，agent 每 60 秒輪詢 | 460.0 s | 358.1 s | −22.2% |
| Wan 14B，不輪詢（`--wait`） | 325.8 s | 314.9 s | −3.3% |
| Wan 14B 全步數 | 1,342.0 s | 1,302.7 s | −2.9% |

- **效益看計算密度**：步數少、時間花在權重串流與分頁的工作才吃得到 Gen5；計算密集的幾乎無感。
- **32 GB RAM 才是瓶頸**：每張圖搬 12–31 GB、每段影片搬 116 GB，幾乎都是分頁流量。
  agent 每 60 秒輪詢一次，就要把 LLM 換出再換回；**舊碟配不輪詢（325.8 s）比新碟配輪詢（358.1 s）還快。**
- 碟本身：多執行緒峰值 10.98 GB/s，但單執行緒 1 MiB 循序讀只有 2.96 GB/s，和 Gen3 一樣。
  實際工作負載的峰值約 4.6 GB/s，瓶頸在存取型態（單執行緒、佇列淺），不在介面。

### 建議

1. 先讓 agent 等結果而不是輪詢——零成本，效益大於換碟。
2. 下一筆硬體預算給 **RAM**。
3. LLM 用 mmap 載入，換出時就是丟掉乾淨頁，不必寫 pagefile。

## 二、預熱 page cache：Linux 上的冷讀

### 條件

Intel Core Ultra 9 275HX、30.7 GB RAM、RTX 5080 Laptop、Linux 實機、ADATA T7P5 Gen5 x4（ext4）。
讀一個 20.5 GB 的 Qwen3.5-35B-A3B GGUF；每個「冷」數字都用 `mincore()` 驗證、重複 3 次。

### 結果

| 項目 | 值 |
|---|---|
| 裝置上限（O_DIRECT、QD32、128 KB 分散讀） | 8.41 GB/s |
| 最佳預熱方式：4 執行緒 `read()` | **5.95 GB/s**（軟體路徑天花板 6.37 GB/s 的 93%） |
| 同方法在 Windows | 5.36 GB/s |
| `mmap(MAP_POPULATE)` | 3.88 GB/s |
| `readahead(2)`、`fadvise(WILLNEED)` | kernel 只照做 0.8%／0% |

對執行中的 llama-server：用真實的記憶體壓力把 GGUF 擠出 page cache（模擬影像生成拿走 RAM 再還回來）。

| 階段 | 不預熱 | 先預熱 | 倍率 |
|---|---|---|---|
| prefill（32 tok） | 1,980 ms | 651 ms | **3.0×** |
| decode（64 tok） | 21.7 tok/s | 44.4 tok/s | **2.0×** |

不預熱時，裝置實際只跑 814 MB/s：MoE 推論是依 routing 分散地讀入專家，受 IOPS 限制，不是頻寬。

### 方法上的兩件事

- 一開始在 ZFS 上量到 22 GB/s，其實是 ARC 快取。**冷讀一定要驗證。**
- `posix_fadvise(DONTNEED)` 趕不走執行中行程已 mmap 的頁，只有真實的記憶體壓力能回收。

## 相關

- [Offload](../memory/offload.md)
- [把 GPU 讓給其他程式](gpu-handoff.md)
- [TRUSTA 企業級 SSD](../products/trusta-ssd.md)
