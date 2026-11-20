---
title: vLLM 原生 KV cache offload 到 DRAM／NVMe
summary: 16 GB 顯卡上 Qwen3.5-35B 滿 context、8 併發、多人共用文件三種情境都驗證成功；重訪從 445 秒變 6 秒、1,676 秒變 37.6 秒。DRAM 層放不下要載回的前綴時，讀了也命中 0%。
updatedAt: 2026-11-20
measuredAt: 2026-09-23
---

# vLLM 原生 KV cache offload 到 DRAM／NVMe

## 條件

| 項目 | 值 |
|---|---|
| 硬體 | RTX 5060 Ti 16 GB（PCIe Gen5 x8）、125 GB DRAM、ADATA T7P5 3.84 TB Gen5 NVMe（ext4）、Linux |
| vLLM | 0.28.0，離線 `LLM()` API（`vllm serve` 未測） |
| 主模型 | Qwen3.5-35B-A3B GPTQ-8bit（hybrid 架構），`--cpu-offload-gb 32 --cpu-offload-params experts --max-num-seqs 64`，GPU 上只留 6.4 GB 權重 |
| 對照模型 | Gemma-4-E4B fp8（SWA hybrid） |
| 量測 | 磁碟讀取量取自裝置計數，不是 page cache |

## 結果

| 情境 | 不 offload | 有 offload | 備註 |
|---|---|---|---|
| 滿 context 146K × 2 條輪流擠出 GPU，重訪 | 444.9 s／條 | **6.04 s／條**（DRAM 8 GB + NVMe） | 從 NVMe 讀 3,049 MB |
| 8 併發 × 18K，16 條輪流，第 2 次重訪 | 1,676 s | **37.6 s（44.5×）** | 命中 98.1% |
| 8 人問同一份 135K 文件，清空 GPU 後再問 | TTFT 190–194 s | **6.9–13.8 s**（DRAM 6 GB） | 命中 99.3% |
| 同上，但 DRAM 層只有 2 GB | — | 191–196 s | **讀了 8,954 MB，命中 0%** |
| NVMe I/O（20.6 MB block） | — | 寫 7.1、讀 8.1 GB/s | |
| 專家 offload：UVA 對 Prefetch | — | prefill 831 對 2,000 tok/s；decode **17.8 對 0.76** tok/s | **用 UVA** |

## 結論

1. **DRAM 層是閘門。** NVMe 的 block 必須先升級進 DRAM 才能送到 GPU，所以 DRAM 層要放得下「最長要重用的前綴」。
   第四列就是放不下的樣子：資料讀了，但一點都用不到。
2. **offload 省的是重算時間，不是同時服務人數。** 正在計算的請求，KV 仍然必須全在 GPU。
3. **prefix caching 必須開著**，否則 offload 靜默失效。
4. **NVMe 層不會自己刪檔**，也沒有 `fsync`。後端提供上限設定與清理端點，見 [vLLM](../engines/vllm.md)。
5. **跨重啟重用需要固定 `PYTHONHASHSEED`**，後端啟動 vLLM 時已經設好；整機重開後的實測命中率是 99.3%。

## 更新：vLLM 0.32，100 人，DRAM 加 SSD（2026-11）

同樣的問題，換成更大的場景：GPU 的 KV cache 刻意壓小，100 個人輪流對話。

| 項目 | 值 |
|---|---|
| 硬體 | 兩張各 48 GB 的 GPU（tensor parallel 2）、20 核工作站等級 CPU、503 GiB DRAM；KV tier 放在 Gen5 NVMe SSD 上，與模型碟分開 |
| vLLM | 0.32.0，`vllm serve`；請求打 `/v1/completions`、streaming；重用以 `cached_tokens` 判斷 |
| 模型 | Qwen3.5-35B-A3B GPTQ-8bit（41 GB），權重全在 GPU |
| 設定 | model length 262,144、`max_num_seqs` 128、**GPU KV 限制在 816,244 tokens**（8 GiB；不限制是 2,139,300，工作集放得下，offload 不會被用到） |
| 負載 | 100 人，每人一份不同的 16,000 token 文件，同時送出「整段歷史＋新問題」，共 3 輪；工作集約 160 萬 token，是 GPU KV 的 2 倍。`max_tokens` 16 |
| SSD 的組別 | 每秒清掉 page cache，讓讀取真的來自磁碟 |

100 個請求一輪的時間，以及重用了多少：

| 設定 | 第 1 輪 | 第 2 輪 | 第 3 輪 | 第 2 輪命中 | 命中人數 |
|---|---|---|---|---|---|
| 不 offload | 111.1 s | 111.2 s | 111.5 s | 0% | 0 |
| DRAM 8 GiB | 114.0 s | 113.8 s | 114.6 s | 0% | 0 |
| DRAM 32 GiB | 113.5 s | 113.5 s | 112.9 s | 0% | 0（第 3 輪 1） |
| DRAM 8 GiB＋SSD | 113.8 s | 70.1 s | 71.6 s | 40.5% | 41 |
| **DRAM 32 GiB＋SSD** | 114.2 s | **8.1 s** | **5.7 s** | **98.7%** | **100** |

DRAM 32 GiB＋SSD 的第 2 輪首字平均 4.5 s（p95 7.8 s），不 offload 是 56.2 s（p95 105.5 s）。
一個人的 200,000 token 文件，在 100 人的幾輪把它擠出 GPU 之後再問：沒有 SSD 層是 25.5 s，有的話是 **1.8 s**（重用 199,584 tokens）。

- **省下的時間全部來自 SSD 層。** 只有 DRAM 時，即使 32 GiB，命中率仍接近 0：工作集輪流進出，比它小的 DRAM LRU 會整批互相擠掉。
- **DRAM 層太小會讓 SSD 層打折。** 8 GiB 時，SSD 層讀了 529 GB，只有 30.6 GB 送上 GPU；100 個請求同時載入的 block，還沒用就被擠掉。
  32 GiB 時讀了 47.3 GB（寫 52.7 GB），100 人都命中。
- 成本：第 1 輪慢 2～3%（要寫 tier）；SSD 讀取的峰值約 3.1～3.6 GB/s。
- 不 offload 時的時間就是重算：這個 35B 約 8,400 tok/s（200K 的文件冷讀 23.7 s）。

只試了 8 與 32 GiB 兩點；「DRAM 層必須容得下同時載入的請求的 block」是由計數器推論的，不是掃描出來的。
GGUF 模型沒能試：vLLM 0.28 與 0.32 都沒有 GGUF loader。

## 限制

- 第二、三列的測試用了 `max_num_batched_tokens=2048`（腳本預設值設錯），**絕對秒數比預設設定慢約 4 倍**；倍率仍可參考。
- 長時間持續寫入、磁碟寫滿都沒有測；prompt 是隨機 token。

## 相關

- [Offload](../memory/offload.md)
- [vLLM](../engines/vllm.md)
- [8 GB 顯存預算下的 vLLM 與 llama.cpp](vllm-vs-llamacpp-8gb.md)
