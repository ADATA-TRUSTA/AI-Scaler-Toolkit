---
title: vLLM 原生 KV cache offload 到 DRAM／NVMe
summary: 16 GB 顯卡上 Qwen3.5-35B 滿 context、8 併發、多人共用文件三種情境都驗證成功；重訪從 445 秒變 6 秒、1,676 秒變 37.6 秒。DRAM 層放不下要載回的前綴時，讀了也命中 0%。
updatedAt: 2026-10-29
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

## 限制

- 第二、三列的測試用了 `max_num_batched_tokens=2048`（腳本預設值設錯），**絕對秒數比預設設定慢約 4 倍**；倍率仍可參考。
- 長時間持續寫入、磁碟寫滿都沒有測；prompt 是隨機 token。

## 相關

- [Offload](../memory/offload.md)
- [vLLM](../engines/vllm.md)
- [8 GB 顯存預算下的 vLLM 與 llama.cpp](vllm-vs-llamacpp-8gb.md)
