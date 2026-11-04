# vLLM 0.28 原生 KV Cache Offload（DRAM／NVMe）實測報告

測試日期：2026-09-23。所有數字都是這台機器上的實測值，碟讀取量取自 `/sys/block/nvme0n1/stat`
裝置計數，不是 page cache。前一代方案 LMCache 已從專案移除（commit c26172a），它的舊量測可從該 commit 之前的 `docs/lmcache_benchmark.md` 取得。

## 1. 結論

1. **vLLM 0.28 原生就能把 KV cache offload 到 NVMe，而且能被重用。** Qwen3.5-35B-A3B（GPTQ 8-bit，
   expert 權重 offload 到 DRAM）在 16 GB GPU 上，滿 context、8 併發、多人共用同一份文件三種情境都驗證成功。
2. **DRAM 層是閘門。** NVMe 的資料必須先搬進 DRAM 層才能送到 GPU。DRAM 層放不下要載回的前綴時，
   會讀了大量資料卻完全命中不了（2 GB：讀 9 GB、命中 0%；6 GB：讀 2.8 GB、命中 99.3%）。
3. **offload 省的是「重算時間」，不是「同時服務人數」。** 正在計算的請求，KV 仍然必須全部放在 GPU 上。
   多人要同時服務，靠的是 prefix caching 讓相同前綴只存一份。
4. **NVMe tier 沒有容量上限，也不會刪檔。** 會一直長到碟滿，需要自己規劃清理。
5. **expert 權重 offload 請用 UVA。** Prefetch 的 prefill 快 2.4 倍，但 decode 慢 23 倍。

## 2. 測試環境

| 項目 | 規格 |
|---|---|
| GPU | RTX 5060 Ti 16 GB，PCIe **Gen5 x8**（同時驅動桌面，約佔 0.9 GB） |
| DRAM | 125 GB |
| NVMe | T7P5 3.8 TB **Gen5**（`/dev/nvme0n1`，ext4），fs tier 也寫在這顆碟上 |
| 軟體 | vLLM **0.28.0**，離線 `LLM()` API（`vllm serve` 模式未測） |
| 模型 | Qwen3.5-35B-A3B GPTQ-8bit（GDN hybrid）、Gemma-4-E4B fp8（SWA hybrid） |

35B 的載入方式：`--cpu-offload-gb 32 --cpu-offload-params experts --max-num-seqs 64`，
GPU 上只留 6.4 GB 權重，其餘 expert 放在 DRAM，用 UVA 直接跨 PCIe 讀取。

## 3. 運作機制（依 vLLM 0.28 原始碼）

```
GPU（計算中的 KV）──寫穿──▶ DRAM 層（cpu_bytes_to_use，LRU）──串接寫入──▶ NVMe fs tier（無上限）
       ▲                          │
       └──── 載回：NVMe → DRAM（升級，需要空位）→ GPU ────┘
```

- **切塊與命名**：KV 以 block 為單位（35B 為 1,056 token），檔名是**鏈式內容 hash**
  （`<hhh>/<hh>_g<group>/<hash>.bin`），相同前綴對到同一個檔。外層目錄還有一層 hash，
  包含模型、dtype、TP/PP 設定，換設定就對不上。
- **寫入**：GPU → DRAM 完成後，立刻串接寫到 NVMe（`tiering/manager.py`：「Always offload to all tiers」）。
  **只存 prompt**（`offload_prompt_only` 預設 True）、**只存填滿的 block**。
- **讀取**：查詢 = 檔案存不存在。命中後先把整個 block 複製進 DRAM 層（promotion），
  再從 DRAM DMA 到 GPU。**沒有 GPUDirect Storage**。第一次查到會回 `HIT_PENDING`，該請求可能先重算。
- **I/O 型態**：每個 block 一個檔、單次 `write()`／`readv()`、O_DIRECT、讀寫各 16 條執行緒。
  **沒有 `fsync`**。
- **跨重啟**：理論上可重用，但**必須固定 `PYTHONHASHSEED`**，否則 hash 起點每次隨機，
  靜默失效且碟持續長大（**未實測**）。

## 4. 實測結果

### 4.1 Qwen3.5-35B，滿 context 146,784，2 條請求輪流擠出 GPU

`max_num_batched_tokens=2048`，每條 prompt 146,528 token。GPU KV 容量 149,796 token，一次只放得下一條。

| 設定 | 冷啟 | 重訪 | 命中 | 從 NVMe 讀 |
|---|---|---|---|---|
| 不 offload | 884.3 s/條 | 444.9 s/條 | 49.7% | — |
| DRAM 24 GB | 883.4 s/條 | **6.06 s/條** | 99.5% | 0 |
| DRAM 8 GB + NVMe | 887.7 s/條 | **6.04 s/條** | 99.5% | **3,049 MB** |

NVMe 上共寫了 23.88 GB（1,104 個檔，每檔 20.6 MB）。

### 4.2 Qwen3.5-35B，8 併發 × 18,304 token，16 條輪流

`max_num_batched_tokens=2048`。GPU 同時約放得下 8 條，另外 8 條必然把前面的擠掉。

| pass | 不 offload | DRAM 4 GB + NVMe | 命中 | 從 NVMe 讀 |
|---|---|---|---|---|
| 0 冷啟 | 1,699 s | 1,719 s | 0% | 0 |
| 1 重訪 | 1,685 s | **131 s（12.9×）** | 91.9% | 7,699 MB |
| 2 重訪 | 1,676 s | **37.6 s（44.5×）** | 98.1% | 3,460 MB |

pass 2 沒命中的 5,632 token 剛好等於 16 條 × 352 token 的尾巴（未填滿的 block 不存）。

### 4.3 Qwen3.5-35B，8 人問同一份 135,168 token 的文件

每人問題 900 token，`max_num_batched_tokens=8192`。P3 是清空 GPU 快取後再問新問題。

| 階段 | 不 offload | NVMe，DRAM 2 GB | NVMe，DRAM 6 GB（先驅逐出 DRAM） |
|---|---|---|---|
| P1 冷啟 TTFT | 190.7–194.6 s | 186.9–192.1 s | 185.7–190.9 s |
| P2 文件在 GPU TTFT | 6.5–12.5 s | 6.6–12.6 s | 6.6–12.7 s |
| **P3 清空 GPU 後 TTFT** | 189.9–193.9 s | 191.0–196.2 s | **6.9–13.8 s** |
| P3 命中 | 86.9%（文件重算） | 86.9%（**讀了用不上**） | **99.3%** |
| P3 從 NVMe 讀 | 0 | 8,954 MB | 2,833 MB |
| P3 同時排程人數 | 6 | 5 | **8** |

- P1 的命中 86.9% = 7 × 135,168：文件只算一次，另外 7 人共用。
- 這份文件在 offload 層佔 **185 格／4.0 GB**（attention 128 格，3 個 GDN group 各 19 格）。
  DRAM 2 GB 只有 99 格，放不下，所以升級上來的 block 還沒湊滿就被擠掉。

### 4.4 NVMe 本身的 I/O 速度

直接呼叫 vLLM 的 `batch_store_block`／`batch_load_block`（同一份 C 實作，O_DIRECT，讀寫各 16 條執行緒）。

| block 大小 | 只寫 | 只讀 | 同時讀寫 |
|---|---|---|---|
| 20.6 MB（35B） | 7.1 GB/s | 8.1 GB/s | 讀 4.2 + 寫 4.2 GB/s |

LMCache 在同一顆碟上最高 1.8 GB/s；vLLM 原生路徑約是它的 4.5 倍，已經碰到硬體上限。
每階段只寫了 10 GB，可能還在 SLC 快取內，長時間持續寫入的速度未測。

### 4.5 expert 權重 offload：UVA vs Prefetch（35B，KV offload 開啟）

`max_num_batched_tokens=8192`。

| 項目 | UVA | Prefetch |
|---|---|---|
| prefill 18,304 token | 22.0 s（**831 tok/s**） | 9.2 s（**2,000 tok/s**） |
| decode，1 人 | **17.8 tok/s** | 0.76 tok/s |
| decode，8 人合計 | **46.1 tok/s** | 6.06 tok/s |
| KV 從 DRAM 載回 18k（TTFT） | 1.9 s | 1.4 s |
| DRAM → GPU 載入 432 MB | 16.5 ms（26.2 GB/s） | 16.6 ms（26.0 GB/s） |
| GPU KV block 數 | 210 | 155 |

### 4.6 Gemma-4-E4B（SWA 架構，1,519 token × 20 條，工作集約為 GPU KV 的 2 倍）

`max_num_batched_tokens=2048`，數字為每條請求的平均時間。

| 模型 | 不 offload | DRAM 12 GB | DRAM 2 GB + NVMe | DRAM 0.5 GB + NVMe |
|---|---|---|---|---|
| Gemma-4-E4B（SWA） | 214.8 ms | **25.6 ms（8.39×）** | 29.0 ms（全在 DRAM） | 31–58 ms，命中 84–99% |

- 「DRAM 2 GB + NVMe」下 NVMe 讀取是 0，但 fs tier 仍然寫了 1.77 GB：證實是寫穿，不是溢出才寫。
- SWA 層只需要最近 512 token，這 30k token 的工作集在 offload 層只有 1.77 GB，2 GB 的 DRAM 層就放得下。

### 4.7 這張卡實際放得下的 context（vLLM 自己報告的 GPU KV 容量）

| 模型 | config 上限 | GPU KV 容量 | 備註 |
|---|---|---|---|
| Gemma-4-E4B fp8 | 131,072 | 12,572 token | fp8 後權重仍有 11.24 GiB |
| Qwen3.5-35B（expert offload） | 262,144 | 149,796 token | GDN 架構 KV 很省 |

## 5. 上線建議

| 項目 | 建議 |
|---|---|
| prefix caching | **必開**，沒開的話 offload 會靜默失效 |
| `cpu_bytes_to_use` | 至少放得下「最長要重用的前綴」的 offload 資料，再加併發餘裕。35B 的 135k 文件約 4.0 GB |
| `max_num_batched_tokens` | 維持預設 8,192。2,048 會讓 35B prefill 慢約 4 倍，hybrid 模型的 offload 佔用也會變成約 3 倍 |
| `PYTHONHASHSEED` | 固定一個值，否則跨重啟無法命中 |
| NVMe 用量 | 自行監控與清理，fs tier 不會刪檔 |
| expert offload | 用 UVA；prefetch 只適合長輸入、極短輸出 |
| 容量估算 | 35B：GPU 約 36 KB/token；offload 約 24 KB/token（chunk 8,192）或 78 KB/token（chunk 2,048） |

設定範例（離線 API 已實測；`vllm serve` 用的是相同的參數名稱，但未實測）：

```bash
PYTHONHASHSEED=0 vllm serve <model> \
  --enable-prefix-caching \
  --cpu-offload-gb 32 --cpu-offload-params experts --max-num-seqs 64 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":6442450944,
      "spec_name":"TieringOffloadingSpec",
      "secondary_tiers":[{"type":"fs","root_dir":"/path/to/nvme/vllm_kv_fs"}]}}'
```

## 6. 發現的問題

| 問題 | 影響 | 繞法 |
|---|---|---|
| Prefetch 對 `desc_act=False` 的 GPTQ 模型 assert 失敗 | `offload_params experts` 會匹配到空的 `w13_g_idx`，0 元素張量的 `is_pinned()` 永遠是 False | 只列出 `w13_qweight w2_qweight w13_scales w2_scales`。上游是否已回報未查 |
| hybrid 模型 `max_num_seqs` 過大 | expert offload 後 Mamba state block 只剩 190 個，預設 256 會讓 CUDA graph 拒絕啟動 | `--max-num-seqs 64` |
| fs tier 無 `fsync` | 斷電可能留下長度正確但內容不完整的 block，偵測不到 | 目前無 |

## 7. 測試限制

- 4.1、4.2 用了 `max_num_batched_tokens=2048`（腳本預設值設錯），**絕對秒數比預設設定慢約 4 倍**；
  加速倍率仍可參考，因為對照組用的是同樣的設定。
- 重啟後重用、`vllm serve` 模式、長時間持續寫入 NVMe、碟寫滿後的行為，都沒有測。
- UVA vs Prefetch 的 KV 載回只測了單一請求，與權重搬運在時間上幾乎沒有重疊。
- 所有測試都用隨機 token 當 prompt，只有 4.3 刻意讓多人共用前綴。
