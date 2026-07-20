# 效能實測

[← 回到 README](../README.zh-TW.md)

## 📊 效能實測 (TPS / TTFT)

使用 **Llama Engine** (`llama-server`) 實測。所有測試的共同條件：`n_ctx = 4096`、輸入 `50–108` tokens、輸出 `50` tokens。**TPS** = 每秒輸出 token 數（越高越好）；**TTFT** = 首個 token 延遲（越低越好）。數據來自內部測試，會因驅動、編譯版本與 prompt 而有差異。

### 測試機器

| 代號 | GPU | OS | CPU | VRAM | DRAM |
|------|-----|----|-----|------|------|
| **A** | RTX 5060 Ti | Ubuntu 24.04 | Core Ultra 5 225 | 16 GB | DDR5-4800 / 128 GB |
| **B** | RTX 6000 Ada | Ubuntu 24.04 | Xeon w5-3535X | 48 GB | DDR5-4800 / 512 GB |
| **C** | RTX 5080 | Windows 11 | Core Ultra 9 275HX | 16 GB | DDR5-6400 / 32·64 GB |
| **D** | Intel iGPU | Windows 11 | Core Ultra 7 355 | 共用記憶體 | DDR5-6400 / 32·64 GB |

### 摘要 — 各模型最佳結果

每格顯示該機器達到的**最佳 TPS** 與對應 TTFT。**粗體** = 該模型最快的機器。`–` = 未測試。

| 模型 (大小) | A · 5060 Ti | B · 6000 Ada | C · 5080 | D · iGPU |
|-------------|-------------|--------------|----------|----------|
| GPT-OSS-20B-FP16 (13.8 GB) | **90 / 0.2s** | **155 / 0.1s** | 133 / 0.5s | 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M (11.6 GB) | – | – | **179 / 0.3s** | 20 / 0.9s |
| Qwen3-8B-Q8_0 (8.7 GB) | – | – | **73 / 0.05s** | 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M (22 GB) | 57.5 / 0.4s | **162 / 0.1s** | 80 / 0.5s | 20 / 1s |
| Gemma4-26B-A4B-Q8_0 (26.9 GB) | 29 / 1s | **114 / 0.8s** | 45 / 0.7s | 15 / 1s |
| Gemma4-31B-Q4_K_M (18.3 GB) | 3 / 2.5s | **7.5 / 0.8s** | 6 / 1.5s | 4 / 2s |
| Gemma4-31B-Q8_0 (32.6 GB) | 1.9 / 3.5s | **24.5 / 0.2s** | 3.5 / 2s | 2 / 2s |
| GPT-OSS-120B&sup1; (~63–65 GB) | 15.5 / 1.5s | **48 / 0.5s** | 35 / 3.5s | 9.5 / 15s |

&sup1; **A/B** 跑的 120B 為 **FP16 (65.4 GB)**；**C/D** 為 **Q8 / Q4_K_M (~63 GB)**。量化不同，無法直接等價比較。

### 各機器完整數據

<details>
<summary><b>A · RTX 5060 Ti 16 GB</b> — 完整結果 (128 GB DRAM)</summary>

| 模型 | 設定 (VRAM / DRAM) | TPS | TTFT |
|------|---------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 90 | 0.2s |
| GPT-OSS-120B-FP16 | auto (14.5 / 50.9 GB) | 15.5 | 1.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (12 / 53.4 GB) | 14 | 1.7s |
| Qwen3.5-35B-Q4_K_M | auto (15.7 / 7.5 GB) | 57.5 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (14.1 / 9.1 GB) | 53 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 38 | 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=17 (15.7 / 13 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6 / 22.7 GB) | 20 | 1s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 3 | 2.5s |
| Gemma4-31B-Q8_0 | auto (15.8 / 19.4 GB) | 1.9 | 3.5s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.8 / 20.4 GB) | 1.8 | 3.7s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.7 / 25.5 GB) | 1.5 | 4.5s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (5 / 30.2 GB) | 1.2 | 5.5s |

</details>

<details>
<summary><b>B · RTX 6000 Ada 48 GB</b> — 完整結果 (512 GB DRAM)</summary>

| 模型 | 設定 (VRAM / DRAM) | TPS | TTFT |
|------|---------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 155 | 0.1s |
| GPT-OSS-120B-FP16 | auto (46 / 19.4 GB) | 48 | 0.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=27 (19 / 46.4 GB) | 24 | 1s |
| GPT-OSS-120B-FP16 | n-cpu-moe=29 (16.3 / 49.1 GB) | 20 | 1.3s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (11.6 / 53.8 GB) | 18 | 1.3s |
| Qwen3.5-35B-Q4_K_M | auto (23.2 GB) | 162 | 0.1s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (13.7 / 9.5 GB) | 67 | 0.3s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 45 | 0.5s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 114 | 0.8s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=16 (16.7 / 12 GB) | 42 | 0.4s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6.1 / 22.6 GB) | 24 | 0.6s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 7.5 | 0.8s |
| Gemma4-31B-Q8_0 | auto (35.2 GB) | 24.5 | 0.2s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.5 / 20.7 GB) | 4.5 | 1.3s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.4 / 25.8 GB) | 4 | 1.7s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (4.7 / 30.5 GB) | 3 | 2s |

</details>

<details>
<summary><b>C · RTX 5080 16 GB</b> — 完整結果 (Windows 11，依 DRAM 容量)</summary>

各 DRAM 設定下的 TPS / TTFT。**64 GB + moe** 欄使用 `--n-cpu-moe=N`。

| 模型 | 設定 | 32 GB | 64 GB | 64 GB + moe |
|------|------|-------|-------|-------------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 133 / 0.5s | 133 / 0.5s | – |
| GPT-OSS-20B-FP16 | 0 GPU / 25 CPU | 15 / 3s | 19 / 2s | – |
| GPT-OSS-20B-Q4_K_M | GPU 全載 (12 GB) | 179 / 0.3s | 179 / 0.3s | – |
| GPT-OSS-120B-Q8 | auto (14.6 GB) | 5 / 60s | 33 / 4s | – |
| GPT-OSS-120B-Q8 | 10 GPU / 27 CPU | 2.5 / 77s | 5 / 7s | 10 / 7s |
| GPT-OSS-120B-Q8 | 8 GPU / 29 CPU | 3 / 70s | 23 / 5s | 34 / 4s |
| GPT-OSS-120B-Q8 | 5 GPU / 32 CPU | 2.5 / 80s | 20 / 6s | 32 / 4s |
| GPT-OSS-120B-Q4_K_M | 8 GPU / 29 CPU | 3.3 / 70s | 23.5 / 5s | 35 / 3.5s |
| Qwen3-8B-Q8_0 | GPU 全載 (9.5 GB) | 73 / 0.05s | 73 / 0.05s | – |
| Qwen3-8B-Q8_0 | 15 GPU / 22 CPU | 16 / 1s | 16 / 1s | – |
| Qwen3-8B-Q8_0 | 0 GPU / 37 CPU | 10 / 1.3s | 10 / 1.3s | – |
| Qwen3.5-35B-Q4_K_M | auto (15.5 GB) | 70 / 1s | 80 / 0.5s | – |
| Qwen3.5-35B-Q4_K_M | 20 GPU / 21 CPU | 33 / 1.5s | 33 / 1s | 71 / 0.5s |
| Qwen3.5-35B-Q4_K_M | 0 GPU / 41 CPU | 19 / 2s | 19 / 2s | 52 / 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (15.7 GB) | 43 / 1.5s | 45 / 0.7s | – |
| Gemma4-26B-A4B-Q8_0 | 15 GPU / 16 CPU | 30 / 2s | 30 / 1.5s | 44 / 0.5s |
| Gemma4-26B-A4B-Q8_0 | 0 GPU / 31 CPU | 18 / 3.5s | 18 / 2.5s | 33 / 0.5s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU | 6 / 2s | 6 / 1.5s | – |
| Gemma4-31B-Q8_0 | auto (15.8 GB) | 3.5 / 2s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU | 3.5 / 3s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU | 3 / 60s | 3 / 3s | – |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU | NA | 2.5 / 3s | – |

</details>

<details>
<summary><b>D · Intel Core Ultra 7 355 (iGPU)</b> — 完整結果 (Windows 11，依 DRAM 容量)</summary>

格式：`iGPU / CPU layer 切分 — TPS / TTFT`。

| 模型 | 32 GB DRAM | 64 GB DRAM |
|------|------------|------------|
| GPT-OSS-20B-FP16 | iGPU 全載 (13 GB) — 15 / 0.9s | iGPU 全載 (13 GB) — 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M | iGPU 全載 (12 GB) — 20 / 0.9s | iGPU 全載 (12 GB) — 20 / 0.9s |
| GPT-OSS-120B-Q8 | NA | iGPU 20L (33 GB) / CPU 17 — 7.5 / 17s |
| GPT-OSS-120B-Q4_K_M | iGPU 10 / CPU 27 — 1.9 / 60s | iGPU 20L (33 GB) / CPU 17 — 9.5 / 15s |
| Qwen3-8B-Q8_0 | iGPU 全載 (10 GB) — 9.5 / 1s | iGPU 全載 (10 GB) — 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M | iGPU 30L (20 GB) / CPU 11 — 13 / 3s | iGPU 全載 (23 GB) — 20 / 1s |
| Gemma4-26B-A4B-Q8_0 | iGPU 20L (20 GB) / CPU 11 — 10.5 / 6s | iGPU 全載 (29 GB) — 15 / 1s |
| Gemma4-31B-Q4_K_M | iGPU 全載 (22 GB) — 3.6 / 6s | iGPU 全載 (22 GB) — 4 / 2s |
| Gemma4-31B-Q8_0 | iGPU 30L (18 GB) / CPU 31 — 0.05 / 60s | iGPU 全載 (33 GB) — 2 / 2s |

</details>

---

