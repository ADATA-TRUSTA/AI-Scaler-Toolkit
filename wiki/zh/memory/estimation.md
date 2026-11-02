---
title: 記憶體估算
summary: 兩套估算器（HF 模型與 GGUF 模型）、GGUF 估算的 analytic 與 oracle 兩條路線、五條 GGUF 端點各回答什麼，以及從 llama.cpp 抄來的放置規則。
updatedAt: 2026-10-29
---

# 記憶體估算

## 兩套估算器，不要混用

| | HF 模型 | GGUF 模型 |
|---|---|---|
| 對象 | transformers 載入的模型 | llama.cpp 服務的 GGUF |
| 端點 | `POST /inference/estimate_memory` | `POST /inference/estimate_memory/gguf` 與四個變體 |
| 回答 | 大概要多少記憶體 | **哪幾層在哪個裝置、這組設定塞不塞得下** |

兩者的精度完全不同。

## HF 模型：三條退路

依序嘗試：讀 config 推算 → 在 `meta` device 上建一次模型 → 讀 safetensors 的 index。
第一條對少見架構會失準；第二條需要能實例化；第三條要模型已下載。

連不上 Hugging Face 時只探一次（2 秒），之後記住，不會每個請求都等 timeout。
gated repo 或離線時，錯誤會告訴你提供 `HF_TOKEN` 或先下載模型。

## GGUF 模型：兩條路線

| 路線 | 怎麼做 | 代價 | 精度 |
|---|---|---|---|
| **analytic**（預設） | 從 GGUF 的 tensor table 用純 Python 重現 llama.cpp 的放置與 KV 計算 | 幾乎為零 | 權重與 KV **精確**；compute buffer 用校準公式，**刻意偏高** |
| **oracle**（`verify: true`） | 執行 `llama-fit-params -fitp on`，即 llama.cpp 自己的 dry-run 載入 | 每個資料點一個行程（約 0.4 秒） | **精確** |

sweep 會試很多組合，每組起一個行程不可行，所以預設用 analytic；
要最後確認某一組行不行，再加 `verify: true`。

準確度實測：對照 `llama-fit-params`，6 種架構的權重與 KV 誤差都在 1 MiB 內、從不低估；
compute buffer 高估 0–26%。見 [GGUF 記憶體估算的準確度](../benchmarks/gguf-estimation-accuracy.md)。

## 五條 GGUF 端點

路徑都在 `/inference/estimate_memory/` 底下。

| 端點 | 問題 | 重要的回應欄位 |
|---|---|---|
| `gguf` | 這組設定要多少 | `memory_breakdown_mib`、`placement`、`per_layer_mib` |
| `gguf/check` | **這組設定塞得下嗎** | `fits`、`verdict`、`summary`、`suggestion` |
| `gguf/recommend` | **幫我配一組** | `recommended`、`utilization`、**`llama_server_args`** |
| `gguf/sweep` | 不同組合的對照表 | `rows`、`truncated` |
| `gguf/plan` | 每種 KV 型別各給一組方案 | `plans`、`kv_cache_types` |

**`recommend` 回的 `llama_server_args` 可以直接貼進 `llama_server_extra_args`**——
這是「載不起來怎麼辦」最短的解法。實測它給的參數用掉預算的 96.6–97.8%。

`gguf/check` 的對象不是 GGUF 時回 `400`，`detail` 是 dict 並帶 `hint`，說明這條端點只檢查 GGUF。
每個裝置預留的空間沿用 `llama-fit-params` 的 `-fitt` 預設值。

## llama.cpp 的放置規則

這幾條規則是逐條照 llama.cpp 的原始碼寫的，也解釋了 llama.cpp 一些反直覺的行為：

| 規則 | 後果 |
|---|---|
| 有 `n_layer + 1` 格，**GPU 拿最後的 `n_gpu_layers` 格**，最後一格是 output 層 | **output 層比任何一個 block 先上 GPU**；`-ngl 1` 只放 output 層，不是「第一層」 |
| token embeddings **永遠留在 CPU** | 不計入 GPU 預算 |
| `--n-cpu-moe N` 把前 N 層的專家 tensor 改放 CPU | 專家推到 RAM，**attention 與 KV 留在 GPU** |
| KV cache 跟著擁有那一層的裝置 | |
| `n_ctx` 向上補到 256 的倍數 | |

> 這些規則是照抄的，llama.cpp 改了它們就會失準；compute buffer 的校準公式也是一次性的。
> 升級 llama.cpp 後要重新對照。

### 實測出來、容易誤解的行為

| 行為 | 實測 |
|---|---|
| 不開 flash attention 時 compute buffer 暴增 | 32B、32K ctx：344 → 5,256 MiB。所以 `recommend` 一律寫 `-fa on` |
| `-ncmoe` 不增加 compute buffer，降 `-ngl` 會 | |
| tied embedding 的模型會多一份 `token_embd` | gemma-4-26B-A4B Q4_K_M：`-ngl 1` 時 GPU 剛好多 748 MiB |
| hybrid linear attention 只有部分層有 KV | Qwen3.5-9B、c=4096：KV 178 MiB，不是全部當 attention 算的 512 MiB |

## 相關

- [Offload](offload.md)
- [llama.cpp 引擎](../engines/llama-server.md)
- [GGUF 記憶體估算的準確度](../benchmarks/gguf-estimation-accuracy.md)
