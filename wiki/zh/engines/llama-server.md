---
title: llama.cpp 引擎（llama_server）
summary: 吃 GGUF、管理一個 llama-server 子行程；參數怎麼組、哪些旗標不能自己寫、slot 快取、mmproj，以及調校的實測。
updatedAt: 2026-10-29
---

# llama.cpp 引擎（`llama_server`）

**吃 GGUF。** 後端管理一個 `llama-server` 子行程，生成由主行程直接用 HTTP 打過去。

## 兩種二進位檔

| 形態 | 來源 | 怎麼呼叫 |
|---|---|---|
| 統一的 `llama` | 官方預編譯版 | 加 `serve` 子指令 |
| 獨立的 `llama-server` | 從原始碼編譯 | 直接給參數 |

兩者參數相同，後端用檔名判斷。用哪一個由 `LLAMA_SERVER_BINARY` 決定；
服務啟動時會在 log 印出它與來源。

## 參數怎麼組

```
[binary] [serve] -m <model> --host --port -np -c -b --alias
         [-ngl] [-ncmoe] [--device] [--slot-save-path] [--mmproj] [--jinja]
         --api-key <內部 key> --cors-origins <本服務>
         <llama_server_extra_args>
```

**`llama_server_extra_args` 一律放最後**；llama-server 對重複的旗標是最後一個生效，所以你寫的值會蓋掉自動帶的。

| 自動帶的 | 不帶的情況 |
|---|---|
| `-ncmoe` | `n_cpu_moe` 是 0 |
| `--device` | 沒設 `llama_server_device` |
| `--mmproj` | 沒偵測到，或你已經在 extra args 寫了 |
| `--jinja` | 你已經寫了 `--jinja` 或 `--no-jinja` |

### 不能自己寫的旗標

以下旗標由服務管理，寫在 `llama_server_extra_args` 會在載入前被拒絕（422）：
`--host`、`--port`、`--api-key`、`--cors-*`、`--ssl-*`、`--path`、`--api-prefix`、`--slot-save-path`、`--log-file` 等。

理由：llama-server 預設接受任何 origin，這些旗標可以把它移出 loopback、打開 CORS、或讓它讀寫檔案。
同理，`llama_server_url` 也只接受 loopback 位址。

### 內部 API key

後端每次啟動 llama-server 都帶一把 key（`LLAMA_SERVER_API_KEY`；沒設就**每次隨機產生**），
這樣連這台機器上的網頁都無法直接使用那台 llama-server。

> 副作用：若有外部程式直接打 llama-server 的埠，後端重啟後 key 就變了。
> 要嘛設固定的 `LLAMA_SERVER_API_KEY`，要嘛改走後端的 `/v1/`。

### 為什麼明確加 `--jinja`

**thinking 的開關（`enable_thinking`）與 `reasoning_content` 的抽取都需要 GGUF 的 jinja chat template。**
新版 llama-server 預設就開，舊版沒有，所以明確傳。沒有 `--jinja` 就沒有 `reasoning_content`。

### `--device` 的用途

固定 offload 裝置。例如同時有 NVIDIA 與 Intel GPU 的機器，要用 Intel 時指定 `Vulkan1`，
否則 Vulkan 會先選到 NVIDIA。

## slot 快取：純文字模型的對話可以存檔

| 模型 | `prefill_strategy` | 行為 |
|---|---|---|
| 純文字 | **`slot`** | 載入時還原每個 slot 的 KV、卸載時存回 |
| 多模態 | `cache_prompt` | 不動 slot，只用 prompt 快取 |

還原結果在 `/inference/status` 的 `slot_restore_summary`（`{restored, total, slots}`）。

> 實測重點：**還原後要在原本的對話後面接著加**才接得上。
> 重問同一句需要倒回 recurrent 狀態，而 slot 檔沒有 checkpoint，只能整段重算。
> 見 [三種切換方式](../benchmarks/gpu-handoff.md)。

## 回報的中繼資料

| 欄位 | 是什麼 |
|---|---|
| `llama_capabilities` | 伺服器自己說支援什麼 |
| `llama_model_meta` | 含 `n_ctx_train`（模型自己的上限） |
| **`served_context_length`** | **這次載入實際給的窗口** |
| `prefill_strategy`、`slot_restore_summary` | 見上 |

`served_context_length` 與 `n_ctx_train` 並排，就看得出限制來自引擎還是模型。

## mmproj：沒有它視覺就是不會動

多模態 GGUF 的視覺投影檔由後端從模型檔與 extra args 推導。
**偵測到但檔案不存在會直接報錯**，不會默默略過。

> gemma-4 上線時，視覺要在 extra args 加 `--image-min-tokens 280`（實測）。

## 調校的實測

條件：RTX 5060 Ti 16 GB、Intel Core Ultra 5 225（6 個 P-core + 4 個 E-core）、Linux。

| 事實 | 實測 |
|---|---|
| **大小核 CPU 要手動指定 `-t`，而且只用 P-core** | Qwen3.5-35B-A3B Q4_K_M、`-ncmoe 20`：預設 14.1 tok/s → `-t 6` 23.5 → **`-t 10` 反而掉到 9.8**。llama.cpp 平均分工，E-core 會變成每層的瓶頸。用 `llama_server_extra_args` 加 `-t <P-core 數>` |
| **不要餵 BF16 GGUF** | Qwen3-4B 全放 GPU：BF16 49 tok/s，Q8_0 88 tok/s。兩者都達到顯存頻寬上限的約 83%，BF16 只是檔案大一倍 |
| **MoE 放不下時用 `n_cpu_moe`** | 專家權重推到 RAM，attention 與 KV 留在 GPU。見 [記憶體估算](../memory/estimation.md) |
| **在 AST 之外量 llama-server 要設 `OMP_WAIT_POLICY=active`** | 專家在 CPU 時，AST 內的 decode 比單獨執行快約 16%（47.5 對 40.7 tok/s）。差別來自 OpenMP 的等待策略：AST 的環境剛好載入了 PyTorch 附的 libgomp。做 A/B 比較時兩邊要一致 |

## 啟動失敗

後端收集子行程最近的 stderr（最多 12 行／2000 字），依結束碼組一句摘要，
並判斷是否為 OOM，不會把所有失敗都說成 OOM。

## 相關

- [推論引擎怎麼選](index.md)
- [記憶體估算](../memory/estimation.md) — `-ngl`、`-ncmoe` 怎麼算
- [GGUF 轉換](../models/gguf-conversion.md)
- [對話](../api/chat.md) — thinking 與 `reasoning_content`
