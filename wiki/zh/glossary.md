---
title: 術語表
summary: 在這個專案裡意思與一般用法不同、或同一個字在不同地方指不同東西的術語。
updatedAt: 2026-10-29
---

# 術語表

**只收兩種詞**：在這個專案裡意思和一般用法不同的，以及**同一個字在不同地方指不同東西**的。
一般的 LLM 術語不收。

## 同名不同義

### `engine`

| 在哪 | 意思 |
|---|---|
| 載入設定的 `engine` 欄位 | `transformers`／`llama_server`／`vllm`／`strata` 四者之一 |
| 引擎類別 | 四者做的事不一樣：**只有 transformers 在 worker 裡生成**，其餘三個只負責起停伺服器 |
| 介面上的文字 | 顯示 `llama_cpp`，是給人看的名稱，API 的值是 `llama_server` |

### `offload`

| 情境 | 指什麼 |
|---|---|
| `device_map`（transformers） | 權重放哪裡（GPU／CPU／磁碟） |
| `vllm_cpu_offload_params` | 只把 **MoE 專家權重**放 CPU |
| `vllm_kv_offloading_size` | **KV cache** 放 DRAM，不是權重 |
| DeepSpeed ZeRO-3 offload | **訓練時**的參數與 optimizer state |

**看到 offload 先問「offload 什麼」。** 詳見 [Offload](memory/offload.md)。

### `quantization`

| 引擎 | 值 | 怎麼指定 |
|---|---|---|
| transformers | `int4`／`int8`／`none`（bitsandbytes） | `quantization` 欄位 |
| llama.cpp、Strata | 寫在 GGUF 裡（`Q4_K_M` 等） | 選哪個 GGUF 檔 |
| vLLM | AWQ／GPTQ／FP8… | `vllm_quantization` 欄位 |

### `status`

| 哪裡 | 值 |
|---|---|
| 下載任務 | `pending`／`running`／`cancelling`／`completed`／`failed`／`cancelled` |
| 已登錄的模型 | `ready`／`deleting`／`delete_failed` |
| 訓練 | `idle`／`initializing`／`running`／`completed`／`stopped`／`error` |

`cancelling` 與 `cancelled` 不同：前者是取消已收到、背景工作還沒停。

### `job` 與 `task`

沒有語意差別，只是命名不一致：下載用 `task_id`，GGUF 轉換與訓練用 `job_id`。

### context 長度

| 欄位 | 意思 |
|---|---|
| `max_context_length` | 模型清單記錄的上限 |
| `n_ctx` | llama-server 啟動時要求的窗口 |
| **`served_context_length`** | **引擎這次載入實際給的窗口** |
| `n_ctx_train` | 模型訓練時的上限（llama.cpp 回報） |

後兩個並排，就看得出限制來自引擎還是模型。

## 專案自己的詞

| 詞 | 意思 |
|---|---|
| **worker 行程** | 用 `spawn` 起的子行程，分推論與訓練兩種。卸載模型就是結束它，VRAM 才收得回來 |
| **server engine** | `llama_server`、`vllm`、`strata`：對外提供 OpenAI 相容 HTTP 的那三個。生成不經過 worker |
| **phase** | 訓練的細粒度階段（`loading_model`、`training`、`saving`…）；`status` 是粗粒度的 |
| **slot** | llama-server 的併發槽（`-np`）。純文字模型的 slot KV 可以存檔與還原 |
| **mmproj** | 多模態 GGUF 的視覺投影檔。**少了它，視覺就是不會動，而且不會報錯** |
| **sleep／wake** | 放掉 GPU 但不卸載模型。vLLM 會丟掉 KV；Strata 連對話一起保住 |
| **run config** | Strata 的模型描述檔（`strata-<model>.json`），由 Strata 的 setup 產生 |

## 相關

- [全貌](overview.md)
- [推論引擎怎麼選](engines/index.md)
- [Offload](memory/offload.md)
