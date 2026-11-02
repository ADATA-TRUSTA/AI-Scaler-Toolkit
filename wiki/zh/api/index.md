---
title: API 總覽
summary: 後端 54 條路由的分組與用途；路由清單本身在 docs/usage.md，本頁補充各組的行為與限制。
updatedAt: 2026-10-29
---

# API 總覽

完整的路由清單與命令列範例在 [`docs/usage.md`](../../../docs/usage.md#api-路由)；
互動式文件在服務的 `/docs`（只在本機開得到）。本頁補充**各組的行為與限制**。

## 基本事實

| 事實 | 說明 |
|---|---|
| 位址 | 桌面程式：`http://127.0.0.1:8571`；從原始碼執行：`SERVICE_PORT`，預設 `8000` |
| 路徑前綴 | **沒有 `/api`**，全部掛在根路徑。經反向代理時的前綴由代理加上 |
| 誰能呼叫 | 本機全開；別台機器要 key，預設只有 `/v1/`。見 [遠端存取](remote-access.md) |
| 錯誤格式 | 見 [錯誤模型](errors.md) |

## 分組

| 組 | 路徑 | 說明 |
|---|---|---|
| 服務 | `/health`、`/` | `/health` 回報可用的引擎與是否能微調；是**唯一對網路全開**的路徑 |
| 對話 | `/v1/chat/completions`、`/v1/models` | OpenAI 相容。見 [對話](chat.md) |
| 推論生命週期 | `/inference/load_model`、`unload_model`、`status`、`stop_generation`… | 一次一個模型 |
| sleep／wake | `/inference/sleep`、`/inference/wake` | 放掉 GPU 但不卸載（vLLM、Strata） |
| 磁碟 KV 層 | `/inference/kv_cache/disk`、`/cleanup` | vLLM 的 NVMe KV 用量與清理 |
| 記憶體估算 | `/inference/estimate_memory`、`/gguf`、`/gguf/{check,recommend,sweep,plan}` | 見 [記憶體估算](../memory/estimation.md) |
| 模型 | `/config/models`、`/download`、`/convert`… | 下載、刪除、轉 GGUF |
| 列舉與範例 | `/config/{quantization,offload}_types`、`/config/training_methods`、`/examples/*` | |
| 訓練 | `/training/*` | Linux。見 [微調總覽](../training/index.md) |
| 文件（RAG） | `/rag/docs`、`/rag/search` | |
| 系統 | `/system/resources` | CPU、RAM、GPU、磁碟；`?mode=spec|usage` |

`POST /inference/chat` 已廢棄，回 `410`；請改用 `/v1/chat/completions`。

## 推論生命週期

### 一次一個模型

| 情況 | `POST /inference/load_model` 回 |
|---|---|
| 同一個模型、關鍵設定相同 | `200 {"status": "already_loaded"}` |
| 不同模型或設定 | **`409`**，要先卸載 |
| 正在載入中 | `409` |
| 這台不支援該引擎 | `501` 與原因 |

> **「關鍵設定」只有六個欄位**（`quantization`、`device_map`、`max_memory`、`offload_folder`、`torch_dtype`、
> `model_total_memory`）。只改其他欄位重載會回 `already_loaded`，**不會生效**；先卸載再載入。

worker 異常時狀態轉為 `error`，**復原是手動的**：重新 `load_model`，或呼叫 `POST /inference/force_cleanup_gpu`。
`GET /inference/error_details` 回 worker 記錄的錯誤（含 `is_oom` 與 traceback）。

### sleep／wake

只有 `vllm`（載入時要開 `vllm_enable_sleep_mode`）與 `strata` 支援；其他引擎回 `409`。
差別在 KV：vLLM 丟掉，Strata 連對話一起保住。

| 請求 | 回應 |
|---|---|
| `POST /inference/sleep {"mode": "wait"}` | 等進行中的生成結束再睡 |
| `POST /inference/sleep {"mode": "abort"}` | 立刻中止再睡 |
| 引擎那邊失敗 | `502` |
| 睡著時收到生成請求 | 自動喚醒；`auto_wake` 關閉時回 `503` 與 `Retry-After: 1` |

**後端不會自己決定何時睡。** 由呼叫端協調（例如 agent 要讓 GPU 給影像生成時）。
wake 可能因為 GPU 記憶體被別人佔走而失敗（`502`），要從 `/inference/status` 重讀狀態。

## 模型下載

| 行為 | 說明 |
|---|---|
| `POST /config/models/download` | 立刻回 `task_id`，背景下載。**打錯的 model id 不會立刻失敗**，要等背景任務連上 Hub |
| GGUF repo 沒給 `filename` | **會下載全部量化版本**（可能好幾 GB） |
| `POST /config/models/download/{task_id}/cancel` | 任務先轉 `cancelling`，停下後轉 `cancelled` |
| 下載任務 | 只存在記憶體，**重啟後消失** |
| `DELETE /config/models/{label}` | 回 `202`，背景刪檔；模型先進入 `deleting`，失敗時 `delete_failed`；重複刪同一個回 `409` |

> 開著 `HF_XET_ENABLED` 時，Xet 的傳輸無法中途中斷，取消可能要等到目前的檔案下載完。

## RAG

`/rag/*` 管理文件與搜尋；`/v1/chat/completions` 另有 `use_rag`、`rag_top_k`、`rag_query`、`rag_include_sources` 欄位。
網頁介面目前沒有 RAG 的操作畫面。

## 相關

- [對話](chat.md)
- [錯誤模型](errors.md)
- [遠端存取與信任邊界](remote-access.md)
- [服務架構](../architecture.md)
