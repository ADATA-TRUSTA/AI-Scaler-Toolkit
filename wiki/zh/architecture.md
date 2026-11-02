---
title: 服務架構
summary: 後端的三個行程、兩條推論路徑、狀態存放的位置，以及哪些限制是真的擋、哪些只是警告。
updatedAt: 2026-10-29
---

# 服務架構

本頁講**行程、狀態、互斥**，也就是「為什麼這裡不能那樣做」。
API 清單見 [API 總覽](api/index.md)；訓練流程見 [微調總覽](training/index.md)。

## 一分鐘版本

```
                    ┌──────────────────────────────────┐
                    │  FastAPI 主行程（uvicorn，單一）   │
                    └────┬───────────────────┬─────────┘
                         │ spawn             │ spawn
                         ▼                   ▼
              ┌────────────────────┐  ┌────────────────────┐
              │ 推論 worker 行程     │  │ 訓練 worker 行程     │
              │ transformers 在這裡 │  │ 一次一個 job         │
              └────────────────────┘  └────────────────────┘

   llama-server／vLLM／Strata 不走 worker：
   主行程直接用 async HTTP 打「被管理的那台伺服器」
```

**三個行程、兩條推論路徑。** 其餘都是細節。

## 為什麼要拆行程

**模型放在獨立行程，OOM 時 VRAM 才收得回來。** 行程結束，CUDA context 跟著消失；
同一個行程裡 `del model` 加 `torch.cuda.empty_cache()` 做不到這件事，OOM 之後的狀態本來就不乾淨。
訓練放在獨立行程，則是為了讓訓練的阻塞或 OOM 不會凍住 API。

所以**卸載 transformers 模型就是結束 worker 行程**；VRAM 一定收得回來。

### `spawn` 是強制的

`app.py` 在幾乎所有 import 之前就設定 `multiprocessing.set_start_method("spawn")`。
`fork` 會把已初始化的 CUDA context 複製到子行程，那是未定義行為；
只要有任何模組先碰了 CUDA，再改 start method 就來不及。

代價是子行程什麼都不繼承，每個 worker 要自己重新 import、重新讀設定。

## 兩條推論路徑

| 引擎 | 路徑 | 主行程怎麼拿到結果 |
|---|---|---|
| `transformers` | **worker 佇列** | 經由 multiprocessing queue，由一條 dispatcher 執行緒分給各請求 |
| `llama_server`、`vllm`、`strata` | **async HTTP** | 主行程直接 `await` 那台伺服器 |

三個 server engine 本來就提供 OpenAI 相容 HTTP，所以主行程直接打過去；
一個請求就是 event loop 上的一次 `await`。兩條路徑回傳給呼叫端的**形狀相同**。

### 這個差異在哪裡看得到

| 現象 | 原因 |
|---|---|
| `reasoning_content` 只有 server engine 有 | 上游伺服器會把 reasoning 分開；transformers 路徑沒有伺服器端的 reasoning parser。**不是 bug** |
| `/inference/error_details` 對 server engine 沒內容 | 它讀的是 worker 記錄的錯誤，server engine 的生成不經過 worker |
| `/inference/status` 各引擎回的欄位不同 | transformers 回逐模組放置；llama.cpp 回層數分配與 `served_context_length`；vLLM 回 endpoint；Strata 只回裝置。**呼叫端要容忍缺值** |

### 埠號必須在主行程決定

llama-server 預設的埠被佔用時要換一個。這件事必須在主行程做、**在 worker 拿到設定副本之前**。
如果放在引擎裡做，worker 改的是自己那份副本，主行程讀的仍是舊埠號——
**模型載入成功，每次生成都連不上**。

這正是 spawn 架構的代價：兩個行程各有一份設定，**任何要兩邊一致的決定都只能在主行程做完再傳下去**。

## 狀態放在哪裡

排查問題時先問這個：

| 狀態 | 存哪 | 服務重啟後 |
|---|---|---|
| 已載入的模型與設定 | 主行程記憶體 + worker | **消失** |
| 下載任務 | 主行程記憶體 | **消失** |
| 訓練即時狀態 | 記憶體 | **消失** |
| 訓練事件流 | **磁碟** `LOG_DIR/training/{job_id}/` | 保留 |
| 對話歷史（`session_id`） | Redis；**沒設定就用記憶體** | 看有沒有 Redis |
| 已登錄的模型 | 磁碟 JSON | 保留 |

Redis 只在設定了才使用。**沒設定時記憶體就是正式的儲存，不是連線失敗的退路**；
所以沒有 Redis 也不會出現連線錯誤的警告。

服務啟動時**不會載入任何模型**。worker 在第一次用到時才啟動。

## 互斥：哪些是真的擋，哪些只是警告

### 真的擋

| 規則 | 行為 |
|---|---|
| 一次只能載一個推論模型 | 相同模型且關鍵設定相同 → `200 already_loaded`；不同模型或設定 → **`409`**；正在載入中 → `409` |
| 一次只能跑一個訓練 job | 拒絕開始 |
| 同時生成數有上限 | `MAX_CONCURRENT_GENERATIONS`，預設 8 |
| 每次生成有逾時 | `DEFAULT_GENERATION_TIMEOUT`，預設 1800 秒 |

> **「關鍵設定相同」只比六個欄位**：`quantization`、`device_map`、`max_memory`、
> `offload_folder`、`torch_dtype`、`model_total_memory`（加上模型路徑或名稱）。
> **只改其他欄位重新載入，會被當成「已載入」而回 200，不會真的重載。**
> 要讓其他設定生效，先卸載再載入。

### 只是警告

**訓練與推論可以同時使用同一張 GPU。** 開始訓練時若有模型已載入，伺服器只在 log 留一行警告，
API 呼叫端看不到。在 16 GB 的卡上這通常就是 OOM。**開始訓練前先卸載模型。**

## 誰能連進來

兩層 middleware 在路由之前判斷：CORS 只接受本機的頁面；
另一層依來源分三套規則——本機、別台機器（要 key，預設只有 `/v1/`）、瀏覽器裡的網頁。
詳見 [遠端存取與信任邊界](api/remote-access.md)。

這一層刻意寫成純 ASGI middleware，**串流回應（chat、訓練 log）才能原樣通過**。

## 相關

- [推論引擎怎麼選](engines/index.md)
- [API 總覽](api/index.md)
- [微調總覽](training/index.md)
- [遠端存取與信任邊界](api/remote-access.md)
