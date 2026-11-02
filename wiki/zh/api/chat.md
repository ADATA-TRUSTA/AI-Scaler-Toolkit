---
title: 對話（OpenAI 相容）
summary: /v1/chat/completions 在 OpenAI 協定之外加了哪些欄位、thinking 與 reasoning_effort 怎麼處理、tool call 與 reasoning_content 的形狀，以及 Qwen3.5 的取樣覆寫。
updatedAt: 2026-10-29
---

# 對話（OpenAI 相容）

`POST /v1/chat/completions`，支援串流（SSE）與 tool call。任何 OpenAI SDK 都能直接用：

| 設定 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1`（桌面程式） |
| 模型 | `trusta-ast-default`，或省略——服務的就是目前載入的那一個 |
| API key | 本機任意字串；別台電腦用設定頁顯示的 key |

## OpenAI 之外的欄位

| 欄位 | 用途 |
|---|---|
| `top_k`、`repetition_penalty` | 取樣 |
| `enable_thinking` | 思考開關（見下） |
| `chat_template_kwargs` | 直接傳給 chat template |
| `session_id`、`reset_history` | 伺服器端對話歷史（Redis 或記憶體） |
| `total_timeout` | 這次請求的逾時（秒） |
| `use_rag`、`rag_top_k`、`rag_query`、`rag_include_sources` | RAG |
| `request_id` | 呼叫端自己的追蹤 id |

用 OpenAI SDK 時，這些欄位放在 `extra_body`。

## 回應裡值得知道的兩件事

| 事 | 說明 |
|---|---|
| **思考過程在 `reasoning_content`** | 不是包在 `content` 裡的 `<think>` 標記。只有 server engine（llama.cpp、vLLM、Strata）會有；transformers 路徑沒有 |
| **串流的 tool call 一定帶 `index`** | OpenAI 協定要求的欄位；缺了的話某些 SDK 會把整筆 tool call 丟掉 |

`stream_options: {"include_usage": true}` 時，串流結束前多送一個帶 token 統計的 chunk。

## thinking：三個來源，有優先順序

```
chat_template_kwargs["enable_thinking"]  →  enable_thinking  →  reasoning_effort
```

### `reasoning_effort`

OpenAI 的標準欄位，後端做兩件事：

**一、在沒有明確開關時折成 `enable_thinking`。** 很多 template 只認布林值。

| `reasoning_effort` | 折成 |
|---|---|
| `none`、`minimal` | `enable_thinking = false` |
| 其他任何值（`low`、`medium`、`high`、`max`…） | `enable_thinking = true` |
| 沒給 | 不干涉 template |

「關」是白名單、其他都算「開」，因為分級的詞彙一直在增加；反過來列舉會把新出現的等級靜默當成關。

**二、往引擎轉送，但限制在 `low`／`medium`／`high`。**

| 收到 | 轉送 |
|---|---|
| `none`、`minimal` | **不送**（改用 `enable_thinking=false` 關閉） |
| `low`、`medium`、`high` | 原樣 |
| `max` 或更高 | `high` |

理由：vLLM 渲染 gpt-oss 時只接受這三級，給其他值會報錯；而停不下思考的模型會拒絕 `none`。
代價是有更高等級的模型會被限制在 `high`。

### vLLM 有 reasoning parser 時預設關

見 [vLLM](../engines/vllm.md)。

## Qwen3.5 家族的取樣覆寫

後端依模型名稱與架構辨識 Qwen3.5 家族，並**覆寫呼叫端送的值**：

| 覆寫 | 條件 | 理由 |
|---|---|---|
| `repetition_penalty > 1.0` → `1.0`（並記 warning） | **一律**，含 tool 請求 | Qwen3.5（尤其 MoE）在 >1.0 時會產生混雜語言的亂碼 |
| `temperature=1.0`、`top_p=0.95`、`top_k=20` | 一般請求 | 對齊官方的 generation_config |
| 不覆寫取樣 | **帶 `tools` 的請求** | tool call 是結構化輸出，偏離呼叫端的取樣只會損失可靠性 |

最後一條是量測出來的：Qwen3.5-35B-A3B 同一個請求與 tool，直連 llama-server 產生 153 個 tool-call delta，
覆寫取樣時是 0 個（改成回一個 markdown code block）。

tool 請求而且沒指定 `enable_thinking` 時：**`max_tokens` 夠大（或沒給）才開思考**。
thinking 是 tool call 會發生的原因，但它也燒預算——`max_tokens=512` 的請求可能把整個預算花在 `<think>` 裡、永遠不回答。

Qwen3.5 的 thinking 由 template 預填 `<think>` 啟動，只認 `chat_template_kwargs`，沒有 `/think` 這種軟開關。

## 上傳圖片：JSON 或 multipart

標準做法是把圖片的 data URL 寫進 `messages[].content`。
後端另外接受 **`multipart/form-data`**，給不想自己做 base64 的呼叫端：

| form 欄位 | 內容 |
|---|---|
| `payload`（或 `request`） | JSON 字串，就是原本的請求 body |
| `image`、`images`、`file`、`files`、`attachment`、`attachments` | 檔案；會接到最後一則 user message |

form 的值都是字串，後端會依欄位轉型（`max_tokens`、`top_k` 是整數；`temperature`、`top_p` 是浮點數）。

## 模型睡著時

| 情況 | 回應 |
|---|---|
| 經後端睡著、`auto_wake` 開（預設） | 先喚醒再生成 |
| 經後端睡著、`auto_wake` 關 | `503` + `Retry-After: 1` |
| 別的程式直接對 vLLM 打了 `/sleep` | 立刻失敗並說明模型睡著，不會等到逾時 |

## 隱私

服務的 log 在 INFO 等級只記錄請求的**形狀**（模型、訊息數、是否串流、`max_tokens`），不記對話內容；
完整內容只在 DEBUG 等級記錄。

## 相關

- [錯誤模型](errors.md) — 串流中的錯誤
- [llama.cpp 引擎](../engines/llama-server.md) — `--jinja` 與 `reasoning_content`
- [vLLM](../engines/vllm.md) — tool parser 偵測
