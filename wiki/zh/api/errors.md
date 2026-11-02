---
title: 錯誤模型
summary: 狀態碼在這個服務裡的實際語意、detail 的三種形狀、串流開始之後的錯誤怎麼送，以及 OOM 與 vLLM 啟動失敗的分類。
updatedAt: 2026-10-29
---

# 錯誤模型

## `detail` 有三種形狀

**呼叫端必須三種都處理**，否則會把錯誤訊息吞掉。

```jsonc
// 字串（最常見）
{"detail": "Model with label 'foo' not found"}

// list[dict]：422 驗證錯誤；句子在 msg、欄位在 loc
{"detail": [{"loc": ["body", "vllm_max_model_len"],
             "msg": "Input should be a valid integer, got a number with a fractional part",
             "type": "int_parsing"}]}

// dict：507 OOM、410 已廢棄、400 GGUF check
{"detail": {"error": "Out of Memory (OOM)", "message": "...", "is_oom": true,
            "suggestions": ["Use POST /inference/force_cleanup_gpu ...", "..."]}}
```

> `422` 的 `msg` 常帶 pydantic 的 `"Value error, "` 前綴；跨欄位的規則（例如「這個 offload 設定不會有任何作用」）
> 的 `loc` 不指向任何欄位。

## 狀態碼的語意

| 碼 | 意思 |
|---|---|
| `400` | 參數不合法，或不符合領域規則（例如對非 GGUF 做 GGUF 估算） |
| `404` | 指定的資源不存在（task、job、模型、檔案） |
| **`409`** | **目前的狀態不允許**：模型已載入或正在載入、訓練進行中、重複刪除、sleep／wake 狀態不對、引擎不支援 sleep |
| `410` | `POST /inference/chat` 已廢棄 |
| `422` | 請求驗證失敗 |
| `500` | 未預期的例外 |
| **`501`** | **這台機器不支援**：非 Linux 上微調、vLLM 未安裝 |
| `502` | sleep／wake 時引擎那邊失敗 |
| `503` | 模型睡著且 auto-wake 關閉，附 `Retry-After: 1` |
| **`507`** | **OOM**，`detail` 帶 `is_oom` 與 `suggestions` |

三個刻意的選擇：

- **`409` 而不是 `400`**：請求沒寫錯，等一下再送就會成功。
- **`501` 而不是 `403`**：平台不支援不是權限問題。`detail` 是一句可以直接顯示給使用者的原因。
- **`507` 而不是 `500`**：OOM 是資源不足，而且可以給建議。

> 大部分 `500` 的 `detail` 是底層例外的文字，**不一定適合直接給使用者看**。

### 驗證層的碼

遠端存取的檢查在路由之前，`detail` 一律是字串：

| 碼 | 什麼時候 |
|---|---|
| `401` | 別台機器沒帶 key 或 key 錯 |
| `403` | 別的網站的網頁、`Host` 不是 localhost 或 IP、路徑不在允許清單、遠端存取沒開 |
| `429` | 同一位址錯太多次 key，被鎖定 |

逐條說明見 [從症狀反查](../operations/troubleshooting.md#別台機器連不上)。

## 串流開始之後的錯誤

串流一旦開始，HTTP 狀態碼已經是 200。之後的錯誤以 OpenAI 風格的 error 物件送進事件流，然後結束串流：

```
data: {"error": {"message": "CUDA out of memory...", "type": "server_error"}}
```

| 路徑 | 錯誤在哪 |
|---|---|
| 一般請求（含非串流 chat） | HTTP 狀態碼 + `detail` |
| 串流 chat 中途 | SSE 事件的 `error.message` |

用戶端斷線與生成失敗被刻意分開處理：只有連線類的例外會被當成「用戶端斷了」，
其他錯誤一律變成結構化的 SSE 錯誤——否則生成失敗會看起來像「連線莫名斷掉」。

## OOM 在哪裡看得到

| 來源 | 形狀 |
|---|---|
| 非串流 chat | `507` + `detail.is_oom: true` + `suggestions` |
| `GET /inference/error_details` | `{"has_error", "is_oom", "error_traceback", "process_alive"}` |
| vLLM 啟動失敗 | 分類為 `oom` |

## vLLM 啟動失敗分十一類

```
oom · config · port_busy · model_not_found · cuda_mismatch · gpu_fault
shared_library_missing · toolchain_missing · chat_template · quantization · unknown
```

**先看分類，不要直接假設是 OOM。** 詳見 [vLLM](../engines/vllm.md)。

## 相關

- [對話](chat.md)
- [遠端存取與信任邊界](remote-access.md)
- [從症狀反查](../operations/troubleshooting.md)
