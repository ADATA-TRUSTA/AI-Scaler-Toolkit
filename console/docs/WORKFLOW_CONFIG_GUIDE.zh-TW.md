# Workflow 設定指南

English version: [WORKFLOW_CONFIG_GUIDE.md](WORKFLOW_CONFIG_GUIDE.md)

## 概觀

一份 workflow 設定檔定義了完整「載入 -> 對話 -> 卸載」流程所需的所有參數，包括：

- 推論模型設定（使用哪個引擎與哪個模型）
- 對話設定（prompts、生成參數、串流／非串流）
- 後端 URL 與 timeout

## 設定檔結構

```json
{
  "description": "Configuration description",
  "inference_config": "configs/inference/transformers.json",
  "chat_settings": "configs/chat/batch.json",
  "backend_url": "http://your-backend-host",
  "timeout": 600,
  "verbose": false
}
```

### 欄位說明

| 欄位               | 型別    | 必填 | 說明                                                       |
| ------------------ | ------- | ---- | ---------------------------------------------------------- |
| `description`      | string  | 否   | 這份設定的說明文字                                         |
| `inference_config` | string  | 是   | 推論模型設定檔的路徑                                       |
| `chat_settings`    | string  | 否   | 對話設定檔的路徑。省略時使用預設 prompts                   |
| `backend_url`      | string  | 否   | 後端 URL。預設取 `.env` 中的 `BACKEND_URL`                 |
| `timeout`          | integer | 否   | 請求逾時秒數。預設為 600                                   |
| `verbose`          | boolean | 否   | 是否印出詳細日誌。預設為 false                             |

## 內建的 Workflow 設定

### 1. default.json

- 用途：標準 Transformers 推論
- 推論引擎：Transformers（全精度／量化）
- 對話設定：基本對話設定（4 個 prompt）
- 建議使用場景：快速測試與一般推論

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json
```

### 2. vllm.json

- 用途：高速推論（vLLM 引擎）
- 推論引擎：vLLM
- 對話設定：平行批次設定（8 個 prompt）
- 建議使用場景：高吞吐量情境

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/vllm.json
```

### 3. transformers_offload.json

- 用途：搭配 GPU/CPU offload 的大模型推論
- 推論引擎：Transformers with offload
- 對話設定：基本對話設定
- 建議使用場景：VRAM 較少的環境與較大的模型

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/transformers_offload.json
```

### 4. llama.json

- 用途：GGUF 格式推論（llama.cpp）
- 推論引擎：llama.cpp/llama_server profile
- 對話設定：基本對話設定
- 建議使用場景：量化模型與以 CPU 為主的推論

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/llama.json
```

### 5. parallel_batch_16.json

- 用途：高併發批次推論
- 推論引擎：vLLM
- 對話設定：平行批次設定（16 個 prompt）
- 建議使用場景：大批次工作負載與效能測試

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/parallel_batch_16.json
```

## 使用方式

### 方式 1：使用 workflow 設定（建議）

```bash
# 預設 workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json

# vLLM workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/vllm.json
```

### 方式 2：以命令列參數覆寫

以 workflow 設定為基礎，再用命令列參數覆寫指定的值：

```bash
# vLLM workflow，但改用不同的後端 URL
uv run examples/workflow/full.py \
    --workflow-config configs/workflow/vllm.json \
    --backend-url http://localhost:8000

# 預設 workflow，但改用不同的對話設定
uv run examples/workflow/full.py \
    --workflow-config configs/workflow/default.json \
    --chat-settings configs/chat/parallel_8.json
```

### 方式 3：不使用 workflow 設定（全部明確指定）

```bash
uv run examples/workflow/full.py \
    --config configs/inference/vllm.json \
    --chat-settings configs/chat/batch.json \
    --backend-url http://your-backend-host \
    --timeout 900
```

## 對話設定格式

對話設定檔定義 prompt 與生成行為：

```json
{
  "backend_url": "http://your-backend-host",
  "api_key": "your-api-key",
  "model": "trusta-ast/trusta-ast-default",
  "system_prompt": "You are a helpful assistant.",
  "stream": false,
  "temperature": 0.5,
  "top_p": 0.9,
  "max_tokens": 800,
  "continue_on_error": true,
  "extra_body": {
    "top_k": 50,
    "repetition_penalty": 1.1,
    "total_timeout": 300,
    "enable_thinking": false
  },
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"]
}
```

所有欄位與其預設值都定義在 `ai_scaler_console/helpers/chat.py` 的 `ChatSettings`。請直接看那裡 — 寫在這裡的表格只會漂掉。

## Workflow 執行順序

1. 載入模型（PHASE 1）

- 檢查目前是否有模型正在載入
- 卸載先前已載入的模型
- 依 `inference_config` 載入新模型
- 監控載入進度直到就緒

2. 對話請求（PHASE 2）

- 載入 `chat_settings`
- 依序執行每個 prompt
- 支援串流與非串流回應模式
- 印出 token usage 統計

3. 卸載模型（PHASE 3）

- 釋放模型資源
- 讓系統回到乾淨狀態

## 建立自訂 Workflow 設定

```bash
# 複製預設設定
cp configs/workflow/default.json configs/workflow/custom.json

# 編輯自訂設定
# - 更新 inference_config 路徑
# - 更新 chat_settings 路徑
# - 調整 backend_url 與 timeout

# 執行自訂 workflow
uv run examples/workflow/full.py --workflow-config configs/workflow/custom.json
```

如果這個 workflow 還需要自己的 prompt，先寫一份對話設定檔，再讓 `chat_settings` 指向它。

## 疑難排解

### 問題：找不到設定檔

解法：確認路徑相對於專案根目錄是正確的。

### 問題：模型載入失敗

解法：確認 `inference_config` 中的模型路徑與引擎設定。

### 問題：對話請求逾時

解法：提高 workflow 設定中的 `timeout`。

### 問題：需要更詳細的日誌

解法：在 workflow 設定中加上 `"verbose": true`。

## 參考

- 推論設定範例：configs/inference/
- 對話設定範例：configs/chat/
- 完整 workflow 腳本：examples/workflow/full.py
