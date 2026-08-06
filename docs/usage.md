# 使用與驗證

[← 回到 README](../README.zh-TW.md)

## 啟動後如何驗證

### 健康檢查

#### Linux

```bash
curl http://127.0.0.1:8000/health
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 查看可用模型列表

#### Linux

```bash
curl http://127.0.0.1:8000/config/models
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/config/models
```

### 先載入模型，再送出聊天請求

聊天前請先呼叫 `/inference/load_model` 載入模型；若模型尚未載入完成，聊天請求可能會回傳模型未就緒。

#### Linux

```bash
curl http://127.0.0.1:8000/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen3-4B"
  }'
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/inference/load_model `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model_name":"Qwen/Qwen3-4B"}'
```

確認模型載入完成後，再送聊天請求：

#### Linux

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trusta-ast-default",
    "messages": [{"role": "user", "content": "你好，請簡單自我介紹"}],
    "stream": false
  }'
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"trusta-ast-default","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

---

## 常用 API 路由

### 基本

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

當模型已載入時，`/v1/models` 會回傳補全別名 `trusta-ast-default`；在 `/v1/chat/completions` 請求中使用相同的別名即可。

### 推論管理

- `POST /inference/load_model`
- `POST /inference/unload_model`
- `GET /inference/status`
- `POST /inference/stop_generation`
- `GET /inference/error_details`
- `POST /inference/cleanup_generation_memory`
- `POST /inference/force_cleanup_gpu`
- `POST /inference/chat` —— **已棄用**，回傳 HTTP 410，請改用 `/v1/chat/completions`

### 記憶體估算

- `POST /inference/estimate_memory` —— 以參數量估算，適用 HF checkpoint
- `GET /inference/estimate_memory/{model_name}`
- `POST /inference/estimate_memory/gguf` —— GGUF / llama.cpp 專用
- `POST /inference/estimate_memory/gguf/check`
- `POST /inference/estimate_memory/gguf/plan`
- `POST /inference/estimate_memory/gguf/recommend`
- `POST /inference/estimate_memory/gguf/sweep`

GGUF 模型請使用 `gguf` 系列；`/inference/estimate_memory` 是以參數量估算的，並不適用。

### 訓練管理

- `POST /training/start`
- `GET /training/status`
- `GET /training/status/{session_id}/history`
- `POST /training/stop`
- `GET /training/error_details`
- `POST /training/force_cleanup_gpu`
- `GET /training/{job_id}/log` —— 結構化事件回補
- `GET /training/{job_id}/log/stream` —— SSE 即時日誌

### 列舉值與設定範例

- `GET /config/quantization_types`
- `GET /config/offload_types`
- `GET /config/training_methods`
- `GET /examples/inference`
- `GET /examples/training`
- `GET /examples/conversion`

### 模型設定與下載

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`
- `GET /config/models/downloads` —— 所有下載任務
- `DELETE /config/models/{label}` —— 可帶 `?delete_files=`
- `POST /config/models/refresh_context_lengths`
- `POST /config/models/convert`
- `GET /config/models/convert/{job_id}`

### 系統資訊

- `GET /system/resources`

---

## 前端說明

專案已內含前端靜態檔：

- `frontend/dist`
- `frontend/dist_client`

目前後端會掛載 `frontend/dist` 到 `/frontend/`，根路徑 `/` 會在 `index.html` 存在時自動轉址到 `/frontend/`。

因此：

- 不需要安裝 Node.js
- 不需要另外執行前端 build
- 只要服務啟動成功，直接開 `http://127.0.0.1:8000/` 即可

---

## 日誌設定

```dotenv
LOG_TO_FILE=true
LOG_DIR=<project>/logs
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
```

啟用後會產生：

- `service.log`
- `service.log.YYYY-MM-DD`（日切輪替）

---

