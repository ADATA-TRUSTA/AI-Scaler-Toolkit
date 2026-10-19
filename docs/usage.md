# 使用與驗證

[← 回到 README](../README.zh-TW.md)

## 應用程式

視窗裡是網頁介面：對話、模型清單（下載、登錄、轉成 GGUF），以及 Linux 上的微調。設定欄旁邊的 **Show chat** 按鈕會打開對話區。

關掉視窗**不會**停止服務：應用程式會留在系統匣（Windows 的通知區域），因為其他應用程式可能正在用它的 API。系統匣圖示的選單：

| 項目 | 作用 |
|---|---|
| Open window | 叫回視窗 |
| Open full interface in browser | 用瀏覽器開同一個介面 |
| Settings | 設定視窗（見下） |
| Quit | 停止服務並結束應用程式 |

同時只會有一份在執行；再開一次會把已經開著的視窗叫到前面。

### 設定（Settings）

| 區塊 | 用途 |
|---|---|
| Startup | 開機／登入時自動啟動（預設開啟），讓服務隨時可供其他應用程式使用。第一次設定完成前，登入時不會做任何事 |
| Ports | 應用程式的 port，預設 `8571`。被別的程式佔用時可以改 |
| Network | **Let another computer use this machine's model**，以及別台電腦要用的 API key（見下） |
| Environment | **Reinstall the environment**：移除已安裝的 Python 套件與 GPU 執行環境，再重新安裝。模型會保留。設定或載入模型失敗、重開也沒用時使用 |
| Updates | 顯示目前版本。**Check now** 向 GitHub 查有沒有新版；**Update now** 下載、檢查、安裝，並重新開啟應用程式 |
| Components | 選用的第三方工具，裝在你自己的帳號底下（見下） |

**Let another computer use this machine's model**（Settings → Network）：把服務開放給區域網路，讓其他電腦也能用這裡的模型。其他電腦只能使用 OpenAI 相容的 API（`/v1`），而且要帶那裡顯示的 **API key**：可以和載入的模型對話，但載入模型、下載、微調、網頁介面和 API 文件都只有這台電腦能用。第一次開啟時會自動產生 key。可以按 **Copy** 複製、按 **Use my own key…** 改成自己設定的 key（至少 12 個字元，可用英文字母、數字與 `. _ ~ + / = -`），或在 key 外流時按 **Make a new key** 重新產生；換了之後，還在用舊 key 的電腦要改成新的才能連。拿到 key 的人都能使用模型，所以只在你信任的網路上開啟。如果被防火牆擋住，那裡的 **Cannot connect?** 會提供新增規則的按鈕。

### 元件（Components）

- **Hermes Agent**：可以使用這台電腦模型的 AI agent。按 **Install**，再按 **Open**。**Config AST Endpoint** 會把這個應用程式的端點寫進 Hermes 的設定，讓 Hermes 使用這裡目前載入的模型（要先載入模型）。如果你已經在 Hermes 選了別的 provider，它不會覆蓋你的選擇，而是顯示手動切換的指令。要讓另一台電腦的 Hermes 使用，先開啟 Settings → Network，卡片就會顯示要在那台電腦執行的指令，位址和 key 都已經帶好。
- **ComfyUI**：圖片與影片生成。選它要跑在哪張顯示卡上，按 **Use this card**（或 **Install and use**）；CPU 列在最後，作為備用。圖片與影片的技能會自己在 `127.0.0.1:8188` 啟動 ComfyUI。

---

## 連接其他應用程式

服務使用 OpenAI 的 API。在任何接受「OpenAI 相容」provider 的應用程式或程式助理裡：

| 設定 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1` |
| 模型 | `trusta-ast-default`（代表目前載入的模型），或 `/v1/models` 回傳的模型名稱 |
| API key | 本機填任何字串（本機不需要 key）；從別台電腦連，填 Settings → Network 顯示的 key |

服務一律用目前載入的模型回答，不看請求裡指定的名稱，所以換了模型 `trusta-ast-default` 照樣能用。

---

## 用命令列確認

範例用的是應用程式的 port `8571`。從原始碼啟動的服務用它 `.env` 裡的 `SERVICE_PORT`（`.env.example` 是 `8000`）。

### 健康檢查

```bash
curl http://127.0.0.1:8571/health
```

```powershell
Invoke-RestMethod http://127.0.0.1:8571/health
```

### 已登錄的模型

```bash
curl http://127.0.0.1:8571/config/models
```

### 載入模型，再對話

載入在背景進行：`POST /inference/load_model` 會立刻回傳，等 `GET /inference/status` 回報 `"loaded": true` 就好了。Hugging Face 模型：

```bash
curl http://127.0.0.1:8571/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-4B"}'
```

GGUF 檔，用 llama.cpp：

```bash
curl http://127.0.0.1:8571/inference/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/model.gguf", "model_name": "my-model", "engine": "llama_server", "n_gpu_layers": -1, "n_ctx": 8192}'
```

接著：

```bash
curl http://127.0.0.1:8571/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trusta-ast-default",
    "messages": [{"role": "user", "content": "你好，請簡單介紹你自己。"}],
    "stream": false
  }'
```

PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8571/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"trusta-ast-default","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

完整的 API 與請求內容在 `http://127.0.0.1:8571/docs`。

---

## API 路由

### 基本

- `GET /`
- `GET /health`
- `GET /v1/models` — 目前載入的模型，用它自己的名稱
- `GET /v1/models/{model}` — 單一模型；接受 `trusta-ast-default` 與模型的其他別名
- `POST /v1/chat/completions`

### 推論管理

- `POST /inference/load_model`
- `POST /inference/unload_model`
- `GET /inference/status`
- `POST /inference/stop_generation`
- `GET /inference/error_details`
- `POST /inference/cleanup_generation_memory`
- `POST /inference/force_cleanup_gpu`
- `POST /inference/chat` — **已淘汰**，回傳 HTTP 410；請用 `/v1/chat/completions`

### 記憶體估算

- `POST /inference/estimate_memory` — 依參數量估算，用於 Hugging Face checkpoint
- `GET /inference/estimate_memory/{model_name}`
- `POST /inference/estimate_memory/gguf` — GGUF／llama.cpp 的估算
- `POST /inference/estimate_memory/gguf/check`
- `POST /inference/estimate_memory/gguf/plan`
- `POST /inference/estimate_memory/gguf/recommend`
- `POST /inference/estimate_memory/gguf/sweep`

GGUF 模型請用 `gguf` 這一組；`/inference/estimate_memory` 是依參數量估算，不適合 GGUF。

### 訓練（Linux）

- `POST /training/start`
- `GET /training/status`
- `GET /training/status/{session_id}/history`
- `POST /training/stop`
- `GET /training/error_details`
- `POST /training/force_cleanup_gpu`
- `GET /training/{job_id}/log` — 結構化事件回補
- `GET /training/{job_id}/log/stream` — SSE 即時 log

### 列舉值與設定範例

- `GET /config/quantization_types`
- `GET /config/offload_types`
- `GET /config/training_methods`
- `GET /examples/inference`
- `GET /examples/training`
- `GET /examples/conversion`

### 模型：設定、下載、轉檔

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`
- `POST /config/models/download/{task_id}/cancel`
- `GET /config/models/downloads` — 所有下載工作
- `DELETE /config/models/{label}` — 可加 `?delete_files=`
- `POST /config/models/refresh_context_lengths`
- `POST /config/models/convert`
- `GET /config/models/convert/{job_id}`

### 文件（RAG）

- `POST /rag/docs`
- `GET /rag/docs`
- `DELETE /rag/docs/{doc_id}`
- `GET /rag/search`

### 系統資訊

- `GET /system/resources` — CPU、記憶體（含每條 DRAM 模組）、GPU 與磁碟

---

## Log

| | Windows | Linux |
|---|---|---|
| 應用程式 | `%LOCALAPPDATA%\AIScalerToolkit\logs\` | `~/.local/share/ai-scaler-toolkit/logs/` |

- `launcher.log` — 應用程式本身：啟動、更新、元件
- `provision.log` — 最近一次的設定
- `service.log`，每天輪替為 `service.log.YYYY-MM-DD` — 服務與它執行的引擎

從原始碼啟動的服務，`service.log` 寫在它 `.env` 的 `LOG_DIR`。
