# AI Scaler Console

English version: [README.md](README.md)

AI Scaler（Trusta AST）後端的 Headless（無圖形介面）Python 客戶端。
本專案是 **Trusta-AST-Frontend** 的 API-only 對應版本：相同後端、相同模型生命週期，只是改由 console／腳本驅動，而非網頁 UI。

涵蓋範圍：

- 在後端載入／卸載模型（`/inference/load_model`、`/inference/unload_model`）。
- 透過 **OpenAI 相容** 端點（`<backend_url>/v1/chat/completions`）與已載入的模型對話，支援純文字或圖片。
- 微調訓練（`/training/start`、`/training/stop`、`/training/status`）。

舊版 `/inference/chat` 的 Python 包裝（`AIClient.chat`、`ChatRequest`、`ChatStreamChunk` 等）已移除。所有對話一律透過 OpenAI Python SDK 呼叫後端的 OpenAI 相容端點，與前端的 `runChatCompletions` 行為一致。

`configs/inference/` 底下的模型設定檔會與 Trusta-AST-Frontend 使用的 load-model payload 保持一致。
更詳細的 API 文件請參考 Trusta-AST-Backend API reference。

## 0. 環境設定

本專案以 **uv** 管理：

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安裝套件與相依項目。`uv sync` 會以 editable 模式安裝 `ai_scaler_console`，因此範例可以從任何工作目錄執行：

```bash
uv sync
```

接著從範本建立本機設定：

```bash
cp .env.example .env
```

`.env` 不納入版本控制。只有 `BACKEND_URL` 是必填：

```ini
BACKEND_URL=http://127.0.0.1:8000
```

其餘選填變數都記錄在 `.env.example`：

| 變數          | 預設值                  | 何時需要                                                                   |
| ------------- | ----------------------- | -------------------------------------------------------------------------- |
| `BACKEND_URL` | `http://127.0.0.1:8000` | 一律需要 — 後端的 base URL，不含 `/v1` 後綴。                              |
| `API_KEY`     | `EMPTY`                 | 目前不需要。後端不驗證請求，只是 OpenAI SDK 不接受空字串的 key。           |
| `VERIFY_SSL`  | `true`                  | 只有當後端使用自簽憑證提供 https 時才設為 `false`。                        |

## 1. 建議的無 UI 工作流程

### 1.1 載入模型

從 `configs/inference/` 中挑一份設定檔（transformers／llama_server／vLLM 都支援，欄位與前端 payload 相同）。

> **執行前：** 編輯選用的設定檔，把 `model_path` 設成實際的模型位置。`llama_server` 與 `vllm` 的設定必須填後端主機上的絕對路徑；`transformers` 的設定可直接填 HuggingFace model ID。

```bash
uv run examples/model/load.py --config configs/inference/transformers.json
```

```bash
uv run examples/model/load.py --config configs/inference/transformers_offload.json
```

```bash
uv run examples/model/load.py --config configs/inference/llama.json
```

```bash
uv run examples/model/load.py --config configs/inference/vllm.json
```

若只想讓單次執行指向不同後端，可在命令列覆寫 `BACKEND_URL`：

```bash
uv run examples/model/load.py --config configs/inference/vllm.json --backend-url http://your-backend-host
```

### 1.1.1 取得模型清單

取得前端 `/config/models` API 所使用的統一模型清單：

```bash
uv run examples/model/list.py
```

```bash
uv run examples/model/list.py --json
```

回應會被解析成 `ModelListResponse`，包含三個分組再加一個快速參考：

- `base_models`
- `finetuned_models`
- `llama_gguf_models`
- `last_finetuned` — 後端 `last_finetuned_model.json` 的原始內容，沒有則為 `null`

> 這三組都來自後端的 model registry，而 registry 已不再預載任何 base model。全新的後端在尚未註冊模型前會回傳空清單，這是正常現象，不是連線失敗。載入模型與微調都不經過 registry，直接吃 HuggingFace ID 或路徑。

### 1.2 對話（文字）

模型載入完成後：

```bash
uv run examples/chat/basic.py
```

此腳本透過 OpenAI Python SDK 呼叫 `<backend_url>/v1/chat/completions`。SDK 原生認得的欄位（`messages`、`temperature`、`top_p`、`max_tokens`、`tools`、`tool_choice`、`stream` 等）以一般參數傳入；AST 專屬欄位則走 `extra_body`，與前端做法相同。完整的十二個欄位列在 [`examples/chat/basic.py`](examples/chat/basic.py) 開頭的 docstring。

### 1.2.1 批次對話（多個 prompt）

一個指令執行多個 prompt：

```bash
uv run examples/chat/batch.py
```

```bash
uv run examples/chat/batch.py --settings configs/chat/batch.json
```

腳本會從 `configs/chat/batch.json` 讀取 prompts，逐筆送往 `<backend_url>/v1/chat/completions`，並印出每個 prompt、回應以及各請求的 usage 統計。

### 1.2.2 平行對話

若要同時執行 N 個 prompt 並並排觀察，請使用平行範例。它會開啟一個終端 UI，每個 prompt 一個 pane，各自串流自己的 worker 行程：

```bash
uv run examples/chat/parallel.py
```

```bash
uv run examples/chat/parallel.py --settings configs/chat/parallel_16.json
```

```bash
uv run examples/chat/parallel.py --columns 4
```

按 `q` 離開。

這是單一跨平台腳本 — Windows、Linux、macOS 皆可 — 而且因為 UI 是 TUI 而非桌面視窗，透過 SSH 也能使用。每個 prompt 會產生一個獨立的 OS 行程，這也是這些 worker 真正互相獨立的客戶端、而非共用同一連線池的 coroutine 的原因。

在設定檔中設定 `stream` 以控制回應模式：

- `stream: false` → 每個 worker 取得完整（非串流）回應
- `stream: true` → 每個 pane 逐 token 串流，worker 結束時回報 time-to-first-token

備註：

- `model` 固定為 `trusta-ast/trusta-ast-default` — 這是後端 OpenAI 相容層的綁定名稱，與底層實際載入哪個模型無關。
- `api_key` 是 SDK 要求的欄位，但後端不驗證請求，任何字串都可以。若省略，就會使用 `.env` 中 `API_KEY` 的預設值（`EMPTY`）。

### 1.3 對話（圖片／多模態）

適用於多模態模型（以 transformers 執行的 Gemma 3／Qwen-VL，或在 load 設定中指定 `vllm_mm_image_limit` 的 vLLM）：

```bash
uv run examples/chat/image.py
```

預設使用內附的範例圖片 `examples/assets/test_image.jpg`。想換成自己的圖片，可修改腳本中的 `SAMPLE_IMAGE`（或設定檔中的 `images` 欄位）。

腳本會送出 OpenAI 相容的 `image_url` content part，並示範：

- 本機檔案 → 自動編碼為 Base64 Data URI
- HTTP／HTTPS URL → 原樣傳遞
- 單一訊息中夾帶多張圖片
- 非串流版本，可取得完整回應與 usage 統計

### 1.3.1 批次圖片對話（多個多模態任務）

一個指令執行多個圖片對話任務：

```bash
uv run examples/chat/image_batch.py
```

```bash
uv run examples/chat/image_batch.py --settings configs/chat/image_batch.json
```

腳本會從 `configs/chat/image_batch.json` 讀取 `tasks`。
每個 task 包含一段文字 prompt 與一張或多張圖片（本機路徑或 URL）。

### 1.4 卸載模型

```bash
uv run examples/model/unload.py
```

```bash
uv run examples/model/unload.py --backend-url http://your-backend-host
```

### 1.5 完整流程（載入 → 對話 → 卸載）

```bash
uv run examples/workflow/full.py --workflow-config configs/workflow/default.json
```

可用的 workflow 設定：`default.json`、`llama.json`、`vllm.json`、`transformers_offload.json`、`parallel_batch_16.json`。每一份都會指定一個 inference 設定與一個 chat settings 檔案。詳見 [docs/WORKFLOW_CONFIG_GUIDE.zh-TW.md](docs/WORKFLOW_CONFIG_GUIDE.zh-TW.md)。

### 1.6 變更後端 URL

解析順序，先命中者優先：

1. 命令列的 `--backend-url`（`load.py`、`unload.py`、`list.py`、`full.py`）。
2. 所使用 settings JSON 中的 `backend_url`（batch、image-batch、parallel 與 workflow 設定都接受此欄位）。
3. `.env` 中的 `BACKEND_URL`。

最常見的情境 — 把所有範例改指向新主機 — 只要改 `.env` 的 `BACKEND_URL`，其他都不用動。內附的 settings 檔刻意不寫 `backend_url`，就是為了讓它們沿用這個值。

## 2. 微調

> **需要 Linux 後端。** 微調即使是單 GPU 也會初始化 distributed process group，而該
> rendezvous 在 Windows 上不成立。Windows 後端會以 `501` 拒絕並附上說明；WSL2 的
> distro 算 Linux。console 本身在哪個系統都能跑——關鍵是它連的**後端**的作業系統，
> 可由 `GET /health` 的 `platform` 與 `training_supported` 查得。推論、對話與 GGUF
> 轉換不受影響。

```bash
uv run examples/finetune/train.py
```

```bash
uv run examples/finetune/train.py --settings configs/finetune/run.json
```

```bash
uv run examples/finetune/train.py --force
```

`--force` 會先中斷既有的訓練工作，再啟動新的訓練。

### 2.1 追蹤訓練日誌

啟動後，腳本會以 SSE 追蹤後端的結構化訓練日誌（`/training/{job_id}/log/stream`），
後端每發出一個生命週期階段、metric 或錯誤就即時印出。job id 由 `/training/start`
以 `result.session_id` 回傳。

```bash
uv run examples/finetune/train.py --show-resource
```

```bash
uv run examples/finetune/train.py --poll
```

- `--show-resource` 會一併印出每秒的 GPU／DRAM／SSD 取樣；預設隱藏，因為它一秒一筆。
- `--poll` 退回每 3 秒輪詢 `/training/status`。後端沒有回傳 job id 時也會自動走這條路。

held-out 評估的 metric 只存在於串流上——它們以 `split: "eval"` 的 `metric` 事件穿插在
同一條串流裡，`/training/status` 完全看不到。console 會標上 `[eval]`，避免和訓練曲線
混在一起。連線中斷會用 `Last-Event-ID` 從最後一個 `seq` 續傳；404 代表該工作的日誌目錄
已不存在，視為致命錯誤而不重試。

要讀已完成工作的日誌，或在斷線後補齊，直接用
`AIClient.get_training_log(job_id, since=...)`；完整 traceback 則用
`AIClient.get_training_error_details()`。

## 3. 設定檔

`configs/` 底下附的檔案就是規格本身 — 直接打開你要用的那一份。
它們是純 JSON、無法寫註解，所以下面只列出執行前必須自己改的部分。

### 3.1 推論模型設定（`configs/inference/*.json`）

這些設定對應前端的 load-model payload，由 `engine` 欄位決定使用哪一套 schema。
建立 `InferenceConfig` 時 `base_model` 會被正規化為 `model_name`；任何 `InferenceConfig` 上定義的欄位（見 `ai_scaler_console/config_models.py`）都可以加進 JSON。

| 設定檔                      | `model_path`                                  | 依硬體調整                                                 |
| --------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| `transformers.json`         | HuggingFace ID（後端自動下載）或本機目錄      | `quantization`、`device_map`                               |
| `transformers_offload.json` | 同上                                          | `max_memory` — GPU/CPU 的記憶體分配                        |
| `llama.json`                | **必填**：後端上 `.gguf` 檔案的絕對路徑       | `n_gpu_layers`（`0` = 只用 CPU）、`n_ctx`                  |
| `vllm.json`                 | **必填**：後端上模型目錄的絕對路徑            | `vllm_gpu_memory_utilization`、`vllm_tensor_parallel_size` |

> 絕對路徑指的是 **後端主機** 上的路徑，不是執行這些腳本的機器。
> 多模態輸入必須設 `vllm_mm_image_limit`；不使用時請移除或設為 `null`。

### 3.2 微調 settings（`configs/finetune/run.json`）

`run.json` 只有一個欄位 `finetune_config_path`，指向 `configs/finetune/training_*.json` 底下的訓練設定。`backend_url` 可填但為選填，省略時使用 `.env` 的值。

專案附了兩份訓練設定作為範例。使用前必須填寫：

- `dataset_path`：`.json`／`.jsonl` 資料集檔案（每行格式：`{"prompt": "...", "completion": "..."}`），或 Hugging Face Hub 的 dataset id。只要不是本機 `.json`／`.jsonl` 路徑，一律當成 Hub id 處理 — Hub id 不能帶 config／subset 或 split 選擇器，也不會做本機檔案的格式偵測，因此要自行設定 `prompt_field`／`completion_field`／`text_field`（或 `messages_field`）對應該資料集的欄位。
- `model_name`：base model 的 HuggingFace ID 或本機絕對路徑。
- `output_dir`：微調後 adapter／模型的輸出目錄（會自動建立）。
- `use_deepspeed`：單 GPU 或 CPU 訓練請設 `false`；DeepSpeed 相關欄位只在後端支援時才有效。

> 訓練跑在後端，所以 `dataset_path` 與 `output_dir` 是在**後端主機**上解析的 — 附的設定指向的是後端自己的 `dataset/dataset.jsonl`。

### 3.3 對話 settings（`configs/chat/*.json`）

| 檔案               | 使用者                         | 結構                          |
| ------------------ | ------------------------------ | ----------------------------- |
| `batch.json`       | `examples/chat/batch.py`       | `prompts`（4 個 prompt）      |
| `parallel_8.json`  | `examples/chat/parallel.py`    | `prompts`（8 個 prompt）      |
| `parallel_16.json` | `examples/chat/parallel.py`    | `prompts`（16 個 prompt）     |
| `image_batch.json` | `examples/chat/image_batch.py` | `tasks`（prompt + images）    |

所有欄位與其預設值都定義在 [ai_scaler_console/helpers/chat.py](ai_scaler_console/helpers/chat.py) 的 `ChatSettings` — 請直接看那裡，不要相信文件裡的複本。從 dataclass 看不出來的部分：

- `prompts` 必須是非空的字串串列；圖片版的 `tasks` 必須是非空的 `{prompt, images}` 串列，每張圖片可以是本機路徑或 `https://` URL。
- `continue_on_error: false` 會讓批次在第一個失敗的 prompt 停止。
- `backend_url` 與 `api_key` 為選填，省略時使用 `.env` 的值。
- AST 專屬參數（`top_k`、`repetition_penalty`、`total_timeout`、`enable_thinking` 等）放在 `extra_body` 底下。

## 4. 專案結構

```
ai_scaler_console/              # 可安裝的套件
|-- client.py                   #   AIClient：模型生命週期 + 訓練端點
|-- config_models.py            #   Pydantic models（InferenceConfig、TrainingConfig 等）
|-- exceptions.py               #   AIClientError 及相關例外
|-- settings.py                 #   從 .env 解析出的執行期設定
|-- openai_client.py            #   OpenAI 相容 client 工廠
`-- helpers/
    |-- chat.py                 #   載入 chat settings + 執行單一 prompt
    |-- config_loader.py        #   JSON -> InferenceConfig / TrainingConfig
    |-- finetune_initializer.py #   微調啟動流程
    |-- model_loader.py         #   載入／卸載編排（狀態輪詢）
    |-- parallel_ui.py          #   平行範例使用的 Textual pane grid
    `-- training_handler.py     #   微調進度監控

examples/
|-- chat/                       # basic、batch、image、image_batch、parallel
|-- model/                      # load、unload、list
|-- finetune/                   # train
|-- workflow/                   # full（載入 -> 對話 -> 卸載）
`-- assets/                     # 視覺範例用的 test_image.jpg

configs/
|-- chat/                       # 對話範例的 prompt 組合
|-- finetune/                   # 微調 run settings + 訓練設定
|-- inference/                  # 與前端一致的 load-model payload
`-- workflow/                   # 端到端流程定義

docs/                           # WORKFLOW_CONFIG_GUIDE.md（+ .zh-TW.md）
internal/                       # 內部 benchmark，不屬於對外發佈的客戶端
```

## 5. 開發

Lint、format 與型別檢查設定都放在 `pyproject.toml`，並與 trust-hmst-backend 對齊。

```bash
uv run ruff check
```

```bash
uv run ruff format
```

```bash
uv run pyright
```

若要在每次 commit 自動執行以上檢查：

```bash
uv run pre-commit install
```

`internal/` 不納入 ruff 與 pyright。它放的是 long-prompt KV-offload benchmark 與長時間穩定性測試 — 屬於內部工具，不是客戶端的一部分，也不會發佈。

### 測試

```bash
uv run pytest
```

`tests/unit/` 底下的測試全部離線執行：HTTP 以 `httpx2.MockTransport` 偽造，平行對話測試則把 worker 固定指向不可連線的位址，因此結果不會受本機是否剛好有後端在跑而影響。

`tests/integration/` 需要實際運行的後端，預設不會被選中。請先啟動後端、載入模型，然後：

```bash
uv run pytest --integration
```

需要推論的整合測試在沒有載入模型時會自行 skip，因此在只有後端、沒有模型的情況下，傳輸層的測試仍然會執行。
