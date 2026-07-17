# AI Scaler Toolkit

<img src="Trusta-AST-Frontend/dist/Trusta-16.ico" alt="Trusta Icon" width="24" height="24" style="vertical-align: middle; margin-right: 8px;">
<img src="Trusta-AST-Frontend/dist_client/Adata.ico" alt="Adata Icon" width="24" height="24" style="vertical-align: middle; margin-left: 8px;">

## 🌐 Language / 語言

[🇬🇧 English](README.md) | [🇹🇼 繁體中文](README.zh-TW.md)

---

AI Scaler Toolkit 是一個以 FastAPI 為核心的 LLM 後端服務，提供：

- 模型載入 / 卸載
- OpenAI 相容聊天介面
- 推論與串流回應
- 訓練工作啟動與狀態查詢
- 模型下載
- 前端靜態頁面掛載

本文件涵蓋 Linux 與 Windows 兩種平台；初始化快速流程放在最前面，更多細節請往後看。

> Linux 範例路徑：`/home/test/project/AI-Scaler-Toolkit`
>
> Windows 範例路徑：`C:\Users\<user>\project\AI-Scaler-Toolkit`

---

## 1. 第一次初始化快速流程

### Linux（預設建議走 CUDA）

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

mkdir -p logs .cache/huggingface
cp .env.example .env
# 編輯 .env，優先修改 HF_HOME、LOG_DIR、SERVICE_HOST、SERVICE_PORT

TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
bash deploy/linux/run_service.sh
```

啟動成功後，直接用瀏覽器開啟：

- `http://127.0.0.1:8000/`

若主機沒有 NVIDIA CUDA 環境，再改用 `TRUSTA_ACCEL=xpu`。

> **提醒**：目前 fine-tune 功能僅支援 **Linux + CUDA** 環境。

### Windows（建議先走 XPU）

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

New-Item -ItemType Directory -Force logs, .cache\huggingface
Copy-Item .env.example .env
notepad .env

.\deploy\windows\setup_env.ps1 -Accel xpu
.\deploy\windows\run_service.bat
```

啟動成功後，直接用瀏覽器開啟：

- `http://127.0.0.1:8000/`

若要改走 NVIDIA CUDA，將 `-Accel xpu` 改成 `-Accel cuda`。

> 詳細安裝需求、`.env` 說明、模型載入方式與驗證方式，請看後續章節。

---

## 2. 專案結構

```text
AI-Scaler-Toolkit/
├─ deploy/
│  ├─ linux/
│  │  ├─ run_service.sh
│  │  ├─ setup_env.sh
│  │  └─ stop_service.sh
│  ├─ windows/
│  │  ├─ run_service.bat
│  │  └─ setup_env.ps1
│  └─ docker/
├─ docs/
├─ dataset/
├─ examples/
├─ service/
│  ├─ app.py
│  ├─ settings.py
│  ├─ pyproject.toml
│  └─ configs/
├─ tests/
├─ Trusta-AST-Frontend/
│  ├─ dist/
│  └─ dist_client/
├─ wiki/
├─ logs/
├─ .github/
├─ .env.example
├─ pytest.ini
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

---

## 3. 系統需求

### 基本需求（兩平台共同）

| 工具 | 說明 |
|------|------|
| Git | 含 submodule 支援 |
| Python 3.12+ | 執行服務本體 |
| `uv` | Python 套件 / 環境管理 |
| C/C++ 編譯工具鏈 | 編譯 llama.cpp 時需要 |
| cmake | 編譯 llama.cpp 時需要 |

### Linux（Ubuntu / Debian）

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake python3 python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安裝完成後重新登入 shell，或手動載入 `uv`：

```bash
source "$HOME/.local/bin/env"
```

### Windows

1. **Git for Windows**：<https://git-scm.com/download/win>
2. **Python 3.12+**：<https://www.python.org/downloads/windows/>
  - 安裝時勾選 **Add Python to PATH**
3. **uv**（PowerShell）：
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
4. **Visual Studio Build Tools 2022**（僅編譯 llama.cpp 時需要）
5. 若使用 Intel XPU / iGPU，建議安裝 Intel oneAPI Runtime / Toolkit

### 選配需求

- NVIDIA GPU：需安裝對應驅動與 CUDA 執行環境
- Intel XPU / iGPU：Linux / Windows 皆可，需安裝對應驅動與執行環境
- 私有 Hugging Face 模型：需準備 `HF_TOKEN`

> **注意**：vLLM 僅支援 **Linux + CUDA**；Windows 平台無法使用 vLLM engine。
>
> **注意**：fine-tune 功能目前也僅支援 **Linux + CUDA**；Windows、Linux + XPU、純 CPU 環境皆不支援。
>
> 前端靜態檔已包含在專案內，不需要另外安裝 Node.js 或重新建置前端。

---

## 📊 效能實測 (TPS / TTFT)

使用 **Llama Engine** (`llama-server`) 實測。所有測試的共同條件：`n_ctx = 4096`、輸入 `50–108` tokens、輸出 `50` tokens。**TPS** = 每秒輸出 token 數（越高越好）；**TTFT** = 首個 token 延遲（越低越好）。數據來自內部測試，會因驅動、編譯版本與 prompt 而有差異。

### 測試機器

| 代號 | GPU | OS | CPU | VRAM | DRAM |
|------|-----|----|-----|------|------|
| **A** | RTX 5060 Ti | Ubuntu 24.04 | Core Ultra 5 225 | 16 GB | DDR5-4800 / 128 GB |
| **B** | RTX 6000 Ada | Ubuntu 24.04 | Xeon w5-3535X | 48 GB | DDR5-4800 / 512 GB |
| **C** | RTX 5080 | Windows 11 | Core Ultra 9 275HX | 16 GB | DDR5-6400 / 32·64 GB |
| **D** | Intel iGPU | Windows 11 | Core Ultra 7 355 | 共用記憶體 | DDR5-6400 / 32·64 GB |

### 摘要 — 各模型最佳結果

每格顯示該機器達到的**最佳 TPS** 與對應 TTFT。**粗體** = 該模型最快的機器。`–` = 未測試。

| 模型 (大小) | A · 5060 Ti | B · 6000 Ada | C · 5080 | D · iGPU |
|-------------|-------------|--------------|----------|----------|
| GPT-OSS-20B-FP16 (13.8 GB) | **90 / 0.2s** | **155 / 0.1s** | 133 / 0.5s | 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M (11.6 GB) | – | – | **179 / 0.3s** | 20 / 0.9s |
| Qwen3-8B-Q8_0 (8.7 GB) | – | – | **73 / 0.05s** | 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M (22 GB) | 57.5 / 0.4s | **162 / 0.1s** | 80 / 0.5s | 20 / 1s |
| Gemma4-26B-A4B-Q8_0 (26.9 GB) | 29 / 1s | **114 / 0.8s** | 45 / 0.7s | 15 / 1s |
| Gemma4-31B-Q4_K_M (18.3 GB) | 3 / 2.5s | **7.5 / 0.8s** | 6 / 1.5s | 4 / 2s |
| Gemma4-31B-Q8_0 (32.6 GB) | 1.9 / 3.5s | **24.5 / 0.2s** | 3.5 / 2s | 2 / 2s |
| GPT-OSS-120B&sup1; (~63–65 GB) | 15.5 / 1.5s | **48 / 0.5s** | 35 / 3.5s | 9.5 / 15s |

&sup1; **A/B** 跑的 120B 為 **FP16 (65.4 GB)**；**C/D** 為 **Q8 / Q4_K_M (~63 GB)**。量化不同，無法直接等價比較。

### 各機器完整數據

<details>
<summary><b>A · RTX 5060 Ti 16 GB</b> — 完整結果 (128 GB DRAM)</summary>

| 模型 | 設定 (VRAM / DRAM) | TPS | TTFT |
|------|---------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 90 | 0.2s |
| GPT-OSS-120B-FP16 | auto (14.5 / 50.9 GB) | 15.5 | 1.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (12 / 53.4 GB) | 14 | 1.7s |
| Qwen3.5-35B-Q4_K_M | auto (15.7 / 7.5 GB) | 57.5 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (14.1 / 9.1 GB) | 53 | 0.4s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 38 | 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=17 (15.7 / 13 GB) | 29 | 1s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6 / 22.7 GB) | 20 | 1s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 3 | 2.5s |
| Gemma4-31B-Q8_0 | auto (15.8 / 19.4 GB) | 1.9 | 3.5s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.8 / 20.4 GB) | 1.8 | 3.7s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.7 / 25.5 GB) | 1.5 | 4.5s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (5 / 30.2 GB) | 1.2 | 5.5s |

</details>

<details>
<summary><b>B · RTX 6000 Ada 48 GB</b> — 完整結果 (512 GB DRAM)</summary>

| 模型 | 設定 (VRAM / DRAM) | TPS | TTFT |
|------|---------------------|-----|------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 155 | 0.1s |
| GPT-OSS-120B-FP16 | auto (46 / 19.4 GB) | 48 | 0.5s |
| GPT-OSS-120B-FP16 | n-cpu-moe=27 (19 / 46.4 GB) | 24 | 1s |
| GPT-OSS-120B-FP16 | n-cpu-moe=29 (16.3 / 49.1 GB) | 20 | 1.3s |
| GPT-OSS-120B-FP16 | n-cpu-moe=32 (11.6 / 53.8 GB) | 18 | 1.3s |
| Qwen3.5-35B-Q4_K_M | auto (23.2 GB) | 162 | 0.1s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=21 (13.7 / 9.5 GB) | 67 | 0.3s |
| Qwen3.5-35B-Q4_K_M | n-cpu-moe=41 (5.1 / 18.1 GB) | 45 | 0.5s |
| Gemma4-26B-A4B-Q8_0 | auto (28.7 GB) | 114 | 0.8s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=16 (16.7 / 12 GB) | 42 | 0.4s |
| Gemma4-26B-A4B-Q8_0 | n-cpu-moe=31 (6.1 / 22.6 GB) | 24 | 0.6s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU (10 / 9 GB) | 7.5 | 0.8s |
| Gemma4-31B-Q8_0 | auto (35.2 GB) | 24.5 | 0.2s |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU (14.5 / 20.7 GB) | 4.5 | 1.3s |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU (9.4 / 25.8 GB) | 4 | 1.7s |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU (4.7 / 30.5 GB) | 3 | 2s |

</details>

<details>
<summary><b>C · RTX 5080 16 GB</b> — 完整結果 (Windows 11，依 DRAM 容量)</summary>

各 DRAM 設定下的 TPS / TTFT。**64 GB + moe** 欄使用 `--n-cpu-moe=N`。

| 模型 | 設定 | 32 GB | 64 GB | 64 GB + moe |
|------|------|-------|-------|-------------|
| GPT-OSS-20B-FP16 | GPU 全載 (13 GB) | 133 / 0.5s | 133 / 0.5s | – |
| GPT-OSS-20B-FP16 | 0 GPU / 25 CPU | 15 / 3s | 19 / 2s | – |
| GPT-OSS-20B-Q4_K_M | GPU 全載 (12 GB) | 179 / 0.3s | 179 / 0.3s | – |
| GPT-OSS-120B-Q8 | auto (14.6 GB) | 5 / 60s | 33 / 4s | – |
| GPT-OSS-120B-Q8 | 10 GPU / 27 CPU | 2.5 / 77s | 5 / 7s | 10 / 7s |
| GPT-OSS-120B-Q8 | 8 GPU / 29 CPU | 3 / 70s | 23 / 5s | 34 / 4s |
| GPT-OSS-120B-Q8 | 5 GPU / 32 CPU | 2.5 / 80s | 20 / 6s | 32 / 4s |
| GPT-OSS-120B-Q4_K_M | 8 GPU / 29 CPU | 3.3 / 70s | 23.5 / 5s | 35 / 3.5s |
| Qwen3-8B-Q8_0 | GPU 全載 (9.5 GB) | 73 / 0.05s | 73 / 0.05s | – |
| Qwen3-8B-Q8_0 | 15 GPU / 22 CPU | 16 / 1s | 16 / 1s | – |
| Qwen3-8B-Q8_0 | 0 GPU / 37 CPU | 10 / 1.3s | 10 / 1.3s | – |
| Qwen3.5-35B-Q4_K_M | auto (15.5 GB) | 70 / 1s | 80 / 0.5s | – |
| Qwen3.5-35B-Q4_K_M | 20 GPU / 21 CPU | 33 / 1.5s | 33 / 1s | 71 / 0.5s |
| Qwen3.5-35B-Q4_K_M | 0 GPU / 41 CPU | 19 / 2s | 19 / 2s | 52 / 0.6s |
| Gemma4-26B-A4B-Q8_0 | auto (15.7 GB) | 43 / 1.5s | 45 / 0.7s | – |
| Gemma4-26B-A4B-Q8_0 | 15 GPU / 16 CPU | 30 / 2s | 30 / 1.5s | 44 / 0.5s |
| Gemma4-26B-A4B-Q8_0 | 0 GPU / 31 CPU | 18 / 3.5s | 18 / 2.5s | 33 / 0.5s |
| Gemma4-31B-Q4_K_M | 20 GPU / 41 CPU | 6 / 2s | 6 / 1.5s | – |
| Gemma4-31B-Q8_0 | auto (15.8 GB) | 3.5 / 2s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 20 GPU / 41 CPU | 3.5 / 3s | 3.5 / 3s | – |
| Gemma4-31B-Q8_0 | 10 GPU / 51 CPU | 3 / 60s | 3 / 3s | – |
| Gemma4-31B-Q8_0 | 0 GPU / 61 CPU | NA | 2.5 / 3s | – |

</details>

<details>
<summary><b>D · Intel Core Ultra 7 355 (iGPU)</b> — 完整結果 (Windows 11，依 DRAM 容量)</summary>

格式：`iGPU / CPU layer 切分 — TPS / TTFT`。

| 模型 | 32 GB DRAM | 64 GB DRAM |
|------|------------|------------|
| GPT-OSS-20B-FP16 | iGPU 全載 (13 GB) — 15 / 0.9s | iGPU 全載 (13 GB) — 15 / 0.9s |
| GPT-OSS-20B-Q4_K_M | iGPU 全載 (12 GB) — 20 / 0.9s | iGPU 全載 (12 GB) — 20 / 0.9s |
| GPT-OSS-120B-Q8 | NA | iGPU 20L (33 GB) / CPU 17 — 7.5 / 17s |
| GPT-OSS-120B-Q4_K_M | iGPU 10 / CPU 27 — 1.9 / 60s | iGPU 20L (33 GB) / CPU 17 — 9.5 / 15s |
| Qwen3-8B-Q8_0 | iGPU 全載 (10 GB) — 9.5 / 1s | iGPU 全載 (10 GB) — 9.5 / 1s |
| Qwen3.5-35B-Q4_K_M | iGPU 30L (20 GB) / CPU 11 — 13 / 3s | iGPU 全載 (23 GB) — 20 / 1s |
| Gemma4-26B-A4B-Q8_0 | iGPU 20L (20 GB) / CPU 11 — 10.5 / 6s | iGPU 全載 (29 GB) — 15 / 1s |
| Gemma4-31B-Q4_K_M | iGPU 全載 (22 GB) — 3.6 / 6s | iGPU 全載 (22 GB) — 4 / 2s |
| Gemma4-31B-Q8_0 | iGPU 30L (18 GB) / CPU 31 — 0.05 / 60s | iGPU 全載 (33 GB) — 2 / 2s |

</details>

---

## 4. 從 GitHub 下載專案

本專案含 Git submodule，**clone 時請加上 `--recursive`**。

#### Linux

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

#### Windows（PowerShell）

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone --recursive <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

> 如果已用一般方式 clone，補執行：
> ```bash
> git submodule update --init --recursive
> ```

---

## 5. 第一次下載後必做設定

### 5.1 建立必要資料夾

#### Linux

```bash
mkdir -p logs .cache/huggingface
```

#### Windows（PowerShell）

```powershell
New-Item -ItemType Directory -Force logs, .cache\huggingface
```

### 5.2 建立 `.env`

專案啟動時，`service/settings.py` 會優先讀取專案根目錄的 `.env`；如果 `.env` 不存在，才會退回 `.env.example`。

**原則：請優先修改 `.env`，不要直接改 `service/settings.py`。**

#### Linux

```bash
cp .env.example .env
```

#### Windows（PowerShell）

```powershell
Copy-Item .env.example .env
```

接著用文字編輯器開啟 `.env`，至少建議確認以下內容：

#### Linux 範例

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

HF_HOME=/home/test/project/AI-Scaler-Toolkit/.cache/huggingface
TIKTOKEN_RS_CACHE_DIR=/home/test/project/AI-Scaler-Toolkit
LOG_DIR=/home/test/project/AI-Scaler-Toolkit/logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true
UVICORN_USE_COLORS=true
```

#### Windows 範例

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000
UVICORN_RELOAD=false

HF_HOME=C:\Users\<user>\project\AI-Scaler-Toolkit\.cache\huggingface
TIKTOKEN_RS_CACHE_DIR=C:\Users\<user>\project\AI-Scaler-Toolkit
LOG_DIR=C:\Users\<user>\project\AI-Scaler-Toolkit\logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
UVICORN_ACCESS_LOG=true
UVICORN_USE_COLORS=true
```

若要啟用 `.env.example` 中原本被註解掉的選配設定，請把行首的 `#` 拿掉，再填入實際值。例如：

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxx
LLAMA_SERVER_URL=http://127.0.0.1:5001
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
```

若已經編譯好 llama.cpp，也可在 `.env` 補上：

```dotenv
LLAMA_SERVER_BINARY=./service/utils/llama.cpp/build/bin/llama-server
```

Windows 範例：

```dotenv
LLAMA_SERVER_BINARY=./service/utils/llama.cpp/build/bin/Release/llama-server.exe
```

> 路徑可以用正斜線 `/` 或反斜線 `\`，Python 的 `pathlib` 兩者皆支援。

### 5.3 常見需要手動調整的路徑設定

| 變數 | 用途 | 預設值 |
|------|------|--------|
| `HF_HOME` | Hugging Face 模型與快取 | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS 快取 | 專案根目錄 |
| `LOG_DIR` | 日誌輸出目錄 | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | llama-server 可執行檔路徑 | 自動尋找 build 輸出 |
| `VLLM_SERVER_PROJECT_DIR` | vLLM 隔離環境目錄 | `service/inference/engines/vllm_server` |

### 5.4 dmidecode 權限設定（Linux 選配）

後端 `/system/resources` API 呼叫 `dmidecode` 取得 DRAM 規格。非 root 環境下若要完整資訊，可設定 sudoers 規則：

```bash
sudo visudo -f /etc/sudoers.d/ast-dmidecode
```

加入以下一行（將 `<user>` 替換成實際執行服務的帳號）：

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

若跳過此步驟，API 仍可正常運作，DRAM 規格欄位可能顯示為空。

## 6. 建立 Python 環境

### Linux

#### 建議：使用自動腳本

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
```

若主機沒有 NVIDIA CUDA 環境，可改用 XPU：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh
```

跳過 vLLM 環境建置：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda TRUSTA_SETUP_VLLM=0 bash deploy/linux/setup_env.sh
```

腳本會在 `service/.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

> **提醒**：若要使用 fine-tune，請使用 **Linux + CUDA** 安裝流程。

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel xpu
```

若要改用 CUDA：

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel cuda
```

腳本會在 `service\.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

### 6.1 若要使用 llama-server

`llama.cpp` 來源在 submodule `service/utils/llama.cpp`，需自行編譯。

#### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit/service/utils/llama.cpp
cmake -B build
cmake --build build -j
```

若有 CUDA GPU 可加上：

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

#### Windows（PowerShell，需 Visual Studio Build Tools 2022）

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit\service\utils\llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
```

若有 CUDA GPU：

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build build --config Release -j
```

編譯完成後，在 `.env` 設定 `LLAMA_SERVER_BINARY` 指向實際輸出路徑。

官方編譯說明：<https://github.com/ggml-org/llama.cpp#build>

### 6.2 若要使用 vLLM engine（僅 Linux + CUDA）

`setup_env.sh` 在 CUDA 主機上會自動建立 vLLM 隔離環境；Windows 不支援 vLLM engine。

## 7. 啟動服務

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
bash deploy/linux/run_service.sh
```

若需要讓 `/system/resources` API 透過 `dmidecode` 讀取完整 DRAM 資訊，可改用具備權限的方式啟動：

```bash
cd /home/test/project/AI-Scaler-Toolkit
sudo bash deploy/linux/run_service.sh
```

> 建議優先依照前文設定 `sudoers`，只授權特定 `dmidecode` 指令；除非必要，不建議長期以 root 執行整個服務。


### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\run_service.bat
```

啟動腳本會直接使用 `service/.venv` 內的 Python；**不需要手動啟動虛擬環境**。

### 預設服務位址

| 位址 | 說明 |
|------|------|
| `http://127.0.0.1:8000/` | 前端首頁（若前端檔案存在，會自動轉到 `/frontend/`） |
| `http://127.0.0.1:8000/health` | 健康檢查 |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

## 8. 啟動後如何驗證

### 8.1 健康檢查

#### Linux

```bash
curl http://127.0.0.1:8000/health
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 8.2 查看可用模型列表

#### Linux

```bash
curl http://127.0.0.1:8000/config/models
```

#### Windows（PowerShell）

```powershell
Invoke-RestMethod http://127.0.0.1:8000/config/models
```

### 8.3 先載入模型，再送出聊天請求

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

## 9. 常用 API 路由

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
- `POST /inference/estimate_memory`
- `POST /inference/stop_generation`

### 訓練管理

- `POST /training/start`
- `GET /training/status`
- `POST /training/stop`

### 模型設定與下載

- `GET /config/models`
- `POST /config/models/download`
- `GET /config/models/download/{task_id}`

### 系統資訊

- `GET /system/resources`

---

## 10. 前端說明

專案已內含前端靜態檔：

- `Trusta-AST-Frontend/dist`
- `Trusta-AST-Frontend/dist_client`

目前後端會掛載 `Trusta-AST-Frontend/dist` 到 `/frontend/`，根路徑 `/` 會在 `index.html` 存在時自動轉址到 `/frontend/`。

因此：

- 不需要安裝 Node.js
- 不需要另外執行前端 build
- 只要服務啟動成功，直接開 `http://127.0.0.1:8000/` 即可

---

## 11. 日誌設定

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

## 12. 維護更新

每次從 GitHub 拉新版本後，建議重新同步依賴。

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh
```

若改用 CUDA：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash deploy/linux/setup_env.sh
```

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel xpu
```

若改用 CUDA：

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\deploy\windows\setup_env.ps1 -Accel cuda
```

---

## 13. 常見問題

### Q1. 執行啟動腳本時出現找不到 `.venv`

代表尚未建立 Python 環境，請先執行：

- Linux：`TRUSTA_ACCEL=xpu bash deploy/linux/setup_env.sh`
- Windows：`.\deploy\windows\setup_env.ps1 -Accel xpu`

### Q2. 新機器下載後，模型快取還指到舊路徑

請優先檢查 `.env`：

```dotenv
HF_HOME=<你的快取路徑>
TIKTOKEN_RS_CACHE_DIR=<你的專案根目錄>
```

### Q3. API 已經啟動，但首頁沒有前端畫面

請先確認專案內的前端資料夾已有檔案：

- `Trusta-AST-Frontend/dist/index.html`
- `Trusta-AST-Frontend/dist/assets/`

若 `dist` 內已有檔案，直接重新開啟 `http://127.0.0.1:8000/` 即可。

### Q4. 只需要後端 API，不需要前端

直接使用以下路徑即可：

- `/docs`
- `/health`
- `/v1/chat/completions`

### Q5. Windows 上 `setup_env.ps1` 執行失敗

先確認：

- 已安裝 Python 3.12+
- 已安裝 `uv`
- 若使用 XPU，已安裝 Intel oneAPI

必要時可先調整 PowerShell 執行原則：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q6. Windows 上 `-Accel cuda` 建環境失敗

請確認已安裝 CUDA Toolkit，且驅動版本相容。若不是 NVIDIA GPU，請改用：

```powershell
.\deploy\windows\setup_env.ps1 -Accel xpu
```

### Q7. 用 DeepSpeed 做 fine-tune 時出現 offload / buffer / swapper 類錯誤怎麼辦？

本專案內建的 DeepSpeed profile 位於：

- `service/configs/deepspeed/zero3_offload_cpu_cpu.json`
- `service/configs/deepspeed/zero3_offload_cpu_disk.json`
- `service/configs/deepspeed/zero3_offload_disk_cpu.json`
- `service/configs/deepspeed/zero3_offload_disk_disk.json`

若錯誤訊息提到 `buffer`、`swapper`、NVMe offload、aio / offload queue 類問題，最常見的第一步是調高 `zero_optimization.offload_param.buffer_count`。

例如：

```json
"offload_param": {
  "device": "nvme",
  "buffer_count": 6
}
```

若仍不穩，再逐步調到 `8`，不要一次改太大。

如果 `offload_optimizer.device` 也是 `nvme`，也可以一併調整 `zero_optimization.offload_optimizer.buffer_count`。

> 實務上若看到類似「buffer 0 / swapper 1」錯誤，通常先從 `offload_param.buffer_count` 開始調最有效。

#### DeepSpeed 常見可調參數

| 參數 | 位置 | 何時調整 | 常見效果 / 取捨 |
|------|------|----------|------------------|
| `buffer_count` | `zero_optimization.offload_param.buffer_count` | NVMe/CPU offload 過程出現 buffer、swapper、queue 類錯誤 | 增加可用緩衝數，通常可改善 offload 穩定性，但會多吃記憶體 / 主記憶體 |
| `buffer_count` | `zero_optimization.offload_optimizer.buffer_count` | optimizer offload 到 NVMe 時出現類似錯誤 | 與上面類似，但作用在 optimizer state |
| `buffer_size` | `zero_optimization.offload_param.buffer_size` | 磁碟 I/O 過碎、吞吐不穩、頻繁 offload | 調大可減少切片次數，但會增加單次緩衝占用 |
| `max_in_cpu` | `zero_optimization.offload_param.max_in_cpu` | NVMe 太慢、希望多保留一部分參數在 RAM | 提高可減少磁碟讀寫，但會增加系統 RAM 使用 |
| `pin_memory` | `zero_optimization.offload_param.pin_memory` / `zero_optimization.offload_optimizer.pin_memory` | CPU RAM 夠、希望改善 CPU↔GPU 傳輸 | 可能提升傳輸效率，但也可能增加 RAM 壓力；記憶體吃緊時可關掉 |
| `stage3_prefetch_bucket_size` | `zero_optimization.stage3_prefetch_bucket_size` | ZeRO-3 prefetch 過大導致顯存 / RAM 壓力，或 I/O 不順 | 調小較保守、較穩；調大可能提升吞吐 |
| `reduce_bucket_size` | `zero_optimization.reduce_bucket_size` | 通訊或聚合過程吃太多顯存 | 調小可降瞬時記憶體壓力，但可能影響速度 |
| `stage3_param_persistence_threshold` | `zero_optimization.stage3_param_persistence_threshold` | 參數頻繁搬移造成效能不穩 | 調整可改變常駐策略，需配合模型大小實測 |
| `sub_group_size` | `zero_optimization.sub_group_size` | 超大模型分組處理時記憶體或效能不理想 | 調小較保守，調大可能提升效率 |
| `contiguous_gradients` | `zero_optimization.contiguous_gradients` | 梯度碎片化或記憶體配置問題 | 通常維持 `true`；若遇特殊相容性問題可測試切換 |
| `overlap_comm` | `zero_optimization.overlap_comm` | 通訊重疊造成不穩、記憶體尖峰、效能異常 | 關閉可提升穩定性，但可能變慢 |

#### 不只 DeepSpeed config，也常一起調整的訓練參數

若單改 DeepSpeed 仍不穩，通常也要一起降低訓練壓力：

| 參數 | 位置 | 建議方向 |
|------|------|----------|
| `per_device_train_batch_size` | 訓練請求 body | 顯存不足時先往下調 |
| `gradient_accumulation_steps` | 訓練請求 body 或 DeepSpeed config | batch size 降低後，可往上補回有效 batch |
| `max_seq_length` | 訓練請求 body | 序列太長時先降，例如 `4096 -> 2048` |
| `gradient_checkpointing` | 訓練請求 body | 記憶體不足時建議保持啟用 |

#### 建議調整順序

1. 先確認使用的是哪個 profile（CPU offload / NVMe offload）。
2. 若錯誤含 `buffer` / `swapper`，先調 `offload_param.buffer_count`。
3. 若 NVMe offload 仍不穩，再調 `offload_optimizer.buffer_count`、`buffer_size`、`max_in_cpu`。
4. 若是顯存 / RAM 不足，再降低 `per_device_train_batch_size`、`max_seq_length`。
5. 若是吞吐或穩定性問題，再微調 `stage3_prefetch_bucket_size`、`reduce_bucket_size`、`overlap_comm`。

若使用 NVMe offload，另外也建議確認：

- `nvme_path` 所在磁碟空間足夠
- 盡量使用 SSD / NVMe，不要放在慢速 HDD
- 避免與其他高 I/O 工作共用同一顆磁碟
