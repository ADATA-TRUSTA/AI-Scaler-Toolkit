# 安裝與設定

[← 回到 README](../README.zh-TW.md)

## 系統需求

### 基本需求（兩平台共同）

| 工具 | 說明 |
|------|------|
| Git | 用於 clone 與安裝期抓取 llama.cpp |
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

## 從 GitHub 下載專案

一般 `git clone` 即可 —— 本專案已無 Git submodule。（`llama.cpp` 於安裝期抓取,見下方。）

#### Linux

```bash
mkdir -p /home/test/project
cd /home/test/project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

#### Windows（PowerShell）

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit
```

---

## 第一次下載後必做設定

### 建立必要資料夾

#### Linux

```bash
mkdir -p logs .cache/huggingface
```

#### Windows（PowerShell）

```powershell
New-Item -ItemType Directory -Force logs, .cache\huggingface
```

### 建立 `.env`

專案啟動時，`backend/service/settings.py` 會優先讀取 `backend/`(service/ 所在目錄)的 `.env`；如果 `.env` 不存在，才會退回 `.env.example`。

**原則：請優先修改 `.env`，不要直接改 `backend/service/settings.py`。**

#### Linux

```bash
cp backend/.env.example backend/.env
```

#### Windows（PowerShell）

```powershell
Copy-Item backend\.env.example backend\.env
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
LLAMA_SERVER_BINARY=./backend/service/utils/llama.cpp/build/bin/llama-server
```

Windows 範例：

```dotenv
LLAMA_SERVER_BINARY=./backend/service/utils/llama.cpp/build/bin/Release/llama-server.exe
```

> 路徑可以用正斜線 `/` 或反斜線 `\`，Python 的 `pathlib` 兩者皆支援。

### 常見需要手動調整的路徑設定

| 變數 | 用途 | 預設值 |
|------|------|--------|
| `HF_HOME` | Hugging Face 模型與快取 | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS 快取 | 專案根目錄 |
| `LOG_DIR` | 日誌輸出目錄 | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | llama-server 可執行檔路徑 | 自動尋找 build 輸出 |
| `VLLM_SERVER_PROJECT_DIR` | vLLM 隔離環境目錄 | `backend/service/inference/engines/vllm_server` |

### dmidecode 權限設定（Linux 選配）

後端 `/system/resources` API 呼叫 `dmidecode` 取得 DRAM 規格。非 root 環境下若要完整資訊，可設定 sudoers 規則：

```bash
sudo visudo -f /etc/sudoers.d/ast-dmidecode
```

加入以下一行（將 `<user>` 替換成實際執行服務的帳號）：

```
<user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
```

若跳過此步驟，API 仍可正常運作，DRAM 規格欄位可能顯示為空。

## 建立 Python 環境

### Linux

#### 建議：使用自動腳本

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
```

若主機沒有 NVIDIA CUDA 環境，可改用 XPU：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=xpu bash backend/scripts/linux/setup_env.sh
```

跳過 vLLM 環境建置：

```bash
cd /home/test/project/AI-Scaler-Toolkit
TRUSTA_ACCEL=cuda TRUSTA_SETUP_VLLM=0 bash backend/scripts/linux/setup_env.sh
```

腳本會在 `backend/service/.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

> **提醒**：若要使用 fine-tune，請使用 **Linux + CUDA** 安裝流程。

### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel xpu
```

若要改用 CUDA：

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\setup_env.ps1 -Accel cuda
```

腳本會在 `backend\service\.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

### 若要使用 llama-server

`llama.cpp` 來源於安裝期抓取（`TRUSTA_SETUP_LLAMA=1 bash backend/scripts/linux/setup_env.sh`,Windows 用 `.\backend\scripts\windows\setup_env.ps1 -SetupLlama`）到 `backend/service/utils/llama.cpp`,再自行編譯：

#### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit/backend/service/utils/llama.cpp
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
cd C:\Users\<user>\project\AI-Scaler-Toolkit\backend\service\utils\llama.cpp
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

### 若要使用 vLLM engine（僅 Linux + CUDA）

`setup_env.sh` 在 CUDA 主機上會自動建立 vLLM 隔離環境；Windows 不支援 vLLM engine。

## 啟動服務

### Linux

```bash
cd /home/test/project/AI-Scaler-Toolkit
bash backend/scripts/linux/run_service.sh
```

若需要讓 `/system/resources` API 透過 `dmidecode` 讀取完整 DRAM 資訊，可改用具備權限的方式啟動：

```bash
cd /home/test/project/AI-Scaler-Toolkit
sudo bash backend/scripts/linux/run_service.sh
```

> 建議優先依照前文設定 `sudoers`，只授權特定 `dmidecode` 指令；除非必要，不建議長期以 root 執行整個服務。


### Windows

```powershell
cd C:\Users\<user>\project\AI-Scaler-Toolkit
.\backend\scripts\windows\run_service.bat
```

啟動腳本會直接使用 `backend/service/.venv` 內的 Python；**不需要手動啟動虛擬環境**。

### 預設服務位址

| 位址 | 說明 |
|------|------|
| `http://127.0.0.1:8000/` | 前端首頁（若前端檔案存在，會自動轉到 `/frontend/`） |
| `http://127.0.0.1:8000/health` | 健康檢查 |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

