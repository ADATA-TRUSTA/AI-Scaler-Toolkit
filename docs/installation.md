# 安裝與設定

[← 回到 README](../README.zh-TW.md)

## 用安裝檔安裝（一般使用者）

從 GitHub 的 [Releases](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/releases) 下載自己平台的檔案。每個版本都有兩個檔案，各自對應一種架構：

| 平台 | 檔案 | 需求 |
|------|------|------|
| Windows | `AIScalerToolkit-<版本>-x64-Setup.exe` | Windows 10／11，x64。不需要系統管理員權限 |
| Linux | `AIScalerToolkit-<版本>-x86_64.AppImage` | x86_64，glibc 2.35 以上（例如 Ubuntu 22.04 以上） |

**GPU**：NVIDIA 需 compute capability 7.5 以上（RTX 20 系列、GTX 16 系列以後）與 R580 以上的驅動；Intel Arc／iGPU 亦可。目前只提供 x64／x86_64，沒有 ARM 版本。

**Linux 先裝 FUSE**：AppImage 要靠 FUSE 2 才能執行，Ubuntu 24.04 預設沒有，沒裝的話檔案連開都開不起來。第一次使用前先執行一次：

```bash
sudo apt install libfuse2t64      # Ubuntu 24.04 以上
sudo apt install libfuse2         # Ubuntu 22.04
```

**Linux 第一次啟動**：AppImage 會把缺少的系統套件一次裝好，只跳一次系統密碼視窗：

- WebKitGTK（畫視窗用）、系統匣圖示
- 編譯工具 `build-essential`、`cmake`：llama.cpp 會在你的電腦上編譯，針對這台 CPU 最佳化
- **Intel 顯示卡**另外會裝：Level Zero 與 OpenCL（`libze1`、`libze-intel-gpu1`、`intel-opencl-icd`，PyTorch 用它們使用 GPU）、`libze-dev`（微調訓練要用）、Vulkan 驅動與工具（`mesa-vulkan-drivers`、`vulkan-tools`），以及 llama.cpp 編 Vulkan 版所需的 `libvulkan-dev`、`glslc`、`spirv-headers`。Windows 上這些會隨 Intel 驅動一起裝好，Ubuntu 預設沒有
- **新款 Intel 顯示卡（Lunar Lake、Panther Lake、Arc B 系列）要先加 Intel 的顯卡套件來源**：Ubuntu 24.04 內建的版本（2023 年）太舊，裝了也偵測不到 GPU。第一次啟動前先執行一次：

  ```bash
  sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
  ```

  AppImage 不會替你加套件來源，因為它會影響之後整個系統的更新。Ubuntu 22.04 的套件庫沒有 Level Zero，Intel 顯示卡請用 24.04 以上
- NVIDIA 顯示卡的驅動是專有軟體，要自己先裝好（`nvidia-smi` 能執行即可）

沒有圖形介面、或想先自己裝好時：

```bash
sudo apt install libfuse2t64 libwebkit2gtk-4.1-0 gir1.2-webkit2-4.1 gir1.2-ayatanaappindicator3-0.1 build-essential cmake
# Intel 顯示卡再加上（新款先加上面的 PPA）：
sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd libze-dev mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc spirv-headers
chmod +x AIScalerToolkit-*-x86_64.AppImage
./AIScalerToolkit-*-x86_64.AppImage
```

兩個平台第一次啟動都會下載 GPU 執行環境（數 GB），之後啟動就快了。

以下是**從原始碼安裝**（開發者）的說明。

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

`setup_env` 會把它驗證過的 llama 執行檔路徑寫進 `.env`，通常不需要手動設定。若要指向自己編譯的版本，可在 `.env` 補上：

```dotenv
LLAMA_SERVER_BINARY=./backend/service/utils/llama-bin/llama-server
```

Windows 範例：

```dotenv
LLAMA_SERVER_BINARY=./backend/service/utils/llama-bin/llama-server.exe
```

> 路徑可以用正斜線 `/` 或反斜線 `\`，Python 的 `pathlib` 兩者皆支援。

### 常見需要手動調整的路徑設定

| 變數 | 用途 | 預設值 |
|------|------|--------|
| `HF_HOME` | Hugging Face 模型與快取 | `<project>/.cache/huggingface` |
| `TIKTOKEN_RS_CACHE_DIR` | TikToken / GPT-OSS 快取 | 專案根目錄 |
| `LOG_DIR` | 日誌輸出目錄 | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | llama-server 可執行檔路徑 | `backend/service/utils/llama-bin/`，找不到才退回 PATH |
| `VLLM_SERVER_PROJECT_DIR` | 其 `.venv` 裝有 vLLM 的專案目錄 | backend 專案根目錄（vLLM 已改由 `vllm` extra 裝進本專案環境，不再是獨立子專案） |

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

腳本會在 `backend/.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

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

腳本會在 `backend\.venv` 建立環境；**不需要另外手動啟動虛擬環境**。

### 若要使用 llama-server

安裝流程會取得**官方預編**的 `llama` 執行檔 —— 不需要編譯器，也沒有任何 C++ 建置步驟。
此步驟**預設就會執行**，而且只補齊實際缺少的部分，所以直接執行 `setup_env` 即可：

#### Linux

```bash
bash backend/scripts/linux/setup_env.sh
```

#### Windows（PowerShell）

```powershell
.\backend\scripts\windows\setup_env.ps1
```

它會做兩件事：

1. 透過官方的 `ggml-org/llama-install.sh` 安裝整合後的 `llama` 執行檔（以
   `llama serve` / `llama quantize` 方式使用），Linux 安裝到 `~/.local/bin/llama`，
   Windows 則是 `%LOCALAPPDATA%\Microsoft\WindowsApps\llama.exe`。可用 `.env` 中的
   `LLAMA_SERVER_BINARY` 覆寫路徑。
2. 以 sparse fetch 只取得 GGUF 轉換工具（`convert_hf_to_gguf.py`、
   `convert_lora_to_gguf.py`、`conversion/`、`gguf-py`）到
   `backend/service/utils/llama.cpp`，約 1.7 MB 的純 Python —— 因為這些腳本並不隨
   預編執行檔一起發佈。C++ 原始碼不會被抓取。

llama 的建置版本與 torch 的加速器**互相獨立**，由 `TRUSTA_LLAMA_BACKEND`（Linux）或
`-LlamaBackend`（Windows）決定：`auto`（預設）在偵測到 NVIDIA 顯卡時選用原生 CUDA 版本，
否則使用通用的 Vulkan 版本（Intel / AMD / NVIDIA 顯卡都能辨識）。若要在有 NVIDIA 顯卡的機器上
強制使用通用版本，請傳入 `vulkan`。

版本固定在 `b10107`，可用 `TRUSTA_LLAMA_VERSION`（Linux）或 `-LlamaVersion`（Windows）覆寫。

兩個部分都只在缺少時才安裝，因此重複執行 `setup_env` 成本很低，也不會破壞既有環境：

| 模式 | Linux | Windows | 行為 |
|---|---|---|---|
| auto *（預設）* | — | — | 只安裝缺少的部分 |
| skip | `TRUSTA_INSTALL_LLAMA=0` | `-Llama skip` | 完全不做任何事 |
| force | `TRUSTA_INSTALL_LLAMA=1` | `-Llama force` | 即使已存在也重新安裝 |

執行檔的偵測會依序參考 `LLAMA_SERVER_BINARY`（環境變數或 `backend/.env`）、預設安裝位置、
最後是 `PATH` 上的 `llama`。轉換工具則必須 `convert_hf_to_gguf.py`、
`convert_lora_to_gguf.py` 與 `gguf-py/` 三者齊備才算存在，因此抓取到一半的 checkout 會被修復
而不是跳過。

> **已經有自己的 llama？** 只要把 `LLAMA_SERVER_BINARY` 指向它，auto 模式就會偵測到並完全不動它。
> 只有 `force` 會覆蓋，而官方安裝程式寫入的是上述固定位置，所以被取代的是那一份。

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

啟動腳本會直接使用 `backend/.venv` 內的 Python；**不需要手動啟動虛擬環境**。

### 預設服務位址

| 位址 | 說明 |
|------|------|
| `http://127.0.0.1:8000/` | 前端首頁（若前端檔案存在，會自動轉到 `/frontend/`） |
| `http://127.0.0.1:8000/health` | 健康檢查 |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

