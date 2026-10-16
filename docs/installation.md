# 安裝與設定

[← 回到 README](../README.zh-TW.md)

> 有在用 AI 程式助理嗎？可以請它幫你裝：見 [AI_SETUP.md](AI_SETUP.md)。

## 安裝發行版（一般使用者）

到 GitHub 的 [Releases](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/releases) 頁面下載你平台的檔案。每個版本有兩個，各對應一種架構：

| 平台 | 檔案 | 需求 |
|----------|------|----------|
| Windows | `AIScalerToolkit-<版本>-x64-Setup.exe` | Windows 10/11，x64。不需要系統管理員權限 |
| Linux | `AIScalerToolkit-<版本>-x86_64.AppImage` | x86_64，glibc 2.35 以上（例如 Ubuntu 22.04 以上） |

**GPU**：NVIDIA 需要 compute capability 7.5 以上（RTX 20、GTX 16 系列以後）以及 R580 以上的驅動；Intel Arc 與內顯也可以。只有 x64／x86_64 版本，沒有 ARM 版。

**磁碟**：第一次啟動會下載約 6-10 GB（Python 套件與 GPU 執行環境）；請至少留 20 GB，再加上你要下載的模型。

每個 release 頁面都列出兩個檔案的 SHA-256。Windows 安裝檔由 ADATA Technology Co., Ltd. 簽章。

### Windows

執行 `Setup.exe`。它只裝給你自己的帳號，裝在 `%LOCALAPPDATA%\Programs\AIScalerToolkit`，並加入開始功能表（桌面捷徑可選）。從其中一個開啟 **AI Scaler Toolkit**。

### Linux

**第一次啟動前**：把檔案設成可執行（`chmod +x`，或在檔案內容勾選「允許以程式執行檔案」），否則雙擊不會有反應。把它放在之後不會移動的地方（例如 `~/Applications/`）：選單項目與開機自動啟動都指向這個檔案。AppImage 透過 FUSE 掛載自己，用的是 Ubuntu 預設就有的 `fuse3`，不需要 `libfuse2`。如果出現 `fusermount: not found`，執行 `sudo apt install fuse3`。

**第一次啟動**：AppImage 會一次裝好缺少的系統套件，只問一次密碼：

- WebKitGTK（畫視窗用）與系統匣圖示
- 編譯工具 `build-essential` 與 `cmake`：llama.cpp 會在你的電腦上編譯，針對你的 CPU 最佳化
- **有 Intel GPU 時**還有：Level Zero 與 OpenCL（`libze1`、`libze-intel-gpu1`、`intel-opencl-icd`，PyTorch 透過它們使用 GPU）、`libze-dev`（微調需要）、`intel-ocloc`、Vulkan 驅動與工具（`mesa-vulkan-drivers`、`vulkan-tools`），以及 llama.cpp 的 Vulkan 版編譯時要用的 `libvulkan-dev`、`glslc`、`spirv-headers`。Windows 上這些隨 Intel 驅動一起裝；乾淨的 Ubuntu 一個都沒有
- **較新的 Intel GPU（Lunar Lake、Panther Lake、Arc B 系列）要先加入 Intel 的顯示套件來源**：Ubuntu 24.04 內建的版本（2023 年）太舊，完全看不到這些 GPU。第一次啟動前做一次：

  ```bash
  sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
  ```

  AppImage 不會替你加套件來源，因為那會改變之後每次系統更新會裝什麼。Ubuntu 22.04 的套件庫沒有 Level Zero，所以 Intel GPU 請用 24.04 以上
- NVIDIA 驅動是專有軟體，要自己先裝好（`nvidia-smi` 要能執行）

### 第一次啟動做了什麼

兩個平台的第一次啟動都會先顯示設定視窗，再開啟應用程式：

1. 檢查電腦（磁碟空間、視窗執行環境、GPU）
2. 安裝對應 GPU 的 Python 套件（NVIDIA 用 CUDA、Intel 用 XPU）與 llama.cpp 執行環境。同時有 NVIDIA 與 Intel 顯卡時會用 NVIDIA；要改用 Intel 見 [FAQ Q14](faq.md#q14-同時有-nvidia-和-intel-顯卡用的是哪一張)
3. 啟動服務，並下載一個小的入門模型（MiniCPM5-1B），讓模型清單不是空的

大約 5-15 分鐘，網路慢會更久。如果中途中斷，再開一次應用程式即可：它會從中斷的地方繼續，只裝缺少的部分。之後啟動只要幾秒。

服務在 `http://127.0.0.1:8571`（只限這台電腦）。視窗、設定與 API 見[使用說明](usage.md)。

### 更新

AST 的 Settings → Updates → **Check now**，再按 **Update now**。應用程式會下載新版，用 release 公布的 SHA-256 檢查（Windows 另外檢查簽章），安裝後自動重新開啟。新版需要不同的 Python 套件時，它第一次啟動會把環境更新到位（幾秒到幾分鐘）。

也可以直接下載較新的版本，照上面的方式安裝覆蓋舊版。模型與設定都會保留。

### 解除安裝

**Windows**：設定 → 應用程式 → AI Scaler Toolkit → 解除安裝。它會移除程式與它安裝的環境，並詢問是否一併刪除下載的模型。

**Linux**：沒有解除安裝程式。先結束應用程式（系統匣圖示 → Quit），再刪除：

- AppImage 本身
- `~/.local/share/ai-scaler-toolkit`（環境）與 `~/.config/ai-scaler-toolkit`（設定）
- `~/.local/share/applications/ai-scaler-toolkit*.desktop`、`~/.local/share/icons/hicolor/256x256/apps/ai-scaler-toolkit*.png`、`~/.config/autostart/ai-scaler-toolkit.desktop` 與 `~/Desktop/ai-scaler-toolkit.desktop`

下載的模型留在 `~/.cache/huggingface`，要自己刪才會刪掉。

### 檔案位置

| | Windows | Linux |
|---|---|---|
| 程式 | `%LOCALAPPDATA%\Programs\AIScalerToolkit` | AppImage 本身 |
| 環境、log | `%LOCALAPPDATA%\AIScalerToolkit`（`logs\launcher.log`、`provision.log`、`service.log`） | `~/.local/share/ai-scaler-toolkit`（`logs/…`） |
| 設定 | `%LOCALAPPDATA%\AIScalerToolkit\state.json` | `~/.config/ai-scaler-toolkit/state.json` |
| 下載的模型 | `%USERPROFILE%\.cache\huggingface` | `~/.cache/huggingface` |

模型快取是 Hugging Face 的標準位置，所以和這台電腦上其他 Hugging Face 工具共用。

### 沒有桌面（Linux 伺服器）

`--headless` 不開視窗，直接設定並執行服務。它沒辦法問密碼，所以要先裝好系統套件：

```bash
sudo apt install build-essential cmake
# 有 Intel GPU 時（較新的 GPU 先加上面的 PPA）：
sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd libze-dev intel-ocloc mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc spirv-headers
chmod +x AIScalerToolkit-*-x86_64.AppImage
./AIScalerToolkit-*-x86_64.AppImage --headless
```

第一次執行會完成設定，之後服務持續在 `127.0.0.1:8571` 執行，按 Ctrl+C 停止。沒有 FUSE 的機器，設定 `APPIMAGE_EXTRACT_AND_RUN=1`。

---

## 從原始碼安裝（開發者）

這頁以下是給要改程式碼的人。原始碼直接執行後端：沒有設定視窗，`.env` 要自己寫，服務的 port 也由它決定（`.env.example` 是 `8000`）。

### 需求

| 工具 | 說明 |
|------|-------------|
| Git | clone，以及設定時抓 llama.cpp 的轉檔腳本 |
| `uv` | Python 與環境管理；它會自己裝 Python 3.12 |
| Linux：`build-essential`、`cmake` | Linux 上的 llama.cpp 從原始碼編譯 |

Linux（Ubuntu / Debian）：

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Windows（PowerShell）：先裝 [Git for Windows](https://git-scm.com/download/win)，再裝 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Windows 上的 llama.cpp 不用編譯：設定時會下載官方的預編版。

> **功能支援**：vLLM 只在 **Linux** 上，NVIDIA（CUDA）預設安裝，Intel 要自己開啟（`TRUSTA_SETUP_VLLM=1`）。微調只在 **Linux** 上（CUDA 或 Intel XPU）。Windows 上的引擎是 Transformers 與 llama.cpp。要在有 NVIDIA 顯卡的 Windows 電腦上用 vLLM 或微調，可以在 WSL2（Ubuntu 發行版）裡執行 Linux 版：例如照上面的方式用 `--headless` 執行 AppImage。
>
> repo 已經包含前端的靜態檔，不需要 Node.js，也不用另外 build 前端。

### 取得程式碼

一般的 `git clone` 就夠了，沒有 submodule。

```bash
git clone https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit.git
cd AI-Scaler-Toolkit
```

### 建立 `.env`

服務啟動時，`backend/service/settings.py` 會讀 `backend/.env`。檔案不存在時會改讀 `backend/.env.example`，也就是範例裡的值，包括放在專案裡面的模型快取。請複製一份，至少設好路徑：

```bash
cp backend/.env.example backend/.env          # Windows：Copy-Item backend\.env.example backend\.env
```

```dotenv
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8000

HF_HOME=/home/<user>/.cache/huggingface      # Windows：C:\Users\<user>\.cache\huggingface
LOG_DIR=/home/<user>/AI-Scaler-Toolkit/logs

LOG_TO_FILE=true
LOG_FILE_NAME=service.log
LOG_BACKUP_COUNT=14
```

選填的設定在 `.env.example` 裡是註解掉的；要用的話，拿掉開頭的 `#` 填入值，例如私有的 Hugging Face 模型要設 `HF_TOKEN=hf_xxx`。路徑用 `/` 或 `\` 都可以。

| 變數 | 用途 | 預設 |
|------|------|--------|
| `HF_HOME` | Hugging Face 模型與快取 | `<project>/.cache/huggingface` |
| `LOG_DIR` | log 輸出目錄 | `<project>/logs` |
| `LLAMA_SERVER_BINARY` | `llama-server` 執行檔 | 由 `setup_env` 寫入；`backend/service/utils/llama-bin/` |
| `VLLM_SERVER_PROJECT_DIR` | 放 vLLM 的 `.venv` 的專案 | backend 專案本身（vLLM 透過 `vllm` extra 安裝） |

### 建立 Python 環境

Linux，NVIDIA：

```bash
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
```

Linux，Intel：`TRUSTA_ACCEL=xpu`。不裝 vLLM：`TRUSTA_SETUP_VLLM=0`。

Windows：

```powershell
.\backend\scripts\windows\setup_env.ps1 -Accel cuda      # Intel 用 -Accel xpu
```

腳本會建立 `backend/.venv`（不用手動啟用）、把 llama.cpp 裝到 `backend/service/utils/llama-bin/` 並記進 `.env`、把 llama.cpp 的 GGUF 轉檔腳本抓到 `backend/service/utils/llama.cpp/`，有 Intel GPU 時還會實際檢查 GPU 能不能計算。每一部分只在缺少時才裝，所以重跑很快。

**llama.cpp**：版本 `b10107`，可用 `TRUSTA_LLAMA_VERSION`（Linux）或 `-LlamaVersion`（Windows）改。使用哪個 build 與 torch 的加速器分開決定，用 `TRUSTA_LLAMA_BACKEND`（Linux）或 `-LlamaBackend`（Windows）：`auto`（預設）有 NVIDIA 顯卡時用 CUDA，否則用 Vulkan。重裝用 `TRUSTA_INSTALL_LLAMA=1` 或 `-Llama force`；跳過用 `TRUSTA_INSTALL_LLAMA=0` 或 `-Llama skip`。要用自己的 llama，把 `LLAMA_SERVER_BINARY` 指過去，設定就不會動它。

### 啟動服務

```bash
bash backend/scripts/linux/run_service.sh          # Windows：.\backend\scripts\windows\run_service.bat
```

| 網址（`SERVICE_PORT=8000` 時） | 說明 |
|------|------|
| `http://127.0.0.1:8000/` | 網頁介面（轉到 `/frontend/`） |
| `http://127.0.0.1:8000/health` | 健康檢查 |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

要從原始碼執行桌面啟動器，見 [deploy/README.md](../deploy/README.md)。
