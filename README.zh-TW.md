# AI Scaler Toolkit

<p align="center">
  <img src="assets/trusta-icon.png" alt="Trusta" width="32" height="32">
  &nbsp;&nbsp;
  <img src="assets/adata-icon.png" alt="ADATA" width="32" height="32">
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="README.zh-TW.md">繁體中文</a>
</p>

---

AI Scaler Toolkit 讓你在自己的電腦上，用 NVIDIA 或 Intel 的 GPU 執行大型語言模型。它是 Windows 與 Linux 的桌面應用程式：

- 在視窗裡和模型對話，支援串流輸出；能看圖的模型也可以傳圖片
- 從 Hugging Face 下載模型，並轉成 GGUF
- 三種引擎：Transformers、llama.cpp，以及 vLLM（Linux）
- OpenAI 相容的 API，在 `http://127.0.0.1:8571/v1`，讓你的其他應用程式和程式助理也能用同一個模型
- 微調（Linux）與文件搜尋（RAG）
- 可以幫你安裝的選用元件：Hermes Agent 與 ComfyUI

資料都不會離開你的電腦。

---

## 1. 安裝

### 讓 AI 幫你裝

有在用 AI 程式助理嗎（Claude Code、Cursor、Codex、GitHub Copilot 等）？把下面這段貼給它：

```text
Install AI Scaler Toolkit on this PC for me: https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit - follow docs/AI_SETUP.md in that repository.
```

它會檢查你的 GPU、驅動和磁碟空間，從最新版下載對應的檔案並核對，安裝好後告訴你怎麼連接其他應用程式。

### 或自己裝

從[最新版](https://github.com/ADATA-TRUSTA/AI-Scaler-Toolkit/releases/latest)下載你平台的檔案：

| 平台 | 檔案 | 接著 |
|---|---|---|
| Windows 10/11（x64） | `AIScalerToolkit-<版本>-x64-Setup.exe` | 執行它。不需要系統管理員權限 |
| Linux（x86_64，Ubuntu 22.04 以上） | `AIScalerToolkit-<版本>-x86_64.AppImage` | `chmod +x` 後執行 |

**需要**：compute capability 7.5 以上的 NVIDIA GPU（RTX 20／GTX 16 系列以後）與 580 以上的驅動，或 Intel Arc 顯示卡、Intel 內顯；以及約 20 GB 的磁碟空間。Linux 上用較新的 Intel GPU 時，要先加入 Intel 的顯示套件來源（[說明](docs/installation.md#linux)）。

**第一次啟動**會把所有東西裝好：安裝對應顯示卡的 Python 套件與 GPU 執行環境（約 6-10 GB、5-15 分鐘）和一個小的入門模型，然後開啟對話視窗。Linux 上會問一次密碼，用來安裝系統套件。之後啟動只要幾秒。

**更新**：AST 的 Settings → Updates → Check now → Update now。

其他內容，包括解除安裝與沒有桌面的 Linux 伺服器：見[安裝說明](docs/installation.md)。

---

## 2. 從其他應用程式使用

在任何接受「OpenAI 相容」provider 的應用程式裡設定：

| 設定 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8571/v1` |
| 模型 | `trusta-ast-default`（代表目前載入的模型） |
| API key | 任何字串 |

Hermes Agent 從 Settings → Components 裝好後，按一個按鈕就設定完成：**Config AST Endpoint**。更多內容見[使用說明](docs/usage.md)。

---

## 3. 專案結構

```text
AI-Scaler-Toolkit/
├─ backend/                 # 服務本體（FastAPI），從 backend repo 同步
│  ├─ service/              # app.py、settings.py、推論引擎、訓練、設定檔
│  ├─ scripts/
│  │  ├─ linux/             # setup_env.sh、run_service.sh、stop_service.sh
│  │  ├─ windows/           # setup_env.ps1、run_service.bat
│  │  └─ xpu_smoke.py       # Intel GPU 檢查，由 setup_env 執行
│  ├─ tests/
│  ├─ pyproject.toml
│  └─ .env.example          # 服務讀的是 backend/.env
├─ console/                 # 無介面的 Python 用戶端
├─ frontend/                # 預先 build 好的網頁介面（由 backend 提供）
│  ├─ dist/
│  └─ dist_client/
├─ deploy/                  # 桌面應用程式、Windows 安裝檔、AppImage
├─ docs/
├─ wiki/
├─ .github/
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

要改程式碼：見[安裝 → 從原始碼安裝](docs/installation.md#從原始碼安裝開發者)與 [CONTRIBUTING](.github/CONTRIBUTING.md)。

---

## 📚 完整文件

- **[安裝與設定](docs/installation.md)** — 需求、第一次啟動、更新、解除安裝、Linux 伺服器、從原始碼安裝
- **[使用與驗證](docs/usage.md)** — 視窗、設定、元件、連接其他應用程式、API
- **[維護與常見問題](docs/faq.md)** — 更新與疑難排解
- **[AI_SETUP.md](docs/AI_SETUP.md)** — 給 AI 程式助理的安裝說明（英文）

> English docs: [Installation](docs/installation.en.md) · [Usage](docs/usage.en.md) · [FAQ](docs/faq.en.md)
