# AI Scaler Toolkit

<p align="center">
  <img src="frontend/dist/Trusta-16.ico" alt="Trusta" width="32" height="32">
  &nbsp;&nbsp;
  <img src="frontend/dist_client/Adata.ico" alt="ADATA" width="32" height="32">
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="README.zh-TW.md">繁體中文</a>
</p>

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
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

mkdir -p logs .cache/huggingface
cp .env.example .env
# 編輯 .env，優先修改 HF_HOME、LOG_DIR、SERVICE_HOST、SERVICE_PORT

TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
bash backend/scripts/linux/run_service.sh
```

啟動成功後，直接用瀏覽器開啟：

- `http://127.0.0.1:8000/`

若主機沒有 NVIDIA CUDA 環境，再改用 `TRUSTA_ACCEL=xpu`。

> **提醒**：目前 fine-tune 功能僅支援 **Linux + CUDA** 環境。

### Windows（建議先走 XPU）

```powershell
mkdir C:\Users\<user>\project
cd C:\Users\<user>\project
git clone <YOUR_GITHUB_REPOSITORY_URL> AI-Scaler-Toolkit
cd AI-Scaler-Toolkit

New-Item -ItemType Directory -Force logs, .cache\huggingface
Copy-Item .env.example .env
notepad .env

.\backend\scripts\windows\setup_env.ps1 -Accel xpu
.\backend\scripts\windows\run_service.bat
```

啟動成功後，直接用瀏覽器開啟：

- `http://127.0.0.1:8000/`

若要改走 NVIDIA CUDA，將 `-Accel xpu` 改成 `-Accel cuda`。

> 詳細安裝需求、`.env` 說明、模型載入方式與驗證方式，請看後續章節。

---

## 2. 專案結構

```text
AI-Scaler-Toolkit/
├─ backend/                 # 後端服務，從上游 backend repo 同步
│  ├─ service/
│  │  ├─ app.py
│  │  ├─ settings.py
│  │  ├─ pyproject.toml
│  │  └─ configs/
│  ├─ tests/
│  ├─ pytest.ini
│  └─ scripts/
│     ├─ linux/
│     │  ├─ run_service.sh
│     │  ├─ setup_env.sh
│     │  └─ stop_service.sh
│     ├─ windows/
│     │  ├─ run_service.bat
│     │  └─ setup_env.ps1
│     └─ docker/
├─ console/                 # 無頭 Python client
├─ frontend/                # 預先編譯的前端（由後端提供）
│  ├─ dist/
│  └─ dist_client/
├─ docs/
├─ examples/
│  └─ datasets/
├─ wiki/
├─ .github/
├─ .env.example
├─ LICENSE
├─ README.md
└─ README.zh-TW.md
```

---

## 📚 完整文件

- **[安裝與設定](docs/installation.md)** — 系統需求、下載、`.env`、Python 環境、啟動服務
- **[使用與驗證](docs/usage.md)** — 健康檢查、載入模型、常用 API 路由、前端與日誌
- **[效能實測](docs/benchmarks.md)** — 各 GPU / 量化的 TPS / TTFT
- **[維護與常見問題](docs/faq.md)** — 更新與疑難排解

> English docs: [Installation](docs/installation.en.md) · [Usage](docs/usage.en.md) · [Benchmarks](docs/benchmarks.en.md) · [FAQ](docs/faq.en.md)
