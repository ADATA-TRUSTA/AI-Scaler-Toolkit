# 已知的依賴套件漏洞

這份文件列出掃描（osv-scanner 2.7.0，2026-10-27）在本專案安裝的套件上回報、但刻意保留的漏洞：原因是目前還沒有
修補版本，或是有修補版本但其他依賴套件還不允許升級。每一項都說明這個套件在本專案怎麼使用、為什麼這種用法碰不到
那個漏洞，以及什麼時候該重新檢查。

當修補版本可以安裝時（那時就升級），或是這裡描述的用法有變動時，請重新檢查對應的項目。

English version: [dependency_advisories.md](dependency_advisories.md)

---

## chromadb 1.5.9：選裝，`chroma` extra

**預設不安裝。** RAG 的文件存在 SQLite，用 FTS5 全文檢索（不支援時退回 `LIKE` 搜尋），也可以搭配
sentence-transformers 的向量做排序。Chroma 是同一組 `/rag/*` 路由可以改用的另一種向量資料庫。

要使用時，在 `setup_env` 選好的 extra 之外再加上這個 extra，然後在 `.env` 開啟：

```bash
uv sync --extra cuda --extra vllm --extra chroma   # 保留這台機器原本用的 extra
```

```bash
RAG_BACKEND=chroma        # 或 RAG_USE_CHROMA=true
```

如果設定了上面任一項、卻沒有安裝 `chroma` extra，服務啟動時會記錄一條警告，並繼續使用 SQLite。

### 尚未修補的漏洞

| 漏洞編號 | 內容 |
|---|---|
| GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 | 透過 Chroma 伺服器的 collections 端點，不需登入就能注入程式碼 |
| GHSA-36p7-vc44-83pf / PYSEC-2026-3814 | 已登入的使用者透過更新 collection 的端點注入程式碼 |
| GHSA-2wm9-hf6c-p5cr / PYSEC-2026-3813 | 任何已登入的使用者都能讀寫其他租戶的 collection |
| GHSA-xph7-9rjv-w5fr / PYSEC-2026-3815 | `SimpleRBACAuthorizationProvider` 不檢查權限屬於哪個租戶 |

### 為什麼在這裡碰不到

這四個漏洞都在 Chroma 自己的 HTTP 伺服器上：它的 `/api/v2/...` 端點，以及它的驗證和授權元件。本專案從來不啟動
那個伺服器。`service/rag_manager.py` 用的是 `chromadb.PersistentClient`，在服務自己的程序裡讀寫本機資料夾，
不會開任何 Chroma 的連接埠。本專案自己的 `/rag/*` 路由，除非設定 `SERVICE_REMOTE_ACCESS=all`，否則只開放給本機。

以下情況這個結論就不成立：有人在服務旁邊執行 `chroma run`（Chroma 的獨立伺服器），或是把 `PersistentClient`
換成連到共用 Chroma 伺服器的 `HttpClient`。

---

## accelerate 1.14.0：一定會安裝

| 漏洞編號 | 內容 |
|---|---|
| GHSA-4j2p-28q2-5m79 / PYSEC-2026-3804 | 透過分片 checkpoint 索引檔的 `weight_map` 造成路徑穿越和阻斷服務 |

問題出在 `load_checkpoint_in_model` 和 `load_checkpoint_and_dispatch`：它們讀取 checkpoint 索引檔裡的分片路徑時
沒有檢查，所以特製的 checkpoint 可以指向任何檔案（`../`、絕對路徑），或指向一個會讓載入永遠卡住的具名管道。

### 在本專案怎麼使用

- `infer_auto_device_map`：把模型的各層分配到 GPU 和 CPU
  （`service/inference/engines/transformers_engine.py`、`service/inference/model_utils.py`）。
- `init_empty_weights`：建立不配置權重的模型，用來估算記憶體（`service/inference/memory_estimator.py`）。
- `AlignDevicesHook`、`SequentialHook`：掛上 LoRA adapter 時，找出 offload 的基礎模型放在哪個裝置
  （`service/inference/load_model/peft_loader.py`）。

### 為什麼在這裡碰不到

本專案沒有呼叫這兩個函式；`uv.lock` 裡這些版本的 transformers、peft、trl、sentence-transformers、
bitsandbytes 也都沒有呼叫（2026-10-27 搜尋它們安裝後的原始碼確認）。transformers 載入分片 checkpoint 用的是
它自己的程式碼。而且就算經過這兩個函式，攻擊也需要使用者在這台機器上載入被動過手腳的 checkpoint。

以下情況這個結論就不成立：本專案的程式開始呼叫這兩個函式之一，或是升級 transformers、peft 後，它們改成透過
這兩個函式載入。升級那兩個套件時，請重新搜尋一次安裝後的套件。

---

## setuptools 80.10.2：上游已修補，被 vllm 擋住

| 漏洞編號 | 內容 | 修補版本 |
|---|---|---|
| GHSA-h35f-9h28-mq5c / CVE-2026-59890 / PYSEC-2026-3447 | 在 macOS 的 APFS/HFS+ 上，檔名只差在 Unicode 正規化（NFC/NFD）時，`MANIFEST.in` 的排除規則會漏掉，sdist 可能夾帶原本要排除的檔案 | 83.0.0 |
| GHSA-xmj3-9gf7-h52w | 在 Windows 上，wheel 的 `namespace_packages.txt` 可以讓 setuptools 的舊式 wheel 安裝（`Wheel.install_as_egg`）把 `__init__.py` 寫到目標資料夾之外 | 84.1.0 |

跟上面兩項不同，這兩個漏洞都已經有修補版本，只是還裝不了：本專案沒有直接指定 setuptools，它是 torch 和 vllm 帶進來
的，而 vllm 0.32.0 要求 `setuptools<81`。所以 uv 在所有平台上都解析成 80.10.2，Windows 也一樣（透過 torch）。

### 在本專案怎麼使用

`service/` 和 `scripts/` 都沒有 import `setuptools` 或 `pkg_resources`。它只在從原始碼建置套件時才會用到，而
`uv.lock` 裡要從原始碼建置的只有兩個，都只在 Linux 上安裝：`deepspeed`（在 uv 隔離的建置環境裡建置，用的是那裡
自己的 setuptools，不是這一份）和 `causal-conv1d`（列在 `no-build-isolation-package`，所以會用已安裝的 80.10.2 建置）。

### 為什麼在這裡碰不到

- sdist 的漏洞是在 macOS 上*製作* sdist 準備發布時才會發生。本專案不建置也不發布 sdist，也不支援 macOS。
- Windows 的漏洞需要由 setuptools 自己來安裝 wheel。在已安裝的 setuptools 裡，唯一會走到那裡的是
  `fetch_build_eggs`，也就是從原始碼建置時處理舊式的 `setup_requires`。uv 用自己的安裝程式安裝 wheel，而在
  Windows 上，本專案沒有任何套件會用已安裝的 setuptools 從原始碼建置（2026-10-27 在 Windows 的 venv 確認）。
  唯一會用到它的 `causal-conv1d` 只在 Linux 上建置。

以下情況這個結論就不成立：vllm 允許 `setuptools>=81`（那時就立刻升到 84.1.0 以上，每次升級 vllm 都要檢查這點）、
Windows 的安裝開始有套件在沒有建置隔離的情況下從原始碼建置，或是本專案開始建置並發布 sdist。
