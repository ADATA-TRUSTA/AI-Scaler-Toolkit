# INSTALL.md — 在一台全新電腦上裝起來（Windows / Linux）

[English](INSTALL.md) | 繁體中文

這份文件假設你已經有：一台有 **NVIDIA GPU** 的機器、**Hermes 已安裝並跑得起來**、
以及（可選但這套設計就是為它而生的）一顆**地端常駐 LLM**，由 **TRUSTA-AST 後端**代管
（預設 `http://127.0.0.1:8571`）。這包不直接接觸推理引擎，只透過該後端的 API 詢問狀態。

安裝一句話版本：

```
# Windows
.\install.ps1

# Linux
./install.sh
```

其餘篇幅都在講**為什麼**、以及出錯時看哪裡。

---

## 0. 這套東西會動到哪些目錄

先講清楚會動到哪裡。**只有三個地方**，其中你需要親手改的只有一個檔案。

| 用途 | Windows | Linux | 誰建立 |
|---|---|---|---|
| **Hermes home** | `%LOCALAPPDATA%\hermes` | `~/.hermes` | Hermes 自己（已存在） |
| **Skill**（Hermes 讀得到的技能） | `%LOCALAPPDATA%\hermes\skills\{imggen,vidgen}\` | `~/.hermes/skills/{imggen,vidgen}/` | `install` 複製進去 |
| **引擎**（規劃器、執行器、workflow） | `%LOCALAPPDATA%\hermes\comfy-engine\` | `~/.hermes/comfy-engine/` | `install` 複製進去 |
| **設定檔**（★ 你唯一要改的） | `…\comfy-engine\hermes-comfy.conf` | `…/comfy-engine/hermes-comfy.conf` | `install` 產生 |
| **ComfyUI 本體＋模型** | 你自己裝的位置 | 你自己裝的位置 | 你自己 |
| **Hermes 設定** | `%LOCALAPPDATA%\hermes\config.yaml` | `~/.hermes/config.yaml` | 改一行 `terminal.timeout` |

幾件值得知道的事：

- **Hermes home 的位置是 Hermes 自己決定的**，兩個平台不一樣（上表第一列），
  安裝器用的是和 Hermes 完全相同的判斷規則，包括 `HERMES_HOME` 環境變數覆寫。
  用 Hermes profile 的話，加 `--hermes-home <路徑>`。
- **原本這台參考機把引擎放在 `~/.hermes/comfy-launcher`**。在 Windows 上那只是個剛好同名的資料夾，
  但**在 Linux 上 `~/.hermes` 就是 Hermes home 本身** —— 引擎會直接躺在 Hermes 的家目錄裡，
  容易和 Hermes 自己的檔案混淆。所以這包改用 `<hermes home>/comfy-engine`，兩邊語意一致。
- **沒有任何絕對路徑寫死在程式裡了。** 全部由 `engine/paths.py` 解析，
  順序是「環境變數 → `hermes-comfy.conf` → 各平台預設值」。

---

## 1. 先決條件

### 兩個平台都需要

| 項目 | 說明 |
|---|---|
| GPU + 驅動 | **廠商中立**。規劃器只問「已被佔用多少、還剩多少」。**先原廠工具，都沒有才問服務 API**，依序:`VIDGEN_GPU_QUERY_CMD` → `nvidia-smi` → `intel-ze`(Intel Level Zero) → `xpu-smi`(Intel CLI) → `rocm-smi`(AMD) → ComfyUI 的 `/system_stats` → TRUSTA-AST 的 `/system/resources`。七個都失敗才判 FAIL |
| TRUSTA-AST 後端 | 可選。有的話規劃器就知道「模型載了沒、是哪個引擎、佔多少顯存」；連不到就**假設 LLM 在**並照常保留顯存（保守方向）。位址用 `VIDGEN_AST_URL` |
| Python 3.9+ | 跑 skill 與引擎用。ComfyUI 自己那顆 python 是另一顆，見下 |
| ComfyUI | 需要夠新的版本才有 MiniMax-H3 的核心節點（參考機 **0.31.1**） |
| `ComfyUI-GGUF` 自訂節點 | 預設的影片路徑（Wan 14B GGUF）和 qwen/flux 生圖都靠它 |
| 磁碟 | 模型最少 40 GB、全裝 155 GB。見 `MODELS.zh-TW.md` |
| `ffmpeg` + `ffprobe` | **`dramagen` 需要**（其他兩個 skill 不用）。它把每鏡的 clip 接成 `final.mp4` 並混入字幕軌。要在 PATH 上，`doctor.py` 會檢查 |
| 一個 OpenAI 相容的 LLM 端點 | **`dramagen` 需要**。`story.py` 打 `/v1/chat/completions` 把故事前提寫成 `drama.json`。預設沿用 `VIDGEN_AST_URL`（+`/v1`），要指到別處就設 `DRAMAGEN_LLM_URL` |

VRAM 建議 **12 GB 以上**。參考機是 RTX 5080 Laptop 16 GB，扣掉常駐 LLM 的 7 GB
之後剩約 7.5 GB 給影片。卡更小就要把 `VIDGEN_LLM_VRAM_BUDGET` 調小，否則每次都被拒跑
（調到 7.0 以下需要同時設 `VIDGEN_UNSAFE_OVERRIDE=1`，見下）。

### Windows

- ComfyUI 建議用 **portable 版**（`ComfyUI_windows_portable`），
  它自帶 `python_embeded\python.exe`，那顆才有 torch 和 PyAV。
- 不需要另外裝 CUDA toolkit。

### Linux

- ComfyUI 通常是 `git clone` + venv：

  ```bash
  git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI
  cd /opt/ComfyUI
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
  ./venv/bin/pip install av        # H3 的音訊混流需要 PyAV，requirements 不一定帶
  ```

  安裝器會自動找 `<COMFY_HOME>/venv/bin/python` 或 `.venv`。
- **`av`（PyAV）務必裝**：`vidgen r2v` 產生的是「影像 + 原生立體聲」，
  `h3_media.py` 要用它把兩者混流成 mp4。少了它 H3 會在最後一步失敗。

---

## 2. 安裝

### Windows

```powershell
cd <解壓縮出來的 hermes-comfy-pack>
.\install.ps1
```

被執行原則擋住的話：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

ComfyUI 不在預設位置就明講：

```powershell
.\install.ps1 --comfy-home D:/AI/ComfyUI_windows_portable
```

### Linux

```bash
cd <解壓縮出來的 hermes-comfy-pack>
chmod +x install.sh
./install.sh --comfy-home /opt/ComfyUI
```

### 跟著 git checkout 更新：`--link`

如果這台機器是用 git 追這包（開發機，或會自己 `git pull` 更新的產品 checkout），
改用連結安裝，不要複製：

```
python install.py --link
```

引擎和 skill 腳本直接從 checkout 執行，不複製，也不會往 checkout 寫任何檔案。
只有每台機器不同的東西放在 Hermes home：用本機路徑產生的 `SKILL.md`、`hermes-comfy.conf`，
以及 `<hermes home>/comfy-state/` 底下的執行狀態（工作紀錄、log、輸出）。之後更新只要 `git pull`：
安裝器會加上 `post-merge`、`post-checkout`、`post-rewrite` 三個 hook（接在既有 hook 後面，不覆蓋），
自動重新產生文件；每個 skill 啟動時也會檢查，checkout 若在沒有 hook 的情況下變了也會自己刷新
（手動是 `install.py --refresh`）。原本是複製安裝的機器會保留設定：舊的 conf 成為新 conf 的基礎，
舊的 skill 資料夾移到 `comfy-state/skill-backups/`。

### 想先看它要做什麼

```
python install.py --dry-run
```

### 安裝器實際做的事

1. 複製 `engine/` → `<hermes home>/comfy-engine/`（引擎程式 + 12 個 workflow）
2. 產生 `hermes-comfy.conf`，把偵測到的 `COMFY_HOME` 填進去
3. 複製 `skills/` → `<hermes home>/skills/`，並**把 SKILL.md 裡的絕對路徑改寫成這台機器的實際路徑**
4. 在每個 skill 目錄寫一個 `engine_dir.txt`，讓 skill 執行時找得到引擎
5. 建立 `<hermes home>/motion-venv`，給 `motion` skill 專用的 Python 環境（依 `skills/motion/requirements.txt`
   安裝 OCR 與條碼套件；不會動到 Hermes 自己的 venv），只有那個檔案變動時才重建。有 uv 就用 uv 建（依序找 `UV`、`<hermes home>/bin/uv`、PATH），兩邊的直譯器都不需要 pip；
   沒有才用 pip
6. 把這包的 ComfyUI 節點複製到 `custom_nodes/`，然後跑 `tools/doctor.py`

第 3 點是關鍵：SKILL.md 會叫模型執行**一行完整的絕對路徑指令**，
所以那些路徑必須是真的。原始檔是 `SKILL.md.in`，裡面是 `@@PY@@`、`@@VIDGEN@@` 這種佔位符。

重跑安裝器是安全的：既有檔案會先備份成 `<名稱>.bak.<時間戳>`，
`hermes-comfy.conf` 則會保留不動（要重產就先刪掉它）。

---

## 3. 要改的設定

### ★ `hermes-comfy.conf`（在 `<hermes home>/comfy-engine/`）

安裝器已經填好大部分。實務上只有兩個值需要你確認：

```ini
# 1. ComfyUI 在哪。兩種佈局都會自動辨識：
#    Windows portable：<COMFY_HOME>/ComfyUI/main.py + <COMFY_HOME>/python_embeded/
#    Linux git clone ：<COMFY_HOME>/main.py         + <COMFY_HOME>/venv/
COMFY_HOME=/opt/ComfyUI

# 2. ★ 換機器一定要重設這個：常駐 LLM 佔用的 VRAM。
#    這是「無條件保留」的額度，就算 LLM 當下沒載入也一樣 —— 否則趁它不在時
#    把記憶體用掉，等它回來就會被擠爆。
#    這個值只能往上調：低於預設 7.0 的設定會被忽略（不讓被拒跑的工作
#    自己把守衛調小），被忽略時會在 stderr 提示。所以「沒有常駐 LLM、
#    想用整張卡」要兩個一起設：
#        VIDGEN_UNSAFE_OVERRIDE=1
#        VIDGEN_LLM_VRAM_BUDGET=0
VIDGEN_LLM_VRAM_BUDGET=7.0
```

其餘（`COMFY_PYTHON`、`HERMES_COMFY_ENGINE`、各種記憶體邊界）都有可用預設值，
`doctor.py` 說對就別動。完整說明在 `hermes-comfy.conf.example` 的註解裡。

### Hermes 的 `config.yaml`

```yaml
terminal:
  timeout: 600
```

影片算圖動輒好幾分鐘。預設值太短會在算圖途中把前景指令砍掉 ——
**算圖本身不會失敗**（worker 是 detached 的，會自己跑完並寫出 mp4），
但你會拿不到前景結果，只能再用 `gen.py status <job_id>` 接回來。
`doctor.py` 會檢查這一項。

### 常駐 LLM 的啟動方式（如果有）

**讓 AST 底下的推論引擎用 mmap 載入權重**（多數 GGUF loader 的預設行為）。
這樣權重是乾淨的 file-backed 頁，OS 可以隨時免費丟棄再從 SSD 讀回，
規劃器會把這部分算成「可用記憶體」，於是能放行更多工作。
若權重是髒的私有頁（關掉 mmap，或引擎直接讀檔而不是映射），丟棄要寫 swap/pagefile，
規劃器會自動算成 0（這個判斷是**量出來的**，不用你設定），結果就是能生的影片變短。

> **⚠️ 推論引擎可能會叫你做相反的事，別照它的。** 用 MoE 的 CPU offload
> （`-ncmoe` / `--n-cpu-moe`）載入時，有些引擎會印出這種警告：
>
> ```
> W tensor overrides to CPU are used with mmap enabled
>   - consider using --no-mmap for better performance
> ```
>
> 它是對的 —— **對 LLM 自己的載入與推論速度而言**。但這台機器上 LLM 不是唯一的用戶：
> 關掉 mmap，那些權重就變成不可免費回收的髒頁，規劃器的可回收額度掉到 0，
> 影片的長度上限跟著縮。**兩邊的最佳解是相反的，你得選一邊。**
>
> 實測（5060 Ti 16GB / 128GB RAM，30B-A3B 以 `-ncmoe 39` 載入）：維持 mmap 時
> 可回收頁 **9.8 GB**、RAM 預算 **94.9 GB**；關掉 mmap 時是 **0 GB**、**80.6 GB**。
>
> 這包預設站在影片這一側。如果你這台的主要用途是 LLM 推論、影片只是偶爾跑，
> 那就反過來關掉 mmap，並知道影片長度會變短。

> 舊版有個 `VIDGEN_LLM_MLOCK` 旗標要你「宣告」權重有沒有被鎖在 RAM，**已經移除**。
> 那是宣告而不是量測：宣告錯了會放寬正好用來保護 agent 的那道邊際，而且對多數
> 引擎預設就是錯的。現在 Linux 直接從 `VmLck` 量，鎖住的頁自然算成 0。

---

## 4. 驗證

```
python tools/doctor.py
```

它會逐項列出：解析後的每個路徑、ComfyUI 與其 python、`ComfyUI-GGUF` 節點、
引擎的 10 支檔案、skill 是否就位、GPU/RAM 實測值、`VIDGEN_LLM_VRAM_BUDGET` 是否設得離譜、
ComfyUI 是否已在跑、`terminal.timeout`、以及**24 個模型逐一在不在、缺的該放哪個目錄**。

```
python tools/doctor.py --models          # 只看模型
python tools/doctor.py --engine <路徑>   # 檢查指定的引擎目錄
```

FAIL 一定要處理；WARN 多半是「你沒裝那個功能的模型」，可以無視。

接著實際跑一次（把 `<skills>` 換成你的 skills 目錄）：

```
python <skills>/imggen/gen.py zimage "a red maple leaf on wet slate"
python <skills>/vidgen/gen.py plan
```

第一個回 `{"status":"ok","outputs":["…png"]}`；第二個會即時回報現在能生多長的影片，不算圖。

---

## 5. 開始使用

裝完**重啟 Hermes**，它才會掃到新的 skill。之後直接用自然語言：

- 「生一張戴太空頭盔的柴犬，扁平向量插畫」→ 走 `imggen`
- 「讓這張圖動起來」→ 走 `vidgen i2v`
- 「用這個人當主角做一段有聲音的影片」→ 走 `vidgen r2v`（MiniMax-H3）

兩個 SKILL.md 裡有完整的模型選型指引與踩過的坑，Hermes 會自己讀。
其中有幾條是實測換來的，別繞過：

- **沒有起始圖時不要用 `t2v`** —— 這套設定裡的 14B 是 I2V 模型，純文字會產出**純雜訊**，
  而且照樣回傳 `ok:true` 加一個 mp4 路徑。一律走「先 `imggen` 生靜圖 → 再 `i2v`」兩段式。
  `gen.py` 已經會直接擋下 `t2v`。
- **算圖期間不要再下 prompt** —— 見下一節。

---

## 6. 記憶體讓渡：這套設計的核心，跨平台差異也在這裡

這套東西最特別的地方，是它假設**同一台機器上還住著一顆你正在用的 LLM**，而且
**驅動這整條流程的 agent 就是那顆 LLM**。

流程是寫死的：

```
啟動受保護的 ComfyUI → 送出 prompt → 確認它真的出現在 queue_running
                                          → 這時才把 LLM 佔的 DRAM 交出去
```

順序不能反。如果在 ComfyUI 還沒真的開始算圖時就先釋放，一旦啟動失敗或被拒跑，
記憶體白丟了、影片一格沒生，而 agent 已經要從 SSD 重讀自己的權重 —— 表現出來就像整個 Hermes 卡死。
`memplan.release_llm_dram()` 是唯一會做釋放的地方，而且它**自己打 `/queue` 去確認**，不相信呼叫端。

**釋放做的是什麼**：對 LLM 行程的**乾淨、file-backed 的 mmap 權重頁**做回收。
行程照跑、VRAM 不動、模型沒有被卸載，只是下次推論要從 SSD 重讀。

要回收哪個行程**完全由 AST 決定，而且是由作業系統確認的**。做法刻意不含任何名字，
三條路徑依序試：

1. **從行程樹找**（主要路徑）。`VIDGEN_AST_URL` 就透露了一件事：後端在哪個 port。
   問作業系統誰在那個 port 上 listen，那顆就是 AST 本身；再往下**遞迴**走它的子孫，
   那些就是它啟動的東西。遞迴不是可有可無：後端若是透過 worker 去 spawn 引擎，
   引擎會落在**第二層**，只看直接子行程會停在 worker、剛好漏掉真正握著權重的那顆。
   親子關係還會用**建立時間**確認一次——Windows 不會清掉孤兒行程裡那個已死的父 pid，
   而那個 pid 可能已經被回收給別人用了，比年紀就能把這種假子孫濾掉。
2. **用 port 佐證，不用 port 尋找**。AST 回報的 payload 裡凡是像 port 的數字都拿去
   對照「誰在 listen」，但**只有當那顆行程同時也是 AST 的子孫時才採信**，而且只用來
   排序與標記，不用來納入或排除。單看 port 太鬆——`n_ctx: 64000`、`n_batch: 512`
   也都是整數，照著去修剪會打到無關的服務。
3. **payload 裡的絕對路徑**（後援）。取檔名當候選、再問行程表誰在跑
   （模型權重等資料檔會被排除；含斜線的敘述文字因為不是絕對路徑也會被排除）。
   這條留著是因為後端不在本機、或不在同一個 OS 時（例如 WSL 分裝）根本沒有行程樹
   可走，但它回報的 binary 名字仍然可以在這一側比對。

找到幾顆就修剪幾顆。後端自己 spawn 引擎時，worker 與引擎各握著一部分 DRAM，
只修剪最大的那顆會把其餘的留在原地。

所以這包**不知道也不需要知道**：AST 用哪個欄位回報、那個欄位叫什麼、底下跑的是
哪個推論引擎。AST 換引擎、換欄位名，這邊都不用動。

理由是這包配的是 AST，而 AST 底下的引擎是可換的，所以這包試過的每一個寫死的
名字，都在寫它的那台機器上是對的、換一台就是錯的——而且都用同一種方式靜默失效：
`find_pid()` 找不到、讓渡什麼都沒做。哪個引擎在跑不是這包有資格回答的問題。

模型跑在 AST 自己的 worker 裡（沒有獨立的引擎行程）**也涵蓋在內**：那時走訪停在
worker，而 worker 就是正確的對象。這正是舊做法拿不到任何候選、靜靜什麼都不做的情況。

真正的「沒有回收對象」只剩一種：那個 port 上沒有人在 listen，也就是後端根本不是
這台機器上的行程。這時 DRAM 回收關閉、規劃器把可回收量算成 0（保守方向：影片短
一點，不是把 agent 擠掉），`memplan` 會在 **stderr** 說一次。

`VIDGEN_LLM_PROCESS` 仍然是最優先的覆寫，給 AST 管不到的部署用。它**不經過**上面
那道「確認在跑」的檢查——你指名了一個沒在跑的行程，應該被告知，而不是被靜靜忽略。

> 順帶一提：AST 的 `/inference/cleanup_generation_memory` **不是**這件事的替代品，
> 它清的是 KV cache 與中間激活；`/inference/force_cleanup_gpu` 更是會重啟 worker，
> 那正是這包存在的目的所要避免的驅逐。所以回收仍走 OS 層，只是對象由 API 決定。

| | Windows | Linux |
|---|---|---|
| 量「可用 RAM」 | `GlobalMemoryStatusEx` | `/proc/meminfo` 的 **MemAvailable** |
| 量 LLM 的可回收量 | `WorkingSetSize − PrivateUsage`（PSAPI） | `VmRSS − (RssAnon + VmLck)`（`/proc/PID/status`） |
| 實際回收 | `EmptyWorkingSet` | `process_madvise(MADV_PAGEOUT)` |
| 回收失敗的後果 | **有影響**：Windows 不會主動幫你回收別的行程的乾淨頁，記憶體就這樣被佔著 | **幾乎沒有**：Linux 本來就會在記憶體壓力下丟掉乾淨的 file-backed 頁 |

**Linux 特有的一點**：`process_madvise(MADV_PAGEOUT)` 對「別的行程」動手需要
`CAP_SYS_NICE`，**而且身為它的父行程也不算數**，跟 Yama 的 `ptrace_scope` 無關。
（實測：`ptrace_scope=1` 下對自己親手 spawn 的子行程仍然是 `EPERM`；補上該 capability
之後同一顆就成功，395 MB → 10 MB。見 `tests/test_tree_live.py`。）
所以沒有權限被擋下是**常態，不是故障** —— 它只是少了一個「提前把工作做掉」的
最佳化，預算計算依然正確，影片照生。

### Linux 上到底有哪些「釋放記憶體」的手段（實測結果）

以下數字量測自 WSL2／kernel 6.18，對象是一個持有 512 MB mmap 檔案頁的行程
（等同一顆 mmap 載入 GGUF 的推論引擎）：

| 手段 | 結果 | 需要 | 備註 |
|---|---|---|---|
| `process_madvise(MADV_PAGEOUT)` | ✅ **522 → 8 MB** | `CAP_SYS_NICE` 或 ptrace 權限 | 引擎內建用的就是這個，**精準只針對目標行程** |
| 同上，但無權限 | ❌ EPERM，毫無變化 | — | `ptrace_scope=1` 的預設值就會擋掉 |
| cgroup v2 `echo N > memory.reclaim` | ✅ **522 → 6 MB** | root + **行程要在該 cgroup 裡出生** | 是 cgroup **整體**操作，不是針對單一行程 |
| `/proc/sys/vm/drop_caches` | 全系統 page cache | root | 太鈍，會影響所有程式 |
| `/proc/<pid>/clear_refs` | **不會釋放任何記憶體** | — | 只重設 accessed 位元，常見誤解 |
| `/proc/<pid>/reclaim` | 不存在 | — | 只有 Android／廠商核心才有 |

**兩個實測踩到的陷阱，值得先知道：**

1. **把行程「移進」cgroup 沒有用。** cgroup v2 的頁面是計費給「第一個把它讀進來的 cgroup」，
   所以事後搬進去的話 `memory.current` 會是 0，每次 reclaim 都回 EIO，看起來像功能壞掉。
   行程必須**一開始就在那個 cgroup 裡啟動**——由 systemd 管理的服務本來就是
   （`system.slice/<unit>`），所以正常部署下沒問題。
2. **`memory.reclaim` 要求的量若超過實際計費量，就算真的回收了也會回傳 EIO。**
   要用實際回收量（RSS 掉了多少）判斷成敗，不要用離開碼。

### 直接可用的 helper

包裡附了 **`tools/linux-reclaim-helper.sh`**，兩種手段依序嘗試，接上既有的逃生口即可：

```ini
# hermes-comfy.conf
VIDGEN_LLM_RECLAIM_CMD=/path/to/tools/linux-reclaim-helper.sh
```

引擎會把 pid 接在後面呼叫。它會**先確認那個 cgroup 裡只有目標行程**才走 cgroup 路徑——
否則 reclaim 會波及同一個 cgroup 裡的所有程式，那正是這整套設計要避免的事
（實測：未改設定的 WSL distro 把所有東西都放在 `/init.scope`，沒有這道護欄就會把整個 session 掃掉）。

要讓它有權限，最單純的是加一條 sudoers：

```
your-user ALL=(root) NOPASSWD: /path/to/tools/linux-reclaim-helper.sh
```

然後把 `VIDGEN_LLM_RECLAIM_CMD` 設成 `sudo /path/to/tools/linux-reclaim-helper.sh`。

> **值得再說一次**：這在 Linux 上是**最佳化，不是必需品**。Linux 本來就會在配置壓力下
> 丟掉乾淨的 file-backed 頁；沒有這支 helper，規劃器的預算計算依然正確，只是少了提前釋放。
> Windows 沒有這個特性，那邊才非做不可。

---

### 會把整張顯示卡吃滿的引擎：借出去，而不是保護

上面都假設 LLM 只佔顯示卡的固定一部分。有些引擎刻意把整張卡用滿 —— Strata（在 TRUSTA-AST 後面）會把
expert cache 一路擴大到只剩約 0.3 GB，16 GB 的卡用掉 14.3 GB。替它保留這些空間就沒有東西可以拿來算圖，
所以這類引擎 —— 依實測判斷、不看引擎名稱：AST 後面任何佔用 VRAM 超過 `VIDGEN_LLM_VRAM_BUDGET`（預設 7 GB）的 LLM ——
每次算圖改成**借用**顯示卡
（`engine/llmsleep.py`）：

1. 等 LLM 閒置 —— agent 最後一句話（「已開始算圖，請稍候」）可能還在輸出；
2. `POST /inference/sleep`：引擎把 cache 以外的 VRAM 複製到 page-locked 的主記憶體，對話保留；
3. 如果 sleep 被拒 —— Windows 上備份必須放得進可用的 **commit** 再加 2 GiB，而 Strata 的 resident 模式
   載入時就 page-lock 了 13.5 GiB 的 experts，32 GB 的機器上永遠放不下 —— 就改用 `POST /inference/unload_model`
   （AST 會先把對話存成 slot 檔，下次載入時還原）；
4. 算圖；結束後喚醒它，或用同一份設定 `POST /inference/load_model`（約 35 秒）。

planner 會把這顆 LLM 的 VRAM（從每個程序的 GPU 計數器讀取；AST 對 Strata 回報 0）—— 以及要卸載時它的 RAM ——
算成可用，`start_comfy.py` 也不會替它保留。借出中的 LLM 記錄在 `jobs/LLM_LENT.json`。
`recast` 整輪只借一次，結束時才還。

| 設定 | 效果 |
|---|---|
| `VIDGEN_LLM_SLEEP=off` | 永不借出；跟其他引擎一樣保留 VRAM |
| `VIDGEN_LLM_RETURN=defer` | 這個工作結束後先不要還（`recast` 為它的各個 pass 設定的） |
| `gen.py llm-return`（vidgen） | 立刻把借出的 LLM 還回去 —— 用在中途被中斷的時候 |

借出期間，**agent 什麼都沒辦法回答**：在算圖結束前它是休眠或卸載狀態。

---

## 6.5 WSL 拆分安裝（Hermes 在 Ubuntu、ComfyUI 留在 Windows）

這是一種特別的佈局：**Hermes 與這套引擎跑在 WSL 的 Ubuntu 裡，而 ComfyUI、模型檔、
常駐 LLM 全部留在 Windows。** 兩邊靠兩條通道溝通，不複製任何東西：

```
WSL Ubuntu                                   Windows
  Hermes + imggen/vidgen + engine    ──HTTP──▶  ComfyUI :8188
                                     ──/mnt/c─▶  models/、input/、output/
                                        ？        TRUSTA-AST + LLM（見下面的限制）
```

### 安裝

```bash
# 在 Ubuntu 裡（先把這包複製或掛載進來）
cd /mnt/c/Users/<你>/comfyui-skill
./tools/wsl_bootstrap.sh --dry-run     # 先看它偵測到什麼
./tools/wsl_bootstrap.sh
```

它會偵測 Windows 家目錄（問 Windows 要 `%USERPROFILE%`，不用猜）、找出 ComfyUI、
探測連得到的位址、裝好 apt 套件與 Hermes、把 PyAV 裝進 Hermes 的 venv、跑 `install.py`，
最後把拆分架構專用的設定寫進 `hermes-comfy.conf`。

### 兩個一定要處理的網路問題

**1. 要開 WSL 鏡像網路（mirrored）。** WSL2 預設是 NAT 模式，Ubuntu 有自己的網路命名空間，
所以 **Windows 的 `127.0.0.1` 不是 Ubuntu 的 `127.0.0.1`**。ComfyUI 只綁 `127.0.0.1`，
NAT 模式下 WSL 連不到它。開鏡像網路後兩邊共用 loopback，不用開放任何埠：

```ini
# %USERPROFILE%\.wslconfig
[wsl2]
networkingMode=mirrored
```

改完在 Windows 執行 `wsl --shutdown` 讓設定生效，再在 Windows 啟動 ComfyUI：

```
python <windows-engine>\start_comfy.py
```

**為什麼不能把 ComfyUI 開到網路上**：ComfyUI 完全沒有驗證。綁到 `0.0.0.0` 之後，
同網段任何人都能用你的 GPU 跑工作、讀走所有產出的圖片與影片，裝了 ComfyUI-Manager
的話還能安裝自訂節點，等於在你的電腦上執行任意程式。所以 `start_comfy.py` 不提供
`--listen`，`COMFY_EXTRA_ARGS` 裡帶 `--listen` 也會被拒絕。別台電腦的 Hermes 只透過
AI Scaler Toolkit 的 `/v1`（要帶 API key）使用對話模型；圖片與影片在 Hermes 所在的那台電腦上產生。

**2. 不要把主機位址寫死。** 鏡像網路下直接用 `127.0.0.1`。引擎的
`paths.comfy_url_candidates()` 先試設定值；在 NAT 模式下才會再補當下的預設閘道
（Windows 主機，每次 WSL 重啟都可能不同），`_resolve_comfy_url()` 挑第一個回應的並記住。
ComfyUI 只綁 `127.0.0.1` 時，閘道那條連不到，所以實際上要靠鏡像網路。

### ⚠️ 記憶體保護在跨 OS 邊界後會退化（必讀）

這套東西的核心價值是保護同機常駐的 LLM。**LLM 在 Windows、引擎在 Ubuntu 時，
保護只剩一半**，而且是可預期的一半：

| 機制 | 拆分後還有效嗎 | 為什麼 |
|---|---|---|
| **VRAM 保留額度**（`VIDGEN_LLM_VRAM_BUDGET`） | ✅ **完全有效** | 這是無條件保留，不需要看到 LLM 行程；而且 WSL 的 `nvidia-smi` 讀的是**同一張實體卡**，Windows 端的佔用一樣算得到 |
| ComfyUI 的 `--reserve-vram` | ✅ 有效 | 由 Windows 端的 `start_comfy.py` 設定，與 WSL 無關 |
| **LLM 的 DRAM 可回收量計算** | ❌ **失效** | Ubuntu 的 `/proc` 看不到 Windows 行程,找不到 PID → 可回收量算 0 |
| **LLM 的 DRAM 主動回收** | ❌ 失效 | 無法對 Windows 行程呼叫 `EmptyWorkingSet` |

**失效的那兩項是「偏保守」而不是「危險」**：可回收量算 0 只會讓 RAM 預算變小、更容易拒跑，
不會讓它去搶 LLM 的記憶體。真正保命的 VRAM 那條完好無損。`doctor.py` 會把這件事標成 WARN 並說明。

**如果你要把 DRAM 那條也補回來**，`VIDGEN_LLM_RECLAIM_CMD` 就是為此保留的逃生口：
指向一個透過 WSL interop 呼叫 Windows 端 PowerShell 的小工具即可（pid 會接在指令後面）。

**另外一個要知道的數字**：Ubuntu 的 `/proc/meminfo` 報的是 **WSL 虛擬機分到的記憶體**
（預設是主機的一半），不是 Windows 的實體 RAM。預設的 GGUF 路徑增量 RAM 只有約 1 GB，
所以實務上不影響判斷,但**別把 doctor 報的 RAM 數字當成 Windows 的可用記憶體**。
要讓兩邊一致就在 `.wslconfig` 裡用 `memory=` 設定上限。

### 其餘的都是設定值，不是程式碼

好消息是拆分架構下需要改的其實只有設定：

```ini
COMFY_REMOTE=1                        # 別想在這裡啟動 ComfyUI(自動偵測也會設)
COMFY_URL=http://127.0.0.1:8188      # 鏡像網路下 Windows 的 loopback 就是這裡
COMFY_ROOT=/mnt/c/Users/<你>/ComfyUI_windows_portable/ComfyUI
COMFY_PYTHON=/home/<你>/.hermes/hermes-agent/venv/bin/python
VIDGEN_AST_URL=http://127.0.0.1:8571  # 後端也在 Windows 那邊，同樣走 loopback
```

`COMFY_PYTHON` 指向 **Ubuntu 自己的** python 是刻意的：它只用來跑 `h3_media.py`，
而那支只需要 **PyAV + Pillow，不需要 torch**。指向 Windows 的 `python_embeded\python.exe`
反而會壞（Linux 路徑餵給 Windows 執行檔）。`paths.py` 現在會拒絕在非 Windows 上挑 `.exe`。

輸入圖片是用**複製到 `COMFY_INPUT`** 的方式暫存,所以 `/mnt/c/.../ComfyUI/input` 就夠了,
不需要任何檔案共享設定。

### Hermes 的 LLM endpoint

開了鏡像網路之後，WSL 連 `127.0.0.1` 就是連 Windows，所以 Hermes 直接指向本機的
AI Scaler Toolkit 後端：

```
hermes setup      # base_url 填 http://127.0.0.1:8571/v1
```

對後端來說這是**本機**連線，**不需要 API key，也不需要打開 AI Scaler Toolkit 的
「Let another computer use this machine's model」**。後端自己就提供 OpenAI 相容的 `/v1`，
所以指它即可 —— **不要繞過它去打推理引擎的私有 port**：那個 port 只聽 `127.0.0.1`，
而且有只有後端知道的內部 key，直接連會被拒絕；走後端則是同一個 API，換模型、換引擎時你這邊不用動。

設定檔的做法(`~/.hermes/config.yaml`)是在 `custom_providers` 加一筆再讓 `model.provider` 指過去。
**最省事且最不會出錯的方式是把 Windows 那份 `config.yaml` 複製過來** —— 鏡像網路下 `base_url`
不用改。`model.provider` 是用 `custom_providers[].name` 推導出來的 slug
(`custom:local-(127.0.0.1:8571)`),自己另外新增一筆很容易 slug 對不上而變成 unknown provider。

**不能開鏡像網路的話**：AST 這邊可以在 Settings → Network 打開「Let another computer use
this machine's model」，用 Windows 的位址加上那裡顯示的 API key 連 `/v1`（provider 的 `api_key`
填那把 key）。但 ComfyUI 只聽 `127.0.0.1`，NAT 模式下 WSL 一樣連不到，圖片與影片功能還是要鏡像網路。

**驗證(在 Ubuntu 裡)**:

```bash
curl -s http://127.0.0.1:8571/health          # -> {"status":"healthy",...}
curl -s http://127.0.0.1:8571/v1/models       # -> 目前載入的模型
curl -s http://127.0.0.1:8188/system_stats    # -> ComfyUI 有在跑時才會回應
```

> **混合推理模型的注意事項**:Qwen3.5 這類模型會把思考過程放在 `reasoning_content`。
> 用 `max_tokens` 太小(例如 16)測試時,額度會被思考 token 吃光,`content` 回傳空字串
> 看起來像壞掉。測試請給 256 以上。

---

## 7. 疑難排解

| 症狀 | 原因與處理 |
|---|---|
| `doctor.py` 說 `main.py` 找不到 | `COMFY_HOME` 指錯。要指到**包含** ComfyUI 的那層：portable 是含 `ComfyUI\` 和 `python_embeded\` 的資料夾；Linux 是 clone 出來的 ComfyUI 本身 |
| `ComfyUI's python not found` | 設 `COMFY_PYTHON`。它**不是**跑 skill 的那顆，是有 torch + PyAV 的那顆 |
| `custom node ComfyUI-GGUF` FAIL | 見 `MODELS.zh-TW.md` 最後一節，git clone 進 `custom_nodes/` |
| `no GPU measurement source answered` | 七個來源全滅。裝上廠商工具（`nvidia-smi` 通常在 Linux `/usr/bin`、Windows `C:\Windows\System32`），或先啟動 ComfyUI／AST 後端讓它們的 API 能答，或設 `VIDGEN_GPU_QUERY_CMD` |
| H3 影片最後一步失敗 | ComfyUI 那顆 python 缺 PyAV：`<COMFY_PYTHON> -m pip install av` |
| Windows 上 `Python was not found` | 打到 Microsoft Store 的假殼了。SKILL.md 裡的指令**連直譯器都寫絕對路徑**就是為了這個，照抄別改 |
| Windows 上路徑變成 `C:Users<你>…` | Hermes 的 terminal 是 bash，反斜線被吃掉了。用正斜線（Windows 的 Python 吃得下） |
| 每次都被拒跑、說 VRAM 不夠 | `VIDGEN_LLM_VRAM_BUDGET` 對你的卡太大。設成 LLM 實際佔用量即可；要調到 7.0 以下（含沒有 LLM 的 0）還要加 `VIDGEN_UNSAFE_OVERRIDE=1`，否則會被忽略並在 stderr 提示 |
| 指令逾時了 | **不要重送。** 算圖跑在獨立的 detached worker 上，會自己跑完。用 `gen.py status <job_id>` 接回來（`--wait` 一開始就把 job_id 印在 stderr 了） |
| 算影片時任何訊息都失敗（本機佔滿 GPU 的 LLM） | 正常：LLM 把顯示卡借給算圖，結束後會自己回來（約 35 秒）。如果算圖中途被中斷、模型一直沒載回，執行 vidgen 的 `gen.py llm-return` |
| Hermes 看不到新 skill | 重啟 Hermes |
| Hermes 的 curator 動了我的 skill | 改用 `install.py --external`，把這包的 `skills/` 目錄註冊到 `config.yaml` 的 `skills.external_dirs`。外部目錄的 skill 受保護，curator 不會改寫 |

---

## 8. 從這台參考機搬過去

引擎與 skill 這包已經涵蓋。額外要帶的只有兩樣：

1. **模型**（154.5 GB，或你只要的那部分）—— 見 `MODELS.zh-TW.md`
2. **`ComfyUI-GGUF`** 自訂節點

**不需要帶**的：`jobs/`、`out/`、`logs/`（執行期產物，會自動重建）。

搬完之後 `VIDGEN_LLM_VRAM_BUDGET` 要按新機器的 GPU 重設，
`vidgen` SKILL.md 裡那張「各解析度最長單段」的表是在 16 GB 卡上量的，**只能當量級參考** ——
以 `gen.py plan` 的即時回報為準，引擎本來就會依當下餘量自己擋。
