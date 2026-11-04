# comfyui-skill (hermes-comfy-pack)

[English](README.md) | 繁體中文

讓 [Hermes](https://github.com/NousResearch/hermes-agent) agent **在自己的機器上**生圖、生影片、做介紹動畫的六個技能——
圖片和影片在地端 ComfyUI 上算，動畫用程式產生——打包成可以搬到別台機器的形式。支援 **Windows 與 Linux**。
不含模型權重（另外下載，見 [`MODELS.zh-TW.md`](MODELS.zh-TW.md)）。

| 技能 | 做什麼 |
|---|---|
| **`imggen`** | 生圖與修圖。`qwen21`（Qwen-Image-2.1：prompt 遵循最好、能畫中文字、可吃參考圖）、`qwen21_edit`（修改圖片：換衣服或背景、把多個角色合成在一起）、`qwen21_move`（移動人物在畫面中的位置）、`qwen`、`flux`（Apache 授權，可商用）、`zimage`（快）、`realvis`（寫實）、`img2img` |
| **`vidgen`** | 生影片。Wan 2.2 I2V（快、無聲）與 MiniMax-H3（**原生立體聲音訊**；可用參考圖鎖住角色長相、釘住第一幀，或照著參考影片重演並把裡面的人換掉） |
| **`recast`** | 一支參考影片 + 人物照片 → 換成這些人的同一支影片。動作、鏡頭、構圖、背景與原始音訊都跟著影片走。逐鏡頭算圖，**每個結果都會自動檢查，檢查不過就重抽 seed** |
| **`motion`** | 截圖與照片 → 說明／宣傳動畫，用程式做出來：表格列逐一出現、數值加框、平移與縮放、從畫面裁出的面板、音效。**畫面上的每個字都維持原本的像素**（沒有重新打字，也沒有模型畫的東西），所以影片中的數字就是真實的數字。不需要 GPU |
| **`dramagen`** | 一句話前提 → 一支短劇。LLM 先寫分鏡表，再驅動 `imggen` 與 `vidgen` 算出靜態圖與片段，最後接成帶字幕的成片 |

`recast` 與 `dramagen` 都不自己碰 ComfyUI —— 它們只是去跑 `vidgen/gen.py` 與 `imggen/gen.py`，
所以下面講的記憶體保護在它們身上原封不動地成立。

**[看實際運作 →](docs/showcase/README.zh-TW.md)** 每個技能都由跑在**地端** LLM 上的 Hermes 驅動，全在同一台筆電上：
照原樣輸入的請求、後續修改，以及結果。

本 README 中各項說法背後的詳細量測數據在 **[`TEST_RESULTS.zh-TW.md`](docs/TEST_RESULTS.zh-TW.md)**。

## 這包和「隨便寫個腳本打 ComfyUI API」差在哪

差在它假設**同一台機器上還住著一顆你正在用的 LLM**，而且驅動這整條流程的 agent *就是* 那顆 LLM。

- **記憶體規劃器**（`memplan.py`）：算圖前先量真實 VRAM/RAM，決定權重放 VRAM / DRAM / SSD 哪一層。
  裝不下就**拒跑並說明原因**，而不是把同機的 LLM 擠爆。
- **受保護的啟動器**（`start_comfy.py`）：依實測的外部 VRAM 佔用計算 `--reserve-vram`，
  讓 ComfyUI 吃不到 LLM 的 VRAM。ComfyUI 只綁定 `127.0.0.1`，付費的雲端 API 節點預設停用
  （`--disable-api-nodes`；曾有 agent 繞過 `gen.py` 去呼叫它們）。
- **有順序的記憶體讓渡**：LLM 的 DRAM 只在 ComfyUI 確認已經開始算圖之後才交出去。
  提早釋放的代價是 agent 得從 SSD 重讀自己的權重，看起來就像整個 Hermes 卡死。
- **為佔滿整張卡的引擎出借 GPU**（`llmsleep.py`）：有些地端引擎刻意把自己配置成吃下整張 GPU
  （Strata，透過 TRUSTA-AST，實測在 16 GB 的卡上佔 14.3 GB）。替這類引擎保留 VRAM 會讓算圖沒有空間可用，
  所以改成每次算圖都先等 agent 回完話、讓 LLM 睡眠——RAM 不夠放睡眠備份時就改為卸載；對話由後端保留——
  接著算圖，再把它載回來（參考機器上約 35 s）。
- **雙顯示卡**（`gpushare`、`llmsleep.py`）：實測 LLM 在 ComfyUI 以外的另一張卡上時，不讓它睡眠，也不替它保留 VRAM。
  要連 LLM 那張卡一起用時，`gpushare` 開一段睡眠時段，和另一張卡同時算圖，最後把鏡頭接成一支影片。
- **把實測踩過的坑寫進 SKILL.md**：`t2v` 會產出純雜訊卻回報成功、參考圖被縮到臉只剩 78 px、
  逾時後重送導致同一段算圖付了兩次成本、H3 憑空生出原片沒有的剪接點⋯⋯每一條都寫成模型讀得到的規則，
  並附上促成這條規則的量測。

## 搭配 AI Scaler Toolkit

AI Scaler Toolkit 本身就帶著這包（在它公開的目錄樹的 `hermes-skills/` 底下），不需要 clone。
在它的設定頁 **Components**：

1. **Hermes Agent** —— 先裝它，skills 是裝進它裡面的。
2. **Hermes image & video skills** —— 按 Install。會用 Hermes 的 Python 跑這包的 `install.py`、用 uv 建好
   `motion` 的環境，除了 motion 的套件之外什麼都不用下載。`motion` 裝完就能用。
3. **ComfyUI** —— 選它要跑在哪張卡上。這會裝好 ComfyUI 和 `ComfyUI-GGUF` 節點，再重裝一次 skills，讓它們找得到 ComfyUI。

AI Scaler Toolkit 更新時會帶來新版的 skills，卡片上會提示，按 Reinstall 就會套用。
不管哪種方式，模型權重都不包含在內：見 [`MODELS.zh-TW.md`](MODELS.zh-TW.md)。

## 快速開始

```
git clone https://github.com/TRUSTA-SW-FAE/comfyui-skill.git
cd comfyui-skill
```

接著用下面其中一種方式安裝：

| | 指令 | 之後怎麼更新 |
|---|---|---|
| **連結安裝**（git checkout 建議用這個） | `python install.py --link` | `git pull` 就好，不必做別的。git hook 會重新產生 skill 文件，引擎和腳本直接從 checkout 執行 |
| **複製安裝** | Windows `.\install.ps1`，Linux `./install.sh --comfy-home /opt/ComfyUI` | 拉下或下載新版後，再跑一次安裝程式 |

兩種方式都會順便建好 `motion` 專用的 Python 環境（`<hermes home>/motion-venv`），並複製這包的 ComfyUI 節點。然後：

```
python tools/doctor.py --models   # 檢查每個路徑、節點，以及 27 個模型是否都在
```

再重啟 ComfyUI（載入節點）和 Hermes（載入技能）。

完整說明看 **[`INSTALL.zh-TW.md`](INSTALL.zh-TW.md)**，模型清單看 **[`MODELS.zh-TW.md`](MODELS.zh-TW.md)**。

## 內容物

```
comfyui-skill/
├── INSTALL.md (+ .zh-TW.md)    安裝與設定（Windows / Linux）★ 先看這個
├── MODELS.md (+ .zh-TW.md)     27 個模型（+ 選用的 Turbo LoRA）：檔名、大小、資料夾、授權、哪個功能要用
├── docs/TEST_RESULTS.md (+ .zh-TW.md)  量了什麼、在什麼環境上量、數字多少
├── THIRD_PARTY_NOTICES.md      內附第三方內容的授權；哪些模型不可商用
├── install.py                  跨平台安裝器（.sh / .ps1 只是薄包裝）
├── install.sh / install.ps1
├── hermes-comfy.conf.example   ★ 唯一需要你編輯的設定檔的範本
├── engine/                     → 裝到 <hermes home>/comfy-engine/
│   ├── paths.py                所有路徑的唯一真相來源
│   ├── astapi.py               唯一讀取 TRUSTA-AST 後端狀態的地方
│   ├── llmsleep.py             把佔滿 GPU 的 LLM 的 VRAM 借給算圖，算完再還回去
│   ├── linkcheck.py            讓連結安裝的 skill 文件跟著 checkout 保持最新
│   ├── osmem.py                作業系統抽象層（行程、記憶體、各行程 VRAM、回收、detach）
│   ├── sysmon.py               真實 VRAM / RAM
│   ├── memplan.py              記憶體感知規劃器 + 讓渡的順序保證
│   ├── video_gen.py            影片執行器（非同步 job、detached worker）
│   ├── start_comfy.py          受 VRAM 保護的 ComfyUI 啟動器
│   ├── comfyguard.py           檢查執行中的 ComfyUI 有保護 LLM、且跑在選定的 GPU 上
│   ├── h3_media.py             H3 的影音混流與鏡頭偵測（要在 ComfyUI 那顆 python 下跑）
│   ├── h3_resume.py / h3_wait_join.py   續跑與接合 H3 分段
│   ├── prewarm_llm.py          算圖後把 LLM 的 GGUF 讀回 page cache
│   ├── verify_safetensors.py   檢查 safetensors 下載是否完整
│   └── workflows/*.json        12 個驗證過的 workflow
├── comfyui-plugins/            → 複製到 <ComfyUI>/custom_nodes/（H3 參考圖 + 釘住第一幀）
├── skills/                     → 裝到 <hermes home>/skills/
│   ├── imggen/{SKILL.md.in, gen.py}
│   ├── vidgen/{SKILL.md.in, gen.py, references/traps-and-misuse.md.in}
│   ├── recast/{SKILL.md.in, recast.py}
│   ├── motion/{SKILL.md.in, motion.py, lib/, tests/, requirements.txt}   （自己的 venv：<hermes home>/motion-venv）
│   ├── gpushare/{SKILL.md.in, gpushare.py, dispatch.py}
│   └── dramagen/{SKILL.md.in, story.py, produce.py, references/screenwriting.md}
├── docs/showcase/              完全在地端做出的成果，附上產生它們的 prompt
├── bench/linux/                可以在全新 Linux 機器上跑的測試套件
├── tools/
│   ├── doctor.py               安裝後自我檢查
│   ├── wsl_bootstrap.sh        WSL 拆分安裝（Hermes 在 Ubuntu、ComfyUI 留 Windows）
│   └── linux-reclaim-helper.sh Linux 的記憶體回收（cgroup / process_madvise）
└── tests/test_*.py             任何平台都能跑的自我測試：後端 API 解析、ComfyUI 防護、dramagen、
                                多廠商 VRAM、假 /proc 上的 Linux 分支、真實的行程樹、WSL 路徑解析
```

## 顯示卡廠商中立

引擎問的只有一件事：**這張卡已經被佔用多少、還剩多少**；只有剩下的才給生成用。
**先走原廠工具，都沒有才問服務 API**，依序嘗試：

`VIDGEN_GPU_QUERY_CMD`（自訂）→ `nvidia-smi` → `intel-ze`（Intel Level Zero）
→ `xpu-smi`（Intel CLI）→ `rocm-smi`（AMD）
→ **ComfyUI 的 `/system_stats`**（torch 在哪個後端就報哪個，不需任何額外工具）
→ **TRUSTA-AST 的 `/system/resources`**（後端自己的量測，最後手段）

前面幾個讀的是驅動，因此看得到卡上**所有**行程的佔用——那才是規劃器需要的數字。
`intel-ze` 排在 `xpu-smi` 前面，因為 Level Zero loader 隨 Intel 顯示驅動一起裝，
而 `xpu-smi` 是要另外安裝的 CLI。它會**跑在子行程裡**，因為有些機器有 loader 卻沒有
對應的 Level Zero driver，這時 loader 會直接當掉而不是回錯誤；那種事不能發生在規劃器裡。
內顯的記憶體只算一次（它就是系統 RAM）。

最後兩條是安全網，讓沒見過的加速器也能零設定運作。
`doctor.py` 會告訴你是哪個來源答的。

## 和地端 LLM 的關係

這包是**為了配 TRUSTA-AST 而設計的**，而 AST 底下的推論引擎是可換的。
所以這包從不去戳任何引擎的 port：「有沒有載入模型、佔了多少 VRAM、有沒有可以回收記憶體的行程」
全部問 AST（預設 `127.0.0.1:8571`；AST 跑在別處時請設定 `VIDGEN_AST_URL`）。換引擎不需要改這邊任何東西。

算圖期間 LLM 會怎樣，取決於引擎：

| 引擎行為 | 算圖期間 |
|---|---|
| 佔用的 VRAM 不超過 `VIDGEN_LLM_VRAM_BUDGET`（預設 7 GB；例如 experts 放在 RAM 的 35B MoE） | 替它保留 VRAM；開始算圖後，把它乾淨的 DRAM 分頁交給算圖 |
| 超過這個量 —— 例如刻意佔滿整張卡的 Strata | 讓它睡眠，睡眠備份放不進 RAM 時改為卸載；算完再載回來。agent 的對話會保留 |
| 連不到 | 規劃器**假設 LLM 在**並保留 VRAM——猜錯的代價是影片短一點，而不是把驅動這條流程的 agent 擠掉 |

`SKILL.md.in` 是模板：裡面的 `@@PY@@`、`@@VIDGEN@@` 等佔位符會在安裝時換成這台機器的絕對路徑，
因為 SKILL.md 會給模型完整的指令去執行。

## 需求

NVIDIA GPU（建議 12 GB+ VRAM）、Python 3.9+、ComfyUI（含 `ComfyUI-GGUF` 自訂節點）、
40–171 GB 磁碟空間給模型。Hermes 需已安裝。`motion` 不需要 GPU 也不需要 ComfyUI，
只要 Python 和 `ffmpeg`（它的套件會裝在自己的 venv 裡）。`qwen21` 的模型需要有
`TextEncodeQwenImage21` 節點的 ComfyUI（在 0.37.0 上驗證過）。`qwen21` 用的 Viggle Turbo 是選用的，
須另外下載（研究授權，不隨包附上）；沒有它時 `gen.py` 會退回較慢的基本取樣器。
見 `MODELS.zh-TW.md`。

## 已驗證的環境（最後更新 2026-10）

| 佈局 | 狀態 |
|---|---|
| **Windows 11**（ComfyUI portable + Hermes + TRUSTA-AST 在同一台筆電，RTX 5080 Laptop 16 GB、32 GB RAM） | ✅ **完全地端**：Hermes 在地端 Strata LLM 上，以自然語言驅動 `motion`、`imggen`、`vidgen`、`recast`，包含後續用 prompt 修改（[展示](docs/showcase/README.zh-TW.md)）；`recast` 也搭配遠端 Qwen3.8-Flash 跑過。連結安裝以 `git pull` 更新 |
| **WSL2 + Ubuntu 24.04.4**（Hermes 在 distro、ComfyUI/模型/LLM 留 Windows） | ✅ 出圖 16 s、i2v 影片 132 s、Hermes agent 自然語言驅動 ×3 |
| **原生 Linux**（Ubuntu 24.04.4、RTX 5060 Ti 16 GB） | ✅ 端到端出圖；Hermes（Qwen3.8-27B 與 Qwen3.6-35B-A3B）驅動了多圖換裝、多輪修改、背景 + 多角色合成，以及移動人物 |

尚未在 Linux 上跑過：自動 GPU 出借路徑（`llmsleep.py`）、`recast` 與 `motion`。

`tests/test_*.py` 與 `skills/motion/tests/test_motion.py` 在任何平台都能跑。到新機器後，先跑 `tools/doctor.py`。

## 授權

本包採用 [Apache License 2.0](LICENSE) 授權，與 TRUSTA-AST 相同。

第三方內容與模型的授權列在 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
**部分模型不可商用**（Qwen-Image-2.1 與 Viggle Turbo 採 Qwen Research License；
見 `MODELS.zh-TW.md`）。權重從不隨本包附上。
