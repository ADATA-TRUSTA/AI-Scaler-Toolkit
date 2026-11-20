---
title: Strata 引擎
summary: 大型 MoE 的專家放 DRAM、熱專家快取在 GPU；deep sleep（NVIDIA）約一秒放掉 GPU 並保住對話。選用元件，安裝檔帶著套件，支援 NVIDIA 與 Intel Arc。
updatedAt: 2026-11-20
---

# Strata 引擎

[Strata](https://github.com/Niko1221/Strata) 是一個專為大型 MoE 設計的推論伺服器。
AST 用的是加上 **deep sleep** 的版本。

## 它解的問題

MoE 模型的專家放 **DRAM**，**熱的專家快取在 GPU**，快取大小吃滿顯卡。

| | vLLM 專家 offload | Strata |
|---|---|---|
| 專家放哪 | 用 `vllm_cpu_offload_params=["experts"]` 指定 | 架構本來就這樣 |
| GPU 上放什麼 | KV cache + 非專家權重 | **熱專家的快取** |
| 放掉 GPU | sleep：**KV 丟掉** | deep sleep：**KV 與對話保住** |

## 能不能用

選用元件。**安裝檔帶著 Strata 的套件，第一次載入 Strata 模型時自動安裝**，不用自己設 `STRATA_HOME`。用哪個套件看 GPU：

| GPU | 套件 | deep sleep |
|---|---|---|
| NVIDIA（先檢查：兩種卡都有就用這個） | CUDA | **有** |
| Intel Arc（Windows 或 Linux） | Intel（SYCL） | **沒有**（見下） |
| 其他 | 沒有；不提供 `strata` 引擎 | |

> 套件是加上 **deep sleep** 的 Strata，已經編好，連同 Strata 的 Python 端；不是 Strata 自己的下載。
> 後端對 CUDA 引擎**永遠**用 `--sleep-mode` 啟動，所以 Strata 自己的 setup 下載的現成引擎（沒有這個旗標）不能用。

從原始碼 checkout 跑後端時，`STRATA_HOME` 指向一份裝好的套件（含 `serve/server.py`），
`strata` 才會出現在 `GET /health` 的 `supported_engines`；設定項目見
[`backend/.env.example`](../../../backend/.env.example) 的 Strata 一節。

### 支援哪些 GGUF

Strata 的 setup 只接受固定的一份 GGUF（以第一個分片的檔名認）：GSQ-RCO（Q2_0、IQ2_XS、IQ3_XXS、IQ3_S）、
Swift 1.5（IQ2_XS、IQ3_XXS、IQ3_S）、Coder（IQ1_M）、Unsloth（UD-IQ4_XS、UD-Q4_K_XL、UD-Q5_K_XL、UD-Q6_K_XL、Q8_0）、
Gyro-S／Gyro-M。其他量化（含 Unsloth 的 UD-IQ3_XXS、UD-Q2_K_XL）會被拒絕。
`GET /config/models` 的每個 GGUF 帶 `strata_supported`，Strata 選單只列這些。

### 第一次載入

這個 GGUF 還沒有 run config 時，第一次載入會（需要的話）先裝套件，再對**原地的 GGUF** 跑 Strata 的 setup，
只另外產生 expert pack、草稿層與 run config。檔名不在清單裡，會在這些之前先被拒絕。之後的載入直接啟動。

## Intel Arc（SYCL）套件

| | CUDA 套件 | Intel Arc 套件 |
|---|---|---|
| deep sleep | 有 | **沒有**：它建在 CUDA driver 的虛擬記憶體上，Level Zero 沒有對應的東西 |
| `POST /inference/sleep`、`/wake` | 放行 | **409** `This Strata engine has no deep sleep…` |
| 圖片讀取器 | GPU 上的 `strata-vision` | GPU 上的 `strata-vision-sycl`，CPU 上的 `strata-vision` 當後備 |
| 就緒訊息的 `device` | `cuda` | `sycl` |

Intel 引擎建成可以在**任何有 AVX2 的 x86-64 CPU** 上跑，不只是建置那一台；分片的模型也保有自己的檔名（Hugging Face 快取的符號連結就是這樣）。
這兩點都在 0.2.11-sleep.2 的套件修好：在較新 CPU 上建的引擎，曾在沒有那些指令的伺服器 CPU 上以非法指令錯誤當掉。

實測數字：[Strata 在 Intel Arc 上](../benchmarks/strata-intel-arc.md)。

> **比 CUDA 路徑新、測得少。** 套件在內顯 Arc 的筆電（Windows）與約 32 GB 的獨立 Arc 顯卡（Linux）上安裝並跑過，但走的是 Strata 自己的啟動程式；
> 經過 AST 後端的完整路徑還沒有在 Intel 機器上跑過。Windows 上後端對 Intel 引擎預設設 `STRATA_DOORBELL_CHECK=1`
> （沒有它引擎不會當，但回覆可能漂掉）。

## 模型怎麼描述：run config

Strata 的模型不是用一堆參數描述，而是用 Strata 的 setup 產生的 **run config**
（`strata-<model>.json`：引擎執行檔、expert pack、GGUF 分片、視覺 encoder）。

```
你的 strata-<model>.json ──複製並套用這次載入的覆寫──► STRATA_RUNTIME_DIR/ ──用這份啟動
```

**你的檔案永遠不會被修改。** 沒有給 `strata_config_path` 時，後端在 `STRATA_HOME`
裡找第一個 GGUF 分片等於 `model_path` 的 run config。

## 載入只開四個欄位

```
strata_config_path · strata_max_context · strata_vram_limit_gb · strata_auto_wake
```

其餘啟動旗標（KV 格式、KV 放置、speculative window、expert cache）留在 run config 裡；要調就改那個檔案。
這和 vLLM 相反：**讓 API 小、讓檔案大**。

### `strata_vram_limit_gb`：給少反而慢，建議不要設

Strata 會把 expert cache 開到吃滿 GPU；這個值是保留給其他程式的空間。

| 可用 VRAM | decode |
|---|---|
| 約 15 GB | 72–84 tok/s |
| 約 8 GB | 45–54 tok/s |

**建議留空，等別的程式要用 GPU 時再 `POST /inference/sleep`。**

## deep sleep：約一秒放掉 GPU，對話還在

永遠開啟——在第一次 sleep 之前沒有任何成本。

| | vLLM sleep | Strata deep sleep |
|---|---|---|
| 權重 | 搬到 pinned CPU memory | 引擎的 VRAM 複製到 pinned host RAM |
| KV cache | **丟掉** | **保住** |

實測（Windows、RTX 5060 Ti 16 GB、GSQ-RCO IQ3_S，約 78 GB 兩個分片）：sleep 1.3 秒，VRAM 從 15,866 降到 1,498 MiB；
wake 1.0 秒，對話接得上。端點與 vLLM 共用：`POST /inference/sleep`、`POST /inference/wake`。
行為細節見 [vLLM](vllm.md) 的 sleep／wake 一節。

256K context 下三種讓出 GPU 方式的比較見 [三種切換方式](../benchmarks/gpu-handoff.md)。

### 對話會存檔，下次載入自動還原

卸載時把 slot 存到 `STRATA_RUNTIME_DIR/slots/<model>.slot-0.bin`，下次載入還原。
沒有存檔不是錯誤，回 `{"restored": false, "reason": "no saved conversation"}`。

> 這和 `session_id` 的伺服器端對話歷史是兩回事：這裡存的是引擎自己的 KV，不是訊息列表。

## 載入時預設的環境變數

```
STRATA_UNBUFFERED_LOAD=1    STRATA_NO_ARENA_THP=1          （所有平台）
STRATA_PLE_AUTO_RAM=0                                       （Windows）
STRATA_DOORBELL_CHECK=1                                     （Windows，Intel 引擎）
```

只在你沒有自己設定時才套用。Windows 的第三個是讓引擎不要自己把大表鎖在 RAM 裡（會讓 sleep 的備份放不下，每次 sleep 都回 409）；第四個就是上面說的 Intel 檢查。

前兩個的理由是一次實測的事故：
透過 page cache 載入 50 GiB 的 expert arena，memory pressure 長時間停在 systemd-oomd 的 50% 門檻之上（實測 69.8%），
**oomd 把整個桌面 session 登出了**。改成 unbuffered 並關掉 transparent huge pages 後，
同一次載入的尖峰是 0.2%，而且**讀得更快**（9.1–10.0 GiB/s，原本 6.4）。

## 執行方式

和 llama-server、vLLM 相同：worker 起伺服器（只聽 `127.0.0.1:STRATA_PORT`），主行程用 HTTP 打過去。
等待模型載好的上限是 `STRATA_HEALTH_TIMEOUT`，預設 900 秒（要讀幾十 GB）。
失敗時附上伺服器 log 的最後 40 行。

## 設定

| 變數 | 預設 | 說明 |
|---|---|---|
| `STRATA_HOME` | 空 | Strata 的安裝目錄（含 `serve/server.py`）。空的：第一次載入時安裝安裝檔帶的套件 |
| `STRATA_INSTALL_DIR` | 空 | 套件裝在哪：空的是專案旁邊；桌面版放在資料目錄底下的 `strata` |
| `STRATA_PYTHON` | — | 執行 `serve/server.py` 的 interpreter，通常是 Strata 自己的 `.venv` |
| `STRATA_ENGINE_BINARY` | — | 引擎執行檔；套件安裝時會幫你寫好 |
| `STRATA_PORT` | `5002` | loopback 埠 |
| `STRATA_HEALTH_TIMEOUT` | `900` | 等模型載好的秒數 |
| `STRATA_RUNTIME_DIR` | `./.cache/strata` | run config 複本、伺服器 log、對話存檔 |

## 相關

- [推論引擎怎麼選](index.md)
- [vLLM](vllm.md) — 另一個能 sleep 的引擎
- [Strata 在 Intel Arc 上](../benchmarks/strata-intel-arc.md)
- [用 AMX 讀 prompt](../benchmarks/strata-amx-prefill.md)
- [Strata 與 vLLM 在一張 48 GB 卡上](../benchmarks/strata-vs-vllm.md)
- [三種切換方式](../benchmarks/gpu-handoff.md)
