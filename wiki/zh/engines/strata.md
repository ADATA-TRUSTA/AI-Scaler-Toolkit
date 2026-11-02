---
title: Strata 引擎
summary: 大型 MoE 的專家放 DRAM、熱專家快取在 GPU；deep sleep 約一秒放掉 GPU 並保住對話。選用元件，只支援 NVIDIA。
updatedAt: 2026-10-29
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

選用元件，**只支援 NVIDIA**。後端 0.3.0 本身不包含 Strata：
`STRATA_HOME` 指向一份裝好的 Strata、而且引擎是支援 `--sleep-mode` 的 deep-sleep build 時，
`strata` 才會出現在 `GET /health` 的 `supported_engines`。設定項目見
[`backend/.env.example`](../../../backend/.env.example) 的 Strata 一節。

> 後端**永遠**用 `--sleep-mode` 啟動 Strata，所以 Strata 自己的 setup 下載的現成引擎（沒有這個旗標）不能用；
> `STRATA_ENGINE_BINARY` 必須指向 deep-sleep build。

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

## 載入時預設的兩個環境變數

```
STRATA_UNBUFFERED_LOAD=1    STRATA_NO_ARENA_THP=1
```

只在你沒有自己設定時才套用。理由是一次實測的事故：
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
| `STRATA_HOME` | 空 | Strata 的安裝目錄（含 `serve/server.py`）；空的時候不提供這個引擎 |
| `STRATA_PYTHON` | — | 執行 `serve/server.py` 的 interpreter，通常是 Strata 自己的 `.venv` |
| `STRATA_ENGINE_BINARY` | — | **必填**：deep-sleep build 的引擎 |
| `STRATA_PORT` | `5002` | loopback 埠 |
| `STRATA_HEALTH_TIMEOUT` | `900` | 等模型載好的秒數 |
| `STRATA_RUNTIME_DIR` | `./.cache/strata` | run config 複本、伺服器 log、對話存檔 |

## 相關

- [推論引擎怎麼選](index.md)
- [vLLM](vllm.md) — 另一個能 sleep 的引擎
- [三種切換方式](../benchmarks/gpu-handoff.md)
