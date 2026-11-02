---
title: 從症狀反查
summary: 看到某個症狀時先查什麼；最前面是症狀會把人帶往錯誤方向的幾個案例。
updatedAt: 2026-10-29
---

# 從症狀反查

**進入點是症狀，不是元件。** 原因的細節在各自的頁面，這裡只做「看到 X 先查 Y」。
安裝與一般使用的問題見 [`docs/faq.md`](../../../docs/faq.md)。

## 症狀會指向錯地方的幾個

| 症狀 | 看起來是 | 實際上是 |
|---|---|---|
| 「訓練失敗」，但 `output_dir` 有權重 | 訓練壞了 | **最後的 GGUF 轉換失敗**。權重已存好，不用重訓；用 `POST /config/models/convert` 手動轉。見 [GGUF 轉換](../models/gguf-conversion.md) |
| vLLM 說找不到模型，但路徑是對的 | 模型路徑錯 | **缺 GPU 編譯器**：Intel 少 `intel-ocloc`（`No such file or directory: 'ocloc'`），NVIDIA 少 `nvcc` |
| 改了設定重新載入，沒有生效 | 設定沒寫進去 | **`already_loaded` 只比六個欄位**。先卸載再載入 |
| 微調過的模型答得比微調前差 | 微調失敗 | **thinking 被打開了**，而訓練資料沒有 think block。用 `enable_thinking: false` 再試 |
| agent 回空字串，像是 merge 失敗 | merge 壞了 | **LoRA scaling 變成 1.0**，adapter 幾乎沒作用 |

## 別台機器連不上

**先看狀態碼與訊息：**

| 回應 | 意思 | 怎麼辦 |
|---|---|---|
| `403` …*available on the computer running AI Scaler Toolkit only* | 這個路徑只限本機。網頁介面與 `/docs` 永遠如此 | 在本機開；或確認要打的是 `/v1/` |
| `403` …*the inference API (/v1/) only* | 路徑在管理 API，而 `SERVICE_REMOTE_ACCESS` 是預設的 `inference` | 要開就設 `all`，**但先讀風險說明** |
| `403` …*not enabled on this service* | `SERVICE_API_KEY` 是空的 | 在設定裡開啟遠端存取並設 key |
| `401` …*A valid API key is required* | key 不對或格式不對 | `Authorization: Bearer <key>`，只認 bearer |
| `403` …*Web pages from other sites cannot use this service* | 跨站網頁，或 `Origin` 是 `null` | 從本機提供的頁面開 |
| `403` …*must be addressed to localhost or an IP address* | `Host` 不是 localhost 也不是 IP | 用 `localhost` 或 IP 連 |
| `429`，連一下就全部被拒 | 一分鐘錯 10 次 key → 鎖 5 分鐘 | 等 5 分鐘，或重啟服務 |
| 經反向代理開網頁介面，每個 API 都 `403` | 代理送了 `X-Forwarded-For` | 見 [放在反向代理後面](reverse-proxy.md) |

`/health` 沒有 key 時只回「它活著」，所以從別台機器看不到可用的引擎清單。

## 載入模型

| 症狀 | 原因 |
|---|---|
| `409 A model is already loaded` | 一次只能載一個，先卸載 |
| `409 A model is already being loaded` | 上一個還在載入 |
| `501` 與一句說明 | 平台不支援（非 Linux 上的 vLLM、vLLM 沒裝）；那句話就是原因 |
| `ValueError: When passing device_map as a string...` | `device_map` 送了 `none`；改用 `cuda:0` 或 `auto` |
| 載入成功但在 CPU 上跑 | 省略 `device_map` 會落在 CPU |
| 多模態 GGUF 沒有視覺 | 少了 mmproj，**而且不會報錯** |
| 比顯卡大的 MoE 載不起來 | vLLM 用 `vllm_cpu_offload_params=["experts"]`；llama.cpp 用 `n_cpu_moe`；或用 [記憶體估算](../memory/estimation.md) 的 `recommend` |
| 模型載入成功，但每次生成都連不上 | llama-server 的埠號問題；確認沒有手動設定 `llama_server_url` 指到別處 |

## vLLM

| 症狀 | 查 |
|---|---|
| 選單裡沒有 vLLM | 主機不是 Linux 或沒裝；看 `GET /health` 的 `supported_engines`。Intel 要 `TRUSTA_SETUP_VLLM=1` 安裝 |
| Intel 內顯上第一次載入成功、第二次失敗 | kernel 的 TTM page pool 留住了最多一半 RAM，vLLM 把它算成已用 |
| `is less than desired gpu memory utilization` | 調低 `vllm_gpu_memory_utilization`，不是換大卡 |
| tool 結果之後的回覆是空的 | reasoning parser 把整個回覆當成思考；有 parser 時 thinking 預設關，確認請求沒有強制打開 |
| 重啟後撞埠 | 殘留的 `vllm serve`。服務啟動時會清；手動起的不算 |
| `/dev/shm` 一直滿 | 被強制結束的 server 留下的 offload 區域 |
| KV offload 目錄越來越大 | vLLM 不會從磁碟層刪檔；設 `vllm_kv_offloading_disk_max_gb` 或呼叫 cleanup |
| 有 NVMe 層卻沒有命中 | DRAM 層放不下要重用的前綴（DRAM 層是閘門） |

## sleep／wake

| 症狀 | 原因 |
|---|---|
| sleep 回 `409` | 這個引擎不支援；只有 vLLM（載入時開 `vllm_enable_sleep_mode`）與 Strata |
| vLLM 的 sleep 載入前就被拒 | `VLLM_SERVER_HOST` 不是 loopback 時不能開 sleep mode |
| **wake 回 `502`** | 睡著時別的程式拿走了 GPU 記憶體。最常見的 wake 失敗 |
| sleep 之後對話不見了（vLLM） | 正常：vLLM 的 sleep 丟掉 KV。要保住對話用 Strata |
| sleep 很久才回 | 預設 `mode: "wait"` 會等進行中的生成結束；要立刻停用 `abort` |

## Strata

| 症狀 | 原因 |
|---|---|
| 選單裡沒有 `strata` | `STRATA_HOME` 沒設，或指向的目錄沒有 `serve/server.py`（Strata 只能裝在 NVIDIA 主機上） |
| `STRATA_ENGINE_BINARY is empty` | 必填，要指向 deep-sleep build |
| 載入 50 GB 的模型時桌面被登出 | systemd-oomd。確認沒有把 `STRATA_UNBUFFERED_LOAD`、`STRATA_NO_ARENA_THP` 設成 0 |
| decode 比預期慢很多 | `strata_vram_limit_gb` 設太低；建議留空 |
| 載入逾時 | `STRATA_HEALTH_TIMEOUT` 預設 900 秒 |

## 微調

| 症狀 | 原因 |
|---|---|
| 約 61 分鐘被 `cudaErrorLaunchTimeout` 中止 | 顯卡同時驅動螢幕。`save_steps` 設小，或把螢幕接到別的卡 |
| GPU 不夠但 DRAM 還很多 | 多模態 full fine-tune 在 16 GB 放不下，CPU offload 救不了；用 LoRA |
| NVMe offload 吃掉幾十 GB DRAM | 不是洩漏；調 `DEEPSPEED_NVME_PINNED_BUDGET_GB` |
| 訓練一開始就 OOM | 推論模型還載著？訓練與推論不互斥 |
| Intel 上推論可以，訓練報 `ze_data was not initialized` | oneCCL 需要 `libze_loader.so`（`libze-dev` 套件） |

## 下載

| 症狀 | 原因 |
|---|---|
| 按了取消，過了很久還在跑 | `HF_XET_ENABLED=1` 時傳輸無法中途中斷 |
| 小模型卻下載了好幾 GB | GGUF repo 沒給 `filename`，會抓全部量化版本 |
| 打錯 model id 卻沒有立刻失敗 | 端點立刻回 `task_id`，要等背景任務連上 Hub |
| 重啟後任務清單空了 | 下載任務只存在記憶體 |

## 對話

| 症狀 | 原因 |
|---|---|
| 思考過程不見了 | 在 `reasoning_content` 欄位，不是 `<think>` 標記 |
| transformers 引擎沒有思考過程 | 正常，那條路徑沒有 reasoning parser |
| 長 context 的第一個回應就斷線 | 代理或用戶端的逾時太短；等第一個 byte 就是整段 prefill |
| 答案停在思考裡 | `max_tokens` 太小 |
| 微調 adapter 好像沒有作用 | 服務注入的預設 system prompt 改變了 prompt；把 `DEFAULT_SYSTEM_PROMPT` 設成空字串 |
| Qwen3.5 的 `repetition_penalty` 沒生效 | 刻意的：>1.0 會被改成 1.0。見 [對話](../api/chat.md) |

## 先查上游，再說壞掉

第三方套件出問題時，**先查它的 issue、changelog、版本相容表**。
例如 vLLM 0.27.1 載不動 gemma-4、GGUF 轉換器的 `global_head_dim` KeyError，都是上游的已知問題。

## 相關

- [跑了才發現的坑](known-pitfalls.md)
- [錯誤模型](../api/errors.md)
- [`docs/faq.md`](../../../docs/faq.md)
