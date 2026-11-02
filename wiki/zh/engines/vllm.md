---
title: vLLM 引擎
summary: 四個引擎裡限制最多也最快的一個：Linux 專用、跑在自己的 venv、tool parser 自動偵測、兩段式 KV offload、sleep/wake、啟動失敗分類。
updatedAt: 2026-10-29
---

# vLLM 引擎

後端管理一個 `vllm serve` 子行程。版本是 **0.28.0**。

## 能不能用：平台與安裝兩道關

1. **不是 Linux** → 不提供
2. **是 Linux 但沒裝** → 也不提供

| `TRUSTA_SETUP_VLLM` | NVIDIA 主機 | Intel 主機 |
|---|---|---|
| 沒設（`auto`） | **裝** | 不裝 |
| `1` | 裝 | **裝** |
| `0` | 不裝 | 不裝 |

Intel 上預設不裝有三個理由：內顯沒有自己的記憶體，vLLM 的「顯存」就是系統 RAM；
kernel 的 TTM page pool 用完後會留住最多一半的 RAM，vLLM 的啟動檢查把它算成已用，
所以**第一次載入成功、第二次失敗**；vLLM 官方的 XPU 支援也不含內顯。

被擋下來時 `POST /inference/load_model` 回 `501` 與一句原因。網頁介面依 `GET /health` 的
`supported_engines` 決定選單，所以正常情況看不到這個選項。

## 跑在自己的 venv

vLLM 有自己的專案目錄與 venv（`VLLM_SERVER_PROJECT_DIR`），不是後端這個。
**換來的是 vLLM 的版本不必和後端綁在一起**——兩者的相依套件會打架。
後端會自己找 vLLM 的 interpreter 與 NVIDIA 動態庫路徑。

## tool call parser：跑一次問出來，不是查表

載入時，後端用模型自己的 chat template 渲染一次 tool call，
再讓 vLLM 的各個 parser 試著讀回來——**讀得回來的那個就是答案**。
答案來自兩個權威（模型的 template 與 vLLM 的 parser），所以 vLLM 改了 parser 名稱或語法也不會失準。

可以用 `vllm_tool_call_parser` 與 `vllm_reasoning_parser` 明確指定，覆寫偵測結果。

## reasoning parser 開著時，thinking 預設關

條件：引擎是 vLLM、有 reasoning parser、而請求沒有指定 `enable_thinking`。

要修的症狀是 **tool 結果之後每次回覆都是空的**：reasoning parser 把 `</think>` 之前的全部當成思考，
沒有 `</think>` 時就是整個回覆；而有些 hybrid template（例如 MiniCPM5）沒指定時什麼都不預填，
模型常直接作答——於是答案連同 tool call 一起被歸成 reasoning。

釘成「關」而不是「開」，是因為微調資料裡的答案通常沒有 think block。
開發者實測：MiniCPM5-1B 的一個 LoRA，thinking 關時 40 題答對 27，打開時 18。

只影響有 reasoning parser 的載入；template 本來就會思考的模型（例如 Qwen3）不受影響。

## MoE 專家 offload：比顯卡大的 MoE 靠這個載起來

| 欄位 | 作用 |
|---|---|
| `vllm_cpu_offload_gb` | 多少 GB 權重放 pinned CPU memory |
| `vllm_cpu_offload_params` | **只** offload 名稱含這些段的參數 |

`["experts"]` 只把 MoE 專家放 CPU、attention 留在 GPU。不給的話會照順序搬任何權重，連 attention 一起，整體變慢。
**比對的是整段名稱**：`experts` 匹配 `mlp.experts.w13_qweight`，`expert` 不匹配。

詳見 [Offload](../memory/offload.md)。

## KV offload：兩段式，DRAM 層是閘門

```
GPU ──逐出──► DRAM 層（vllm_kv_offloading_size）──► NVMe 層（vllm_kv_offloading_disk_path）
```

**磁碟上的 block 必須先複製進 DRAM 層，GPU 才能用。** 所以 DRAM 層要放得下最長要重用的前綴；
`vllm_kv_offloading_size` 沒設，磁碟層根本不會生效。

| 欄位 | 說明 |
|---|---|
| `vllm_kv_offloading_size` | DRAM 層大小（GB）。**多 GPU 時是所有 GPU 的總和** |
| `vllm_kv_offloading_disk_path` | NVMe 層的目錄 |
| `vllm_kv_offloading_disk_max_gb` | NVMe 層的上限（GiB） |

**vLLM 從來不會從磁碟層刪檔**，那個目錄只會長（測試時一個目錄長到 65 GB），所以後端提供上限與清理：

```
GET  /inference/kv_cache/disk           # 用了多少
POST /inference/kv_cache/disk/cleanup   # 刪最久沒用的 block，刪到上限的九成以下
```

邊跑邊刪是安全的：vLLM 查詢 block 就是看檔案在不在，被刪的會當成 miss 重算；
寫入是先寫暫存檔再改名，所以 `.bin` 一定是完整的。

後端啟動 vLLM 時會固定 `PYTHONHASHSEED`，**KV 才能跨重啟重用**；否則 hash 起點每次不同，快取靜默失效。

實測見 [vLLM 原生 KV cache offload](../benchmarks/vllm-kv-offload.md)。

## sleep／wake：放掉 GPU 但不卸載

```
POST /inference/sleep   {"mode": "wait" | "abort"}
POST /inference/wake
```

載入時要開 `vllm_enable_sleep_mode`。vLLM 的 sleep 會把權重搬到 pinned CPU memory，**KV cache 丟掉**。

| 行為 | 說明 |
|---|---|
| `mode: "wait"`（預設） | 等進行中的生成結束 |
| `mode: "abort"` | 立刻中止它們 |
| 睡著時收到生成請求 | 先喚醒；`vllm_auto_wake` 為 `false` 時回 `503` 與 `Retry-After: 1` |
| 已經睡著再 sleep | 什麼都不做 |
| 引擎不支援 | `409` |
| sleep 或 wake 本身失敗 | `502` |

> ⚠️ **wake 會失敗，而且常見**：睡著時別的程式拿走了 GPU 記憶體，醒不回來 → `502`，
> 而 `/inference/status` 仍顯示睡著。**協調者要重讀狀態，不能假設 wake 回來就醒了。**

後端**沒有閒置計時器**——何時睡、何時醒由呼叫端（例如 agent 的 skill）決定；
`auto_wake` 只是安全網，避免協調者忘了喚醒時請求失敗。

### admin 端點有 guard

sleep 需要 vLLM 的 admin 端點，而同一個開關也會掛出 `/collective_rpc`、`/update_weights` 等危險端點。
後端用一個 vLLM middleware 把關：

| 端點 | 本機程式 | 瀏覽器網頁 |
|---|---|---|
| `/sleep`、`/wake_up`、`/is_sleeping`、cache reset | ✅ | ❌ |
| 其他 admin 端點 | 404 | 404 |

`VLLM_SERVER_HOST` 不是 loopback 時，admin 端點一律關閉；這時開 `vllm_enable_sleep_mode` 會在載入前就被拒絕。

## Intel GPU：`gpu_memory_utilization` 會被調低

只在 XPU build 上：介面送的是 `0.9`，但內顯的「顯存」就是系統 RAM，32 GB 的筆電上 0.9 等於 25.7 GB，
永遠不會有那麼多空的，vLLM 會拒絕啟動。後端改成**現在真正空著的比例**、保留 2 GiB 給系統，並記一行 warning。
NVIDIA 原樣傳遞。

## 啟動失敗分十一類

```
oom · config · port_busy · model_not_found · cuda_mismatch · gpu_fault
shared_library_missing · toolchain_missing · chat_template · quantization · unknown
```

存在的理由是**以前什麼失敗都被猜成 OOM**；把缺動態庫當成 OOM，使用者會一直縮小模型然後一直失敗。

`toolchain_missing` 排在 `model_not_found` 前面，因為它的訊息長得像找不到檔案：
Intel 少 `intel-ocloc` 時是 `No such file or directory: 'ocloc'`，CUDA 少 `nvcc` 時是 `could not find nvcc`。

## 殘留清理

| 清什麼 | 何時 |
|---|---|
| 殘留的 `vllm serve` 行程 | 服務啟動時（`VLLM_STARTUP_SWEEP`），避免之後撞埠 |
| `/dev/shm` 的 offload 區域 | 被強制結束的 server 會留下；確認沒有行程還在映射後才刪 |

## 適用範圍

| 事實 | 說明 |
|---|---|
| 16 GB 卡上，8B 級的 bf16 模型放不進 vLLM | 只有 Qwen3-4B 級有餘裕；大模型要量化或專家 offload |
| 8 GB 顯存跑 Qwen3.5-35B-A3B 131K context | GPTQ-Int4 + 專家放 CPU + fp8 KV。見 [8 GB 對照](../benchmarks/vllm-vs-llamacpp-8gb.md) |
| vLLM 0.27.1 載不動 gemma-4 | 0.28.0 已修 |

## 相關

- [推論引擎怎麼選](index.md)
- [Offload](../memory/offload.md)
- [對話](../api/chat.md)
- [錯誤模型](../api/errors.md)
