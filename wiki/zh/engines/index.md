---
title: 推論引擎怎麼選
summary: transformers、llama.cpp、vLLM、Strata 四個推論引擎的取捨，以及它們共通的行為。
updatedAt: 2026-10-29
---

# 推論引擎怎麼選

## 先看這張表

| | `transformers` | `llama_server`（llama.cpp） | `vllm` | `strata` |
|---|---|---|---|---|
| 模型格式 | HF safetensors | **GGUF** | HF safetensors | GGUF（大型 MoE） |
| 平台 | 全平台 | 全平台 | **Linux** | **NVIDIA** 與 **Intel Arc**（Intel 沒有 deep sleep），選用元件 |
| 執行方式 | 在後端的 worker 行程內 | 管理一個子行程 | 管理一個子行程 | 管理一個子行程 |
| 併發 | 弱（一次一個 forward） | `-np` 個 slot | **強**（continuous batching） | 以單人為主 |
| 量化 | bitsandbytes int8／int4 | GGUF 內建 | AWQ／GPTQ／FP8 等 | GGUF 內建 |
| 多模態 | ✅ | ✅（需要 mmproj） | ✅ | 依模型 |
| 微調產物 | **adapter 直接載** | 先 merge 再轉 GGUF | 先 merge（或用 vLLM 的 LoRA） | 先轉 GGUF |
| 放掉 GPU 但不卸載 | ❌ | ❌ | ✅ sleep（**丟掉 KV**） | ✅ deep sleep（**保住對話**） |
| 最適合 | 剛微調完想立刻試 | 單機、記憶體吃緊、要跑大模型 | 多人併發、長 context、KV 重用 | 大型 MoE 單人用，且要和其他程式輪流用 GPU |

**一句話**：微調完先用 `transformers` 驗證，上線用其他三個。
多人併發選 `vllm`、單機吃緊選 `llama_server`、大型 MoE 而且要讓 GPU 給別的程式用選 `strata`。

實際數字見 [8 GB 顯存預算下的 vLLM 與 llama.cpp](../benchmarks/vllm-vs-llamacpp-8gb.md)
與 [三種切換方式](../benchmarks/gpu-handoff.md)。

## 結構上最重要的一件事

四個引擎類別都有 `load_model`、`unload`、`apply_chat_template`、`cleanup_generation_memory`，
但**只有 transformers 實作了生成**。另外三個只負責起停伺服器與整理它的中繼資料；
生成由主行程直接用 async HTTP 打過去。詳見 [服務架構](../architecture.md)。

## 四個引擎共通的事

**一、一次只能載一個模型。** 載入不同模型或不同設定回 `409`，要先卸載。

**二、卸載的意思不同。** transformers 是結束 worker 行程（VRAM 才收得回來）；
server engine 是停掉子行程。llama.cpp 與 Strata 會在這時把對話的 KV 存檔。

**三、`quantization` 欄位只對 transformers 有效。** GGUF 的量化在檔案裡；vLLM 用 `vllm_quantization`。

**四、`/inference/status` 的欄位因引擎而異。** 呼叫端要容忍缺值。

## 各引擎的細節

- [transformers](transformers.md)
- [llama.cpp（llama_server）](llama-server.md)
- [vLLM](vllm.md)
- [Strata](strata.md)

## 相關

- [Offload：四種不同的東西](../memory/offload.md)
- [記憶體估算](../memory/estimation.md)
- [服務架構](../architecture.md)
