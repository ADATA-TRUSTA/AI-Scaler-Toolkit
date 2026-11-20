# AI Scaler Toolkit Wiki（繁體中文）

> English: [`../en/index.md`](../en/index.md)
>
> 內容對應 **後端 0.3.0**。安裝與日常操作請看 [`docs/`](../../docs/installation.md)；
> 這裡寫的是「為什麼會這樣、怎麼調、怎麼在 API 上開發」。

## 從哪裡開始

| 你想知道 | 讀這頁 |
|---|---|
| AST 是什麼、由哪些元件組成 | [全貌](overview.md) |
| 我的模型該用哪個推論引擎 | [推論引擎怎麼選](engines/index.md) |
| 模型比顯卡大，怎麼跑得動 | [Offload：四種不同的東西](memory/offload.md) |
| 這組設定塞不塞得下 | [記憶體估算](memory/estimation.md) |
| 出問題了 | [從症狀反查](operations/troubleshooting.md) |
| 實際跑起來多快 | [實測結果](benchmarks/index.md) |

## 概覽

- [全貌](overview.md) — 一台機器上的本地 LLM 平台；四個元件、三層硬體適配、一次一個模型
- [服務架構](architecture.md) — 三個行程、兩條推論路徑、狀態放在哪裡
- [術語表](glossary.md) — 在這個專案裡意思不同、或同名不同義的詞

## 推論引擎

- [推論引擎怎麼選](engines/index.md) — 四個引擎的取捨一覽
- [transformers](engines/transformers.md) — 微調完直接載 adapter；`device_map` 的注意事項
- [llama.cpp（llama_server）](engines/llama-server.md) — GGUF、slot 快取、mmproj、`--jinja`
- [vLLM](engines/vllm.md) — 併發、KV offload、sleep/wake、tool parser 自動偵測
- [Strata](engines/strata.md) — 大型 MoE 的專家放 DRAM，deep sleep 保住對話（NVIDIA；Intel Arc 沒有 deep sleep）

## 記憶體與模型格式

- [Offload：四種不同的東西](memory/offload.md) — 權重、MoE 專家、KV cache、訓練狀態
- [記憶體估算](memory/estimation.md) — HF 與 GGUF 兩套估算器、五條 GGUF 端點
- [GGUF 轉換與 LoRA merge](models/gguf-conversion.md) — 逐 shard merge、量化、轉換的坑

## 微調（Linux）

- [微調總覽](training/index.md) — 流程、phase、事件流與 SSE、兩個順序不變式
- [多模態微調](training/multimodal.md) — `multimodal_scope` 與四個模型特有的坑
- [DeepSpeed ZeRO-3 offload](training/deepspeed-offload.md) — 五個 profile、NVMe 與 pinned DRAM、版本釘選

## API

- [API 總覽](api/index.md) — 路由分組、單一模型、sleep/wake
- [對話（OpenAI 相容）](api/chat.md) — 擴充欄位、thinking 與 `reasoning_effort`、tool call、模型專屬覆寫
- [錯誤模型](api/errors.md) — 狀態碼的語意、`detail` 的三種形狀、串流中的錯誤
- [遠端存取與信任邊界](api/remote-access.md) — 本機、別台機器、瀏覽器網頁三套規則

## 維運

- [從症狀反查](operations/troubleshooting.md) — 看到 X 先查 Y
- [跑了才發現的坑](operations/known-pitfalls.md) — 讀程式碼看不出來的事
- [放在反向代理後面](operations/reverse-proxy.md) — SSE 緩衝、timeout、`X-Forwarded-For`

## 實測結果

- [實測結果索引](benchmarks/index.md) — 每份都附硬體與條件
  - [vLLM 原生 KV cache offload 到 DRAM／NVMe](benchmarks/vllm-kv-offload.md)
  - [8 GB 顯存預算下的 vLLM 與 llama.cpp](benchmarks/vllm-vs-llamacpp-8gb.md)
  - [GGUF 記憶體估算的準確度](benchmarks/gguf-estimation-accuracy.md)
  - [Gen5 NVMe 對本地 AI 流程的影響](benchmarks/gen5-ssd.md)
  - [把 GPU 讓給其他程式：三種切換方式](benchmarks/gpu-handoff.md)
  - [16 GB 顯卡上用 LoRA 訓練 35B MoE](benchmarks/deepspeed-lora-16gb.md)

## 產品

- [TRUSTA 企業級 SSD](products/trusta-ssd.md) — 產品線與適用場景

## 關於本 wiki

- [關於本 wiki](about.md) — 範圍、寫法、對應版本
- [更新紀錄](log.md)
