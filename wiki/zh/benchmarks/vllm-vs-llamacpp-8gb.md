---
title: 8 GB 顯存預算下的 vLLM 與 llama.cpp
summary: Qwen3.5-35B-A3B 只佔約 8 GB 顯存：多人共用前綴、長 prompt 冷啟、重啟後接回快取是 vLLM 的場（4 個 100K 同時冷啟 125 秒對 1,070 秒）；單人 decode、啟動速度、可回收的 DRAM 是 llama.cpp 的場。
updatedAt: 2026-10-29
measuredAt: 2026-09-24
---

# 8 GB 顯存預算下的 vLLM 與 llama.cpp

目標：LLM 只佔約 8 GB 顯存，其餘留給影像生成（ComfyUI）。兩個引擎都透過 AST 的 `/v1/chat/completions` 量測。

## 條件

硬體：RTX 5060 Ti 16 GB、125 GB DRAM、Linux。

| 項目 | vLLM | llama.cpp |
|---|---|---|
| 版本 | 0.28.0 | b10107 |
| 權重 | GPTQ-Int4（attention 保留 bf16） | Q4_K_M GGUF |
| 專家 | 放 CPU | 放 CPU |
| KV | fp8 | q8_0（沒有 fp8） |
| context／同時請求 | 131,072 · 4 | 同左（`-np 4`，每個 slot 32K） |

## 結果

| 項目 | vLLM | llama.cpp |
|---|---|---|
| 載入模型 | 82 s | 7–12 s |
| 單人 decode（20K context） | 29 tok/s | **48 tok/s** |
| 100K 冷啟 TTFT | **104 s** | 145–259 s |
| 重啟後同一個 20K 前綴 | **1.34 s**（從 NVMe 命中） | 26.0 s（整段重算） |
| 4 個 100K 共用前綴同時冷啟 | **125 s**（4 人的 TTFT 都是 105 s） | 1,070 s（一個接一個 prefill） |
| GPU | 閒置 7.8、峰值 8.7 GB | 6.8–7.7 GB |
| DRAM | 約 30 GB，**鎖住** | 17–21 GB mmap 權重，**可回收** |

- **decode 差 1.7 倍的原因**：llama.cpp 在 CPU 上直接算專家（6 個 P-core 滿載）；
  vLLM 每產生一個 token 都要把專家經 PCIe（Gen5 x8，實測約 22 GB/s）送進 GPU。
- vLLM 的三層 KV：NVMe→DRAM 約 7–8 GB/s、DRAM→GPU 約 26 GB/s，搬資料本身不到 0.2 秒；
  命中之後的時間花在重算沒對齊的尾巴（這個 hybrid 模型的前綴以 2,096 token 為單位）。

## 怎麼選

| 情境 | 選 |
|---|---|
| 1–2 個 agent、服務常駐 | llama.cpp |
| 常重啟、多人同時開始、context 常超過 32K | vLLM |

llama.cpp `-np 4` 時每個 slot 只有 32K；一個帶影片生成 skill 的 agent 就需要約 37K。

## 限制

prompt 是合成資料；每個情境只跑一次；兩邊的量化格式不同。

> 一開始記錄的 vLLM DRAM 用量是 47.6 GB，後來發現是把 `/dev/shm` 的 KV 層算了兩次（同一行程映射兩次），實際約 30 GB。

## 相關

- [推論引擎怎麼選](../engines/index.md)
- [vLLM 原生 KV cache offload](vllm-kv-offload.md)
