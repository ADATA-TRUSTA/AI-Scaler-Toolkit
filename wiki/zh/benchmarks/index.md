---
title: 實測結果
summary: 與 AST 核心（DRAM／SSD offload、引擎選擇、記憶體估算、GPU 切換、低顯存訓練）直接相關的實測，每份都附硬體、模型與設定。
updatedAt: 2026-11-20
---

# 實測結果

**每個數字都只在它的條件下成立。** 換硬體、模型、量化或版本，結果都可能不同。
每頁開頭先寫條件，最後寫限制。

| 頁 | 一句話結論 |
|---|---|
| [vLLM 原生 KV cache offload 到 DRAM／NVMe](vllm-kv-offload.md) | 16 GB 顯卡上 35B 模型滿 context 的對話重訪，從 445 秒變 6 秒；DRAM 層要放得下要重用的前綴 |
| [8 GB 顯存預算下的 vLLM 與 llama.cpp](vllm-vs-llamacpp-8gb.md) | 多人共用前綴、長 prompt、重啟後接回快取是 vLLM 的場；單人 decode、啟動速度是 llama.cpp 的場 |
| [GGUF 記憶體估算的準確度](gguf-estimation-accuracy.md) | 權重與 KV 誤差 ≤1 MiB 且從不低估；`recommend` 給的參數用掉預算的 97% 左右 |
| [Gen5 NVMe 對本地 AI 流程的影響](gen5-ssd.md) | 效益看工作負載有多少時間在等磁碟；預熱 page cache 讓 MoE decode 快 2 倍 |
| [把 GPU 讓給其他程式：三種切換方式](gpu-handoff.md) | 256K 對話下，Strata sleep/wake 拿回 GPU 到答出第一句約 1.8 秒，重新載入約 26–33 秒 |
| [Strata 在 Intel Arc 內顯上](strata-intel-arc.md) | 58 GB 的 Q1 模型在 30～32 GB 共用記憶體的筆電上跑得動：Windows 開 MTP 約 11 tok/s decode、175 tok/s prefill；256K 對話存檔後重啟只要 6 秒還原 |
| [Strata 用 CPU 的 AMX 讀 prompt](strata-amx-prefill.md) | CPU 分擔一部分專家的運算時，4K prompt 讀取快約 90～110%；32K 沒有增益，引擎預設的上限下超過約 3K tokens 時 AMX 完全沒有作用 |
| [Strata 與 vLLM 在一張 48 GB 卡上，專家放 DRAM](strata-vs-vllm.md) | 四人各讀 4K，Strata 23 秒完成、vLLM 150 秒；有 KV cache 的 4K 追問是 2.3 秒對 16～19 秒 |
| [16 GB 顯卡上用 LoRA 訓練 35B MoE](deepspeed-lora-16gb.md) | DeepSpeed 0.19.2 下 peak 3.08 GiB；0.19.3 以後第一步就 OOM |

## 共通的測試機器

除非該頁另外註明，測試機器是：

| 項目 | 規格 |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16 GB（PCIe Gen5 x8，同時驅動桌面） |
| CPU | Intel Core Ultra 5 225（6 P-core + 4 E-core） |
| RAM | 125 GB DDR5 |
| SSD | ADATA T7P5 Gen5 NVMe |
| OS | Ubuntu 24.04 |

筆電的測試（RTX 5080 Laptop 16 GB、Core Ultra 9 275HX）在頁內註明。
