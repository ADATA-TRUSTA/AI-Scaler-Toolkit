---
title: 16 GB 顯卡上用 LoRA 訓練 35B MoE
summary: Qwen3.5-35B-A3B 的 LoRA 在 16 GB 顯卡上以 ZeRO-3 CPU offload 訓練，DeepSpeed 0.19.2 的 peak 是 3.08 GiB；0.19.3 起凍結權重不從 GPU 釋放，第一步就 OOM。AST 因此把 DeepSpeed 釘在 0.19.2。
updatedAt: 2026-10-29
measuredAt: 2026-09-02
---

# 16 GB 顯卡上用 LoRA 訓練 35B MoE

## 條件

| 項目 | 值 |
|---|---|
| 硬體 | RTX 5060 Ti 16 GB、125 GB DRAM、Linux |
| 軟體 | torch 2.13.0+cu130、transformers 5.15.0 |
| 訓練 | Qwen3.5-35B-A3B、LoRA、profile `zero3_offload_cpu_cpu_tight`、seq 512 |

## 結果：只換 DeepSpeed 版本

| | gradient checkpointing 開 | 關 |
|---|---|---|
| DeepSpeed **0.19.2** | **成功 · peak 3.08 GiB**（鋸齒 3.1 ↔ 2.5，不累積） | OOM · 12.07 GiB |
| DeepSpeed 0.19.6 | OOM · 11.99 GiB（階梯 1.5 → 8.3 → 12.0） | OOM · 12.07 GiB |

關掉 checkpointing 不是解法：只是把問題換成 activation 記憶體爆掉。

## 機制

- 0.19.3 為了修一個 crash，在 checkpoint 重算時把「目前 backward 的 owner」也記到凍結參數上；
  重算層自己的 hook 只移除自己的記錄，所以參數要等 owner 的 backward 結束才釋放。
- 上游假設凍結參數是少數，而 **LoRA 正好相反**：base 全部凍結，於是每一層的每個權重都留在 GPU。
- 數字對得上：每層融合後的專家是 1.50 GiB，1.5 + 1.50 × 7 = 12.0 GiB。
- 在 0.19.6 上停用這個機制，就回到鋸齒形、4.03 GiB、跑完 13 步。

AST 的訓練在用 DeepSpeed 時已經使用 reentrant checkpointing，那個 crash 本來就不會發生——
所以 0.19.3 的改動在這個組態下只有成本。

## 誰會受影響

| 情況 | 0.19.5 → 0.19.2 |
|---|---|
| Qwen3.5-35B MoE · LoRA | OOM → 3,237 MB |
| Qwen3-4B dense · LoRA | 7,834 → 4,777 MB（−39%） |
| gemma-4-E4B · LoRA（圖片、NVMe offload） | 沒變 |
| SmolVLM 256M · full fine-tune | 沒變（沒有凍結參數） |

四個條件同時成立才會累積：LoRA／PEFT、gradient checkpointing、ZeRO-3 offload、單層凍結權重夠大。

## 相關

- [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md)
- [微調總覽](../training/index.md)
