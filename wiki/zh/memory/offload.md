---
title: Offload：四種不同的東西
summary: 「offload」在 AST 裡指四件事：transformers 的權重放置、vLLM 的專家權重 offload、vLLM 的 KV cache 分層、DeepSpeed 的訓練狀態。設定、時機與失敗方式都不同。
updatedAt: 2026-10-29
---

# Offload：四種不同的東西

> **看到 offload 先問「offload 什麼」。**

| # | 名稱 | offload 什麼 | 設定 | 何時 |
|---|---|---|---|---|
| 1 | `device_map` | **權重**（整層放 GPU／CPU／磁碟） | `device_map`、`max_memory`、`offload_folder` | transformers 推論 |
| 2 | vLLM 權重 offload | **權重**，可只限 MoE 專家 | `vllm_cpu_offload_gb`、`vllm_cpu_offload_params` | vLLM 推論 |
| 3 | vLLM KV offload | **KV cache** | `vllm_kv_offloading_size`、`vllm_kv_offloading_disk_path` | vLLM 推論 |
| 4 | DeepSpeed ZeRO-3 | **參數、optimizer state、gradient** | DeepSpeed profile、`offload_folder` | **訓練** |

llama.cpp 的 `n_cpu_moe`（`-ncmoe`）語意最接近第 2 種；Strata 則是架構上就把專家放 DRAM。

## 1. `device_map`（transformers）

| 值 | 結果 |
|---|---|
| `cuda:0` | 全放 GPU |
| `auto` | accelerate 分配；**量化且有 `max_memory` 時**會先在 meta device 量結構再依量化後的預算推導 |
| `cpu` | 全放 CPU |
| 省略 | **落在 CPU**，不是 GPU |

詳見 [transformers](../engines/transformers.md)。

## 2. vLLM 的權重 offload

| 欄位 | 作用 |
|---|---|
| `vllm_cpu_offload_gb` | **預算**：多少 GB 權重放 pinned CPU memory，每次 forward 經 PCIe 讀取 |
| `vllm_cpu_offload_params` | **範圍**：只 offload 名稱含這些段的參數 |

**`["experts"]` 是比顯卡大的 MoE 能不能載起來的關鍵**：專家放 CPU、attention 留 GPU。
不給範圍會照順序搬任何權重，連 attention 一起，整體變慢。
比對的是整段名稱：`experts` 匹配 `mlp.experts.w13_qweight`，`expert` 不匹配。
只給範圍而預算是 0 會被拒絕。

> 實測：專家 offload 有 UVA 與 Prefetch 兩種方式；在 PCIe Gen5 x8 的 16 GB 卡上，
> UVA 的 prefill 831 tok/s、decode 17.8 tok/s，Prefetch 是 2,000 與 **0.76** tok/s。**用 UVA。**
> 見 [vLLM 原生 KV cache offload](../benchmarks/vllm-kv-offload.md)。

## 3. vLLM 的 KV offload：兩段式，DRAM 層是閘門

```
GPU ──逐出──► DRAM 層（vllm_kv_offloading_size，GB）──► NVMe 層（vllm_kv_offloading_disk_path）
```

- **順序不能跳**：NVMe 上的 block 必須先複製進 DRAM 層，GPU 才能用。
  DRAM 層要放得下「最長要重用的前綴」，否則讀了也命中 0%。
- 多 GPU 時 `vllm_kv_offloading_size` 是**所有 GPU 的總和**。
- 開 KV offload 會連帶開 prefix caching；**prefix caching 必須開著**，否則 offload 靜默失效。
- **NVMe 層不會自己刪檔**。用 `vllm_kv_offloading_disk_max_gb` 設上限，
  或呼叫 `POST /inference/kv_cache/disk/cleanup`。
- **offload 省的是重算時間，不是同時服務人數**：正在計算的請求，KV 仍然必須全在 GPU。

實測：滿 context 146K 的對話被擠出 GPU 後重訪，從 444.9 秒變 6.04 秒。
見 [vLLM 原生 KV cache offload](../benchmarks/vllm-kv-offload.md)。

## 4. DeepSpeed 的訓練 offload

訓練時把參數、optimizer state、gradient 分片，並 offload 到 CPU 或 NVMe。
內建五個 profile，命名是 `<參數放哪>_<optimizer 放哪>`。

| 要注意的 | 說明 |
|---|---|
| NVMe offload 吃很多 DRAM | **是真的 RSS，不是 page cache、也不是洩漏**：swap buffer 加 fp32 staging。旋鈕是 `DEEPSPEED_NVME_PINNED_BUDGET_GB` |
| 多模態 full fine-tune 在 16 GB 放不下 | backward 時 OOM；**DRAM 夠但 VRAM 不夠，CPU offload 救不了**。用 LoRA |

詳見 [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md)。

## 相關

- [記憶體估算](estimation.md)
- [推論引擎怎麼選](../engines/index.md)
- [術語表](../glossary.md)
