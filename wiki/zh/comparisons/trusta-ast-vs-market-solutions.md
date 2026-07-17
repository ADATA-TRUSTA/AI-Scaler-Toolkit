---
title: Trusta AST vs 市面解決方案
summary: 對比 Trusta AST 與 Ollama、LM Studio、vLLM 等方案的定性差異，重點在 offload 機制與 fine-tuning 整合。競品數值未經本專案實測。
kind: comparison
sources:
  - wiki/sources/inference_manual.md
  - wiki/concepts/offload-mechanism.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
provenanceState: merged
---

# Trusta AST vs 市面解決方案

本文件對比 **Trusta AST** 與現有 LLM 服務方案（Ollama、LM Studio、vLLM、llama.cpp 等）的差異，重點在 **DRAM/SSD offload 機制** 與 **整合式 fine-tuning**。

> ⚠️ **重要聲明**：下方競品欄位為一般公開認知，**未在本專案的相同硬體/模型上實測**。
> 本專案只實測「自己」的數字（見 `tests/benchmark_llama_server_prefill.py`、`tests/benchmark_offload_vram.py`）。
> 因此本頁**不列出競品的 tok/s、訓練時間或成本數字**。

## 定性功能對比

| 功能特性 | Trusta AST | Ollama | LM Studio | vLLM |
|---------|-----------|--------|-----------|------|
| Offload | DRAM/SSD 混合（量化 + device_map / n_gpu_layers / DeepSpeed） | 以 GPU 為主 | 以 GPU 為主 | 以 GPU 為主 |
| Fine-tuning | ✅ 整合式（LoRA/QLoRA、全量） | ❌ 僅推論 | ❌ 僅推論 | 需額外工具（如 Unsloth/Axolotl） |
| OpenAI API | ✅ `/v1/chat/completions`、`/v1/models` | 部分相容 | 部分相容 | ✅ 相容 |
| 進程隔離 | ✅ 推論獨立 worker 進程 | 一般為單進程 | 桌面應用 | 依部署而定 |
| 同時載入模型數 | 單一模型（載入第二個回 HTTP 409） | 依實作 | 依實作 | 依部署而定 |

（競品欄僅供快速參考，實際能力請以各專案官方文件為準。）

## 核心差異

### 1. GPU Offload 機制

Trusta AST 依引擎/情境使用既有框架能力做 offload：

- 推論（Transformers）：`device_map` + `offload_folder` + bitsandbytes 量化。
- 推論（llama-server / GGUF）：`n_gpu_layers` 控制 GPU/CPU 分層。
- 訓練：DeepSpeed ZeRO-3，將 optimizer/參數卸載到 CPU RAM 或 NVMe（四種 profile）。

效益是**降低所需 GPU VRAM**，使較大的模型可在較小的 GPU 上運行；代價是吞吐量下降（尤其 disk offload）。

### 2. Fine-tuning 整合（主要差異）

```
工作流程：原始模型 → offload 載入 → 微調（LoRA/QLoRA/全量）→ 保存 → 自動轉 GGUF → 推論部署
```

- 推論與 fine-tuning 於同一平台、共用 offload 與模型登錄。
- 純推論工具（Ollama、LM Studio）多不含 fine-tuning；vLLM 則需搭配外部訓練工具。

### 3. 多進程隔離

```
主進程 (FastAPI) ←→ Worker 進程 (推論)
    ├─ Session 管理        ├─ 模型載入
    ├─ 請求路由            ├─ 推論計算
    └─ 狀態查詢            └─ 記憶體管理
```

- Worker 因 OOM 等原因結束時，主 HTTP 服務仍存活，狀態轉為 `error`。
- **復原為手動**：需再次 `/inference/load_model` 或 `/inference/force_cleanup_gpu`。

## 使用場景

**適合 Trusta AST：**
- 需要在同一平台做推論 + fine-tuning。
- GPU 資源有限，想用 offload 降低 VRAM 需求以部署較大模型。
- 需要 OpenAI 相容 API。

**較不適合（建議其他方案）：**
- 超高併發（每秒大量請求）→ vLLM + 多 GPU。
- 極低延遲需求 → 純 GPU 部署、模型完全在 VRAM。

## 效能與成本（如何取得可信數字）

- **本服務 tok/s / TTFT / 載入時間**：`tests/benchmark_llama_server_prefill.py`（結果見 `tests/benchmark_llama_server_prefill_results.json`）。
- **VRAM 降幅（offload 前後）**：`tests/benchmark_offload_vram.py`。
- **Fine-tuning 時間**：`tests/stress_tests/stress_test_finetune.py` 可量測本服務自身的 wall-clock。
- **競品對比**：需另行安裝 Ollama / 獨立 vLLM 等並在相同條件下手動實測，本專案不提供該數據。

## 結論

Trusta AST 的差異化在於**「offload 降低 VRAM 需求」+「同平台整合 fine-tuning」+「多進程隔離」**。這些是本服務確定具備的能力；與競品的量化對比若無相同條件實測，應避免給出具體數字。

---

**相關文件**：[[GPU Offload Mechanism]]、[[Trusta AST Inference Service]]、[[Inference Engine Comparison]]  
**最後更新**：2026-07-16
