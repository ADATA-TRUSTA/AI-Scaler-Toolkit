---
title: TRUSTA-AST 核心差異化
summary: Trusta AST 的關鍵差異：DRAM/SSD offload 降低 GPU VRAM 需求、在同一平台整合 fine-tuning、多進程隔離。
kind: concept
sources:
  - wiki/sources/inference_manual.md
  - concepts/offload-mechanism.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
---

# TRUSTA-AST 核心差異化

## 🎯 核心價值主張

1. **DRAM/SSD Offload 機制** — 透過 device_map / offload_folder / n_gpu_layers / DeepSpeed，將模型層或訓練狀態卸載到 DRAM/SSD，**降低所需 GPU VRAM**，使較大的模型可在較小的 GPU 上運行。
2. **整合式 Fine-tuning** — 不只是推論服務，同一平台即可完成 LoRA/QLoRA 與全量微調，並可自動轉出 GGUF 供推論。
3. **多進程隔離** — 推論在獨立 worker 進程執行，OOM 不會拖垮主 HTTP 服務（復原為手動，見下）。

## 🆚 與市面方案的差異（定性）

> 下表僅比較「本服務確定具備」的能力；競品欄為一般認知，**未在本專案實測**，請勿當作 benchmark 引用。

| 面向 | Trusta AST | 一般純推論工具（如 Ollama / LM Studio） |
|------|-----------|------------------------------------------|
| 主要功能 | 推論 + Fine-tuning（同平台） | 多為僅推論 |
| Offload | DRAM/SSD 混合（量化 + device_map / n_gpu_layers / DeepSpeed） | 多以 GPU 為主 |
| OpenAI API | `/v1/chat/completions`、`/v1/models` 相容 | 相容程度不一 |
| 進程隔離 | 推論獨立 worker 進程 | 不一定 |

## 💰 成本效益（如何理解）

成本節省的來源是「**所需 GPU 級距下降**」：開 offload 後模型可在較小 VRAM 的機器運行，因此可用較便宜的硬體。

- **VRAM 降幅（實測）**：在 RTX 5060 Ti（16GB）上，Qwen3-14B（bf16、權重約 27.5GB，單卡放不下）透過 `device_map=auto` + `offload_folder` 執行，峰值 GPU VRAM 僅 **~13.4GB**，相較「全程放 GPU 所需的 ~27.5GB」**降低約 51%**。代價是吞吐——在大量 CPU/磁碟 offload 下，生成降到 **~0.5 tok/s**，所以 offload 的重點是「讓原本跑不動的模型能跑」，不是速度。可用 `backend/tests/benchmark_offload_vram.py` 重現（原始數據：`backend/tests/benchmark_offload_vram_results.json`）。
- **金額節省**：屬**估算**，取決於 GPU 報價假設。依目前初步評估，**訓練成本約可節省 ~80%（估算，非實測金額）**；此數字僅供參考，實際依模型、硬體與使用時數而定。

## 🔑 技術重點

### 1. DRAM/SSD Offload
- 傳統：模型需整包載入 GPU VRAM，需較高階 GPU。
- Trusta AST：以量化 + device_map/offload_folder（推論）或 DeepSpeed ZeRO-3（訓練）將部分資料卸載到 DRAM/SSD，降低 VRAM 需求。

### 2. 整合 Fine-tuning
- 單一平台完成推論與微調，共用 offload 機制與模型登錄；支援 LoRA/QLoRA 與全量微調。

### 3. 多進程隔離
- 推論於獨立 worker 進程；worker 因 OOM 等原因結束時，主服務仍存活，狀態轉為 `error`。
- **復原為手動**：需再次呼叫 `/inference/load_model` 或 `/inference/force_cleanup_gpu`。

## 📊 應用場景

- **中小企業/受限預算**：以量化 + offload 在較小 GPU 上部署 7B–13B 模型。
- **開發環境**：同一平台頻繁微調與測試，快速迭代。
- **混合工作負載**：推論與 fine-tuning 於同一服務處理。

## 💡 總結

Trusta AST 的定位：**在同一平台整合「offload 降低 VRAM 需求」與「fine-tuning」的 LLM 服務**，適合需要在有限 GPU 資源下同時做推論與微調的團隊。

---

**最後更新**：2026-07-16  
**相關文件**：[[GPU Offload Mechanism]]、[[Trusta AST vs Market Solutions]]
