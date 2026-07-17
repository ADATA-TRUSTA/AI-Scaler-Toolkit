---
title: GPU Offload Mechanism
summary: Trusta AST 透過 device_map / offload_folder / n_gpu_layers / DeepSpeed ZeRO-3 將模型或訓練狀態卸載到 DRAM/SSD，降低 GPU VRAM 需求，並整合 fine-tuning 能力。
kind: concept
sources:
  - wiki/sources/inference_manual.md
  - wiki/sources/finetune_manual.md
createdAt: "2026-05-28"
updatedAt: "2026-07-16"
confidence: medium
provenanceState: extracted
---

# GPU Offload 機制

**GPU Offload 機制** 是 Trusta AST 的核心功能之一：把放不下 GPU VRAM 的模型層或訓練狀態卸載到 CPU DRAM 或 SSD/NVMe，讓較小的 GPU 也能載入較大的模型，並在同一平台整合 fine-tuning。

## 核心機制（實際實作）

Offload 並非單一自訂模組，而是依引擎/情境使用既有框架能力：

### 1. 推論：Transformers 引擎

- 以 HuggingFace `device_map`（例如 `"auto"`）自動切分模型到 GPU / CPU；放不下的層透過 `offload_folder` 落到磁碟（SSD/NVMe）。
- 搭配 bitsandbytes 量化（`int4` / `int8` / `nf4` / `fp4`）進一步降低記憶體占用。
- 相關實作：`service/inference/engines/transformers_engine.py`。

### 2. 推論：llama-server 引擎（GGUF）

- 以 `n_gpu_layers`（`n_gpu_layers=-1` 代表全上 GPU，其餘數值代表部分層在 GPU、其餘留在 CPU）控制 GPU/CPU 分配。
- 相關實作：`service/inference/engines/llama_server_engine.py`。

### 3. 訓練：DeepSpeed ZeRO-3 Offload

- Full 參數訓練或大型模型 LoRA 時，可將 optimizer 狀態與參數卸載到 CPU RAM 或 NVMe Disk。
- 內建四種 profile（`service/configs/deepspeed/`）：
  - `zero3_offload_cpu_cpu`、`zero3_offload_cpu_disk`、`zero3_offload_disk_cpu`、`zero3_offload_disk_disk`
- 注意：Disk offload 會顯著拉長訓練時間，屬於「極低記憶體環境」的手段。

> 量化實測：VRAM 降幅與 tok/s 可用 `tests/benchmark_offload_vram.py` 實測產生，
> 結果寫入 `tests/benchmark_offload_vram_results.json`；本頁若引用數字應以該實測為準。
>
> 實測範例（RTX 5060 Ti，16GB）：Qwen3-14B bf16（權重約 27.5GB）透過 `device_map=auto` + `offload_folder` 執行，峰值 **~13.4GB GPU VRAM**（比全程放 GPU 的 ~27.5GB **少約 51%**），在大量 CPU/磁碟 offload 下約 **0.5 tok/s**——即以吞吐換取「讓原本載不進來的模型能跑」。

## 帶來的效益

- **降低 VRAM 需求**：原本需要高階大顯存 GPU 才能整包放進 VRAM 的模型，開 offload 後可在較小 VRAM 的機器上運行。
- **可運行更大模型**：受限環境仍可載入超過單卡 VRAM 的模型（代價是吞吐下降）。
- **推論與訓練共用同一套機制**：同一平台完成，不需切換工具。

> 成本效益：硬體成本節省來自「所需 GPU 級距下降」。**VRAM 降幅可實測**（見上），
> 由 VRAM 降幅換算的金額節省則屬**估算**（取決於 GPU 報價假設），文件引用時請標明為估算值。

## Fine-tuning 整合能力

Trusta AST 不只是推論服務，也在同一平台提供 fine-tuning：

- 推論與 fine-tuning 共用同一套 offload 機制與模型登錄。
- 支援的方式：LoRA / QLoRA、全量微調。
- 訓練完成可自動轉出 GGUF（Q4_K_M）供 llama-server 推論。

## 實際應用場景

- **中小模型（7B–13B）在受限硬體部署**：以量化 + device_map/offload 降低 VRAM 需求。
- **大模型於低記憶體環境**：以 `n_gpu_layers` 部分卸載或 DeepSpeed disk offload 換取「跑得起來」。
- **推論 + fine-tuning 一體工作流**：同一服務完成微調與部署。

## 相關概念

- [[Multi-Process Isolation]] - 多進程隔離架構
- [[Inference Engine Comparison]] - 推理引擎比較
- [[Trusta AST Inference Service]] - 服務整體介紹

---

**最後更新**：2026-07-16  
**相關文件**：[[Trusta AST Inference Service]]、[[Inference Engine Comparison]]
