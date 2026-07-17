# Project Purpose

## Goal

以結構化 wiki 記錄並解釋 **Trusta AST（AI Scaler Toolkit）** 這個以 FastAPI 為核心的 LLM 推理／訓練服務：架構、推理引擎、OpenAI 相容 API、offload 機制與 fine-tuning 能力；並連結 **TRUSTA 企業級 SSD 產品線**（offload 到 SSD/DRAM 的硬體對應）。

## Key Questions

1. 這個服務如何用 offload（device_map / n_gpu_layers / DeepSpeed）降低所需 GPU VRAM，讓較大的模型在較小 GPU 上運行？
2. 三種推理引擎（Transformers / vLLM / llama-server）各自的定位與取捨為何？
3. 服務如何在同一平台整合推理與 fine-tuning，並保持多進程隔離的穩定性？

## Scope

**In scope:**
- Trusta AST 服務的架構、API、引擎、offload、fine-tuning（來源：`docs/` 手冊與 `service/` 程式碼）
- TRUSTA SSD 產品線概述（offload 硬體對應）
- 建立本 wiki 所用的 LLM Wiki 方法論／格式轉換

**Out of scope:**
- 競品（Ollama / 獨立 vLLM 等）的效能／成本實測數字（本專案未在相同條件下量測）
- 未實作的功能（如 worker 自動重啟、同時多模型、動態負載平衡）

## Thesis

> Trusta AST 的核心價值在於「以 offload 降低 GPU VRAM 需求」＋「同平台整合 fine-tuning」＋「多進程隔離」。效能與 VRAM 降幅應以本專案的實測腳本為準；成本節省為估算，需明確標示。
