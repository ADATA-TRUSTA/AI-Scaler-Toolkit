---
type: source
title: Trusta AST 微調服務手冊
created: 2026-04-29
updated: 2026-04-29
tags: [finetuning, llm, api, deepspeed, lora, qlora, gguf]
related: [training-manager, multi-process-isolation, deepspeed-offload-profiles, dataset-format-specifications, trusta-ast-backend]
sources: ["finetune_manual.md"]
---
# Trusta AST 微調服務手冊

本文件作為 **Trusta AST Fine-tune Service** 的技術手冊，這是一套可用於生產環境的大型語言模型（LLM）微調管線。它將 VRAM 管理、格式轉換與損失遮罩（loss masking）等複雜操作，抽象化於一個簡單的 HTTP API 之後。

## 主要能力
- **訓練方法**：支援 LoRA、QLoRA（4-bit 量化）與全參數微調（Full Parameter Fine-tuning）。
- **架構**：採用多行程隔離（multi-process isolation）架構，訓練在獨立的 Worker Process 中執行，以防止 API 服務當機（OOM、掛起）。
- **訓練後處理**：自動將訓練完成的 Hugging Face 權重轉換為 `GGUF Q4_K_M` 量化格式，以便立即使用 `llama-server` 部署。
- **DeepSpeed 整合**：內建 ZeRO-3 Offload 設定檔，讓在 VRAM 有限的硬體上進行訓練成為可能。
- **資料集支援**：可處理單欄位（預訓練）與雙欄位（具備僅完成損失的指令微調）兩種 JSONL 格式。

## 核心架構
本服務採用一個 `TrainingManager`（單例，Singleton）來管理工作階段狀態，並將工作委派給 `TrainingProcessManager`。此管理器會產生（spawn）隔離的 `Worker Processes` 來執行訓練，確保訓練迴圈中的任何失敗都不會影響主 FastAPI 伺服器。

## API 總覽
本服務在 `/training/` 前綴下公開一組 REST API，包含用於啟動訓練、檢查狀態、擷取損失歷史，以及強制清理 GPU 資源的端點。

## 使用方式
請參閱 [[Training Configuration Guide]] 以取得詳細的參數說明，並參閱 [[Dataset Format Specifications]] 以了解資料準備需求。

---FILE: wiki/entities/trusta-ast-backend.md---
---
type: entity
title: Trusta AST Backend
created: 2026-04-29
updated: 2026-04-29
tags: [backend, architecture, trusta, ast]
related: [finetune-manual, inference-service, model-registry]
sources: ["finetune_manual.md"]
---
# Trusta AST Backend

**Trusta AST Backend** 是承載微調服務（Fine-tune Service）與推論服務（Inference Service）的中央系統。它的設計旨在為大型語言模型的操作提供一個穩健、生產級的環境。

## 主要元件
- **Fine-tune Service**：管理 LLM 的訓練生命週期。
- **Inference Service**：處理模型服務與推論請求。
- **Model Registry**：一個集中式註冊表（`models_registry.json`），追蹤基礎模型與微調模型。

## 架構理念
本後端高度仰賴 **Multi-Process Isolation**（多行程隔離）。微調服務與推論服務都在隔離的 Worker Process 中執行其繁重的運算任務。此設計確保：
1. 訓練當機（例如記憶體不足 Out of Memory）不會使 API 伺服器停擺。
2. 行程結束時 VRAM 會被乾淨地釋放。
3. 並行操作可被安全地管理。

如需微調服務能力的詳細文件，請參閱 [[finetune_manual]] 來源。

---FILE: wiki/entities/training-manager.md---
---
type: entity
title: Training Manager
created: 2026-04-29
updated: 2026-04-29
tags: [software-component, singleton, fastapi, concurrency]
related: [training-process-manager, multi-process-isolation, trusta-ast-backend]
sources: ["finetune_manual.md"]
---
# Training Manager

**Training Manager** 是一個位於 `service/training_manager.py` 的單例（Singleton）軟體元件。它作為微調服務的核心邏輯樞紐與門面（facade）。

## 職責
- **工作階段管理**：追蹤進行中訓練工作階段的狀態。
- **並行控制**：使用鎖（locks）來防止可能導致資源衝突的同時訓練啟動。
- **委派**：將實際的訓練執行委派給 [[Training Process Manager]]。

## API 門面
本管理器向 FastAPI 伺服器（`/training/*` 端點）公開多個方法，包含：
- `start_training()`：啟動一個新的工作階段。
- `get_status()`：回傳目前的進度與損失指標。
- `stop_training()`：強制終止 worker 行程。
- `get_history()`：擷取訓練日誌。

它確保主 API 行程保持輕量且反應迅速，而繁重的運算則在背景進行。

---FILE: wiki/concepts/multi-process-isolation.md---
---
type: concept
title: Multi-Process Isolation Architecture
created: 2026-04-29
updated: 2026-04-29
tags: [architecture, reliability, ipc, worker-process]
related: [training-manager, inference-service, trusta-ast-backend]
sources: ["finetune_manual.md"]
---
# Multi-Process Isolation Architecture（多行程隔離架構）

**Multi-Process Isolation** 是 Trusta AST Backend（特別是在 [[Fine-tune Service]] 與 [[Inference Service]] 中）所使用的核心架構模式，用以確保可靠性與資源管理。

## 定義
此模式將 **Main Process**（承載 FastAPI 伺服器）與 **Worker Process**（執行如訓練或推論等繁重任務）分離。兩者之間的通訊透過行程間通訊（Inter-Process Communication，IPC）佇列進行。

## 益處
1. **當機隔離**：如果訓練任務當機或遇到記憶體不足（OOM）錯誤，它只會終止 Worker Process。主 API 伺服器會持續執行並接受請求。
2. **VRAM 管理**：Worker 行程在完成、失敗或明確停止時，保證會結束（並釋放 VRAM）。這可防止 VRAM 洩漏隨時間累積。
3. **乾淨狀態**：新的訓練工作階段總是以全新的 Worker 行程啟動，確保先前執行遺留的殘餘狀態不會影響新的工作。

## 實作
Main Process 中的 [[Training Process Manager]]（TPM）管理這些 Worker Process 的生命週期，透過佇列傳送指令（啟動、停止）並接收狀態更新。

---FILE: wiki/concepts/deepspeed-offload-profiles.md---
---
type: concept
title: DeepSpeed Offload Profiles
created: 2026-04-29
updated: 2026-04-29
tags: [deepspeed, zeo-3, offload, optimization, hardware]
related: [fine-tune-service, qlo-r-a, full-parameter-training]
sources: ["finetune_manual.md"]
---
# DeepSpeed Offload Profiles（DeepSpeed 卸載設定檔）

**DeepSpeed Offload Profiles** 是針對 [[DeepSpeed]] ZeRO-3 最佳化器的預先組態設定，會將最佳化器狀態（optimizer states）與模型參數卸載至 CPU RAM 或 NVMe Disk。此技術可降低 VRAM 佔用量，讓在消費級硬體或較小型 GPU 上進行訓練成為可能。

## 可用設定檔
系統提供四種內建設定檔：

| 設定檔名稱 | 最佳化器卸載 | 參數卸載 | 使用情境 |
| :--- | :--- | :--- | :--- |
| `zero3_offload_cpu_cpu` | CPU RAM | CPU RAM | 標準的低 VRAM 訓練；需要 >64GB RAM。 |
| `zero3_offload_cpu_disk` | CPU RAM | NVMe Disk | 當 RAM 有限但有可用的高速 NVMe SSD 時。 |
| `zero3_offload_disk_cpu` | NVMe Disk | CPU RAM | 特殊情境。 |
| `zero3_offload_disk_disk` | NVMe Disk | NVMe Disk | 極度低記憶體環境；需要高速 NVMe。 |

## 效能取捨
雖然這些設定檔能讓原本會因 VRAM 限制而失敗的訓練得以進行，但它們會顯著影響速度：
- **CPU Offload**：比僅使用 GPU 的訓練慢 3-5 倍。
- **Disk Offload**：比僅使用 CPU 的卸載慢 5-20 倍。

**建議**：在啟用 DeepSpeed 卸載之前，優先考慮 [[QLoRA]] 或 [[LoRA]] 方法，因為它們提供更佳的效能與資源比。

---FILE: wiki/concepts/dataset-format-specifications.md---
---
type: concept
title: Dataset Format Specifications
created: 2026-04-29
updated: 2026-04-29
tags: [data-format, jsonl, sft, pre-training, dataset]
related: [fine-tune-service, completion-only-loss, sft-strategy]
sources: ["finetune_manual.md"]
---
# Dataset Format Specifications（資料集格式規範）

Trusta AST Fine-tune Service 要求以 **JSONL**（JSON Lines）格式提供訓練資料。每一行必須代表單一的訓練範例。系統根據訓練目標的不同，支援兩種不同的欄位配置。

## 模式 1：單欄位（預訓練）
用於標準的預訓練，或需要計算完整序列損失的任務。
- **欄位**：`text`
- **損失計算**：對整個序列計算標準的交叉熵損失（cross-entropy loss）。
- **範例**：
```json
{"text": "### Question:\nHow to set Linux timezone?\n### Answer:\nUse timedatectl..."}
```

## 模式 2：雙欄位（指令微調 / SFT）
用於指令微調（SFT），以防止模型記憶提示（prompt）。
- **欄位**：`prompt` 與 `completion`
- **損失計算**：**Completion-Only Loss**（僅完成損失）。損失會針對 `prompt` 部分進行遮罩，僅在 `completion` 部分計算。
- **需求**：必須搭配 `use_sft_trainer: true` 使用，以運用 `SFTStrategy`。
- **範例**：
```json
{"prompt": "What is Docker volume?", "completion": "Docker volume is a mechanism for persistent data storage..."}
```

## 品質建議
- **最少資料量**：500-1000 筆高品質範例。
- **序列長度**：確保 95% 的範例能容納於 `max_seq_length` 之內。
- **清理**：移除重複項目並確保為 UTF-8 編碼。
- **一致性**：在整個資料集中維持一致的格式（例如系統提示）。

---FILE: wiki/concepts/training-configuration-guide.md---
---
type: concept
title: Training Configuration Guide
created: 2026-04-29
updated: 2026-04-29
tags: [configuration, parameters, lora, qlora, hyperparameters]
related: [fine-tune-service, training-manager, sft-strategy]
sources: ["finetune_manual.md"]
---
# Training Configuration Guide（訓練組態指南）

[[Fine-tune Service]] 接受一個 JSON `TrainingConfig` 物件來定義訓練工作。以下是可用參數與建議設定的完整指南。

## 必填參數
- `model_name`：HF 模型 ID 或註冊表標籤（例如 `Qwen/Qwen3-4B`）。
- `method`：`lora`、`qlora` 或 `full` 其中之一。
- `dataset_path`：JSONL 資料集檔案的路徑。
- `output_dir`：結果的目標目錄（必須為空）。

## LoRA/QLoRA 參數
- `lora_r`：LoRA 適配器（adapters）的秩（rank）（例如 16）。秩越高 = 參數越多。
- `lora_alpha`：縮放因子（通常為 `lora_r * 2`）。
- `lora_dropout`：用於正規化的 dropout 比率（例如 0.05）。
- `lora_target_modules`：要套用適配器的模組清單（例如 `["q_proj", "k_proj"]`）。

## 訓練超參數
- `num_train_epochs`：對資料集進行完整遍歷的次數。
- `per_device_train_batch_size`：每個 GPU 步驟的批次大小。
- `gradient_accumulation_steps`：在更新權重前累積梯度（有效批次 = `batch_size * accum_steps`）。
- `learning_rate`：學習率（例如 LoRA 使用 `2e-4`）。
- `max_seq_length`：最大 token 長度（會截斷較長的輸入）。

## 訓練器選擇
- `use_sft_trainer`：設為 `true` 以使用 TRL 的 `SFTTrainer`（建議用於具備僅完成損失的 SFT）。設為 `false` 則使用原生的 `CausalLMTrainer`。

## DeepSpeed 組態
- `use_deepspeed`：`true` 以啟用 ZeRO-3 卸載。
- `deepspeed_profile`：從內建設定檔中選擇（例如 `zero3_offload_cpu_cpu`）。
- `offload_folder`：若使用磁碟卸載時的目錄。

如需完整的 JSON 結構描述與範例組態，請參閱 [[finetune_manual]] 來源。

---FILE: wiki/concepts/gguf-automatic-conversion.md---
---
type: concept
title: GGUF Q4_K_M Automatic Conversion
created: 2026-04-29
updated: 2026-04-29
tags: [gguf, quantization, llama-cpp, post-training, deployment]
related: [fine-tune-service, trusta-ast-backend, qwen-llama-models]
sources: ["finetune_manual.md"]
---
# GGUF Q4_K_M Automatic Conversion（GGUF Q4_K_M 自動轉換）

Trusta AST Fine-tune Service 的一項關鍵功能是模型權重的 **自動訓練後轉換**。一旦訓練成功完成，系統會自動將 Hugging Face（HF）格式的模型轉換為 **GGUF** 格式，並量化為 **Q4_K_M**。

## 工作流程
1. **訓練完成**：模型權重以標準 HF 格式儲存至 `output_dir`。
2. **資源清理**：Worker Process 釋放所有 VRAM 與模型參照。
3. **轉換**：`conversion_manager` 腳本呼叫 `llama.cpp` 工具以：
   - 將 HF 權重轉換為 GGUF F16。
   - 將 GGUF 模型量化為 Q4_K_M（4-bit）。
4. **輸出**：最終的 `.gguf` 檔案儲存於 `output_dir`，可供 `llama-server` 或其他 GGUF 相容推論引擎使用。

## 益處
- **立即部署**：使用者無需手動執行轉換腳本。
- **最佳化大小**：Q4_K_M 量化在維持合理準確度的同時，顯著縮減模型大小（每個參數約 4 bits）。
- **更快推論**：GGUF 模型針對 CPU 與 GPU 推論進行了最佳化，通常能提供比標準 HF Transformers 更低的延遲。

## 需求
- 環境中必須安裝 `llama.cpp`。
- 需要足夠的磁碟空間（約為模型大小的 3-4 倍）以進行暫時性的 F16 轉換步驟。

---FILE: wiki/index.md---
---
type: overview
title: Wiki Index
created: 2026-04-29
updated: 2026-04-29
tags: [index, navigation]
related: []
sources: []
---
# Wiki Index（Wiki 索引）

此索引依類型列出 Trusta AST Backend wiki 的所有頁面。

## Entities（實體）
- [[trusta-ast-backend]] — 承載微調與推論服務的母系統。
- [[training-manager]] — 管理訓練工作階段的單例元件。
- [[training-process-manager]] — 銜接 API 與 Worker 行程的元件。
- [[sft-strategy]] — 訓練用的策略模式（Strategy pattern）實作。
- [[fastapi-server]] — 服務的 API 進入點。
- [[redis]] — 用於儲存訓練歷史與狀態的資料庫。
- [[llama-cpp-gguf]] — 用於訓練後轉換的工具與格式。
- [[deepspeed]] — 為大型模型啟用 ZeRO-3 卸載的函式庫。
- [[peft]] — 為 LoRA 與 QLoRA 微調提供支援的函式庫。
- [[qwen-llama-models]] — 文件中使用的範例模型。

## Concepts（概念）
- [[multi-process-isolation]] — 用於可靠性與資源管理的架構模式。
- [[deep-speed-offload-profiles]] — 用於低 VRAM 訓練的預先組態 DeepSpeed 設定。
- [[dataset-format-specifications]] — 單欄位與雙欄位模式的 JSONL 需求。
- [[training-configuration-guide]] — TrainingConfig 的詳細參數參考。
- [[gguf-automatic-conversion]] — 用於產生 GGUF Q4_K_M 的訓練後管線。
- [[completion-only-loss]] — 用於指令微調的損失遮罩技術。
- [[strategy-pattern-in-training]] — 在 SFTTrainer 與 CausalLMTrainer 之間動態選擇。

## Sources（來源）
- [[finetune-manual]] — Trusta AST Fine-tune Service 的技術手冊。

## Queries（查詢）
*目前沒有進行中的查詢。*

## Comparisons（比較）
*目前沒有進行中的比較。*

## Synthesis（綜合）
*目前沒有進行中的綜合頁面。*

---FILE: wiki/overview.md---
---
type: overview
title: Project Overview
created: 2026-04-29
updated: 2026-04-29
tags: [overview, trusta-ast, llm, finetuning]
related: []
sources: ["finetune_manual.md"]
---
# Overview（總覽）

本 wiki 記載了 **Trusta AST Backend** 的架構、組態與運作細節。此後端的設計旨在為大型語言模型（LLM）的操作提供一個穩健、可用於生產環境的環境，特別著重於微調與推論服務。系統透過 **Multi-Process Isolation Architecture**（多行程隔離架構）強調可靠性，讓繁重的運算任務在隔離的 Worker Process 中執行，以防止服務當機並確保乾淨的資源管理。

## 核心能力
**Fine-tune Service** 是後端的核心元件，提供一套精簡的 LLM 訓練管線。它支援 **LoRA**、**QLoRA** 與 **全參數微調（Full Parameter Fine-tuning）** 方法，將複雜的組態細節抽象化於一個簡單的 HTTP API 之後。一項關鍵的差異化功能是用於指令微調的 **Completion-Only Loss**（僅完成損失）機制，它確保模型學會生成答案，而不會記憶提示。此外，本服務會自動處理 **訓練後轉換（Post-Training Conversion）**，將 Hugging Face 權重轉換為 **GGUF Q4_K_M** 量化檔案，可供 `llama-server` 部署使用。

## 硬體最佳化
為了在消費級硬體或 VRAM 有限的 GPU 上進行訓練，系統整合了 **DeepSpeed ZeRO-3 Offload**。此功能提供預先組態的設定檔，將最佳化器狀態與參數卸載至 CPU RAM 或 NVMe Disk。雖然這會伴隨效能取捨（較慢的訓練速度），但它顯著降低了全參數微調的進入門檻。本 wiki 記載了這些設定檔及其對效能的影響，以協助使用者在可行性與效率之間做出明智的取捨。

## 資料與組態
系統要求以 **JSONL 格式** 提供訓練資料，支援單欄位（預訓練）與雙欄位（指令）兩種配置。系統維護了一份完整的 **Training Configuration Guide**，詳述 LoRA 秩、學習率與 DeepSpeed 設定等參數。後端也維護一個集中式的 **Model Registry**（`models_registry.json`）來追蹤基礎模型與微調模型，確保微調服務與推論服務之間的無縫整合。

---FILE: wiki/log.md---
---
type: overview
title: Wiki Activity Log
created: 2026-04-29
updated: 2026-04-29
tags: [log]
related: []
sources: []
---
# Wiki Activity Log（Wiki 活動日誌）

## 2026-04-29

- 匯入 `finetune_manual.html`
  - 為 Trusta AST Fine-tune Service Manual 建立來源摘要。
  - 新增實體：Training Manager、Trusta AST Backend。
  - 新增概念：Multi-Process Isolation、DeepSpeed Offload Profiles、Dataset Format Specifications、Training Configuration Guide、GGUF Automatic Conversion。
  - 更新 wiki 索引與總覽以反映新內容。
