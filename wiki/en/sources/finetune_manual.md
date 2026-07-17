---
type: source
title: Trusta AST Fine-tune Service Manual
created: 2026-04-29
updated: 2026-04-29
tags: [finetuning, llm, api, deepspeed, lora, qlora, gguf]
related: [training-manager, multi-process-isolation, deepspeed-offload-profiles, dataset-format-specifications, trusta-ast-backend]
sources: ["finetune_manual.md"]
---
# Trusta AST Fine-tune Service Manual

This document serves as the technical manual for the **Trusta AST Fine-tune Service**, a production-ready pipeline for Large Language Model (LLM) fine-tuning. It abstracts complex operations such as VRAM management, format conversion, and loss masking behind a simple HTTP API.

## Key Capabilities
- **Training Methods**: Supports LoRA, QLoRA (4-bit quantized), and Full Parameter Fine-tuning.
- **Architecture**: Utilizes a multi-process isolation architecture where training runs in a separate Worker Process to prevent API service crashes (OOM, hangs).
- **Post-Training**: Automatically converts trained Hugging Face weights into `GGUF Q4_K_M` quantized format for immediate deployment with `llama-server`.
- **DeepSpeed Integration**: Includes built-in ZeRO-3 Offload profiles to enable training on hardware with limited VRAM.
- **Dataset Support**: Handles both single-field (pre-training) and dual-field (instruction fine-tuning with completion-only loss) JSONL formats.

## Core Architecture
The service employs a `TrainingManager` (Singleton) that manages session state and delegates to a `TrainingProcessManager`. This manager spawns isolated `Worker Processes` to execute training, ensuring that any failure in the training loop does not affect the main FastAPI server.

## API Overview
The service exposes a REST API under the `/training/` prefix, including endpoints for starting training, checking status, retrieving loss history, and force-cleaning GPU resources.

## Usage
Refer to the [[Training Configuration Guide]] for detailed parameter descriptions and the [[Dataset Format Specifications]] for data preparation requirements.

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

The **Trusta AST Backend** is the central system hosting the Fine-tune Service and Inference Service. It is designed to provide a robust, production-grade environment for Large Language Model operations.

## Key Components
- **Fine-tune Service**: Manages the training lifecycle of LLMs.
- **Inference Service**: Handles model serving and inference requests.
- **Model Registry**: A centralized registry (`models_registry.json`) that tracks both base and fine-tuned models.

## Architecture Philosophy
The backend relies heavily on **Multi-Process Isolation**. Both the Fine-tune and Inference services run their heavy computational tasks in isolated Worker Processes. This design ensures that:
1. Training crashes (e.g., Out of Memory) do not take down the API server.
2. VRAM is released cleanly upon process exit.
3. Concurrent operations can be managed safely.

For detailed documentation on the Fine-tune service capabilities, see the [[finetune_manual]] source.

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

The **Training Manager** is a Singleton software component located in `service/training_manager.py`. It acts as the core logic hub and facade for the Fine-tune Service.

## Responsibilities
- **Session Management**: Tracks the state of active training sessions.
- **Concurrency Control**: Uses locks to prevent simultaneous training starts that could lead to resource conflicts.
- **Delegation**: Delegates the actual training execution to the [[Training Process Manager]].

## API Facade
The manager exposes methods to the FastAPI server (`/training/*` endpoints), including:
- `start_training()`: Initiates a new session.
- `get_status()`: Returns current progress and loss metrics.
- `stop_training()`: Forcefully terminates the worker process.
- `get_history()`: Retrieves training logs.

It ensures that the main API process remains lightweight and responsive while heavy computation happens in the background.

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
# Multi-Process Isolation Architecture

**Multi-Process Isolation** is a core architectural pattern used by the Trusta AST Backend (specifically in the [[Fine-tune Service]] and [[Inference Service]]) to ensure reliability and resource management.

## Definition
This pattern separates the **Main Process** (hosting the FastAPI server) from the **Worker Process** (executing heavy tasks like training or inference). Communication between them occurs via Inter-Process Communication (IPC) Queues.

## Benefits
1. **Crash Isolation**: If a training task crashes or encounters an Out-of-Memory (OOM) error, it only kills the Worker Process. The main API server continues to run and accept requests.
2. **VRAM Management**: Worker processes are guaranteed to exit (and release VRAM) upon completion, failure, or explicit stop. This prevents VRAM leaks from accumulating over time.
3. **Clean State**: A new training session always starts with a fresh Worker process, ensuring no residual state from previous runs affects the new job.

## Implementation
The [[Training Process Manager]] (TPM) in the Main Process manages the lifecycle of these Worker Processes, sending commands (start, stop) and receiving status updates via queues.

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
# DeepSpeed Offload Profiles

**DeepSpeed Offload Profiles** are pre-configured settings for the [[DeepSpeed]] ZeRO-3 optimizer that offload optimizer states and model parameters to CPU RAM or NVMe Disk. This technique reduces the VRAM footprint, enabling training on consumer-grade hardware or smaller GPUs.

## Available Profiles
The system provides four built-in profiles:

| Profile Name | Optimizer Offload | Parameter Offload | Use Case |
| :--- | :--- | :--- | :--- |
| `zero3_offload_cpu_cpu` | CPU RAM | CPU RAM | Standard low-VRAM training; requires >64GB RAM. |
| `zero3_offload_cpu_disk` | CPU RAM | NVMe Disk | When RAM is limited but fast NVMe SSD is available. |
| `zero3_offload_disk_cpu` | NVMe Disk | CPU RAM | Specialized scenarios. |
| `zero3_offload_disk_disk` | NVMe Disk | NVMe Disk | Extreme low-memory environments; requires high-speed NVMe. |

## Performance Trade-offs
While these profiles enable training that would otherwise fail due to VRAM constraints, they significantly impact speed:
- **CPU Offload**: 3-5x slower than GPU-only training.
- **Disk Offload**: 5-20x slower than CPU-only offload.

**Recommendation**: Prioritize [[QLoRA]] or [[LoRA]] methods before enabling DeepSpeed offload, as they offer better performance-to-resource ratios.

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
# Dataset Format Specifications

The Trusta AST Fine-tune Service requires training data to be provided in **JSONL** (JSON Lines) format. Each line must represent a single training example. The system supports two distinct field configurations depending on the training goal.

## Mode 1: Single-Field (Pre-training)
Used for standard pre-training or tasks requiring full sequence loss calculation.
- **Field**: `text`
- **Loss Calculation**: Standard cross-entropy loss over the entire sequence.
- **Example**:
```json
{"text": "### Question:\nHow to set Linux timezone?\n### Answer:\nUse timedatectl..."}
```

## Mode 2: Dual-Field (Instruction Fine-tuning / SFT)
Used for Instruction Fine-tuning (SFT) to prevent the model from memorizing the prompt.
- **Fields**: `prompt` and `completion`
- **Loss Calculation**: **Completion-Only Loss**. The loss is masked for the `prompt` part and calculated only on the `completion` part.
- **Requirement**: Must be used with `use_sft_trainer: true` to leverage the `SFTStrategy`.
- **Example**:
```json
{"prompt": "What is Docker volume?", "completion": "Docker volume is a mechanism for persistent data storage..."}
```

## Quality Recommendations
- **Minimum Data**: 500-1000 high-quality examples.
- **Sequence Length**: Ensure 95% of examples fit within `max_seq_length`.
- **Cleaning**: Remove duplicates and ensure UTF-8 encoding.
- **Consistency**: Maintain consistent formatting (e.g., system prompts) across the dataset.

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
# Training Configuration Guide

The [[Fine-tune Service]] accepts a JSON `TrainingConfig` object to define the training job. Below is a comprehensive guide to the available parameters and recommended settings.

## Required Parameters
- `model_name`: The HF model ID or registry label (e.g., `Qwen/Qwen3-4B`).
- `method`: One of `lora`, `qlora`, or `full`.
- `dataset_path`: Path to the JSONL dataset file.
- `output_dir`: Target directory for results (must be empty).

## LoRA/QLoRA Parameters
- `lora_r`: Rank of the LoRA adapters (e.g., 16). Higher rank = more parameters.
- `lora_alpha`: Scaling factor (typically `lora_r * 2`).
- `lora_dropout`: Dropout rate for regularization (e.g., 0.05).
- `lora_target_modules`: List of modules to apply adapters to (e.g., `["q_proj", "k_proj"]`).

## Training Hyperparameters
- `num_train_epochs`: Number of full passes over the dataset.
- `per_device_train_batch_size`: Batch size per GPU step.
- `gradient_accumulation_steps`: Accumulates gradients before updating weights (effective batch = `batch_size * accum_steps`).
- `learning_rate`: Learning rate (e.g., `2e-4` for LoRA).
- `max_seq_length`: Maximum token length (truncates longer inputs).

## Trainer Selection
- `use_sft_trainer`: Set to `true` to use the TRL `SFTTrainer` (recommended for SFT with completion-only loss). Set to `false` for the native `CausalLMTrainer`.

## DeepSpeed Configuration
- `use_deepspeed`: `true` to enable ZeRO-3 offloading.
- `deepspeed_profile`: Select from built-in profiles (e.g., `zero3_offload_cpu_cpu`).
- `offload_folder`: Directory for disk offloading if used.

See the [[finetune_manual]] source for the full JSON schema and example configurations.

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
# GGUF Q4_K_M Automatic Conversion

A key feature of the Trusta AST Fine-tune Service is the **automatic post-training conversion** of model weights. Once training completes successfully, the system automatically converts the Hugging Face (HF) format model into the **GGUF** format, quantized to **Q4_K_M**.

## Workflow
1. **Training Completion**: The model weights are saved to the `output_dir` in standard HF format.
2. **Resource Cleanup**: The Worker Process releases all VRAM and model references.
3. **Conversion**: The `conversion_manager` script invokes `llama.cpp` tools to:
   - Convert HF weights to GGUF F16.
   - Quantize the GGUF model to Q4_K_M (4-bit).
4. **Output**: The final `.gguf` file is saved in the `output_dir`, ready for use with `llama-server` or other GGUF-compatible inference engines.

## Benefits
- **Immediate Deployment**: Users do not need to manually run conversion scripts.
- **Optimized Size**: Q4_K_M quantization significantly reduces model size (approx. 4 bits per parameter) while maintaining reasonable accuracy.
- **Faster Inference**: GGUF models are optimized for CPU and GPU inference, often providing lower latency than standard HF Transformers.

## Requirements
- `llama.cpp` must be installed in the environment.
- Sufficient disk space is required (approx. 3-4x the model size) for the temporary F16 conversion step.

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
# Wiki Index

This index lists all pages organized by type for the Trusta AST Backend wiki.

## Entities
- [[trusta-ast-backend]] — The parent system hosting Fine-tune and Inference services.
- [[training-manager]] — Singleton component managing training sessions.
- [[training-process-manager]] — Component bridging API and Worker processes.
- [[sft-strategy]] — Strategy pattern implementation for training.
- [[fastapi-server]] — API entry point for the service.
- [[redis]] — Database for storing training history and status.
- [[llama-cpp-gguf]] — Tool and format used for post-training conversion.
- [[deepspeed]] — Library enabling ZeRO-3 offload for large models.
- [[peft]] — Library enabling LoRA and QLoRA fine-tuning.
- [[qwen-llama-models]] — Example models used in the documentation.

## Concepts
- [[multi-process-isolation]] — Architecture pattern for reliability and resource management.
- [[deep-speed-offload-profiles]] — Pre-configured DeepSpeed settings for low VRAM training.
- [[dataset-format-specifications]] — JSONL requirements for Single and Dual-field modes.
- [[training-configuration-guide]] — Detailed parameter reference for TrainingConfig.
- [[gguf-automatic-conversion]] — Post-training pipeline for GGUF Q4_K_M generation.
- [[completion-only-loss]] — Loss masking technique for instruction fine-tuning.
- [[strategy-pattern-in-training]] — Dynamic selection between SFTTrainer and CausalLMTrainer.

## Sources
- [[finetune-manual]] — Technical manual for the Trusta AST Fine-tune Service.

## Queries
*No queries currently active.*

## Comparisons
*No comparisons currently active.*

## Synthesis
*No synthesis pages currently active.*

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
# Overview

This wiki documents the architecture, configuration, and operational details of the **Trusta AST Backend**. The backend is designed to provide a robust, production-ready environment for Large Language Model (LLM) operations, specifically focusing on fine-tuning and inference services. The system emphasizes reliability through a **Multi-Process Isolation Architecture**, where heavy computational tasks run in isolated Worker Processes to prevent service crashes and ensure clean resource management.

## Core Capabilities
The **Fine-tune Service** is a central component of the backend, offering a streamlined pipeline for LLM training. It supports **LoRA**, **QLoRA**, and **Full Parameter Fine-tuning** methods, abstracting complex configuration details behind a simple HTTP API. A key differentiator is the **Completion-Only Loss** mechanism for Instruction Fine-tuning, which ensures models learn to generate answers without memorizing prompts. Additionally, the service automatically handles **Post-Training Conversion**, transforming Hugging Face weights into **GGUF Q4_K_M** quantized files, ready for deployment with `llama-server`.

## Hardware Optimization
To enable training on consumer-grade hardware or GPUs with limited VRAM, the system integrates **DeepSpeed ZeRO-3 Offload**. This feature provides pre-configured profiles that offload optimizer states and parameters to CPU RAM or NVMe Disk. While this comes with a performance trade-off (slower training speeds), it significantly lowers the barrier to entry for full parameter fine-tuning. The wiki documents these profiles and their impact on performance to help users make informed trade-offs between feasibility and efficiency.

## Data and Configuration
The system requires training data in **JSONL format**, supporting both single-field (pre-training) and dual-field (instruction) configurations. A comprehensive **Training Configuration Guide** is maintained, detailing parameters for LoRA rank, learning rates, and DeepSpeed settings. The backend also maintains a centralized **Model Registry** (`models_registry.json`) to track base and fine-tuned models, ensuring seamless integration between the Fine-tune and Inference services.

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
# Wiki Activity Log

## 2026-04-29

- Ingested `finetune_manual.html`
  - Created source summary for Trusta AST Fine-tune Service Manual.
  - Added entities: Training Manager, Trusta AST Backend.
  - Added concepts: Multi-Process Isolation, DeepSpeed Offload Profiles, Dataset Format Specifications, Training Configuration Guide, GGUF Automatic Conversion.
  - Updated wiki index and overview to reflect new content.
