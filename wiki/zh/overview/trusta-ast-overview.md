---
title: TRUSTA-AST 概述
summary: TRUSTA-AST 是一個高性能的 LLM 推理服務後端，提供統一的 OpenAI API 接口和多種推理引擎支援。
kind: overview
sources:
  - wiki/entities/trusta-ast-inference-service.md
  - wiki/overview.md
  - wiki/sources/inference_manual.md
createdAt: "2026-05-28"
updatedAt: "2026-05-28"
confidence: high
provenanceState: merged
---

# TRUSTA-AST 概述

**TRUSTA-AST** 是一個**高性能 LLM 推理服務後端**，專為生產環境設計，提供統一的 OpenAI API 接口和多種推理引擎支援。

## 🎯 核心定位

TRUSTA-AST 的主要目標是：
1. **統一接口**：提供 OpenAI 標準 API，簡化客戶端集成
2. **穩定性**：透過多進程隔離架構確保服務高可用性
3. **靈活性**：支援多種推理引擎，適應不同硬體和需求
4. **生產級**：專為高併發、高可靠的生產環境設計

## 🏗️ 核心架構特點

### 1. 多進程隔離 (Multi-Process Isolation)

```
┌─────────────────────────────────────────────────┐
│          Main Process (FastAPI)                 │
│  ├─ HTTP 請求處理                                │
│  ├─ 路由與認證                                   │
│  └─ Session 管理                                 │
└─────────────┬───────────────────────────────────┘
              │ IPC Queue
┌─────────────▼───────────────────────────────────┐
│         Worker Process (Inference)              │
│  ├─ 載入模型加載                                │
│  ├─ VRAM/CPU 記憶體管理                          │
│  └─ 執行推理計算                                │
└─────────────────────────────────────────────────┘
```

**優點：**
- ✅ **穩定性**：Worker 的 OOM 錯誤不會影響主服務
- ✅ **錯誤隔離**：Worker 失敗時主服務仍存活，狀態轉為 `error`；復原為手動（重新 `load_model` 或 `force_cleanup_gpu`）
- ✅ **記憶體管理**：精確控制 VRAM 使用
- ✅ **避免 GIL 問題**：推理任務不阻塞 HTTP 事件迴路

### 2. 推理引擎無關性 (Engine Agnostic)

TRUSTA-AST 支援三種主要推理引擎：

| 引擎 | 適用場景 | 特點 |
|------|---------|------|
| **Hugging Face Transformers** | 一般用途、CPU 離載、低資源 | 默認引擎，支援 int4/int8 量化 |
| **vLLM** | 高併發生產環境 | PagedAttention，超高吞吐量 |
| **llama-server (GGUF)** | 高效 CPU/GPU 推理 | 量化模型，快速啟動 |

### 3. OpenAI API 兼容性

完全遵循 OpenAI API 標準：
- **主要端點**：`POST /v1/chat/completions`
- **支援功能**：
  - SSE 串流回應
  - Session ID 上下文維護
  - Tool Calling (Function Calling)
  - 多模態輸入（取決於引擎支援）

## 🔧 核心組件

### 1. FastAPI Server
- 處理 HTTP 請求、認證、路由
- 不直接執行重型推理任務
- 保持服務穩定性

### 2. ModelManager
- 單例模式協調器
- 管理 API 層與 Worker 進程的通訊
- 控制模型生命週期

### 3. Worker Process
- 獨立的隔離子進程
- 負載模型加載、VRAM 管理、推理執行
- 可安全終止和重啟

### 4. SessionManager
- 管理對話狀態
- 支援多輪對話
- 持久化到 Redis 或記憶體儲存

## 📊 關鍵功能

### 模型管理
- `POST /inference/load_model` - 非同步載入模型
- `POST /inference/unload_model` - 同步卸載模型
- `POST /inference/estimate_memory` - 預估 VRAM 需求（不需載入）
- `POST /inference/force_cleanup_gpu` - 強制清理 GPU 記憶體

### 記憶體管理
- `POST /inference/cleanup_generation_memory` - 清理 KV Cache
- 自動 OOM 檢測（復原為手動）
- 精確的 VRAM 使用監控

### 模型特定優化
- **Qwen3.5 模型專用優化**：
  - 自動強制 `temperature=1.0`
  - 自動強制 `top_p=0.95`
  - 自動禁用 `enable_thinking`
  - 防止生成 artifact

## 🚀 部署場景

### 高吞吐量場景
- 使用 vLLM + 張量平行
- 適合大型模型生產部署
- 支援多 GPU 配置

### 資源受限場景
- 使用 Transformers + CPU Offload
- 或使用 GGUF 模型 + llama-server
- 適合 CPU 推理或有限硬體

### 部署彈性
- 單一服務同一時間載入**一個**模型（載入第二個會回 HTTP 409）
- 可依需求選擇不同引擎（Transformers / vLLM / llama-server）
- 同一時間載入一個模型；切換模型需先卸載再載入

## 📈 技術亮點

### 1. Model-Aware Parameter Adjustment
自動檢測模型類型並調整生成參數：
```python
if "qwen3.5" in model_name.lower():
    params = {
        "temperature": 1.0,      # 穩定輸出分佈
        "top_p": 0.95,          # 最佳 token 選擇
        "top_k": 20,            # 限制詞彙採樣
        "repetition_penalty": 1.0,
        "enable_thinking": False # 防止 artifact
    }
```

### 2. 記憶體預估工具
- 不需實際載入模型即可預估 VRAM 需求
- 避免 OOM 錯誤
- 幫助容量規劃

### 3. Session 管理
- 支援多輪對話上下文
- Redis 持久化儲存
- 自動清理舊會話

## 🔍 與其他系統的比較

### vs 傳統 RAG
| 方面 | TRUSTA-AST | 傳統 RAG |
|------|-----------|---------|
| 知識管理 | 編譯式知識庫 | 檢索式知識庫 |
| 累積性 | 知識會累積 | 每次重新檢索 |
| 結構化 | 結構化 wiki | 非結構化 |
| 查詢速度 | 快（已編譯） | 慢（每次檢索） |

### vs 單進程架構
| 方面 | TRUSTA-AST | 單進程 |
|------|-----------|--------|
| 穩定性 | 高（隔離） | 低（全 crash） |
| 錯誤隔離 | 主服務不受 worker crash 影響 | 全服務 crash |
| 記憶體管理 | 精確 | 困難 |
| 並發能力 | 高 | 受限 |

## 📚 相關文件

- [[Trusta AST Inference Service]] - 詳細服務說明
- [[Multi-Process Isolation]] - 架構模式詳解
- [[OpenAI API Compatibility]] - API 兼容性說明
- [[Qwen3.5 Inference Optimization]] - Qwen3.5 專用優化
- [[Inference Engine Comparison]] - 引擎比較

## 💡 使用建議

### 選擇推理引擎
- **需要最大兼容性** → Transformers
- **需要最高吞吐量** → vLLM
- **需要高效能/低資源** → llama-server (GGUF)

### 最佳實踐
1. 使用 `estimate_memory` 端點預估 VRAM 需求
2. 定期清理 KV Cache 避免記憶體洩漏
3. 為 Qwen3.5 模型依賴自動參數調整
4. 使用 Session ID 維護對話上下文
5. 監控 Worker 進程健康狀態

## 🎓 學習資源

- **官方手冊**：`wiki/sources/inference_manual.md`
- **API 文件**：OpenAI API 標準文件
- **引擎文檔**：
  - vLLM: https://docs.vllm.ai
  - llama.cpp: https://github.com/ggerganov/llama.cpp
  - Transformers: https://huggingface.co/docs

---

**最後更新**：2026-05-28
**知識庫版本**：1.0
**相關概念**：[[LLM 推理]]、[[生產級部署]]、[[API 標準化]]
