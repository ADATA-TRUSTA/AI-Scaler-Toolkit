# TRUSTA-AST 是什麼？

## 📝 簡明回答

**TRUSTA-AST** 是一個**生產級的高性能 LLM 推理服務後端**，它：

1. **提供統一的 OpenAI API 接口** - 讓客戶端可以像使用 OpenAI API 一樣使用
2. **支援多種推理引擎** - Transformers、vLLM、llama-server (GGUF)
3. **確保服務穩定性** - 透過多進程隔離架構
4. **適應不同硬體需求** - 從 CPU 到多 GPU 配置都能支援

---

## 🎯 核心價值

### 對開發者
- ✅ 熟悉的 OpenAI API，學習成本低
- ✅ 支援各種模型，不綁定單一引擎
- ✅ 統一的接口，不需要為不同引擎寫不同代碼

### 對運營者
- ✅ 高可用性，Worker 失敗不會導致整個服務掛掉
- ✅ 精確的記憶體管理，避免 OOM
- ✅ 靈活的部署選項，適應不同硬體

### 對最終用戶
- ✅ 穩定的服務體驗
- ✅ 快速的回應速度
- ✅ 支援多輪對話上下文

---

## 🏗️ 核心架構（圖解）

```
用戶請求
    │
    ▼
┌─────────────────────────────┐
│  FastAPI 伺服器 (主進程)     │
│  - 處理 HTTP 請求              │
│  - 認證與授權                 │
│  - Session 管理               │
└──────────┬──────────────────┘
           │ IPC Queue
           ▼
┌─────────────────────────────┐
│  Worker 進程 (推理進程)        │
│  - 載入模型                   │
│  - 管理 VRAM                  │
│  - 執行推理計算               │
└─────────────────────────────┘
```

**關鍵設計**：主進程和 Worker 進程完全隔離，Worker 的錯誤（如 OOM）不會影響主進程。

---

## 🔧 三大推理引擎

### 1. Hugging Face Transformers
```yaml
engine: transformers
適用場景：
  - 一般用途
  - CPU 離載
  - 低資源環境
特點：
  - 支援 int4/int8 量化
  - CPU/GPU 靈活配置
  - 默認引擎
```

### 2. vLLM
```yaml
engine: vllm
適用場景：
  - 高併發生產環境
  - 多 GPU 配置
  - 需要超高吞吐量
特點：
  - PagedAttention 技術
  - 原生平行請求
  - 支援 AWQ/GPTQ/FP8 量化
```

### 3. llama-server (GGUF)
```yaml
engine: llama_server
適用場景：
  - 高效 CPU/GPU 推理
  - 資源受限環境
  - 快速啟動需求
特點：
  - GGUF 量化模型
  - llama.cpp 基礎
  - 啟動速度快
```

---

## 📊 主要功能

### API 端點
```
POST /v1/chat/completions  # 主要聊天端點 (OpenAI 標準)
POST /inference/load_model # 載入模型
POST /inference/unload_model # 卸載模型
GET  /inference/status     # 檢查模型狀態
POST /inference/estimate_memory # 預估 VRAM 需求
```

### 特色功能
- **串流回應**：SSE 即時 token 輸出
- **Session 管理**：多輪對話上下文
- **Tool Calling**：函數呼叫支援
- **記憶體管理**：自動清理 KV Cache
- **模型優化**：Qwen3.5 自動參數調整

---

## 💡 實際應用場景

### 場景 1：高併發客服聊天機器人
```
使用 vLLM + 多 GPU
- 處理每秒數百個請求
- 保持低延遲
- 自動擴展
```

### 場景 2：本地開發環境
```
使用 llama-server + GGUF
- CPU 推理即可運行
- 啟動速度快
- 資源需求低
```

### 場景 3：依需求切換引擎
```
同一時間載入單一模型（載入第二個回 HTTP 409）；
依需求卸載後改用不同引擎：
- Qwen3 系列使用 Transformers
- 需高吞吐時使用 vLLM
- CPU/受限硬體使用 GGUF + llama-server
```

---

## 🆚 與其他方案比較

### vs 直接使用 OpenAI API
| 方面 | TRUSTA-AST | OpenAI API |
|------|-----------|-----------|
| 成本 | 自有硬體成本 | API 調用費用 |
| 隱私 | 數據完全本地 | 數據送雲端 |
| 自訂 | 可自訂模型 | 固定模型 |
| 控制 | 完整控制 | 有限控制 |

### vs 單進程架構
| 方面 | TRUSTA-AST | 單進程 |
|------|-----------|--------|
| 穩定性 | 高（隔離） | 低 |
| 錯誤隔離 | 主服務不受 worker crash 影響 | 全服務 crash |
| 併發 | 高 | 受限 |
| 管理 | 複雜 | 簡單 |

---

## 🚀 快速開始

### 基本配置
```bash
# 安裝環境（Linux + CUDA 範例；Windows 請用 scripts/windows/ 底下的腳本）
cp .env.example .env
TRUSTA_ACCEL=cuda bash scripts/linux/setup_env.sh

# 啟動服務（服務埠由 .env 的 SERVICE_PORT 控制，預設 8000）
bash scripts/linux/run_service.sh
```

> 注意：引擎與模型**不是**用環境變數指定，而是在 `POST /inference/load_model`
> 請求的 body 中以 `engine` / `model_name` 欄位指定；沒有 `pip install trusta-ast-service` 套件。

### 基本調用
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True
    }
)

for chunk in response.iter_lines():
    print(chunk)
```

---

## 📚 相關文件

- [[Trusta AST Inference Service]] - 詳細服務說明
- [[Multi-Process Isolation]] - 架構模式詳解
- [[OpenAI API Compatibility]] - API 兼容性
- [[Qwen3.5 Inference Optimization]] - 模型優化
- [[Inference Engine Comparison]] - 引擎比較

---

**簡化總結**：TRUSTA-AST 就是讓你**用熟悉的 OpenAI API**，**穩定高效地運行自己的 LLM 模型**，支援多種引擎，適應不同硬體需求。
