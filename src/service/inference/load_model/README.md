# Load Model Module

此模組包含模型載入的工具函數，特別是 PEFT/LoRA 微調模型的處理。

## 目錄結構

```
load_model/
├── __init__.py           # 模組初始化，導出公共 API
├── peft_loader.py        # PEFT/LoRA 模型載入工具
└── README.md            # 此文件
```

## PEFT Loader (`peft_loader.py`)

### 功能

提供 PEFT/LoRA 微調模型的檢測和載入功能。

### 主要函數

#### `is_peft_model(model_path: str) -> bool`

檢測指定路徑是否為 PEFT/LoRA 微調模型。

**參數:**
- `model_path`: 模型路徑

**返回:**
- `True` 如果路徑包含 `adapter_config.json`（PEFT 模型的標誌檔案）
- `False` 否則

**範例:**
```python
from service.inference.load_model import is_peft_model

if is_peft_model("/path/to/model"):
    print("This is a PEFT/LoRA model")
```

#### `load_peft_model(model_path: str, base_model, hf_token: Optional[str] = None)`

載入 PEFT/LoRA 微調模型的 adapters 並合併到 base model。

**參數:**
- `model_path`: LoRA adapter 的路徑
- `base_model`: 已載入的基礎模型實例
- `hf_token`: HuggingFace token（可選）

**返回:**
- 合併了 LoRA adapter 的模型實例

**異常:**
- `RuntimeError`: 如果 PEFT 函式庫未安裝
- `Exception`: 如果載入失敗

**範例:**
```python
from transformers import AutoModelForCausalLM
from service.inference.load_model import load_peft_model

# 載入 base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 載入 LoRA adapters
model = load_peft_model("/path/to/lora_adapter", base_model)
```

#### `read_base_model_name(model_path: str) -> str`

從 PEFT 模型的 `adapter_config.json` 讀取 base model 名稱。

**參數:**
- `model_path`: PEFT 模型路徑

**返回:**
- Base model 的名稱或路徑

**異常:**
- `FileNotFoundError`: 如果 `adapter_config.json` 不存在
- `ValueError`: 如果無法找到 `base_model_name_or_path` 欄位

**範例:**
```python
from service.inference.load_model.peft_loader import read_base_model_name

base_name = read_base_model_name("/path/to/lora_adapter")
print(f"Base model: {base_name}")
```

### 全域變數

#### `PEFT_AVAILABLE: bool`

指示 PEFT 函式庫是否可用。如果 `peft` 套件未安裝，此值為 `False`。

**範例:**
```python
from service.inference.load_model import PEFT_AVAILABLE

if PEFT_AVAILABLE:
    print("PEFT support is enabled")
else:
    print("PEFT not installed. Install with: pip install peft")
```

## 使用方式

### 基本導入

```python
from service.inference.load_model import (
    is_peft_model,
    load_peft_model,
    PEFT_AVAILABLE
)
```

### 完整工作流程範例

```python
from pathlib import Path
from transformers import AutoModelForCausalLM
from service.inference.load_model import (
    is_peft_model,
    load_peft_model,
    PEFT_AVAILABLE
)
from service.inference.load_model.peft_loader import read_base_model_name

model_path = "/path/to/model"

# 檢查是否為 PEFT 模型
if is_peft_model(model_path):
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT not installed")
    
    # 讀取 base model 名稱
    base_model_name = read_base_model_name(model_path)
    print(f"Loading base model: {base_model_name}")
    
    # 載入 base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # 載入 LoRA adapters
    model = load_peft_model(model_path, base_model)
    print("✓ PEFT model loaded successfully")
else:
    # 一般模型：直接載入
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    print("✓ Regular model loaded successfully")
```

## 依賴項

### 必需
- `pathlib`: 路徑處理
- `json`: 解析 adapter_config.json
- `logging`: 日誌記錄
- `typing`: 類型註解

### 可選
- `peft`: LoRA/QLoRA 支援
  - 安裝: `pip install peft`
  - 如果未安裝，`PEFT_AVAILABLE` 將為 `False`

## 整合到主系統

此模組已整合到 `model_inference_process.py`：

```python
# 在 model_inference_process.py 中
from .load_model import is_peft_model, load_peft_model, PEFT_AVAILABLE
from .load_model.peft_loader import read_base_model_name

# 使用範例（在 worker 進程中）
is_peft = is_peft_model(model_source)
if is_peft:
    base_model_name = read_base_model_name(model_source)
    base_model = ModelClass.from_pretrained(base_model_name, **kwargs)
    model = load_peft_model(model_source, base_model, hf_token)
```

## 錯誤處理

### 常見錯誤

1. **PEFT 未安裝**
   ```
   RuntimeError: PEFT library not available. Install with: pip install peft
   ```
   解決：`pip install peft`

2. **adapter_config.json 缺失**
   ```
   FileNotFoundError: adapter_config.json not found in /path/to/model
   ```
   解決：確認路徑正確且為有效的 PEFT 模型

3. **base_model_name_or_path 缺失**
   ```
   ValueError: base_model_name_or_path not found in adapter_config.json
   ```
   解決：檢查 adapter_config.json 格式是否正確

## 擴展性

未來可以在此模組中添加更多模型載入工具：

- `awq_loader.py`: AWQ 量化模型載入
- `gptq_loader.py`: GPTQ 量化模型載入
- `merged_loader.py`: 合併後的模型載入
- 等等...

## 參考文件

- [PEFT 官方文檔](https://huggingface.co/docs/peft)
- [LOAD_LORA_INFERENCE.md](../../../LOAD_LORA_INFERENCE.md) - LoRA 推理使用指南
- [QLORA_EXAMPLE.md](../../../QLORA_EXAMPLE.md) - QLoRA 訓練範例
