# 配置範例 - Hugging Face Transformers 標準格式

本文檔展示了簡化後的配置格式，直接使用 Hugging Face Transformers 的標準參數。

## 推理配置 (InferenceConfig)

### 範例 1: 基本配置 (CPU)
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "none",
  "device_map": "cpu"
}
```

### 範例 2: GPU 配置 + INT8 量化
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "int8",
  "device_map": "auto"
}
```

### 範例 3: 記憶體限制配置
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "int8",
  "device_map": "auto",
  "max_memory": {
    "0": "5GB",
    "cpu": "20GB"
  }
}
```

### 範例 4: Disk Offload 配置
```json
{
  "model_name": "openai/gpt-oss-120b",
  "quantization": "int8",
  "device_map": "auto",
  "max_memory": {
    "0": "5GB",
    "cpu": "20GB"
  },
  "offload_folder": "./offload"
}
```

### 範例 5: 自定義設備映射
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "int4",
  "device_map": {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.norm": "cpu",
    "lm_head": "cpu"
  }
}
```

## 訓練配置 (TrainingConfig)

### 範例 1: LoRA 訓練
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "method": "lora",
  "dataset_path": "./dataset/train.jsonl",
  "output_dir": "./output/lora",
  "quantization": "none",
  "device_map": "auto",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1
}
```

### 範例 2: QLoRA 訓練 + 記憶體限制
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "method": "qlora",
  "dataset_path": "./dataset/train.jsonl",
  "output_dir": "./output/qlora",
  "quantization": "nf4",
  "device_map": "auto",
  "max_memory": {
    "0": "10GB",
    "cpu": "30GB"
  },
  "lora_r": 16,
  "lora_alpha": 32
}
```

### 範例 3: 完整配置 + Disk Offload
```json
{
  "model_name": "openai/gpt-oss-120b",
  "method": "qlora",
  "dataset_path": "./dataset/train.jsonl",
  "output_dir": "./output/qlora",
  "quantization": "nf4",
  "device_map": "auto",
  "max_memory": {
    "0": "5GB",
    "cpu": "20GB"
  },
  "offload_folder": "./offload",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "learning_rate": 2e-4,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}
```

## 主要改變

### 移除的配置
- ❌ `model_offload` (整個物件)
- ❌ `ModelOffloadConfig` 類別

### 新增/保留的配置
- ✅ `device_map`: 直接使用 HF 格式 ("auto", "cpu", "cuda:0", 或自定義字典)
- ✅ `max_memory`: 記憶體限制 (例如: {0: "20GB", "cpu": "50GB"})
- ✅ `offload_folder`: Disk offload 路徑
- ✅ `quantization`: 量化類型 (none, int8, int4, nf4, fp4)

## 對應關係

### 舊格式 → 新格式

#### CPU Offload
```json
// 舊格式
{
  "model_offload": {"type": "cpu"}
}

// 新格式
{
  "device_map": "auto",
  "max_memory": {"cpu": "30GB"}
}
```

#### Disk Offload
```json
// 舊格式
{
  "model_offload": {
    "type": "disk",
    "offload_dir": "./offload"
  }
}

// 新格式
{
  "device_map": "auto",
  "offload_folder": "./offload"
}
```

#### 自定義設備映射
```json
// 舊格式
{
  "model_offload": {
    "device_map": {"GPU0": 12, "CPU": 50}
  }
}

// 新格式
{
  "device_map": "auto",
  "max_memory": {
    "0": "12GB",
    "cpu": "50GB"
  }
}
```

## API 使用範例

### 載入模型
```python
from service import InferenceConfig, model_manager

# 基本配置
config = InferenceConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    quantization="int8",
    device_map="auto"
)

model_manager.load_model(config)
```

### 記憶體限制配置
```python
config = InferenceConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    quantization="int8",
    device_map="auto",
    max_memory={0: "5GB", "cpu": "20GB"},
    offload_folder="./offload"
)

model_manager.load_model(config)
```

### 訓練配置
```python
from service import TrainingConfig, training_manager

config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    method="qlora",
    dataset_path="./dataset/train.jsonl",
    output_dir="./output/qlora",
    quantization="nf4",
    device_map="auto",
    max_memory={0: "10GB", "cpu": "30GB"}
)

training_manager.start_training(config)
```
