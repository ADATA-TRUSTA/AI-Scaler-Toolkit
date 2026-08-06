# Config Examples - Standard Hugging Face Transformers Format

This document shows the simplified config format, which uses the standard Hugging Face Transformers parameters directly.

## Inference Config (InferenceConfig)

### Example 1: Basic config (CPU)
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "none",
  "device_map": "cpu"
}
```

### Example 2: GPU config + INT8 quantization
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "quantization": "int8",
  "device_map": "auto"
}
```

### Example 3: Memory-limited config
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

### Example 4: Disk offload config
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

### Example 5: Custom device map
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

## Training Config (TrainingConfig)

### Example 1: LoRA training
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

### Example 2: QLoRA training + memory limits
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

### Example 3: Full config + disk offload
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

## Key Changes

### Removed
- ❌ `model_offload` (the whole object)
- ❌ the `ModelOffloadConfig` class

### Added/kept
- ✅ `device_map`: HF format used directly ("auto", "cpu", "cuda:0", or a custom dict)
- ✅ `max_memory`: memory limits (e.g. {0: "20GB", "cpu": "50GB"})
- ✅ `offload_folder`: disk offload path
- ✅ `quantization`: quantization type (none, int8, int4, nf4, fp4)

## Mapping

### Old format → new format

#### CPU Offload
```json
// Old format
{
  "model_offload": {"type": "cpu"}
}

// New format
{
  "device_map": "auto",
  "max_memory": {"cpu": "30GB"}
}
```

#### Disk Offload
```json
// Old format
{
  "model_offload": {
    "type": "disk",
    "offload_dir": "./offload"
  }
}

// New format
{
  "device_map": "auto",
  "offload_folder": "./offload"
}
```

#### Custom device map
```json
// Old format
{
  "model_offload": {
    "device_map": {"GPU0": 12, "CPU": 50}
  }
}

// New format
{
  "device_map": "auto",
  "max_memory": {
    "0": "12GB",
    "cpu": "50GB"
  }
}
```

## API Usage Examples

### Loading a model
```python
from service import InferenceConfig, model_manager

# Basic config
config = InferenceConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf", quantization="int8", device_map="auto"
)

model_manager.load_model(config)
```

### Memory-limited config
```python
config = InferenceConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    quantization="int8",
    device_map="auto",
    max_memory={0: "5GB", "cpu": "20GB"},
    offload_folder="./offload",
)

model_manager.load_model(config)
```

### Training config
```python
from service import TrainingConfig, training_manager

config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    method="qlora",
    dataset_path="./dataset/train.jsonl",
    output_dir="./output/qlora",
    quantization="nf4",
    device_map="auto",
    max_memory={0: "10GB", "cpu": "30GB"},
)

training_manager.start_training(config)
```
