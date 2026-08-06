# Load Model Module

This module holds model-loading helpers, in particular for PEFT/LoRA fine-tuned models.

## Directory Layout

```
load_model/
├── __init__.py           # Module init, exports the public API
├── peft_loader.py        # PEFT/LoRA model loading helpers
└── README.md            # This file
```

## PEFT Loader (`peft_loader.py`)

### Purpose

Provides detection and loading of PEFT/LoRA fine-tuned models.

### Main Functions

#### `is_peft_model(model_path: str) -> bool`

Checks whether the given path holds a PEFT/LoRA fine-tuned model.

**Parameters:**
- `model_path`: Model path

**Returns:**
- `True` if the path contains `adapter_config.json` (the marker file of a PEFT model)
- `False` otherwise

**Example:**
```python
from service.inference.load_model import is_peft_model

if is_peft_model("/path/to/model"):
    print("This is a PEFT/LoRA model")
```

#### `load_peft_model(model_path: str, base_model, hf_token: Optional[str] = None)`

Loads the adapters of a PEFT/LoRA fine-tuned model onto the base model.

**Parameters:**
- `model_path`: Path to the LoRA adapter
- `base_model`: Already-loaded base model instance
- `hf_token`: HuggingFace token (optional)

**Returns:**
- Model instance with the LoRA adapter attached

**Exceptions:**
- `RuntimeError`: If the PEFT library is not installed
- `Exception`: If loading fails

**Example:**
```python
from transformers import AutoModelForCausalLM
from service.inference.load_model import load_peft_model

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load the LoRA adapters
model = load_peft_model("/path/to/lora_adapter", base_model)
```

#### `read_base_model_name(model_path: str) -> str`

Reads the base model name from a PEFT model's `adapter_config.json`.

**Parameters:**
- `model_path`: PEFT model path

**Returns:**
- Name or path of the base model

**Exceptions:**
- `FileNotFoundError`: If `adapter_config.json` does not exist
- `ValueError`: If the `base_model_name_or_path` field cannot be found

**Example:**
```python
from service.inference.load_model.peft_loader import read_base_model_name

base_name = read_base_model_name("/path/to/lora_adapter")
print(f"Base model: {base_name}")
```

### Global Variables

#### `PEFT_AVAILABLE: bool`

Indicates whether the PEFT library is available. If the `peft` package is not installed, this is `False`.

**Example:**
```python
from service.inference.load_model import PEFT_AVAILABLE

if PEFT_AVAILABLE:
    print("PEFT support is enabled")
else:
    print("PEFT not installed. Install with: pip install peft")
```

## Usage

### Basic Import

```python
from service.inference.load_model import is_peft_model, load_peft_model, PEFT_AVAILABLE
```

### Full Workflow Example

```python
from pathlib import Path
from transformers import AutoModelForCausalLM
from service.inference.load_model import is_peft_model, load_peft_model, PEFT_AVAILABLE
from service.inference.load_model.peft_loader import read_base_model_name

model_path = "/path/to/model"

# Check whether this is a PEFT model
if is_peft_model(model_path):
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT not installed")

    # Read the base model name
    base_model_name = read_base_model_name(model_path)
    print(f"Loading base model: {base_model_name}")

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype="auto"
    )

    # Load the LoRA adapters
    model = load_peft_model(model_path, base_model)
    print("✓ PEFT model loaded successfully")
else:
    # Regular model: load directly
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    print("✓ Regular model loaded successfully")
```

## Dependencies

### Required
- `pathlib`: Path handling
- `json`: Parsing adapter_config.json
- `logging`: Log output
- `typing`: Type annotations

### Optional
- `peft`: LoRA/QLoRA support
  - Install: `pip install peft`
  - If not installed, `PEFT_AVAILABLE` is `False`

## Integration With the Main System

This module is already wired into `model_inference_process.py`:

```python
# In model_inference_process.py
from .load_model import is_peft_model, load_peft_model, PEFT_AVAILABLE
from .load_model.peft_loader import read_base_model_name

# Usage example (inside the worker process)
is_peft = is_peft_model(model_source)
if is_peft:
    base_model_name = read_base_model_name(model_source)
    base_model = ModelClass.from_pretrained(base_model_name, **kwargs)
    model = load_peft_model(model_source, base_model, hf_token)
```

## Error Handling

### Common Errors

1. **PEFT not installed**
   ```
   RuntimeError: PEFT library not available. Install with: pip install peft
   ```
   Fix: `pip install peft`

2. **adapter_config.json missing**
   ```
   FileNotFoundError: adapter_config.json not found in /path/to/model
   ```
   Fix: confirm the path is correct and is a valid PEFT model

3. **base_model_name_or_path missing**
   ```
   ValueError: base_model_name_or_path not found in adapter_config.json
   ```
   Fix: check that adapter_config.json is well-formed

## Extensibility

More model-loading helpers can be added to this module later:

- `awq_loader.py`: Loading AWQ-quantized models
- `gptq_loader.py`: Loading GPTQ-quantized models
- `merged_loader.py`: Loading merged models
- and so on...

## References

- [PEFT official docs](https://huggingface.co/docs/peft)
- [LOAD_LORA_INFERENCE.md](../../../LOAD_LORA_INFERENCE.md) - LoRA inference guide
- [QLORA_EXAMPLE.md](../../../QLORA_EXAMPLE.md) - QLoRA training example
