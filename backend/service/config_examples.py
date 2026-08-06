"""Example inference/training configuration payloads surfaced in the API docs."""

TRAINING_CONFIG_EXAMPLES = {
    "LoRA Example": {
        "summary": "LoRA fine-tune (single-field mode)",
        "description": "Train on the text field; suits general conversational data",
        "value": {
            "model_name": "qwen3-4b-local",
            "method": "lora",
            "dataset_path": "./dataset/train.jsonl",
            "output_dir": "./output/qwen3_lora_v1",
            "num_gpu": 1,
            "text_field": "text",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 2,
            "max_seq_length": 2048,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        },
    },
    "QLoRA Example": {
        "summary": "QLoRA fine-tune (two-field mode)",
        "description": "Train on prompt + completion; suits question/answer pair data",
        "value": {
            "model_name": "llama3-8b-local",
            "method": "qlora",
            "dataset_path": "./dataset/qa_pairs.jsonl",
            "output_dir": "./output/llama3_qlora_qa",
            "num_gpu": 1,
            "prompt_field": "prompt",
            "completion_field": "completion",
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "warmup_steps": 50,
            "logging_steps": 5,
            "save_steps": 200,
            "save_total_limit": 3,
            "max_seq_length": 1024,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        },
    },
    "Full Fine-tune with DeepSpeed": {
        "summary": "Full-parameter fine-tune (DeepSpeed)",
        "description": "Full-parameter fine-tune with DeepSpeed offload; suits large models",
        "value": {
            "model_name": "qwen3-4b-local",
            "method": "full",
            "dataset_path": "./dataset/train.jsonl",
            "output_dir": "./output/qwen3_full_finetune",
            "num_gpu": 2,
            "text_field": "text",
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5e-5,
            "warmup_steps": 200,
            "logging_steps": 10,
            "save_steps": 1000,
            "save_total_limit": 1,
            "max_seq_length": 2048,
            "use_deepspeed": True,
            "deepspeed_profile": "zero3_offload_disk_disk",
        },
    },
    "LoRA with Custom DeepSpeed Config": {
        "summary": "LoRA + custom DeepSpeed",
        "description": "Use a custom DeepSpeed config file",
        "value": {
            "model_name": "tinyllama-local",
            "method": "lora",
            "dataset_path": "./dataset/custom.jsonl",
            "output_dir": "./output/tinyllama_custom",
            "num_gpu": 1,
            "text_field": "content",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3e-4,
            "max_seq_length": 1024,
            "use_deepspeed": True,
            "deepspeed_config": "./configs/my_deepspeed_config.json",
        },
    },
    "Multimodal (Image) LoRA Fine-tune": {
        "summary": "Multimodal image fine-tune (LoRA)",
        "description": (
            "Train a multimodal model (VLM) to read images. model_name must be a multimodal "
            "checkpoint (e.g. gemma-4-E4B-it, Qwen3.5-35B-A3B). Every dataset record needs "
            "an images field (relative paths resolve against the dataset file's directory) "
            "plus prompt/completion. Prefer prompt/completion over messages: TRL only masks "
            "out image tokens in prompt/completion mode, otherwise the loss covers a large "
            "number of image tokens and dilutes the training signal. max_seq_length must fit "
            "the expanded image tokens (a single image can reach hundreds to thousands of "
            "tokens). The vision tower stays frozen (only the language tower's comprehension "
            "is optimized); training also emits an mmproj GGUF for image inference in "
            "llama.cpp."
        ),
        "value": {
            "model_name": "gemma-4-e4b-it-local",
            "method": "lora",
            "dataset_path": "./dataset/image_train.jsonl",
            "output_dir": "./output/gemma4_image_lora",
            "num_gpu": 1,
            "prompt_field": "prompt",
            "completion_field": "completion",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "warmup_steps": 10,
            "logging_steps": 5,
            "save_steps": 500,
            "max_seq_length": 2048,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "use_deepspeed": True,
            "deepspeed_profile": "zero3_offload_cpu_cpu",
            "eval_split_ratio": 0.1,
            "eval_steps": 50,
        },
    },
    "LoRA with Held-out Test Set": {
        "summary": "LoRA + test set evaluation",
        "description": (
            "Split a held-out test set off the training set automatically, evaluate "
            "periodically during training and run a final evaluation at the end. "
            "eval_split_ratio=0.1 holds out 10% as the test set; eval_steps controls how "
            "often evaluation runs (falls back to logging_steps when left unset). Test set "
            "loss/accuracy is returned separately from the training metrics (eval_logs in "
            "history, split=eval on log events), so the two curves can be compared to spot "
            "overfitting."
        ),
        "value": {
            "model_name": "qwen3-4b-local",
            "method": "lora",
            "dataset_path": "./dataset/train.jsonl",
            "output_dir": "./output/qwen3_lora_eval",
            "num_gpu": 1,
            "prompt_field": "prompt",
            "completion_field": "completion",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "save_steps": 500,
            "max_seq_length": 2048,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "eval_split_ratio": 0.1,
            "eval_steps": 50,
        },
    },
}

INFERENCE_CONFIG_EXAMPLES = {
    "INT8 Quantization with CPU Offload": {
        "summary": "INT8 quantization + CPU offload",
        "description": "INT8 quantization with part of the model offloaded to the CPU; suits limited VRAM",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "quantization": "int8",
            "model_total_memory": "10GB",
            "device_map": "auto",
            "max_memory": {"0": "5GB", "cpu": "5GB"},
            "torch_dtype": "auto",
        },
    },
    "CPU Only Execution": {
        "summary": "CPU-only execution",
        "description": "Run the model entirely on the CPU; slower but needs no GPU",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "quantization": "none",
            "model_total_memory": "15GB",
            "device_map": "cpu",
        },
    },
    "Large Model with Disk Offload": {
        "summary": "Large model + disk offload",
        "description": "Run a very large model using both CPU and disk offload",
        "value": {
            "model_name": "openai/gpt-oss-120b",
            "quantization": "int8",
            "model_total_memory": "120GB",
            "device_map": "auto",
            "max_memory": {"0": "5GB", "cpu": "20GB"},
            "offload_folder": "./offload",
        },
    },
    "Llama Server Engine": {
        "summary": "Llama Server (concurrent users)",
        "description": "Start/stop the llama-server subprocess via load/unload, with -np multi-slot concurrency",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "model_path": "/path/to/model.gguf",
            "engine": "llama_server",
            "llama_server_auto_start": True,
            "llama_server_binary": "llama-server",
            "llama_server_host": "127.0.0.1",
            "llama_server_port": 8080,
            "llama_server_np": 4,
            "llama_server_health_timeout": 300,
            "llama_server_url": "http://127.0.0.1:8080",
            "llama_server_model": "Qwen/Qwen3-4B",
            "llama_server_mmproj": "/path/to/mmproj.gguf",
            "llama_server_timeout": 300,
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "n_batch": 512,
        },
    },
    "vLLM Engine": {
        "summary": "vLLM OpenAI-compatible Server",
        "description": "The backend starts the vLLM server on load_model and stops it on unload_model",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "engine": "vllm",
            "vllm_gpu_memory_utilization": 0.8,
            "vllm_max_model_len": 4096,
            "vllm_enforce_eager": False,
            "vllm_cpu_offload_gb": 8,
        },
    },
    "Intel XPU (Intel GPU)": {
        "summary": "Intel XPU inference",
        "description": "Run inference on an Intel GPU (Arc/Iris Xe), with INT8 quantization support",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "quantization": "int8",
            "model_total_memory": "10GB",
            "device_map": "auto",
            "max_memory": {"xpu:0": "6GB", "cpu": "8GB"},
            "torch_dtype": "auto",
        },
    },
    "Intel XPU Multi-Device": {
        "summary": "Intel XPU multi-device",
        "description": "Run inference across several Intel GPUs (mixed offload)",
        "value": {
            "model_name": "Qwen/Qwen3-8B",
            "quantization": "int4",
            "model_total_memory": "20GB",
            "device_map": "auto",
            "max_memory": {"xpu:0": "8GB", "xpu:1": "8GB", "cpu": "16GB"},
            "torch_dtype": "auto",
        },
    },
    "Intel XPU Pure mode": {
        "summary": "Pure XPU mode (no CPU offload)",
        "description": "Run entirely on the Intel GPU; suits ample VRAM",
        "value": {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantization": "int4",
            "device_map": "auto",
            "max_memory": {"xpu:0": "12GB"},
            "torch_dtype": "float16",
        },
    },
}

MEMORY_ESTIMATE_EXAMPLES = {
    "Standard Estimation": {
        "summary": "Standard estimate (FP16)",
        "description": "Estimate the memory a 4B model needs in FP16",
        "value": {
            "model_name": "Qwen/Qwen3-4B",
            "quantization": "none",
            "batch_size": 1,
            "sequence_length": 2048,
        },
    },
    "INT8 Estimation": {
        "summary": "INT8 quantization estimate",
        "description": "Estimate the memory a 1.1B model needs under INT8 quantization",
        "value": {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantization": "int8",
            "batch_size": 1,
            "sequence_length": 4096,
        },
    },
}

GGUF_ESTIMATE_EXAMPLES = {
    "Partial Offload": {
        "summary": "Partial offload",
        "description": "A 32B Q8_0 model with 20 layers on the GPU at 32K context",
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "n_gpu_layers": 20,
            "n_ctx": 32768,
        },
    },
    "Quantized KV Cache": {
        "summary": "Quantized KV cache",
        "description": "Full offload with a q8_0 KV cache, which roughly halves context memory",
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "n_gpu_layers": -1,
            "n_ctx": 32768,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        },
    },
    "MoE Expert Offload": {
        "summary": "MoE experts on the CPU",
        "description": (
            "Expert weights of the first 24 layers stay in DRAM while attention and the "
            "KV cache remain on the GPU"
        ),
        "value": {
            "model_path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
            "n_gpu_layers": -1,
            "n_ctx": 16384,
            "n_cpu_moe": 24,
        },
    },
    "From llama-server Arguments": {
        "summary": "Reuse llama_server_extra_args",
        "description": (
            "Pass the frontend's extra arguments as-is; they override the other fields, "
            "just as llama-server appends them last"
        ),
        "value": {
            "model_path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
            "llama_server_extra_args": ["-ngl", "-1", "-c", "32768", "-ncmoe", "24", "-ub", "1024"],
        },
    },
}

GGUF_SWEEP_EXAMPLES = {
    "Default Grid": {
        "summary": "Default sweep",
        "description": (
            "Auto-generated -ngl ladder x common context lengths x [f16, q8_0], compared "
            "against currently free VRAM"
        ),
        "value": {"model_path": "/models/Qwen3-32B-Q8_0.gguf"},
    },
    "Custom Grid": {
        "summary": "Custom sweep range",
        "description": "Explicit -ngl and context combinations with an explicit VRAM budget",
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "n_gpu_layers_grid": [0, 10, 20, 30, 40, 65],
            "n_ctx_grid": [4096, 32768],
            "kv_quant_grid": ["f16", "q8_0"],
            "gpu_budget_mib": 15839,
        },
    },
}

GGUF_RECOMMEND_EXAMPLES = {
    "Auto Fit": {
        "summary": "Automatic recommendation",
        "description": "Solve for the largest workable -ngl and context given free VRAM/DRAM",
        "value": {"model_path": "/models/Qwen3-32B-Q8_0.gguf"},
    },
    "Fixed Context": {
        "summary": "Pinned context",
        "description": "Keep 32K context and only adjust -ngl / -ncmoe / KV quantization",
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "n_ctx": 32768,
            "allow_ctx_reduction": False,
        },
    },
    "Manual GPU Cap": {
        "summary": "Fill a manual GPU cap",
        "description": (
            "Cap VRAM at 12 GiB and let the search spend almost all of it: -ngl is "
            "maximised first, then the context grows into whatever budget is left. Leave "
            "n_ctx unset so the context is free to grow"
        ),
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "gpu_budget_mib": 12288,
            "margin_mib": 512,
            "n_ctx_min": 4096,
            "target_utilization": 0.95,
        },
    },
}

GGUF_PLAN_EXAMPLES = {
    "GPU Size Menu": {
        "summary": "One plan per GPU size",
        "description": (
            "Solve for 5, 10 and 15 GiB of VRAM and offer f16 / q8_0 / q4_0 KV cache in "
            "each, so the caller can pick per GPU class"
        ),
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "gpu_budgets_mib": [5120, 10240, 15360],
            "kv_cache_types": ["f16", "q8_0", "q4_0"],
            "margin_mib": 512,
        },
    },
    "Long Context Options": {
        "summary": "Options for a pinned long context",
        "description": (
            "Hold 128K context and let each KV cache type answer how much offload it "
            "affords on one budget"
        ),
        "value": {
            "model_path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
            "gpu_budgets_mib": [15360],
            "n_ctx": 131072,
            "kv_cache_types": ["f16", "q8_0", "q4_0"],
        },
    },
    "Fast Analytic Menu": {
        "summary": "Skip verification",
        "description": (
            "Analytic figures only; no llama.cpp process is started, so a full menu is "
            "instant at the cost of the compute buffer's margin of error"
        ),
        "value": {
            "model_path": "/models/Qwen3-32B-Q8_0.gguf",
            "gpu_budgets_mib": [5120, 10240, 15360],
            "verify": False,
        },
    },
}

CONVERSION_CONFIG_EXAMPLES = {
    "Auto Detect (Recommended)": {
        "summary": "Auto-detect the output type",
        "description": "Prefer `auto` and let llama.cpp pick a common 16-bit type from the model weights.",
        "value": {
            "model_path": "/path/to/hf-or-merged-model",
            "output_path": "/path/to/output/model-auto.gguf",
            "outtype": "auto",
        },
    },
    "F16 Common": {
        "summary": "F16 half precision",
        "description": "One of the most common full-model GGUF types, balancing size against compatibility.",
        "value": {
            "model_path": "/path/to/hf-or-merged-model",
            "output_path": "/path/to/output/model-f16.gguf",
            "outtype": "f16",
        },
    },
    "BF16 Common": {
        "summary": "BF16 half precision",
        "description": "Suits models whose original weights are already bfloat16, such as Qwen3.5 weights.",
        "value": {
            "model_path": "/path/to/hf-or-merged-model",
            "output_path": "/path/to/output/model-bf16.gguf",
            "outtype": "bf16",
        },
    },
    "Q8_0 Quantized": {
        "summary": "Q8_0 quantization",
        "description": "A common high-quality quantization format: smaller than F16/BF16 yet still accurate.",
        "value": {
            "model_path": "/path/to/hf-or-merged-model",
            "output_path": "/path/to/output/model-q8_0.gguf",
            "outtype": "q8_0",
        },
    },
    "F32 Full Precision": {
        "summary": "F32 full precision",
        "description": "The largest output, normally used only for debugging, verification or strict precision needs.",
        "value": {
            "model_path": "/path/to/hf-or-merged-model",
            "output_path": "/path/to/output/model-f32.gguf",
            "outtype": "f32",
        },
    },
}
