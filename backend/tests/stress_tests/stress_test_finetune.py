import requests
import time
import json
import os
import sys
import logging
from typing import Dict, Any, List
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"stress_test_finetune_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "dataset.jsonl")

# Define test configurations
# Note: Ensure these base models exist in models_registry.json
TEST_CONFIGS = [
    # Full Fine-tuning
    {"model_name": "Qwen/Qwen3-4B", "method": "full", "label": "qwen3-4b-full"},
    {"model_name": "Qwen/Qwen3-8B", "method": "full", "label": "qwen3-8b-full"},
    
    # LoRA
    {"model_name": "Qwen/Qwen3-4B", "method": "lora", "label": "qwen3-4b-lora"},
    {"model_name": "Qwen/Qwen3-8B", "method": "lora", "label": "qwen3-8b-lora"},
    {"model_name": "Qwen/Qwen3-14B", "method": "lora", "label": "qwen3-14b-lora"},
    
    # QLoRA
    {"model_name": "Qwen/Qwen3-32B", "method": "qlora", "label": "qwen3-32b-qlora"},
]

def check_service_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        logger.info("Service is healthy")
        return True
    except Exception as e:
        logger.error(f"Service health check failed: {e}")
        return False

def wait_for_training(timeout=3600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/training/status")
            status = resp.json()
            state = status.get("status")
            
            if state == "completed":
                logger.info("Training completed successfully")
                return True, status
            elif state == "failed":
                logger.error(f"Training failed: {status.get('error')}")
                return False, status
            elif state in ["idle", "stopped"]:
                # Should not happen if we just started
                pass
            
            logger.info(f"Training in progress... Status: {state}, Progress: {status.get('progress')}%")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            time.sleep(10)
    
    logger.error("Training timed out")
    return False, None

def wait_for_model_load(timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/inference/status")
            status = resp.json()
            
            if status.get("loaded"):
                logger.info("Model loaded successfully")
                return True
            if status.get("error"):
                logger.error(f"Model load failed: {status.get('error')}")
                return False
                
            logger.info("Model loading...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error checking load status: {e}")
            time.sleep(5)
    return False

def ensure_model_exists(model_name):
    """Ensure the base model is registered."""
    logger.info(f"Checking if model {model_name} exists...")
    resp = requests.get(f"{BASE_URL}/config/models")
    models = resp.json()
    base_models = models.get("base_models", [])
    
    # Check if model_name matches any label or base_model_name
    exists = False
    for m in base_models:
        if m.get("label") == model_name or m.get("base_model_name") == model_name:
            exists = True
            break
            
    if not exists:
        logger.error(f"Model {model_name} not found in registry. Please ensure it is registered in models_registry.json.")
        return False
    
    logger.info(f"Model {model_name} exists.")
    return True

def run_test_cycle(config: Dict[str, Any]):
    model_name = config["model_name"]
    method = config["method"]
    label = config["label"]
    
    if not ensure_model_exists(model_name):
        logger.error(f"Skipping cycle {label} because base model could not be ensured.")
        return False
    
    logger.info(f"=== Starting Test Cycle: {label} ({method}) ===")
    
    # Resolve the actual HF model ID or path from the registry
    # The backend training process expects a valid HF ID or path, not just a label
    actual_model_id = model_name
    try:
        resp = requests.get(f"{BASE_URL}/config/models")
        models = resp.json()
        base_models = models.get("base_models", [])
        for m in base_models:
            # Match by label (preferred) or base_model_name
            if m.get("base_model_name") == model_name:
                # Prefer local_path if it exists, else base_model_name
                actual_model_id = m.get("base_model_name")
                logger.info(f"Resolved model '{model_name}' to '{actual_model_id}'")
                break
    except Exception as e:
        logger.warning(f"Failed to resolve model ID from registry: {e}")
    
    # 1. Start Training
    # Common parameters from user request
    common_params = {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 2,
        "save_steps": 100,
        "save_total_limit": 1,
        "save_tokenizer": True,
        "use_deepspeed": True,
        "deepspeed_profile": "zero3_offload_disk_cpu",
        "offload_folder": os.path.abspath("deepspeed_offload"), # Use local folder
        "prompt_field": "prompt",
        "completion_field": "completion"
    }

    if method == "full":
        # Specifics for Full Finetuning
        train_payload = {
            "model_name": actual_model_id,
            "dataset_path": DATASET_PATH,
            "method": "full",
            "output_dir": f"finetune_output/{label}",
            "max_seq_length": 1024,
            **common_params
        }
    else:
        # Specifics for LoRA/QLoRA
        train_payload = {
            "model_name": actual_model_id,
            "dataset_path": DATASET_PATH,
            "method": method,
            "output_dir": f"finetune_output/{label}",
            "max_seq_length": 256,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": [],
            "use_sft_trainer": True,
            "packing": False,
            **common_params
        }
    
    logger.info(f"Starting training for {model_name}...")
    resp = requests.post(f"{BASE_URL}/training/start", json=train_payload)
    if resp.status_code != 200:
        logger.error(f"Failed to start training: {resp.text}")
        return False
        
    success, status = wait_for_training()
    if not success:
        return False
        
    # 2. Load Fine-tuned Model
    # The finetuned model should be registered automatically, but let's construct the path
    # Or use the label if the system registers it with a predictable label
    # Assuming the system registers it as "finetuned_{timestamp}" or similar.
    # Actually, let's check the registry or just use the output path directly if supported.
    # The system registers it. Let's list models to find it.
    
    time.sleep(2) # Wait for registration
    models_resp = requests.get(f"{BASE_URL}/config/models")
    models = models_resp.json()
    finetuned_models = models.get("finetuned_models", [])
    
    # Find our model by output_dir matching
    target_model = None
    abs_output_dir = os.path.abspath(train_payload["output_dir"])
    for m in finetuned_models:
        if os.path.abspath(m.get("output_dir", "")) == abs_output_dir:
            target_model = m
            break
            
    if not target_model:
        logger.warning("Could not find registered finetuned model. Trying to load by path directly if supported, or skipping inference test.")
        # Construct a config using the path
        load_payload = {
            "model_name": model_name, # Base model
            "model_path": abs_output_dir, # Adapter path
            "quantization": "none",
            "device_map": "auto"
        }
    else:
        load_payload = {
            "model_name": target_model["base_model_name"],
            "model_path": target_model["output_dir"],
            "quantization": "none",
            "device_map": "auto"
        }

    logger.info("Loading fine-tuned model for inference...")
    resp = requests.post(f"{BASE_URL}/inference/load_model", json=load_payload)
    if resp.status_code != 200:
        logger.error(f"Failed to request model load: {resp.text}")
        return False
        
    if not wait_for_model_load():
        return False
        
    # 3. Test Chat
    logger.info("Testing chat...")
    chat_payload = {
        "message": "Hello, tell me a joke.",
        "max_new_tokens": 50,
        "temperature": 0.7
    }
    resp = requests.post(f"{BASE_URL}/inference/chat", json=chat_payload)
    if resp.status_code == 200:
        logger.info(f"Chat response: {resp.json().get('response')}")
    else:
        logger.error(f"Chat failed: {resp.text}")
        return False
        
    # 4. Unload Model
    logger.info("Unloading model...")
    requests.post(f"{BASE_URL}/inference/unload_model")
    time.sleep(5)
    
    # 5. Delete Model (Cleanup)
    if target_model:
        label_to_delete = target_model["label"]
        logger.info(f"Deleting model {label_to_delete}...")
        requests.delete(f"{BASE_URL}/config/models/{label_to_delete}?delete_files=true")
    else:
        # Manual cleanup if not found in registry
        import shutil
        if os.path.exists(abs_output_dir):
            shutil.rmtree(abs_output_dir)
            
    logger.info(f"=== Cycle {label} Completed Successfully ===\n")
    return True

def main():
    if not check_service_health():
        sys.exit(1)
        
    # Create dummy dataset if not exists
    if not os.path.exists(DATASET_PATH):
        logger.info(f"Creating dummy dataset at {DATASET_PATH}")
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            for i in range(10):
                # Use prompt/completion format as requested
                f.write(json.dumps({"prompt": f"Instruction {i}", "completion": f"Output {i}"}) + "\n")

    failed_tests = []
    for config in TEST_CONFIGS:
        if not run_test_cycle(config):
            failed_tests.append(config["label"])
            
    if failed_tests:
        logger.error(f"The following tests failed: {failed_tests}")
        sys.exit(1)
    else:
        logger.info("All stress tests passed!")

if __name__ == "__main__":
    main()
