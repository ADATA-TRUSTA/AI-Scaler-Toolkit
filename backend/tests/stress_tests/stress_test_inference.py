import requests
import time
import sys
import logging
import random
import os
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"stress_test_inference_{timestamp}.log")

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

# Models to test (Must be available in the environment)
MODEL_NAME = "Qwen3 4B"

OFFLOAD_CONFIGS = [
    {"quantization": "none", "offload_folder": None, "device_map": "auto"},
    {"quantization": "int8", "offload_folder": None, "device_map": "auto"},
    {"quantization": "int4", "offload_folder": None, "device_map": "auto"},
    # CPU Offload simulation (requires setting max_memory to force offload usually, 
    # but here we just change params that might trigger different paths)
    {"quantization": "int4", "offload_folder": "./offload_test", "device_map": "auto"},
]

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
                
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error checking load status: {e}")
            time.sleep(2)
    return False

def ensure_model_exists(model_name):
    """Ensure the model is registered."""
    resp = requests.get(f"{BASE_URL}/config/models")
    models = resp.json()
    base_models = models.get("base_models", [])
    finetuned_models = models.get("finetuned_models", [])
    
    for m in base_models + finetuned_models:
        if m.get("label") == model_name or m.get("base_model_name") == model_name:
            return True
            
    logger.error(f"Model {model_name} not found in registry.")
    return False

def get_registry_label(model_name):
    try:
        resp = requests.get(f"{BASE_URL}/config/models")
        models = resp.json()
        for m in models.get("base_models", []) + models.get("finetuned_models", []):
            if m.get("base_model_name") == model_name or m.get("label") == model_name:
                return m["label"]
    except Exception:
        pass
    return model_name

def run_inference_cycle(iteration, config):
    logger.info(f"=== Iteration {iteration} | Config: {config} ===")
    
    target_model = get_registry_label(MODEL_NAME)
    
    # 1. Load Model
    payload = {
        "model_name": target_model,
        "quantization": config["quantization"],
        "device_map": config["device_map"],
        "offload_folder": config["offload_folder"]
    }
    
    logger.info("Loading model...")
    resp = requests.post(f"{BASE_URL}/inference/load_model", json=payload)
    if resp.status_code != 200:
        # If 409, maybe previous unload failed?
        if resp.status_code == 409:
            logger.warning("Model already loaded or loading? Trying to unload first.")
            requests.post(f"{BASE_URL}/inference/unload_model")
            time.sleep(5)
            resp = requests.post(f"{BASE_URL}/inference/load_model", json=payload)
            if resp.status_code != 200:
                logger.error(f"Failed to load model: {resp.text}")
                return False
        else:
            logger.error(f"Failed to load model: {resp.text}")
            return False
            
    if not wait_for_model_load():
        return False
        
    # 2. Chat
    logger.info("Chatting...")
    chat_payload = {
        "message": "What is the capital of France?",
        "max_new_tokens": 20,
        "temperature": 0.1
    }
    resp = requests.post(f"{BASE_URL}/inference/chat", json=chat_payload)
    if resp.status_code == 200:
        logger.info(f"Response: {resp.json().get('response')}")
    else:
        logger.error(f"Chat failed: {resp.text}")
        return False
        
    # 3. Unload
    logger.info("Unloading...")
    requests.post(f"{BASE_URL}/inference/unload_model")
    
    # Wait a bit for cleanup
    time.sleep(3)
    
    # Verify unloaded
    status = requests.get(f"{BASE_URL}/inference/status").json()
    if status.get("loaded"):
        logger.error("Model failed to unload properly")
        return False
        
    return True

def main():
    iterations = 10
    for i in range(1, iterations + 1):
        config = random.choice(OFFLOAD_CONFIGS)
        if not run_inference_cycle(i, config):
            logger.error("Stress test failed at iteration {i}")
            sys.exit(1)
            
    logger.info("Inference stress test completed successfully")

if __name__ == "__main__":
    main()
