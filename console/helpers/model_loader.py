"""
Model Loader/Unloader Handler
Handles pre-load checks, loading, monitoring, and unloading processes for models
"""

import time
from ai_client import AIClient
from config_models import InferenceConfig


def check_and_wait_for_loading(client: AIClient) -> None:
    """Check if any model is currently loading, wait if so"""
    print("\n[Step 1] Checking if any model is currently loading...")
    while True:
        status = client.get_status()
        if status.is_loading:
            print(f"⏳ Model is loading... waiting (model: {status.model_name})")
            time.sleep(2)
        else:
            print("✓ No model is currently loading")
            break


def unload_existing_model(client: AIClient) -> None:
    """Check and unload already loaded model"""
    print("\n[Step 2] Checking for loaded model...")
    status = client.get_status()
    if status.loaded:
        print(f"Model already loaded: {status.model_name}")
        print("🔄 Unloading current model...")
        try:
            client.unload_model()
            print("✓ Model unloaded successfully")
            time.sleep(1)  # Give a short delay to ensure unloading completes
        except Exception as e:
            print(f"✗ Failed to unload model: {e}")
            raise
    else:
        print("No model currently loaded, proceeding to load")


def load_model(
    client: AIClient, infer_config: InferenceConfig, model_name: str
) -> None:
    """Load new model"""
    print(f"\n[Step 3] Loading model: {model_name}")
    print("=" * 60)
    client.load_model(infer_config)
    time.sleep(1)  # Give a short delay to wait for status update


def monitor_loading_progress(client: AIClient) -> None:
    """Monitor model loading progress until completion or failure"""
    print("\n[Step 4] Monitoring model loading progress...")
    while True:
        status = client.get_status()
        is_loading = status.is_loading
        loaded = status.loaded

        print(f"Status: is_loading={is_loading}, loaded={loaded}")

        if not is_loading:
            # Model is no longer loading, check if successfully loaded
            if loaded:
                print("✓ Model loaded successfully!")
                break
            else:
                print("✗ Model failed to load")
                if status.loading_error:
                    print(f"Error: {status.loading_error}")
                raise Exception("Model loading failed")

        # Wait a while before checking again
        time.sleep(2)


def unload_model(client: AIClient) -> None:
    """Unload currently loaded model"""
    print("\n" + "=" * 60)
    print("Unloading model...")
    print("=" * 60)
    try:
        client.unload_model()
        print("✓ Model unloaded successfully")
    except Exception as e:
        print(f"✗ Failed to unload model: {e}")
        raise


def prepare_and_load_model(
    client: AIClient, infer_config: InferenceConfig, model_name: str
) -> None:
    """
    Complete model loading process:
    1. Check if any model is currently loading
    2. Unload existing model
    3. Load new model
    4. Monitor loading progress
    """
    print("\n" + "=" * 60)
    print("Checking current model status before loading...")
    print("=" * 60)

    check_and_wait_for_loading(client)
    unload_existing_model(client)
    load_model(client, infer_config, model_name)
    monitor_loading_progress(client)

    print("=" * 60)
