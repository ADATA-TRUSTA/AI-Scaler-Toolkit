import sys
import os
import argparse
import traceback
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRA_Merger")


def _read_mem_available_gib() -> int | None:
    """Read available system memory in GiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                    available_gib = max(1, available_kib // (1024 * 1024))
                    return available_gib
    except (OSError, ValueError, IndexError):
        return None
    return None


def _resolve_device_map(offload_folder: str | None, requested_device_map: str | None) -> str:
    """Resolve merge device placement.

    With `offload_folder` present, default to CPU merge for stability because
    `device_map="auto"` aggressively fills available GPU memory before using
    CPU/disk, which often OOMs during `merge_and_unload()`.
    """
    if requested_device_map:
        return requested_device_map

    if offload_folder:
        return "cpu"

    return "auto"


def _build_max_memory(device_map: str, offload_folder: str | None) -> dict[int | str, str] | None:
    """Build conservative max_memory only for bounded auto-offload mode."""
    if device_map != "auto" or not offload_folder:
        return None

    max_memory: dict[int | str, str] = {}

    if torch.cuda.is_available():
        max_memory[0] = os.getenv("LORA_MERGE_GPU_MAX_MEMORY", "256MiB")

    cpu_max_memory = os.getenv("LORA_MERGE_CPU_MAX_MEMORY")
    if cpu_max_memory:
        max_memory["cpu"] = cpu_max_memory
    else:
        available_gib = _read_mem_available_gib()
        if available_gib is not None:
            reserved_gib = 4 if available_gib > 8 else 1
            max_memory["cpu"] = f"{max(1, available_gib - reserved_gib)}GiB"

    return max_memory or None


def _normalize_no_split_modules(model) -> None:
    """Normalize nested `_no_split_modules` entries to a flat string list.

    Some MoE models expose nested sets/lists in `_no_split_modules`, which
    causes `accelerate` internals to raise `unhashable type: 'set'` when PEFT
    loads adapters.
    """
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules is None:
        return

    flattened: list[str] = []
    for item in no_split_modules:
        if isinstance(item, (set, list, tuple)):
            flattened.extend(str(value) for value in item)
        else:
            flattened.append(str(item))

    model._no_split_modules = flattened


def streaming_merge(base_model_path: str, adapter_path: str, output_path: str) -> bool:
    """記憶體友善的 PyTorch streaming LoRA 合併方案。
    不載入完整模型，只逐一 shard 或 key 合併與儲存，避免 64GB DRAM 發生 OOM。
    """
    logger.info("Initializing memory-efficient streaming merge...")
    import json
    import gc
    import shutil

    # 1. 解析 adapter 設定以取得 scaling
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        logger.warning(f"No adapter_config.json found at {adapter_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    r = config.get("r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    scaling = lora_alpha / r if r else 1.0

    # 2. 載入 LoRA 權重
    adapter_weights = {}
    adapter_st = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(adapter_st):
        from safetensors.torch import load_file
        logger.info(f"Loading adapter safetensors: {adapter_st}")
        adapter_weights = load_file(adapter_st, device="cpu")
    elif os.path.exists(adapter_bin):
        logger.info(f"Loading adapter bin: {adapter_bin}")
        adapter_weights = torch.load(adapter_bin, map_location="cpu")
    else:
        logger.warning("No adapter weights found in adapter path!")
        return False

    # 3. 建立 base weight key -> (lora_A, lora_B, scaling) 對應
    lora_mappings = {}
    prefix = "base_model.model."
    for key, tensor in adapter_weights.items():
        if "lora_A" in key and key.endswith(".weight"):
            clean_key = key[len(prefix):] if key.startswith(prefix) else key
            # 支援包含 adapter name 如 .lora_A.default.weight 或純 .lora_A.weight 結構
            if ".lora_A." in clean_key:
                parts = clean_key.split(".lora_A.")
                base_key = parts[0]
                adapter_suffix = parts[1] # e.g. "default.weight"
                lora_B_key = f"{prefix}{base_key}.lora_B.{adapter_suffix}" if key.startswith(prefix) else f"{base_key}.lora_B.{adapter_suffix}"
            else:
                base_key = clean_key.replace(".lora_A.weight", "")
                lora_B_key = f"{prefix}{base_key}.lora_B.weight" if key.startswith(prefix) else f"{base_key}.lora_B.weight"

            if lora_B_key in adapter_weights:
                base_weight_key = f"{base_key}.weight"
                lora_mappings[base_weight_key] = (tensor, adapter_weights[lora_B_key], scaling)

    logger.info(f"Loaded {len(lora_mappings)} adapter mappings.")
    if not lora_mappings:
        logger.warning("No active LoRA mappings resolved.")
        return False

    # 4. 判斷 base model 結構，取得 shards 列表
    st_index = os.path.join(base_model_path, "model.safetensors.index.json")
    bin_index = os.path.join(base_model_path, "pytorch_model.bin.index.json")

    shards_to_process = []  # list of tuples: (shard_filename, is_safetensors)
    index_file_to_copy = None

    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(st_index):
        index_file_to_copy = (st_index, "model.safetensors.index.json")
        with open(st_index, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        unique_shards = sorted(list(set(weight_map.values())))
        for s in unique_shards:
            shards_to_process.append((s, True))
    elif os.path.exists(bin_index):
        index_file_to_copy = (bin_index, "pytorch_model.bin.index.json")
        with open(bin_index, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        unique_shards = sorted(list(set(weight_map.values())))
        for s in unique_shards:
            shards_to_process.append((s, False))
    else:
        # 單一權重檔案模式
        single_st = "model.safetensors"
        single_bin = "pytorch_model.bin"
        if os.path.exists(os.path.join(base_model_path, single_st)):
            shards_to_process.append((single_st, True))
        elif os.path.exists(os.path.join(base_model_path, single_bin)):
            shards_to_process.append((single_bin, False))

    if not shards_to_process:
        logger.warning("No model weight files found in base model directory!")
        return False

    # 5. 複製所有非權重之設定檔案與 tokenizer 等檔案到 output_path
    logger.info("Copying config, tokenizer and non-weight files...")
    for item in os.listdir(base_model_path):
        item_path = os.path.join(base_model_path, item)
        if os.path.isdir(item_path):
            continue
        # 跳過索引與權重檔案，這些我們會重新建立或處理
        if (
            item.endswith(".safetensors") or
            item.endswith(".bin") or
            item.endswith(".pth") or
            item.endswith(".pt") or
            item == "model.safetensors.index.json" or
            item == "pytorch_model.bin.index.json"
        ):
            continue
        shutil.copy2(item_path, os.path.join(output_path, item))

    # 複製 index
    if index_file_to_copy:
        shutil.copy2(index_file_to_copy[0], os.path.join(output_path, index_file_to_copy[1]))

    # 優先由 adapter 複製其 tokenizer/config 相關的附加檔案 (若有)
    for item in os.listdir(adapter_path):
        if "tokenizer" in item or "config" in item or "vocabulary" in item:
            item_path = os.path.join(adapter_path, item)
            if os.path.isfile(item_path) and "adapter_config.json" not in item:
                shutil.copy2(item_path, os.path.join(output_path, item))

    # 6. 開始進行串流合併與存檔
    logger.info(f"Processing {len(shards_to_process)} shards...")
    from safetensors.torch import save_file as save_st

    for shard_file, is_st in shards_to_process:
        src_shard_path = os.path.join(base_model_path, shard_file)
        dest_shard_path = os.path.join(output_path, shard_file)
        logger.info(f"Loading shard: {shard_file}")

        if is_st:
            from safetensors.torch import load_file as load_st
            shard_dict = load_st(src_shard_path, device="cpu")
        else:
            shard_dict = torch.load(src_shard_path, map_location="cpu")

        # 確保為一般 dictionary（以提供寫入權限）
        shard_dict = dict(shard_dict)

        updated_count = 0
        for key in list(shard_dict.keys()):
            if key in lora_mappings:
                lora_A, lora_B, scaling = lora_mappings[key]
                base_tensor = shard_dict[key]
                orig_dtype = base_tensor.dtype

                # 轉為 float32 以求數值精度與合併穩定度
                W = base_tensor.to(torch.float32)
                A = lora_A.to(torch.float32)
                B = lora_B.to(torch.float32)

                # B @ A 進行合併
                update = torch.matmul(B, A) * scaling
                W += update

                shard_dict[key] = W.to(orig_dtype)
                updated_count += 1

        if updated_count > 0:
            logger.info(f"Merged {updated_count} tensors in {shard_file}")
        else:
            logger.info(f"Copied {shard_file} as-is (no lora targets present here)")

        # 儲存
        logger.info(f"Saving merged shard: {shard_file}")
        if is_st:
            save_st(shard_dict, dest_shard_path)
        else:
            torch.save(shard_dict, dest_shard_path)

        # 徹底釋放記憶體
        del shard_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Streaming merge successfully completed!")
    return True


def merge_lora(base_model_path, adapter_path, output_path, offload_folder=None, device_map=None):
    # 優先嘗試記憶體極低消耗的串流合併方案
    try:
        success = streaming_merge(base_model_path, adapter_path, output_path)
        if success:
            logger.info("Streaming merge was successful. Skipping traditional PEFT load merger.")
            return
        logger.warning("Streaming merge was not successful. Falling back to traditional PEFT load.")
    except Exception as e:
        logger.warning(f"Streaming merge failed: {e}. Falling back to traditional PEFT merger.\n{traceback.format_exc()}")

    logger.info(f"Loading base model from {base_model_path}")

    resolved_device_map = _resolve_device_map(offload_folder, device_map)
    logger.info(f"Using device_map={resolved_device_map} for LoRA merge")
    
    load_kwargs = {
        "device_map": resolved_device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if offload_folder:
        logger.info(f"Enabling disk offloading to: {offload_folder}")
        load_kwargs["offload_folder"] = offload_folder
        load_kwargs["offload_state_dict"] = True

    max_memory = _build_max_memory(resolved_device_map, offload_folder)
    if max_memory:
        load_kwargs["max_memory"] = max_memory
        logger.info(f"Using bounded max_memory for merge: {max_memory}")

    if resolved_device_map == "cpu":
        logger.info("LoRA merge is running in CPU mode to avoid GPU OOM during merge_and_unload()")

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **load_kwargs
        )

        _normalize_no_split_modules(base_model)

        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("Merging weights...")
        # merge_and_unload 會將 LoRA 權重合併進 base model 並移除 adapter layers
        # 對於 offload 的模型，accelerate 會自動處理權重的載入與儲存
        model = model.merge_and_unload()
        
        logger.info(f"Saving merged model to {output_path}")
        # max_shard_size="10GB" 確保儲存時自動分片，避免產生單一超大檔案
        if hasattr(model, "to"):
            model = model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.save_pretrained(output_path, safe_serialization=True, max_shard_size="10GB")
        
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info("Merge completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during merge: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("base_model_path", help="Path to the base model")
    parser.add_argument("adapter_path", help="Path to the LoRA adapter")
    parser.add_argument("output_path", help="Path to save the merged model")
    parser.add_argument("--offload", help="Path to offload folder for low memory merging", default=None)
    parser.add_argument(
        "--device-map",
        choices=["auto", "cpu"],
        default=os.getenv("LORA_MERGE_DEVICE_MAP"),
        help="Override merge device placement. Default: cpu when --offload is set, otherwise auto.",
    )
    
    args = parser.parse_args()
    
    merge_lora(
        args.base_model_path,
        args.adapter_path,
        args.output_path,
        args.offload,
        args.device_map,
    )
