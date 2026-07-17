import logging
import json
from pathlib import Path
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

def load_training_dataset(dataset_path: str):
    """Load dataset from local file or HuggingFace hub.
    
    Supports multiple formats:
    1. JSON/JSONL files with 'text' field
    2. JSON/JSONL files with 'prompt' and 'completion' fields
    3. HuggingFace dataset names
    
    For prompt-completion format, the function will automatically combine
    them into a single text field for training.
    """
    if dataset_path.endswith((".json", ".jsonl")):
        try:
            # 嘗試載入數據集，並指定正確的欄位
            logger.info(f"[DatasetLoader] Loading dataset from {dataset_path}")
            
            # 使用 load_dataset 載入 JSON/JSONL 檔案
            dataset = hf_load_dataset(
                "json", 
                data_files={"train": dataset_path},
                split="train"
            )
            
            logger.info(f"[DatasetLoader] Loaded {len(dataset)} examples")
            
            # 檢查數據集格式並自動轉換 prompt-completion 格式
            if len(dataset) > 0:
                first_example = dataset[0]
                logger.debug(f"[DatasetLoader] First example keys: {list(first_example.keys())}")
                
                # 模式三：OpenAI chat format（messages 欄位）
                # TRL SFTTrainer >= 0.12 會自動套用 tokenizer.apply_chat_template
                if "messages" in first_example:
                    messages = first_example["messages"]
                    if not isinstance(messages, list) or not all(
                        isinstance(m, dict) and "role" in m and "content" in m
                        for m in messages
                    ):
                        raise ValueError(
                            "Dataset 'messages' field must be a list of {role, content} dicts. "
                            f"Got: {messages[:1]}"
                        )
                    logger.info(
                        "[DatasetLoader] Detected OpenAI chat format (messages field); "
                        "passing through for TRL to apply chat template"
                    )

                # 模式二：prompt + completion 雙欄位
                elif "prompt" in first_example and "completion" in first_example:
                    logger.info("[DatasetLoader] Detected prompt-completion format")

                # 模式一：單一 text 欄位
                elif "text" not in first_example:
                    raise ValueError(
                        f"Dataset must have a 'messages' field (OpenAI chat format), "
                        f"both 'prompt' and 'completion' fields, or a 'text' field. "
                        f"Found fields: {list(first_example.keys())}"
                    )
            else:
                raise ValueError("Dataset is empty")
            
            return dataset
            
        except Exception as e:
            logger.error(f"[DatasetLoader] Failed to load dataset from {dataset_path}: {e}")
            # 提供更詳細的錯誤信息
            
            # 嘗試讀取並驗證 JSON 格式
            try:
                dataset_file = Path(dataset_path)
                if not dataset_file.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
                
                logger.info(f"[DatasetLoader] Validating JSON format in {dataset_path}...")
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logger.info(f"[DatasetLoader] File has {len(lines)} lines")
                    
                    # 檢查每一行是否是有效的 JSON
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # 跳過空行
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as json_err:
                            raise ValueError(
                                f"Invalid JSON at line {i}: {line[:100]}... Error: {json_err}"
                            )
                
                logger.info("[DatasetLoader] JSON format validation passed")
                
            except Exception as validate_err:
                logger.error(f"[DatasetLoader] Dataset validation error: {validate_err}")
                raise ValueError(f"Dataset validation failed: {validate_err}") from e
            
            # 如果驗證通過但還是失敗，拋出原始錯誤
            raise
    
    # HuggingFace 數據集
    return hf_load_dataset(dataset_path)["train"]
