from typing import Dict, Optional, List
import threading
import uuid
import time
import os
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from .model_registry import model_registry
from .utils.token_utils import load_hf_token
from .settings import configure_logging

logger = configure_logging(__name__)

class DownloadTask(BaseModel):
    task_id: str
    model_id: str
    label: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    local_path: Optional[str] = None
    progress: Optional[str] = None # 用于存储简单的进度描述

class DownloadManager:
    """
    Manages background model download tasks.
    Singleton pattern.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.tasks: Dict[str, DownloadTask] = {}
        self._initialized = True

    def start_download(
        self, 
        model_id: str, 
        label: str, 
        cache_dir: Optional[str] = None, 
        force_download: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Start a new download task in a background thread.
        Returns the task_id.
        """
        task_id = str(uuid.uuid4())
        task = DownloadTask(
            task_id=task_id,
            model_id=model_id,
            label=label,
            status="pending",
            start_time=time.time(),
            progress="Initialized"
        )
        with self._lock:
            self.tasks[task_id] = task
        
        # Start thread
        thread = threading.Thread(
            target=self._download_worker,
            args=(task_id, model_id, label, cache_dir, force_download, filename),
            daemon=True
        )
        thread.start()
        
        return task_id

    def _download_worker(
        self, 
        task_id: str, 
        model_id: str, 
        label: str, 
        cache_dir: Optional[str], 
        force_download: bool,
        filename: Optional[str]
    ):
        task = self.get_task(task_id)
        if not task:
            return

        try:
            task.status = "running"
            task.progress = "Downloading from HuggingFace..."
            logger.info(f"[DownloadManager] Starting download task {task_id} for {model_id} (filename={filename})")
            
            hf_token = load_hf_token()
            
            downloaded_path = None
            max_context = None
            size_gb = "unknown"
            
            if filename:
                # GGUF 單檔/多檔下載邏輯
                from huggingface_hub import hf_hub_download, list_repo_files
                import re
                
                # 1. 檢查是否為分片檔案 (e.g., model-00001-of-00005.gguf)
                target_files = [filename]
                
                # 嘗試列出 repo 檔案以找到關聯分片與 mmproj
                all_files = []
                try:
                    task.progress = "Fetching repository file list..."
                    all_files = list_repo_files(repo_id=model_id, token=hf_token)
                except Exception as list_err:
                    logger.warning(f"[DownloadManager] Failed to list repo files: {list_err}")
                
                # 簡單的 heuristic 檢查 "-of-" pattern
                # 假設用戶選的是其中一個分片，我們應該下載全部
                if "-of-" in filename and ".gguf" in filename and all_files:
                    # 提取前綴與目錄結構，例如 "BF16/gemma-3-27b-it-BF16"
                    # 假設 filename 是 "BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf"
                    # 找到所有匹配 *-of-*.gguf 且在同一目錄下的檔案
                    
                    # 取得目錄部分
                    dir_part = os.path.dirname(filename)
                    base_name_part = os.path.basename(filename)
                    
                    # 嘗試找出 common prefix (去除 -00001-of-XXXXX.gguf)
                    # 支援非固定位數的數字 (e.g. 01-of-05 or 00001-of-00005)
                    match = re.match(r"(.*)-\d+-of-\d+\.gguf", base_name_part)
                    if match:
                        prefix = match.group(1)
                        # 篩選所有符合 prefix 且包含 -of- 的檔案
                        related_files = []
                        for f in all_files:
                            # 檢查是否在同一目錄
                            f_dir = os.path.dirname(f)
                            f_name = os.path.basename(f)
                            if f_dir == dir_part and f_name.startswith(prefix) and "-of-" in f_name and f_name.endswith(".gguf"):
                                related_files.append(f)
                        
                        if related_files:
                            target_files = sorted(related_files)
                            logger.info(f"[DownloadManager] Detected split files ({len(target_files)} parts), downloading all...")

                # 檢查並加入 mmproj 檔案 (提供多模態/影像支援)
                # 若倉庫內有包含 "mmproj" 且副檔名為 ".gguf" 的檔案，一併加入下載目標
                if all_files:
                    mmproj_files = [f for f in all_files if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
                    for mf in mmproj_files:
                        if mf not in target_files:
                            logger.info(f"[DownloadManager] Detected Vision/Multimodal projection file: {mf}, adding to download list...")
                            target_files.append(mf)

                # 2. 依次下載所有檔案
                total_files = len(target_files)
                
                for idx, f_name in enumerate(target_files):
                    task.progress = f"Downloading file {idx+1}/{total_files}: {f_name}..."
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=f_name,
                        cache_dir=cache_dir,
                        token=hf_token,
                        resume_download=True,
                        force_download=force_download
                    )
                    # 記錄第一個檔案的路徑作為主要路徑 (通常 loader 只需要第一个，或者会自动找)
                    if idx == 0:
                        downloaded_path = path

                logger.info(f"[DownloadManager] GGUF Model {model_id} downloaded. Main path: {downloaded_path}")
                
                # 3. 讀取 GGUF Metadata (嘗試使用 gguf 庫)
                task.progress = "Reading GGUF metadata..."
                try:
                    # 嘗試計算總大小
                    total_size = 0
                    for f_name in target_files:
                        # 由於 hf_hub_download 返回的是絕對路徑，我们需要重新获取每个文件的路径
                        # 但上面的 loop 没有保存所有路径。
                        # 简单起见，重新获取路径 (cached so fast)
                        p = hf_hub_download(repo_id=model_id, filename=f_name, cache_dir=cache_dir, token=hf_token)
                        total_size += os.path.getsize(p)
                    size_gb = f"{total_size / (1024**3):.1f}GB"
                    
                    # 嘗試讀取 Context Length
                    try:
                        import gguf
                        reader = gguf.GGUFReader(downloaded_path) # 只讀第一個分片通常包含 metadata
                        
                        # 定義優先查找的鍵值
                        ctx_keys = ['llama.context_length', 'qwen.context_length', 'context_length', 'n_ctx']
                        found_ctx = None
                        
                        # 1. 優先查找標準鍵
                        for key in ctx_keys:
                            if key in reader.fields:
                                field = reader.fields[key]
                                if field.parts:
                                    # parts[-1] 包含數據，可能是 list 或 numpy array
                                    raw_val = field.parts[-1]
                                    if isinstance(raw_val, list) and len(raw_val) > 0:
                                        found_ctx = raw_val[0]
                                    elif hasattr(raw_val, 'item'): # numpy scalar
                                        found_ctx = raw_val.item()
                                    elif isinstance(raw_val, (int, float)):
                                        found_ctx = raw_val
                                    
                                    if found_ctx:
                                        logger.info(f"[DownloadManager] Found GGUF context length via key '{key}': {found_ctx}")
                                        break
                        
                        # 2. 泛用查找
                        if not found_ctx:
                            for key in reader.fields:
                                if ('context_length' in key or 'n_ctx' in key) and 'rope' not in key:
                                    field = reader.fields[key]
                                    if field.parts:
                                        raw_val = field.parts[-1]
                                        val = None
                                        if isinstance(raw_val, list) and len(raw_val) > 0:
                                            val = raw_val[0]
                                        elif hasattr(raw_val, 'item'):
                                            val = raw_val.item()
                                            
                                        if isinstance(val, (int, float)):
                                            found_ctx = val
                                            logger.info(f"[DownloadManager] Found GGUF context length via heuristic key '{key}': {found_ctx}")
                                            break
                                            
                        if found_ctx:
                            max_context = int(found_ctx)
                            
                    except ImportError:
                        logger.warning("[DownloadManager] 'gguf' library not found, skipping metadata read. Please install with `pip install gguf`")
                    except Exception as meta_err:
                        logger.warning(f"[DownloadManager] Failed to read GGUF metadata: {meta_err}")

                except Exception as e:
                    logger.warning(f"[DownloadManager] Error processing GGUF size/metadata: {e}")

            else:
                # 標準 snapshot 下載
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=hf_token,
                    resume_download=True,
                    force_download=force_download,
                )
                logger.info(f"[DownloadManager] Model {model_id} downloaded to {downloaded_path}")
            
                task.progress = "Analyzing model configuration..."
                
                # Context length logic - 僅對非 GGUF 嘗試讀取 config.json
                if not filename:
                    try:
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(
                            downloaded_path,
                            token=hf_token,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        max_context = getattr(config, "max_position_embeddings", None)
                        if max_context is None:
                            max_context = getattr(config, "n_positions", None)
                        if max_context is None:
                            max_context = getattr(config, "max_sequence_length", None)
                    except Exception as e:
                        logger.warning(f"[DownloadManager] Failed to get context length: {e}")

            task.progress = "Registering model..."
            
            if filename:
                # GGUF 註冊流程
                # 根據用戶需求:
                # base_model_name -> repo_id (model_id)
                # label -> user custom label
                # filename -> specific filename
                # source -> "hf"
                # local_path -> downloaded_path
                
                model_registry.add_llama_gguf_model(
                    label=label, 
                    base_model_name=model_id, # repo_id
                    size=size_gb,
                    max_context_length=max_context,
                    source="hf",
                    local_path=downloaded_path,
                    filename=filename
                )
            else:
                # Register Standard HF Model
                model_registry.add_base_model(
                    label=label,
                    hf_model_name=model_id,
                    local_path=downloaded_path,
                    max_context_length=max_context,
                )
            
            task.status = "completed"
            task.progress = "Done"
            task.local_path = downloaded_path
            task.end_time = time.time()
            logger.info(f"[DownloadManager] Task {task_id} completed successfully")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress = "Failed"
            task.end_time = time.time()
            logger.error(f"[DownloadManager] Task {task_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[DownloadTask]:
        # Sort by start time desc
        return sorted(self.tasks.values(), key=lambda x: x.start_time, reverse=True)

# Global instance
download_manager = DownloadManager()
