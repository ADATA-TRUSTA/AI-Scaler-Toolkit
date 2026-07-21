"""
Model Registry: keeps a list of recommended base models and locally finetuned models.
- Registry file: service/configs/models_registry.json
- Provides functions to read/update the registry and list models for UI selection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


REGISTRY_PATH = Path("service/configs/models_registry.json")
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_max_context_length_from_hf(
    model_id: str, hf_token: Optional[str] = None
) -> Optional[int]:
    """
    從 Hugging Face 獲取模型的 max context length

    嘗試順序：
    1. 從 HF API 的 config.json 獲取 max_position_embeddings
    2. 從 config.json 獲取 model_max_length
    3. 從 config.json 獲取 n_positions
    4. 從 tokenizer_config.json 獲取 model_max_length

    Args:
        model_id: HuggingFace 模型 ID (例如: "Qwen/Qwen3-4B")
        hf_token: HuggingFace token (可選)

    Returns:
        max context length 或 None（如果無法獲取）
    """
    try:
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # 嘗試從 config.json 獲取
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        try:
            response = requests.get(config_url, headers=headers, timeout=10)
            if response.status_code == 200:
                config = response.json()

                # 嘗試多個可能的欄位名稱
                for key in [
                    "max_position_embeddings",
                    "model_max_length",
                    "n_positions",
                    "max_sequence_length",
                ]:
                    if key in config and isinstance(config[key], int):
                        logger.info(
                            f"Got max_context_length for {model_id}: {config[key]} (from {key})"
                        )
                        return config[key]
        except Exception as e:
            logger.debug(f"Failed to get config.json for {model_id}: {e}")

        # 嘗試從 tokenizer_config.json 獲取
        tokenizer_url = (
            f"https://huggingface.co/{model_id}/raw/main/tokenizer_config.json"
        )
        try:
            response = requests.get(tokenizer_url, headers=headers, timeout=10)
            if response.status_code == 200:
                tokenizer_config = response.json()
                if "model_max_length" in tokenizer_config:
                    max_len = tokenizer_config["model_max_length"]
                    # 有些模型會設置一個非常大的數字（如 1000000000），過濾掉
                    if isinstance(max_len, int) and max_len < 1000000:
                        logger.info(
                            f"Got max_context_length for {model_id}: {max_len} (from tokenizer_config)"
                        )
                        return max_len
        except Exception as e:
            logger.debug(f"Failed to get tokenizer_config.json for {model_id}: {e}")

        logger.warning(f"Could not determine max_context_length for {model_id}")
        return None

    except Exception as e:
        logger.error(f"Error getting max_context_length for {model_id}: {e}")
        return None


# Seed models used only to initialize the registry file when it does not exist yet.
# After creation, the JSON file is the single source of truth.
SEED_BASE_MODELS: List[Dict] = [
    {
        "base_model_name": "openai/gss-opt-20b",
        "label": "OPENAI GSS-Opt-20B",
        "source": "hf",
        "size": "~20B",
        "max_context_length": None,
    },
    {
        "base_model_name": "openai/gss-opt-120b",
        "label": "OPENAI GSS-Opt-120B",
        "source": "hf",
        "size": "~120B",
        "max_context_length": None,
    },
    {
        "base_model_name": "google/gemma-3-4b-it",
        "label": "Gemma 3 4B",
        "source": "hf",
        "size": "~4B",
        "max_context_length": None,
    },
    {
        "base_model_name": "google/gemma-3-12b-it",
        "label": "Gemma 3 12B",
        "source": "hf",
        "size": "~12B",
        "max_context_length": None,
    },
    {
        "base_model_name": "Qwen/Qwen3-4B",
        "label": "Qwen3 4B",
        "source": "hf",
        "size": "~4B",
        "max_context_length": None,
    },
    {
        "base_model_name": "Qwen/Qwen3-14B",
        "label": "Qwen3 14B",
        "source": "hf",
        "size": "~14B",
        "max_context_length": None,
    },
    {
        "base_model_name": "Qwen/Qwen3-32B",
        "label": "Qwen3 32B",
        "source": "hf",
        "size": "~32B",
        "max_context_length": None,
    },
]


@dataclass
class FinetunedModelInfo:
    base_model_name: Optional[str]
    method: Optional[str]
    output_dir: str
    label: Optional[str] = None
    size: Optional[str] = None
    max_context_length: Optional[int] = None
    added_at: str = datetime.now().isoformat()


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.path = REGISTRY_PATH
        self.mutex = Lock()
        self._ensure_file()
        self._initialized = True
        logger.info(f"ModelRegistry initialized at {self.path}")

    def _ensure_file(self):
        if not self.path.exists():
            data = {
                "base_models": SEED_BASE_MODELS,
                "finetuned_models": [],
                "llama_gguf_models": [],
            }
            self._write(data)

    def _read(self) -> Dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read registry, returning empty lists: {e}")
            # Do not silently re-seed here to avoid diverging from user's file.
            return {"base_models": [], "finetuned_models": []}

    def _write(self, data: Dict):
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def _identifier_variants(self, value: Optional[str]) -> set[str]:
        """Return comparable identifier variants for labels and local paths."""
        if not value or not isinstance(value, str):
            return set()

        stripped = value.strip()
        if not stripped:
            return set()

        variants = {stripped, os.path.normpath(stripped)}
        if stripped.startswith("./"):
            variants.add(stripped[2:])
            variants.add(os.path.normpath(stripped[2:]))
        return {item for item in variants if item}

    def _model_identifier_fields(self, item: Dict, model_type: str) -> List[str]:
        if model_type == "base":
            return ["label", "local_path", "base_model_name"]
        if model_type == "finetuned":
            return ["label", "output_dir"]
        if model_type == "llama_gguf":
            return ["label", "local_path", "filename"]
        return ["label"]

    def _matches_model_identifier(self, item: Dict, model_type: str, label: str) -> bool:
        target_variants = self._identifier_variants(label)
        if not target_variants:
            return False

        for field in self._model_identifier_fields(item, model_type):
            if target_variants & self._identifier_variants(item.get(field)):
                return True
        return False

    def _normalize_model_item(self, item: Dict, model_type: str) -> Dict:
        """
        統一 list_models 輸出欄位：
        {
            "model_name": "base_model_name",
            "model_path": "...",
            "label": "label",
            "size": "size",
            "max_context_length": "max_context_length"
        }

        model_path 規則：
        - base_models: local_path；若無則使用 HuggingFace 路徑(base_model_name)
        - finetuned_models: output_dir
        - llama_gguf_models: filename
        """
        base_model_name = item.get("base_model_name")

        if model_type == "base_models":
            model_path = item.get("local_path") or base_model_name
        elif model_type == "finetuned_models":
            model_path = item.get("output_dir")
        elif model_type == "llama_gguf_models":
            model_path = item.get("local_path") or item.get("filename")
        else:
            model_path = None

        return {
            "model_name": base_model_name,
            "model_path": model_path,
            "label": item.get("label"),
            "size": item.get("size"),
            "method": item.get("method"),  # only for finetuned_models
            "max_context_length": item.get("max_context_length"),
        }

    def list_models(self) -> Dict:
        """Return combined registry data suitable for UI consumption."""
        with self.mutex:
            data = self._read()
        # last_finetuned quick reference if exists
        last_info_path = self.path.parent / "last_finetuned_model.json"
        last_finetuned = None
        if last_info_path.exists():
            try:
                with open(last_info_path, "r", encoding="utf-8") as f:
                    last_finetuned = json.load(f)
            except Exception:
                last_finetuned = None

        base_models = [
            self._normalize_model_item(item, "base_models")
            for item in data.get("base_models", [])
        ]
        finetuned_models = [
            self._normalize_model_item(item, "finetuned_models")
            for item in data.get("finetuned_models", [])
        ]
        llama_gguf_models = [
            self._normalize_model_item(item, "llama_gguf_models")
            for item in data.get("llama_gguf_models", [])
        ]

        return {
            "base_models": base_models,
            "finetuned_models": finetuned_models,
            "llama_gguf_models": llama_gguf_models,
            "last_finetuned": last_finetuned,
        }

    def update_base_model_context_length(
        self, model_id: str, hf_token: Optional[str] = None
    ) -> Optional[int]:
        """
        更新指定 base model 的 max_context_length

        Args:
            model_id: 模型 ID (例如: "Qwen/Qwen3-4B")
            hf_token: HuggingFace token (可選)

        Returns:
            更新後的 max_context_length 或 None
        """
        max_len = _get_max_context_length_from_hf(model_id, hf_token)
        if max_len is not None:
            with self.mutex:
                data = self._read()
                base_models = data.get("base_models", [])
                for model in base_models:
                    if model.get("base_model_name") == model_id:
                        model["max_context_length"] = max_len
                        break
                data["base_models"] = base_models
                self._write(data)
                logger.info(f"Updated max_context_length for {model_id}: {max_len}")
        return max_len

    def refresh_all_context_lengths(self, hf_token: Optional[str] = None):
        """
        刷新所有 base models 的 max_context_length

        Args:
            hf_token: HuggingFace token (可選)
        """
        with self.mutex:
            data = self._read()
            base_models = data.get("base_models", [])

            updated_count = 0
            for model in base_models:
                model_id = model.get("base_model_name")
                if model_id and model.get("source") == "hf":
                    max_len = _get_max_context_length_from_hf(model_id, hf_token)
                    if max_len is not None:
                        model["max_context_length"] = max_len
                        updated_count += 1

            data["base_models"] = base_models
            self._write(data)
            logger.info(
                f"Refreshed max_context_length for {updated_count}/{len(base_models)} models"
            )

    def add_base_model(
        self,
        label: str,
        hf_model_name: str,
        local_path: Optional[str] = None,
        size: Optional[str] = None,
        max_context_length: Optional[int] = None,
        source: str = "hf",
    ):
        """
        添加新的基礎模型到 registry

        Args:
            label: 模型的顯示標籤
            hf_model_name: Hugging Face 模型 ID
            local_path: 本地路徑（可選）
            size: 模型大小標籤（如 "~4B"，可選）
            max_context_length: 最大上下文長度（可選）
            source: 模型來源（預設 "hf"，可為 "gguf" 等）
        """
        with self.mutex:
            data = self._read()
            base_models = data.get("base_models", [])

            # 檢查是否已存在（根據 label）
            for model in base_models:
                if model.get("label") == label:
                    logger.warning(
                        f"Base model with label '{label}' already exists, updating..."
                    )
                    model["base_model_name"] = hf_model_name
                    model["source"] = source
                    if local_path:
                        model["local_path"] = local_path
                    if size:
                        model["size"] = size
                    if max_context_length is not None:
                        model["max_context_length"] = max_context_length
                    data["base_models"] = base_models
                    self._write(data)
                    logger.info(f"Updated base model: {label}")
                    return

            # 新增模型
            new_model = {
                "base_model_name": hf_model_name,
                "label": label,
                "source": source,
                "size": size or "unknown",
                "max_context_length": max_context_length,
            }
            if local_path:
                new_model["local_path"] = local_path

            base_models.append(new_model)
            data["base_models"] = base_models
            self._write(data)
            logger.info(f"Added new base model to registry: {label} ({hf_model_name})")

    def add_llama_gguf_model(
        self,
        label: str,
        base_model_name: str,
        size: str = "unknown",
        max_context_length: Optional[int] = None,
        source: str = "local",  # default to local if not specified, user requested "hf" for downloads
        local_path: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """
        添加新的 GGUF 模型到 llama_gguf_models 清單

        Args:
            label: 模型的顯示標籤 (用戶自訂)
            base_model_name: 原始 huggingface repo name (e.g. "openai/gpt-oss-20b-F16") 或 path
            size: 模型大小 (e.g. "13GB")
            max_context_length: 最大上下文長度
            source: 來源類型 "hf" 或 "local"
            local_path: 檔案的絕對路徑
            filename: GGUF 檔案名稱 (若與 label 不同則需指定，用於 HF from_pretrained)
        """
        with self.mutex:
            data = self._read()
            gguf_models = data.get("llama_gguf_models", [])

            # 檢查是否已存在
            for model in gguf_models:
                if model.get("label") == label:
                    logger.warning(
                        f"GGUF model with label '{label}' already exists, updating..."
                    )
                    model["base_model_name"] = base_model_name
                    if size:
                        model["size"] = size
                    if max_context_length is not None:
                        model["max_context_length"] = max_context_length
                    model["source"] = source
                    if local_path:
                        model["local_path"] = local_path
                    if filename:
                        model["filename"] = filename
                    data["llama_gguf_models"] = gguf_models
                    self._write(data)
                    logger.info(f"Updated GGUF model: {label}")
                    return

            # 新增
            new_model = {
                "base_model_name": base_model_name,
                "label": label,
                "size": size,
                "max_context_length": max_context_length,
                "source": source,
                "local_path": local_path or label,  # fallback to label if label is path
            }
            if filename:
                new_model["filename"] = filename

            gguf_models.append(new_model)
            data["llama_gguf_models"] = gguf_models
            self._write(data)
            logger.info(f"Added new GGUF model to registry: {label}")

    def add_finetuned(self, info: FinetunedModelInfo):
        """Add or update a finetuned model entry by output_dir (idempotent)."""
        with self.mutex:
            data = self._read()

            # Auto-fill size/context from base model if missing
            if info.base_model_name and (not info.size or not info.max_context_length):
                base_models = data.get("base_models", [])
                for bm in base_models:
                    if (
                        bm.get("label") == info.base_model_name
                        or bm.get("base_model_name") == info.base_model_name
                    ):
                        if not info.size:
                            info.size = bm.get("size")
                        if not info.max_context_length:
                            info.max_context_length = bm.get("max_context_length")
                        break

            items: List[Dict] = data.get("finetuned_models", [])
            # de-duplicate by output_dir
            found = False
            for it in items:
                if it.get("output_dir") == info.output_dir:
                    it.update({k: v for k, v in asdict(info).items() if v is not None})
                    found = True
                    break
            if not found:
                items.append(asdict(info))
            data["finetuned_models"] = items
            self._write(data)
            logger.info(f"Finetuned model added to registry: {info.output_dir}")

    def delete_model(self, label: str) -> Optional[Dict]:
        """
        Delete a model from registry by label.
        Returns the deleted model info if found, else None.
        Checks both base_models and finetuned_models.
        """
        with self.mutex:
            data = self._read()
            deleted_model = None

            # Check base_models
            base_models = data.get("base_models", [])
            new_base_models = []
            for m in base_models:
                if self._matches_model_identifier(m, "base", label):
                    deleted_model = m
                    deleted_model["type"] = "base"
                else:
                    new_base_models.append(m)

            if deleted_model:
                data["base_models"] = new_base_models
                self._write(data)
                logger.info(f"Deleted base model from registry: {label}")
                return deleted_model

            # Check finetuned_models
            finetuned_models = data.get("finetuned_models", [])
            new_finetuned_models = []
            for m in finetuned_models:
                if self._matches_model_identifier(m, "finetuned", label):
                    deleted_model = m
                    deleted_model["type"] = "finetuned"
                else:
                    new_finetuned_models.append(m)

            if deleted_model:
                data["finetuned_models"] = new_finetuned_models
                self._write(data)
                logger.info(f"Deleted finetuned model from registry: {label}")
                return deleted_model

            # Check llama_gguf_models
            gguf_models = data.get("llama_gguf_models", [])
            new_gguf_models = []
            for m in gguf_models:
                if self._matches_model_identifier(m, "llama_gguf", label):
                    deleted_model = m
                    deleted_model["type"] = "llama_gguf"
                else:
                    new_gguf_models.append(m)

            if deleted_model:
                data["llama_gguf_models"] = new_gguf_models
                self._write(data)
                logger.info(f"Deleted GGUF model from registry: {label}")
                return deleted_model

            return None


# Singleton instance
model_registry = ModelRegistry()
