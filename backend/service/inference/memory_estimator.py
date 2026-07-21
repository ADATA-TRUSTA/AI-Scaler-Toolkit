"""
GPU Memory Estimator for Model Inference
估計模型在不同配置下的 GPU 記憶體需求
支持從 Hugging Face 讀取模型配置並自動計算參數量
支持 MoE (Mixture of Experts) 模型
"""
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Optional torch import for environment detection
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - estimator should still work without torch
    torch = None

# Optional transformers for config loading
try:
    from transformers import AutoConfig  # type: ignore
except Exception:
    AutoConfig = None


# 常見模型的參數量 (單位: Billion parameters)
MODEL_SIZES = {
    # Llama 系列
    "llama-7b": 7.0,
    "llama-13b": 13.0,
    "llama-30b": 30.0,
    "llama-65b": 65.0,
    "llama-2-7b": 7.0,
    "llama-2-13b": 13.0,
    "llama-2-70b": 70.0,
    "llama-3-8b": 8.0,
    "llama-3-70b": 70.0,
    
    # Mistral 系列
    "mistral-7b": 7.0,
    "mixtral-8x7b": 47.0,  # 實際活躍參數
    
    # Qwen 系列
    "qwen3-4b": 4.0,
    "qwen3-14b": 14.0,
    "qwen3-32b": 32.0,
    "qwen3-Next-80B-A3B-Instruct": 78.59,  # MoE: 512 experts, 10 active per token
    
    # Gemma 系列
    "gemma3-4b": 4.0,
    "gemma3-12b": 12.0,

    # TinyLlama
    "tinyllama-1.1b": 1.1,
    
    # ChatGLM 系列
    "chatglm-6b": 6.0,
    "chatglm2-6b": 6.0,
    "chatglm3-6b": 6.0,

    # OpenAI GSS-Opt 系列
    "gss-opt-20b": 20.0,
    "gss-opt-120b": 120.0,
}


class MemoryEstimator:
    """記憶體需求估算器 - 支援從 HuggingFace 讀取配置與 MoE 模型"""
    
    def __init__(self):
        self.model_sizes = MODEL_SIZES
        self._config_cache = {}  # Cache for loaded configs
    
    def _load_model_config(self, model_name: str) -> Optional[Any]:
        """
        從 Hugging Face 載入模型配置
        
        Args:
            model_name: 模型名稱或路徑
        
        Returns:
            模型配置物件或 None
        """
        if model_name in self._config_cache:
            return self._config_cache[model_name]
        
        if AutoConfig is None:
            logger.warning("transformers 未安裝，無法自動載入模型配置")
            return None
        
        try:
            config = AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=True,
                local_files_only=True)
            self._config_cache[model_name] = config
            logger.info(f"成功載入模型配置: {model_name}")
            return config
        except Exception as e:
            logger.warning(f"無法載入模型配置 {model_name}: {e}")
            return None
    
    def _calculate_params_from_config(self, config: Any) -> Tuple[float, bool, Dict]:
        """
        從配置計算模型參數量
        
        Args:
            config: 模型配置物件
        
        Returns:
            (參數量(B), 是否為MoE, 詳細資訊)
        """
        params_info = {
            "total_params": 0,
            "active_params": 0,
            "is_moe": False,
            "num_experts": 0,
            "experts_per_token": 0,
            "architecture": getattr(config, "architectures", ["unknown"])[0] if hasattr(config, "architectures") else "unknown"
        }
        
        # 處理多模態模型（如 Gemma 3）- 使用 text_config
        if hasattr(config, "text_config") and config.text_config is not None:
            logger.info("偵測到多模態模型，使用 text_config 進行計算")
            config = config.text_config
        
        # 檢測 MoE 模型
        is_moe = False
        num_experts = 0
        experts_per_token = 1
        moe_intermediate_size = None
        shared_expert_intermediate_size = None
        
        # Mixtral 風格的 MoE
        if hasattr(config, "num_local_experts"):
            is_moe = True
            num_experts = getattr(config, "num_local_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True
        
        # DeepSeek MoE 風格
        elif hasattr(config, "n_routed_experts"):
            is_moe = True
            num_experts = getattr(config, "n_routed_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True
        
        # Qwen3-Next 風格的 MoE (使用 num_experts)
        elif hasattr(config, "num_experts") and getattr(config, "num_experts", 0) > 1:
            is_moe = True
            num_experts = getattr(config, "num_experts", 8)
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
            moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
            shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", None)
            params_info["num_experts"] = num_experts
            params_info["experts_per_token"] = experts_per_token
            params_info["is_moe"] = True
            params_info["moe_intermediate_size"] = moe_intermediate_size
            params_info["shared_expert_intermediate_size"] = shared_expert_intermediate_size
        
        # 獲取基本模型參數
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        vocab_size = getattr(config, "vocab_size", 32000)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        
        # 計算嵌入層參數
        embedding_params = vocab_size * hidden_size
        
        # 計算注意力層參數 (每層)
        # Q, K, V projections + output projection
        attention_params_per_layer = (
            hidden_size * hidden_size +  # Q
            hidden_size * (hidden_size // num_attention_heads) * num_key_value_heads +  # K (GQA support)
            hidden_size * (hidden_size // num_attention_heads) * num_key_value_heads +  # V (GQA support)
            hidden_size * hidden_size  # output projection
        )
        
        # 計算 FFN 參數 (每層)
        if is_moe:
            # MoE: router + (experts * FFN_size)
            # 每個 expert 包含 gate_proj, up_proj, down_proj
            router_params = hidden_size * num_experts
            
            # 使用 moe_intermediate_size 如果存在（Qwen3-Next 風格）
            expert_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
            
            # 每個 expert 的參數: gate(hidden->intermediate) + up(hidden->intermediate) + down(intermediate->hidden)
            params_per_expert = (hidden_size * expert_intermediate_size) + (hidden_size * expert_intermediate_size) + (expert_intermediate_size * hidden_size)
            expert_params = num_experts * params_per_expert
            
            # Shared expert (如果存在)
            shared_expert_params = 0
            if shared_expert_intermediate_size is not None and shared_expert_intermediate_size > 0:
                shared_expert_params = (hidden_size * shared_expert_intermediate_size) + (hidden_size * shared_expert_intermediate_size) + (shared_expert_intermediate_size * hidden_size)
            
            ffn_params_per_layer = router_params + expert_params + shared_expert_params
            
            # 活躍參數：只有被選中的 experts 參與計算
            active_expert_params = experts_per_token * params_per_expert
            active_ffn_params_per_layer = router_params + active_expert_params + shared_expert_params
        else:
            # 標準 FFN: gate + up + down projections
            ffn_params_per_layer = (hidden_size * intermediate_size) + (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
            active_ffn_params_per_layer = ffn_params_per_layer
        
        # Layer norm 參數 (可忽略不計，但為完整性加入)
        norm_params_per_layer = hidden_size * 2  # pre-attention norm + post-ffn norm
        
        # 總參數計算
        total_layer_params = num_layers * (
            attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        )
        
        active_layer_params = num_layers * (
            attention_params_per_layer + active_ffn_params_per_layer + norm_params_per_layer
        )
        
        # 輸出層 (lm_head)
        output_params = vocab_size * hidden_size
        
        # 總計
        total_params = embedding_params + total_layer_params + output_params
        active_params = embedding_params + active_layer_params + output_params
        
        params_info["total_params"] = total_params / 1e9  # 轉換為 Billion
        params_info["active_params"] = active_params / 1e9
        params_info["hidden_size"] = hidden_size
        params_info["num_layers"] = num_layers
        params_info["intermediate_size"] = intermediate_size
        params_info["vocab_size"] = vocab_size
        
        # 返回 total_params（用於內存估算），is_moe，和詳細資訊
        return params_info["total_params"], is_moe, params_info
    
    def extract_model_size(self, model_name: str) -> Optional[float]:
        """
        從模型名稱中提取參數量，優先嘗試從 HuggingFace 載入配置
        
        Args:
            model_name: 模型名稱，例如 "meta-llama/Llama-2-7b-chat-hf"
        
        Returns:
            參數量 (Billion) 或 None
        """
        # 優先嘗試從 HuggingFace 載入配置
        config = self._load_model_config(model_name)
        if config is not None:
            try:
                size, is_moe, info = self._calculate_params_from_config(config)
                logger.info(f"從配置計算得到參數量: {size:.2f}B (MoE: {is_moe})")
                return size
            except Exception as e:
                logger.warning(f"從配置計算參數量失敗: {e}")
        
        # 回退到名稱匹配
        model_name_lower = model_name.lower()
        
        # 嘗試從預定義的字典匹配
        for key, size in self.model_sizes.items():
            if key in model_name_lower:
                return size
        
        # 嘗試從名稱中提取數字
        # 例如: "7b", "13b", "70b"
        import re
        
        # 匹配 XXb 格式
        match = re.search(r'(\d+\.?\d*)b', model_name_lower)
        if match:
            size = float(match.group(1))
            return size
        
        # 匹配 XX-billion 格式
        match = re.search(r'(\d+\.?\d*)-?billion', model_name_lower)
        if match:
            return float(match.group(1))
        
        logger.warning(f"無法從模型名稱提取參數量: {model_name}")
        return None
    
    def estimate_memory_requirements(
        self,
        model_name: str,
        quantization: str = "none",
        include_activations: bool = True,
        batch_size: int = 1,
        sequence_length: int = 2048,
    ) -> Dict:
        """
        估計模型的記憶體需求（支援 MoE 模型）
        
        Args:
            model_name: 模型名稱
            quantization: 量化類型 (none, int8, int4, nf4, fp4)
            include_activations: 是否包含激活值記憶體
            batch_size: 批次大小
            sequence_length: 序列長度
        
        Returns:
            記憶體估計結果字典
        """
        # 嘗試載入配置獲取完整資訊
        config = self._load_model_config(model_name)
        is_moe = False
        params_info = {}
        
        if config is not None:
            try:
                model_size_b, is_moe, params_info = self._calculate_params_from_config(config)
                logger.info(f"使用配置計算: {model_size_b:.2f}B 參數 (MoE: {is_moe})")
            except Exception as e:
                logger.warning(f"配置計算失敗，回退到名稱提取: {e}")
                model_size_b = self.extract_model_size(model_name)
        else:
            # 提取模型大小（回退方式）
            model_size_b = self.extract_model_size(model_name)
        
        if model_size_b is None:
            return {
                "error": "無法識別模型大小",
                "model_name": model_name,
                "suggestion": "請確認模型名稱正確或 transformers 已安裝"
            }
        
        # 計算模型權重記憶體
        # 對於 MoE，使用 total_params（所有 experts），而非 active_params
        total_params_for_memory = params_info.get("total_params", model_size_b) if is_moe else model_size_b
        model_memory = self._calculate_model_memory(total_params_for_memory, quantization)
        
        # 計算激活值記憶體（推理時的中間結果）
        # 使用實際配置或估計
        hidden_size = params_info.get("hidden_size") if params_info else self._estimate_hidden_size(model_size_b)
        num_layers = params_info.get("num_layers") if params_info else self._estimate_num_layers(model_size_b)
        
        activation_memory = 0
        if include_activations:
            activation_memory = self._calculate_activation_memory_with_config(
                hidden_size, num_layers, batch_size, sequence_length, quantization
            )
        
        # 計算 KV cache 記憶體
        kv_cache_memory = self._calculate_kv_cache_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length
        )

        # 執行環境開銷（Python/Torch/CUDA 等）
        overhead_breakdown = self._estimate_runtime_overhead(quantization)
        overhead_memory = overhead_breakdown["total"]

        # 總記憶體需求
        total_memory = model_memory + activation_memory + kv_cache_memory + overhead_memory

        # 推薦的最低 GPU 記憶體
        recommended_gpu_memory = total_memory * 1.1  # 預留 10% 安全餘量
        
        # 在 offload 混合模式下的最低 GPU 記憶體需求
        min_gpu_with_offload = activation_memory + kv_cache_memory + overhead_memory + (model_memory * 0.1)
        
        result = {
            "model_name": model_name,
            "model_size_billions": round(model_size_b, 2),
            "quantization": quantization,
            "memory_breakdown_gb": {
                "model_weights": round(model_memory, 2),
                "activations": round(activation_memory, 2),
                "kv_cache": round(kv_cache_memory, 2),
                "overhead": round(overhead_memory, 2),
                "total": round(total_memory, 2)
            },
            "overhead_details_gb": {
                "python_runtime": round(overhead_breakdown["python_runtime"], 2),
                "pytorch_framework": round(overhead_breakdown["pytorch_framework"], 2),
                "cuda_context": round(overhead_breakdown["cuda_context"], 2),
                "cuda_libraries": round(overhead_breakdown["cuda_libraries"], 2),
                "transformers_lib": round(overhead_breakdown["transformers_lib"], 2),
                "quantization_lib": round(overhead_breakdown["quantization_lib"], 2),
                "cuda_driver": round(overhead_breakdown["cuda_driver"], 2),
                "total": round(overhead_breakdown["total"], 2)
            },
            "recommendations": {
                "full_gpu_memory_gb": round(recommended_gpu_memory, 2),
                "min_gpu_with_cpu_offload_gb": round(min_gpu_with_offload, 2),
                "min_gpu_with_disk_offload_gb": round(min_gpu_with_offload * 0.7, 2),
            },
            "offload_strategies": self._generate_offload_strategies(
                model_memory, activation_memory, kv_cache_memory, overhead_memory
            ),
            "notes": [
                f"估計基於 {model_size_b:.2f}B 參數模型",
                f"批次大小: {batch_size}, 序列長度: {sequence_length}",
            ]
        }
        
        # MoE 專屬資訊
        if is_moe:
            result["moe_info"] = {
                "is_moe": True,
                "num_experts": params_info.get("num_experts", 0),
                "experts_per_token": params_info.get("experts_per_token", 0),
                "total_params_billions": round(params_info.get("total_params", 0), 2),
                "active_params_billions": round(params_info.get("active_params", 0), 2),
            }
            result["notes"].append(
                f"MoE 模型: {params_info.get('num_experts')} experts, "
                f"每 token 使用 {params_info.get('experts_per_token')} experts"
            )
            result["notes"].append(
                f"總參數: {params_info.get('total_params', 0):.2f}B, "
                f"活躍參數: {params_info.get('active_params', 0):.2f}B"
            )
        
        # 配置資訊
        if params_info:
            result["model_config"] = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "intermediate_size": params_info.get("intermediate_size"),
                "vocab_size": params_info.get("vocab_size"),
                "architecture": params_info.get("architecture", "unknown"),
            }
            result["notes"].append(f"配置來源: Hugging Face ({params_info.get('architecture')})")
        else:
            result["notes"].append("配置來源: 估計值（建議安裝 transformers 以獲取精確配置）")
        
        result["notes"].extend([
            "實際記憶體使用可能因框架與執行環境（含 CUDA/驅動/庫）而異",
            "建議預留 20% 安全餘量",
            ("偵測到 CUDA 環境並已估入額外開銷" if (torch and hasattr(torch, "cuda") and torch.cuda.is_available()) else "目前為 CPU-only 環境，開銷較低")
        ])
        
        return result

    def _estimate_runtime_overhead(self, quantization: str) -> Dict[str, float]:
        """
        估計執行環境的詳細記憶體開銷（GB）。

        包含：
        - Python 進程基本記憶體 (~0.3 GB)
        - PyTorch 框架開銷 (~0.5-1.0 GB，依版本而異)
        - CUDA 上下文初始化 (每 GPU ~0.5-1.0 GB)
        - CUDA 核心庫工作區 (cuBLAS, cuDNN, cuSPARSE ~0.5-1.5 GB)
        - Transformers 函式庫 (~0.2-0.5 GB)
        - 量化函式庫 (bitsandbytes ~0.3-0.5 GB)
        - CUDA 驅動與執行期 (~0.2-0.3 GB)

        Returns:
            包含各項開銷明細與總和的字典
        
        注意：此為經驗值，實際數字會依驅動版本、CUDA 版本、顯卡世代與設定不同。
        """
        overhead = {
            "python_runtime": 0.3,  # Python 進程與核心庫
            "pytorch_framework": 0.0,
            "cuda_context": 0.0,
            "cuda_libraries": 0.0,
            "transformers_lib": 0.2,
            "quantization_lib": 0.0,
            "cuda_driver": 0.0,
        }
        
        # PyTorch 框架開銷 (依是否有 CUDA 支援而異)
        if torch and torch.cuda.is_available():
            overhead["pytorch_framework"] = 0.8  # CUDA 版本較大
            
            # 偵測 GPU 資訊以調整估計值
            try:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                compute_capability = torch.cuda.get_device_capability(0) if device_count > 0 else (0, 0)
                
                # CUDA 上下文 (每個 GPU)
                # 較新的 GPU (Compute Capability >= 8.0, Ampere+) 可能需要更多
                if compute_capability[0] >= 8:  # Ampere (A100, RTX 30xx) 或更新
                    overhead["cuda_context"] = 0.8 * device_count
                elif compute_capability[0] >= 7:  # Volta/Turing (V100, T4, RTX 20xx)
                    overhead["cuda_context"] = 0.6 * device_count
                else:  # 較舊的 GPU
                    overhead["cuda_context"] = 0.5 * device_count
                
                # CUDA 核心庫工作區 (cuBLAS, cuDNN, cuSPARSE 等)
                # 依 GPU 世代與 CUDA 版本調整
                if compute_capability[0] >= 8:
                    # 新世代 GPU 支援更多 CUDA 核心功能 (Tensor Cores, etc.)
                    overhead["cuda_libraries"] = 1.1
                elif compute_capability[0] >= 7:
                    overhead["cuda_libraries"] = 0.8
                else:
                    overhead["cuda_libraries"] = 0.5
                
                # CUDA 驅動與執行期
                overhead["cuda_driver"] = 0.25
                
                logger.debug(f"偵測到 GPU: {device_name} (Compute {compute_capability[0]}.{compute_capability[1]})")
            except Exception as e:
                # 若 GPU 偵測失敗，使用保守估計值
                logger.warning(f"GPU 資訊偵測失敗，使用預設值: {e}")
                overhead["cuda_context"] = 0.6
                overhead["cuda_libraries"] = 0.8
                overhead["cuda_driver"] = 0.25
        else:
            # CPU-only 環境
            overhead["pytorch_framework"] = 0.5
            overhead["cuda_context"] = 0.0
            overhead["cuda_libraries"] = 0.0
            overhead["cuda_driver"] = 0.0
        
        # 量化函式庫 (bitsandbytes, 僅當使用量化時)
        if quantization.lower() in {"int8", "int4", "nf4", "fp4"}:
            # bitsandbytes 需載入 CUDA 核心與量化常數
            overhead["quantization_lib"] = 0.4 if torch and torch.cuda.is_available() else 0.1
        
        # 計算總開銷
        total = sum(overhead.values())
        overhead["total"] = round(total, 2)
        
        return overhead
    
    def _calculate_model_memory(self, model_size_b: float, quantization: str) -> float:
        """
        計算模型權重記憶體需求
        
        Args:
            model_size_b: 模型大小 (Billion parameters)
            quantization: 量化類型
        
        Returns:
            記憶體需求 (GB)
        """
        # 每個參數的位元數
        bits_per_param = {
            "none": 16,      # FP16/BF16
            "fp16": 16,
            "bf16": 16,
            "int8": 8,
            "int4": 4,
            "nf4": 4,
            "fp4": 4,
        }
        
        bits = bits_per_param.get(quantization.lower(), 16)
        
        # 計算記憶體 (GB)
        # 1B parameters * bits_per_param / 8 (bytes) / 1024^3 (GB)
        memory_gb = (model_size_b * 1e9 * bits / 8) / (1024 ** 3)
        
        return memory_gb
    
    def _calculate_activation_memory(
        self, model_size_b: float, batch_size: int, sequence_length: int, quantization: str
    ) -> float:
        """更精確推理激活記憶體估計（回退方法）"""
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        return self._calculate_activation_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length, quantization
        )
    
    def _calculate_activation_memory_with_config(
        self, hidden_size: int, num_layers: int, batch_size: int, 
        sequence_length: int, quantization: str
    ) -> float:
        """使用實際配置計算激活記憶體"""
        bytes_per_element = 2 if quantization == "none" else 1
        
        # 推理時僅需保留少部分中間層記憶體
        raw_memory = (batch_size * sequence_length * hidden_size * num_layers * bytes_per_element) / (1024 ** 3)
        return raw_memory / 30  # 大約減少到訓練的 1/30
    
    def _calculate_kv_cache_memory(self, model_size_b, batch_size, sequence_length) -> float:
        """調整 KV cache 為更準確的實際值（回退方法）"""
        hidden_size = self._estimate_hidden_size(model_size_b)
        num_layers = self._estimate_num_layers(model_size_b)
        return self._calculate_kv_cache_memory_with_config(
            hidden_size, num_layers, batch_size, sequence_length
        )
    
    def _calculate_kv_cache_memory_with_config(
        self, hidden_size: int, num_layers: int, batch_size: int, sequence_length: int
    ) -> float:
        """使用實際配置計算 KV cache"""
        bytes_per_element = 2  # FP16
        
        kv_cache = (2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element) / (1024 ** 3)
        return kv_cache * 0.6  # 實際平均使用約 60%
    
    def _estimate_hidden_size(self, model_size_b: float) -> int:
        """估計隱藏層大小"""
        if model_size_b <= 1:
            return 2048
        elif model_size_b <= 3:
            return 2560
        elif model_size_b <= 7:
            return 4096
        elif model_size_b <= 13:
            return 5120
        elif model_size_b <= 30:
            return 6656
        elif model_size_b <= 70:
            return 8192
        else:
            return 12288
    
    def _estimate_num_layers(self, model_size_b: float) -> int:
        """估計層數"""
        if model_size_b <= 1:
            return 22
        elif model_size_b <= 3:
            return 32
        elif model_size_b <= 7:
            return 32
        elif model_size_b <= 13:
            return 40
        elif model_size_b <= 30:
            return 60
        elif model_size_b <= 70:
            return 80
        else:
            return 120
    
    def _generate_offload_strategies(
        self,
        model_memory: float,
        activation_memory: float,
        kv_cache_memory: float,
        overhead_memory: float
    ) -> list:
        """
        生成不同 GPU 記憶體大小的 offload 策略建議
        """
        strategies = []
        
        # 策略 1: 全 GPU (無 offload)
        full_gpu = model_memory + activation_memory + kv_cache_memory + overhead_memory
        strategies.append({
            "name": "Full GPU (No Offload)",
            "min_gpu_gb": round(full_gpu * 1.1, 2),
            "description": "所有模型權重和計算都在 GPU 上",
            "performance": "最快",
            "config": {"offload": "none"}
        })
        
        # 策略 2: CPU Offload (部分權重)
        cpu_offload_50 = (model_memory * 0.5) + activation_memory + kv_cache_memory + overhead_memory
        strategies.append({
            "name": "CPU Offload (50% weights)",
            "min_gpu_gb": round(cpu_offload_50 * 1.1, 2),
            "description": "50% 模型權重 offload 到 CPU",
            "performance": "中等",
            "config": {"offload": "cpu", "device_map": "auto"}
        })
        
        # 策略 3: CPU Offload (大部分權重)
        cpu_offload_80 = (model_memory * 0.2) + activation_memory + kv_cache_memory + overhead_memory
        strategies.append({
            "name": "CPU Offload (80% weights)",
            "min_gpu_gb": round(cpu_offload_80 * 1.1, 2),
            "description": "80% 模型權重 offload 到 CPU，僅關鍵層在 GPU",
            "performance": "較慢",
            "config": {"offload": "cpu", "device_map": "auto"}
        })
        
        # 策略 4: Disk Offload
        disk_offload = (model_memory * 0.1) + activation_memory + kv_cache_memory + overhead_memory
        strategies.append({
            "name": "Disk Offload (90% weights)",
            "min_gpu_gb": round(disk_offload * 1.1, 2),
            "description": "大部分權重 offload 到硬碟（NVMe 推薦）",
            "performance": "最慢，但 GPU 需求最低",
            "config": {"offload": "disk", "offload_dir": "./offload"}
        })
        
        return strategies


# 創建全局實例
memory_estimator = MemoryEstimator()
