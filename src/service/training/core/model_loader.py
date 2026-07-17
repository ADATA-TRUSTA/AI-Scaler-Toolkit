import torch
import logging
from typing import Optional, Tuple, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from ...config_models import TrainingConfig, TrainingMethod

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model and tokenizer loading for training."""
    
    def __init__(self, config: TrainingConfig, hf_token: Optional[str]):
        self.config = config
        self.hf_token = hf_token

    def load_tokenizer(self):
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=self.hf_token,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        """Load model based on configuration (Full, LoRA, QLoRA)."""
        if self.config.method == TrainingMethod.QLORA:
            return self._prepare_qlora_model()
        else:
            return self._prepare_standard_model()

    def _prepare_standard_model(self):
        """Load standard model (Full or LoRA)."""
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except (AttributeError, ValueError, TypeError) as e:
            # Fallback for custom models (like gpt-oss) where low_cpu_mem_usage causes meta-device initialization issues
            # specifically 'AttributeError: ... object has no attribute ...' during load_shard_file
            logger.warning(f"[ModelLoader] Fast loading failed ({e}). Retrying with low_cpu_mem_usage=False...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            
        base_model.config.use_cache = False

        if self.config.method == TrainingMethod.LORA:
            return self._apply_lora(base_model)
        else:
            return base_model

    def _prepare_qlora_model(self):
        """Create a 4-bit QLoRA model."""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info("[ModelLoader] Loading 4-bit quantized base model for QLoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            token=self.hf_token,
            dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            local_files_only=True,
        )

        base_model.config.use_cache = False

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(base_model)
        
        return self._apply_lora(model)

    def _find_linear_modules(self, model):
        """
        Dynamically find all linear modules for LoRA targeting.
        Refs: https://github.com/artidoro/qlora/blob/main/qlora.py
        """
        import bitsandbytes as bnb
        
        # Add basic Linear
        cls_set = {torch.nn.Linear}
        
        # Add Quantized Linear types if available
        if self.config.method == TrainingMethod.QLORA:
            try:
                from bitsandbytes.nn import Linear4bit
                cls_set.add(Linear4bit)
            except ImportError:
                pass
            try:
                from bitsandbytes.nn import Linear8bitLt
                cls_set.add(Linear8bitLt)
            except ImportError:
                pass

        lora_module_names = set()
        for name, module in model.named_modules():
            if any(isinstance(module, cls) for cls in cls_set):
                names = name.split('.')
                module_name = names[-1]
                # Avoid targeting the output head for stability unless explicitly requested
                if module_name != "lm_head": 
                    lora_module_names.add(module_name)
                    
        return list(lora_module_names)

    def _apply_lora(self, base_model):
        """Apply LoRA adapters."""
        target_modules = self.config.lora_target_modules
        
        # If no target modules specified, try dynamic detection first, then fallback
        if not target_modules:
            target_modules = self._find_linear_modules(base_model)
            if not target_modules:
                 # Fallback to defaults if dynamic detection finds nothing
                if self.config.method == TrainingMethod.QLORA:
                    target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]
                else:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            logger.info(f"[ModelLoader] Auto-detected LoRA target modules: {target_modules}")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        
        model = get_peft_model(base_model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        return model
