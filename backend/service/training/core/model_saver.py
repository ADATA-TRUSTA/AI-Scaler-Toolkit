import os
import logging
import torch
from ...config_models import TrainingConfig, TrainingMethod
from ...model_registry import model_registry, FinetunedModelInfo

logger = logging.getLogger(__name__)

def save_training_results(trainer, tokenizer, config: TrainingConfig):
    """Save model, tokenizer, and update registry."""
    
    # Force offline mode to prevent connection attempts to HF Hub during save
    # This is critical for offline environments where base model is already local
    original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    try:
        # Save model
        # 針對Deepspeed LoRA/QLoRA 進行特殊處理
        if getattr(config, "use_deepspeed", False) and config.method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            logger.info("[ModelSaver] Saving LoRA adapter only (bypassing full DeepSpeed checkpoint)...")
            
            os.makedirs(str(config.output_dir), exist_ok=True)
            
            model_to_save = trainer.model
            while hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            
            import deepspeed
            trainable_params = [p for p in model_to_save.parameters() if p.requires_grad]
            
            try:
                with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        model_to_save.save_pretrained(str(config.output_dir))
                        
                        # 儲存訓練參數
                        try:
                            trainer.args.save_to_json(os.path.join(config.output_dir, "training_args.json"))
                        except Exception:
                            pass
                
                torch.distributed.barrier()
            except Exception as e:
                logger.error(f"[ModelSaver] Failed to save final LoRA adapter: {e}")
                raise e
                
        else:
            # Full finetuning or non-Deepspeed LoRA/QLoRA: 使用標準儲存
            trainer.save_model()

        if config.save_tokenizer:
            tokenizer.save_pretrained(str(config.output_dir))

        # record last finetuned model
        try:
            info = {
                "base_model_name": config.model_name,
                "method": config.method.value,
                "output_dir": str(config.output_dir),
            }
            from pathlib import Path

            cfg_dir = Path("service/configs")
            cfg_dir.mkdir(parents=True, exist_ok=True)
            info_path = cfg_dir / "last_finetuned_model.json"
            info_path.write_text(
                __import__("json").dumps(info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            reg_info = FinetunedModelInfo(
                base_model_name=info["base_model_name"],
                method=info["method"],
                output_dir=info["output_dir"],
                label=str(config.output_dir),
            )
            model_registry.add_finetuned(reg_info)
        except Exception as reg_err:
            logger.warning(f"[ModelSaver] Failed to update model registry: {reg_err}")
            
    finally:
        # Restore original HF_HUB_OFFLINE state
        if original_hf_hub_offline is None:
            del os.environ["HF_HUB_OFFLINE"]
        else:
            os.environ["HF_HUB_OFFLINE"] = original_hf_hub_offline
