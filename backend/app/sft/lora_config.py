"""LoRA and QLoRA configuration management."""

from typing import Optional
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import torch

from .config import SFTConfig


class LoRAConfigManager:
    """Manager for creating LoRA and QLoRA configurations.

    Converts SFTConfig settings into PEFT LoraConfig and
    BitsAndBytesConfig objects for parameter-efficient fine-tuning.
    """

    @staticmethod
    def create_lora_config(config: SFTConfig) -> LoraConfig:
        """Create PEFT LoraConfig from SFTConfig.

        Args:
            config: SFT configuration

        Returns:
            LoraConfig for PEFT

        Example:
            >>> sft_config = SFTConfig(base_model="gpt2", ...)
            >>> lora_config = LoRAConfigManager.create_lora_config(sft_config)
        """
        target_modules = config.lora_target_modules or ["q_proj", "v_proj"]

        return LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    @staticmethod
    def create_qlora_config(config: SFTConfig) -> BitsAndBytesConfig:
        """Create 4-bit quantization config for QLoRA.

        Args:
            config: SFT configuration

        Returns:
            BitsAndBytesConfig for 4-bit quantization

        Example:
            >>> sft_config = SFTConfig(base_model="gpt2", use_qlora=True, ...)
            >>> bnb_config = LoRAConfigManager.create_qlora_config(sft_config)
        """
        if not config.use_qlora:
            raise ValueError("QLoRA config requested but use_qlora=False in config")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def get_target_modules_for_model(model_name: str) -> list[str]:
        """Get recommended target modules for a model architecture.

        Args:
            model_name: Model name or architecture (e.g., "llama", "gpt2")

        Returns:
            List of recommended target module names
        """
        model_lower = model_name.lower()

        # Llama family (Llama-2, Mistral, etc.)
        if any(name in model_lower for name in ["llama", "mistral", "mixtral"]):
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # GPT family
        elif "gpt" in model_lower:
            return ["c_attn", "c_proj"]

        # Default (common attention modules)
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
