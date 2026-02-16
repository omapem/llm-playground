"""Supervised Fine-Tuning (SFT) module for instruction fine-tuning.

Provides components for fine-tuning pre-trained models using:
- LoRA/QLoRA for parameter-efficient fine-tuning
- Template-based instruction formatting (Alpaca, Chat)
- TRL SFTTrainer integration
- W&B experiment tracking
"""

# Components will be imported as they are implemented:
from .config import SFTConfig
from .templates import (
    PromptTemplate,
    AlpacaTemplate,
    ChatTemplate,
    TemplateRegistry,
)
from .lora_config import LoRAConfigManager

# from .dataset import SFTDatasetProcessor
# from .trainer import SFTTrainer

__all__ = [
    "SFTConfig",
    "PromptTemplate",
    "AlpacaTemplate",
    "ChatTemplate",
    "TemplateRegistry",
    "LoRAConfigManager",
]
