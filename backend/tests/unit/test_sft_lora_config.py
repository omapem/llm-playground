"""Tests for LoRA configuration management."""

import pytest
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from app.sft.lora_config import LoRAConfigManager
from app.sft.config import SFTConfig


def test_create_lora_config_basic():
    """Test creating basic LoRA config from SFTConfig."""
    sft_config = SFTConfig(
        base_model="gpt2",
        dataset_name="test",
        dataset_format="alpaca",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    lora_config = LoRAConfigManager.create_lora_config(sft_config)

    assert isinstance(lora_config, LoraConfig)
    assert lora_config.r == 16
    assert lora_config.lora_alpha == 32
    assert lora_config.lora_dropout == 0.1
    assert lora_config.bias == "none"
    assert lora_config.task_type == "CAUSAL_LM"


def test_create_lora_config_default_target_modules():
    """Test LoRA config uses default target modules when not specified."""
    sft_config = SFTConfig(
        base_model="gpt2",
        dataset_name="test",
        dataset_format="alpaca",
    )

    lora_config = LoRAConfigManager.create_lora_config(sft_config)

    # Should default to q_proj and v_proj (PEFT converts to set)
    assert set(lora_config.target_modules) == {"q_proj", "v_proj"}


def test_create_lora_config_custom_target_modules():
    """Test LoRA config uses custom target modules when specified."""
    sft_config = SFTConfig(
        base_model="gpt2",
        dataset_name="test",
        dataset_format="alpaca",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    lora_config = LoRAConfigManager.create_lora_config(sft_config)

    # PEFT converts to set, so compare sets
    assert set(lora_config.target_modules) == {"q_proj", "v_proj", "k_proj", "o_proj"}


def test_create_qlora_config():
    """Test creating QLoRA config with 4-bit quantization."""
    sft_config = SFTConfig(
        base_model="gpt2",
        dataset_name="test",
        dataset_format="alpaca",
        use_qlora=True,
    )

    bnb_config = LoRAConfigManager.create_qlora_config(sft_config)

    assert isinstance(bnb_config, BitsAndBytesConfig)
    assert bnb_config.load_in_4bit is True
    assert bnb_config.bnb_4bit_use_double_quant is True
    assert bnb_config.bnb_4bit_quant_type == "nf4"


def test_create_qlora_config_when_not_enabled():
    """Test QLoRA config raises error when use_qlora=False."""
    sft_config = SFTConfig(
        base_model="gpt2",
        dataset_name="test",
        dataset_format="alpaca",
        use_qlora=False,
    )

    with pytest.raises(ValueError, match="QLoRA config requested but use_qlora=False"):
        LoRAConfigManager.create_qlora_config(sft_config)


def test_get_target_modules_llama():
    """Test getting target modules for Llama models."""
    modules = LoRAConfigManager.get_target_modules_for_model("meta-llama/Llama-2-7b-hf")

    assert "q_proj" in modules
    assert "v_proj" in modules
    assert "k_proj" in modules
    assert "o_proj" in modules
    assert "gate_proj" in modules
    assert "up_proj" in modules
    assert "down_proj" in modules


def test_get_target_modules_gpt():
    """Test getting target modules for GPT models."""
    modules = LoRAConfigManager.get_target_modules_for_model("gpt2")

    assert "c_attn" in modules
    assert "c_proj" in modules


def test_get_target_modules_unknown():
    """Test getting target modules for unknown model uses defaults."""
    modules = LoRAConfigManager.get_target_modules_for_model("unknown-model")

    assert "q_proj" in modules
    assert "v_proj" in modules
    assert "k_proj" in modules
    assert "o_proj" in modules
