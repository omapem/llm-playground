"""Tests for SFT configuration management."""

import pytest
import yaml
from pathlib import Path
from app.sft.config import SFTConfig


def test_sft_config_init_with_required_fields():
    """Test SFTConfig initialization with required fields."""
    config = SFTConfig(
        base_model="gpt2",
        dataset_name="test_dataset",
        dataset_format="alpaca",
    )

    assert config.base_model == "gpt2"
    assert config.dataset_name == "test_dataset"
    assert config.dataset_format == "alpaca"
    # Check defaults
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.max_seq_length == 2048


def test_sft_config_missing_required_field():
    """Test SFTConfig raises error when required field is missing."""
    with pytest.raises(TypeError):
        SFTConfig(base_model="gpt2")  # Missing dataset_name and format


def test_sft_config_invalid_dataset_format():
    """Test SFTConfig raises error for invalid dataset format."""
    with pytest.raises(ValueError, match="dataset_format must be one of"):
        SFTConfig(
            base_model="gpt2",
            dataset_name="test_dataset",
            dataset_format="invalid_format",
        )


def test_sft_config_invalid_validation_split():
    """Test SFTConfig raises error for invalid validation split."""
    with pytest.raises(ValueError, match="validation_split must be in"):
        SFTConfig(
            base_model="gpt2",
            dataset_name="test_dataset",
            dataset_format="alpaca",
            validation_split=1.5,  # Invalid: must be < 1.0
        )


def test_sft_config_qlora_sets_paged_optimizer():
    """Test QLoRA automatically uses paged optimizer."""
    config = SFTConfig(
        base_model="llama-2-7b",
        dataset_name="yahma/alpaca-cleaned",
        dataset_format="alpaca",
        use_qlora=True,
    )

    assert config.use_qlora is True
    assert config.optimizer_type == "paged_adamw_32bit"


def test_sft_config_from_yaml(tmp_path):
    """Test loading config from YAML file."""
    config_data = {
        "base_model": "gpt2",
        "dataset_name": "test_dataset",
        "dataset_format": "alpaca",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 8,
    }

    # Write YAML file
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # Load config
    config = SFTConfig.from_yaml(str(config_path))

    assert config.base_model == "gpt2"
    assert config.dataset_name == "test_dataset"
    assert config.dataset_format == "alpaca"
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.batch_size == 8


def test_sft_config_to_yaml(tmp_path):
    """Test saving config to YAML file."""
    config = SFTConfig(
        base_model="gpt2",
        dataset_name="test_dataset",
        dataset_format="alpaca",
        lora_r=16,
        batch_size=8,
    )

    # Save to YAML
    config_path = tmp_path / "output_config.yaml"
    config.to_yaml(str(config_path))

    # Verify file exists
    assert config_path.exists()

    # Load and verify contents
    with open(config_path) as f:
        loaded_data = yaml.safe_load(f)

    assert loaded_data["base_model"] == "gpt2"
    assert loaded_data["dataset_name"] == "test_dataset"
    assert loaded_data["dataset_format"] == "alpaca"
    assert loaded_data["lora_r"] == 16
    assert loaded_data["batch_size"] == 8


def test_sft_config_from_yaml_missing_file():
    """Test loading from non-existent file raises error."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        SFTConfig.from_yaml("/nonexistent/path/config.yaml")
