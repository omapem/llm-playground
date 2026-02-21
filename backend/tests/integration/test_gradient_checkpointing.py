"""Integration tests for gradient checkpointing.

Tests the gradient checkpointing feature for memory optimization during training.
"""

import pytest
import torch
from unittest.mock import MagicMock

from app.training.config import TrainingConfig
from app.training.trainer import Trainer
from app.transformer import TransformerConfig


class DummyDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, num_samples: int = 100, seq_length: int = 128):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randint(0, 1000, (self.seq_length,))


def test_gradient_checkpointing_flag_in_config():
    """Test that gradient_checkpointing flag exists in config and defaults to True."""
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,  # Disable warmup for test
    )

    # Should default to True for memory optimization
    assert hasattr(config, "gradient_checkpointing")
    assert config.gradient_checkpointing is True


def test_gradient_checkpointing_can_be_disabled():
    """Test that gradient_checkpointing can be explicitly disabled."""
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,
        gradient_checkpointing=False,
    )

    assert config.gradient_checkpointing is False


def test_gradient_checkpointing_enabled_in_trainer():
    """Test that gradient checkpointing is enabled when flag is True."""
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,
        gradient_checkpointing=True,
    )

    dataset = DummyDataset(num_samples=10)
    trainer = Trainer(config, dataset)

    # Verify the method exists and is callable
    assert hasattr(trainer.model, "gradient_checkpointing_enable")
    assert callable(trainer.model.gradient_checkpointing_enable)

    # Verify gradient checkpointing was actually enabled during init
    assert trainer.model.gradient_checkpointing is True
    assert trainer.model.transformer.gradient_checkpointing is True


def test_gradient_checkpointing_not_enabled_when_disabled():
    """Test that gradient checkpointing is not enabled when flag is False."""
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,
        gradient_checkpointing=False,
    )

    dataset = DummyDataset(num_samples=10)
    trainer = Trainer(config, dataset)

    # When disabled, we should not have enabled it
    # This test verifies the initialization logic
    assert trainer.config.gradient_checkpointing is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gradient_checkpointing_reduces_memory():
    """Test that gradient checkpointing reduces memory usage.

    This test is GPU-only and demonstrates memory savings.
    """
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    # Test with checkpointing disabled
    config_no_ckpt = TrainingConfig(
        model_config=model_config,
        max_steps=5,
        batch_size=4,
        gradient_checkpointing=False,
        mixed_precision=None,  # Disable for consistent memory measurement
    )

    dataset = DummyDataset(num_samples=10, seq_length=256)
    trainer_no_ckpt = Trainer(config_no_ckpt, dataset)

    # Clear cache before measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run one forward-backward pass
    batch = next(iter(trainer_no_ckpt.train_loader))
    batch = batch.to(trainer_no_ckpt.device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

    output = trainer_no_ckpt.model(input_ids, labels=target_ids)
    output.loss.backward()

    memory_no_ckpt = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    # Clean up
    del trainer_no_ckpt
    torch.cuda.empty_cache()

    # Test with checkpointing enabled
    config_with_ckpt = TrainingConfig(
        model_config=model_config,
        max_steps=5,
        batch_size=4,
        gradient_checkpointing=True,
        mixed_precision=None,
    )

    trainer_with_ckpt = Trainer(config_with_ckpt, dataset)

    # Clear cache before measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run one forward-backward pass
    batch = next(iter(trainer_with_ckpt.train_loader))
    batch = batch.to(trainer_with_ckpt.device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

    output = trainer_with_ckpt.model(input_ids, labels=target_ids)
    output.loss.backward()

    memory_with_ckpt = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    # Gradient checkpointing should reduce memory usage
    # Typically 30-50% reduction, but we'll be conservative
    assert memory_with_ckpt < memory_no_ckpt, (
        f"Gradient checkpointing should reduce memory: "
        f"{memory_with_ckpt:.3f}GB vs {memory_no_ckpt:.3f}GB"
    )

    # Clean up
    del trainer_with_ckpt
    torch.cuda.empty_cache()


def test_gradient_checkpointing_handles_unsupported_model():
    """Test graceful handling when model doesn't support gradient checkpointing."""
    import logging
    from unittest.mock import patch

    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,
        gradient_checkpointing=True,
    )

    dataset = DummyDataset(num_samples=10)

    # Create a trainer - it should initialize without errors
    # The model supports gradient checkpointing, so no warning
    trainer = Trainer(config, dataset)

    # Verify that gradient checkpointing is enabled
    assert trainer.model.gradient_checkpointing is True

    # Now test with a mock model that doesn't support it
    with patch('app.training.trainer.GPTModel') as MockModel:
        mock_instance = MagicMock()
        mock_instance.to.return_value = mock_instance
        mock_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        # Mock model doesn't have gradient_checkpointing_enable
        del mock_instance.gradient_checkpointing_enable
        MockModel.return_value = mock_instance

        # This should log a warning but not crash
        with patch('app.training.trainer.logger') as mock_logger:
            trainer2 = Trainer(config, dataset)
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "does not support" in mock_logger.warning.call_args[0][0]


def test_gradient_checkpointing_in_yaml_config():
    """Test that gradient_checkpointing can be loaded from YAML config."""
    import tempfile
    import yaml
    from pathlib import Path

    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        model_config=model_config,
        max_steps=10,
        warmup_steps=0,
        gradient_checkpointing=True,
    )

    # Save to YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        config.to_yaml(config_path)

    try:
        # Load from YAML
        loaded_config = TrainingConfig.from_yaml(config_path)
        assert loaded_config.gradient_checkpointing is True

        # Test with False
        config.gradient_checkpointing = False
        config.to_yaml(config_path)
        loaded_config = TrainingConfig.from_yaml(config_path)
        assert loaded_config.gradient_checkpointing is False
    finally:
        # Clean up
        Path(config_path).unlink(missing_ok=True)
