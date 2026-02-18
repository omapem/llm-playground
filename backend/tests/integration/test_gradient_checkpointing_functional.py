"""Functional test for gradient checkpointing during training.

This test verifies that gradient checkpointing works correctly during
forward and backward passes.
"""

import pytest
import torch

from app.transformer import TransformerConfig
from app.transformer.model import GPTModel


def test_gradient_checkpointing_forward_backward():
    """Test that gradient checkpointing works during forward/backward pass."""
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    model = GPTModel(config)
    model.train()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    assert model.gradient_checkpointing is True
    assert model.transformer.gradient_checkpointing is True

    # Create sample input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(input_ids, labels=labels)
    assert output.loss is not None

    # Backward pass - should work with gradient checkpointing
    output.loss.backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_gradient_checkpointing_disable():
    """Test that gradient checkpointing can be disabled."""
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    model = GPTModel(config)
    model.train()

    # Enable then disable
    model.gradient_checkpointing_enable()
    assert model.gradient_checkpointing is True

    model.gradient_checkpointing_disable()
    assert model.gradient_checkpointing is False
    assert model.transformer.gradient_checkpointing is False

    # Forward/backward should still work
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    output.loss.backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_gradient_checkpointing_inference_mode():
    """Test that gradient checkpointing doesn't affect inference."""
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    model = GPTModel(config)
    model.gradient_checkpointing_enable()
    model.eval()

    # Forward pass in eval mode
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
