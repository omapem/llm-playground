"""Model architecture configurator for creating and managing transformer configurations."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .transformer_block import StackedTransformerBlocks
from .embeddings import PositionalEncoding, LearnedPositionalEncoding
from .attention import create_causal_mask


@dataclass
class TransformerConfig:
    """Configuration for a transformer model.

    Attributes:
        hidden_size: Model dimension (must be divisible by num_heads)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        intermediate_size: FFN hidden dimension (typically 4x hidden_size)
        vocab_size: Size of vocabulary for embeddings
        max_position_embeddings: Maximum sequence length supported
        activation: Activation function ('relu', 'gelu', 'gelu_approximate')
        dropout_rate: Dropout probability
        norm_first: Use pre-norm if True, post-norm if False
        position_encoding_type: 'sinusoidal' or 'learned'
        tie_embeddings: Whether to share weights between input and output embeddings
    """

    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    activation: str = "gelu"
    dropout_rate: float = 0.1
    norm_first: bool = True
    position_encoding_type: str = "sinusoidal"
    tie_embeddings: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformerConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )

        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {self.dropout_rate}"
            )

        if self.activation not in ["relu", "gelu", "gelu_approximate"]:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.position_encoding_type not in ["sinusoidal", "learned"]:
            raise ValueError(
                f"Unknown position encoding type: {self.position_encoding_type}"
            )

    def get_head_dim(self) -> int:
        """Get dimension per head."""
        return self.hidden_size // self.num_heads


# Pre-defined configurations
class GPT2Small(TransformerConfig):
    """GPT-2 small configuration."""

    def __init__(self):
        super().__init__(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
            vocab_size=50257,
            max_position_embeddings=1024,
            activation="gelu",
            dropout_rate=0.1,
            position_encoding_type="learned",
        )


class GPT2Medium(TransformerConfig):
    """GPT-2 medium configuration."""

    def __init__(self):
        super().__init__(
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            max_position_embeddings=1024,
            activation="gelu",
            dropout_rate=0.1,
            position_encoding_type="learned",
        )


class Llama7B(TransformerConfig):
    """Llama 7B-like configuration."""

    def __init__(self):
        super().__init__(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            max_position_embeddings=2048,
            activation="gelu",
            dropout_rate=0.0,
            position_encoding_type="sinusoidal",
        )


class ModelConfigurator:
    """Configures and instantiates transformer models.

    Provides utilities for creating, validating, and working with transformer
    model architectures.
    """

    @staticmethod
    def create_model(config: TransformerConfig) -> nn.Module:
        """Create a transformer model from configuration.

        Args:
            config: TransformerConfig instance

        Returns:
            Instantiated transformer model

        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()

        # For now, return the encoder stack
        # Can be extended to include embeddings and head
        model = StackedTransformerBlocks(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            norm_first=config.norm_first,
        )

        return model

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total parameters in a model.

        Args:
            model: PyTorch model

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Estimate model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Approximate model size in MB (assuming fp32)
        """
        total_params = ModelConfigurator.count_parameters(model)
        # Each float32 parameter = 4 bytes
        return (total_params * 4) / (1024 * 1024)

    @staticmethod
    def get_architecture_info(config: TransformerConfig) -> Dict[str, Any]:
        """Get detailed architecture information.

        Args:
            config: TransformerConfig instance

        Returns:
            Dictionary with architecture details
        """
        config.validate()

        # Create a model to count parameters
        model = ModelConfigurator.create_model(config)
        total_params = ModelConfigurator.count_parameters(model)
        model_size_mb = ModelConfigurator.get_model_size_mb(model)

        # Calculate layer-wise parameter distribution
        attention_params = config.num_layers * (
            3 * (config.hidden_size ** 2 + config.hidden_size)  # Q, K, V projections
            + config.hidden_size ** 2 + config.hidden_size  # Output projection
        )

        ffn_params = config.num_layers * (
            config.hidden_size * config.intermediate_size + config.intermediate_size
            + config.intermediate_size * config.hidden_size + config.hidden_size
        )

        norm_params = config.num_layers * 2 * (
            config.hidden_size + config.hidden_size  # weight + bias per norm
        )

        return {
            "config": config.to_dict(),
            "total_parameters": total_params,
            "model_size_mb": model_size_mb,
            "layer_info": {
                "num_layers": config.num_layers,
                "hidden_size": config.hidden_size,
                "head_dim": config.get_head_dim(),
                "num_heads": config.num_heads,
                "intermediate_size": config.intermediate_size,
            },
            "parameter_distribution": {
                "attention_parameters": attention_params,
                "feedforward_parameters": ffn_params,
                "normalization_parameters": norm_params,
                "other_parameters": total_params - attention_params - ffn_params - norm_params,
            },
            "sequence_info": {
                "max_position_embeddings": config.max_position_embeddings,
                "vocabulary_size": config.vocab_size,
            },
            "computation_info": {
                "flops_per_token": self._estimate_flops_per_token(config),
                "activation_memory_mb": self._estimate_activation_memory(config),
            },
        }

    @staticmethod
    def _estimate_flops_per_token(config: TransformerConfig) -> int:
        """Estimate FLOPs per token for inference.

        Args:
            config: TransformerConfig instance

        Returns:
            Estimated FLOPs per token
        """
        # Simplified estimation: 2 * total_parameters per forward pass
        # (1 multiply + 1 add per parameter)
        total_params = (
            config.num_layers
            * (
                # Attention
                4 * config.hidden_size ** 2
                + 2 * config.hidden_size * config.max_position_embeddings
                # FFN
                + 2 * config.hidden_size * config.intermediate_size
            )
        )
        return total_params * 2

    @staticmethod
    def _estimate_activation_memory(config: TransformerConfig) -> float:
        """Estimate activation memory in MB.

        Args:
            config: TransformerConfig instance

        Returns:
            Estimated activation memory for one sample
        """
        # Rough estimate: activations for one sequence
        # (batch_size=1, seq_len, hidden_size)
        seq_len = config.max_position_embeddings
        activation_elements = seq_len * config.hidden_size
        # Add attention weights: (seq_len, seq_len) per head
        activation_elements += config.num_layers * seq_len * seq_len
        # Float32 = 4 bytes
        return (activation_elements * 4) / (1024 * 1024)

    @staticmethod
    def compare_configs(configs: Dict[str, TransformerConfig]) -> Dict[str, Any]:
        """Compare multiple configurations.

        Args:
            configs: Dictionary mapping names to configurations

        Returns:
            Comparison dictionary
        """
        comparison = {}
        for name, config in configs.items():
            comparison[name] = ModelConfigurator.get_architecture_info(config)
        return comparison
