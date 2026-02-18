"""GPT Language Model implementation.

This module provides a complete GPT-style language model combining embeddings,
position encodings, transformer blocks, and language modeling head.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configurator import TransformerConfig
from .transformer_block import StackedTransformerBlocks
from .embeddings import LearnedPositionalEncoding, PositionalEncoding


@dataclass
class ModelOutput:
    """Output from GPT model forward pass.

    Attributes:
        logits: Predicted token logits of shape (batch_size, seq_len, vocab_size)
        loss: Optional scalar loss if labels provided
        hidden_states: Final hidden states from transformer
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class GPTModel(nn.Module):
    """GPT-style language model with embeddings, transformer blocks, and LM head.

    This model combines:
    - Token embeddings: Map token IDs to dense vectors
    - Position embeddings: Add positional information
    - Transformer blocks: Process sequence with self-attention
    - LM head: Project to vocabulary for next token prediction

    Args:
        config: TransformerConfig with model hyperparameters

    Attributes:
        config: Model configuration
        token_embedding: Token embedding layer
        position_embedding: Position encoding layer
        transformer: Stacked transformer blocks
        lm_head: Output projection to vocabulary
    """

    def __init__(self, config: TransformerConfig):
        """Initialize GPT model.

        Args:
            config: TransformerConfig with model hyperparameters

        Raises:
            ValueError: If config is invalid
        """
        super().__init__()

        # Validate config
        config.validate()
        self.config = config

        # Token embeddings: maps token IDs to hidden_size dimensional vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings: add positional information to token embeddings
        if config.position_encoding_type == "learned":
            self.position_embedding = LearnedPositionalEncoding(
                hidden_size=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
                dropout_rate=config.dropout_rate,
            )
        else:
            # Default to sinusoidal
            self.position_embedding = PositionalEncoding(
                hidden_size=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
                dropout_rate=config.dropout_rate,
            )

        # Transformer blocks: core processing layers
        self.transformer = StackedTransformerBlocks(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            norm_first=config.norm_first,
        )

        # Language modeling head: project to vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optional: tie embeddings between input and output
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory optimization.

        Gradient checkpointing trades compute for memory by not storing all
        intermediate activations during the forward pass. This is useful for
        training large models with limited GPU memory.

        Note:
            This will slow down training slightly due to recomputation during
            backward pass, but can reduce memory usage by 40-50%.
        """
        self.gradient_checkpointing = True
        # Enable checkpointing in transformer blocks
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.transformer, "gradient_checkpointing_disable"):
            self.transformer.gradient_checkpointing_disable()

    def _init_weights(self) -> None:
        """Initialize model weights using GPT-2 style initialization.

        Uses normal distribution with mean=0.0 and std=0.02 for weights,
        and zeros for biases.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # GPT-2 weight initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Embedding weight initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask for masking positions
            labels: Optional target token IDs for loss computation.
                    If provided, computes cross-entropy loss.

        Returns:
            ModelOutput with logits, optional loss, and hidden states

        Raises:
            ValueError: If input shapes are invalid
        """
        batch_size, seq_len = input_ids.shape

        # Validate sequence length
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max "
                f"{self.config.max_position_embeddings}"
            )

        # 1. Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        token_embeds = self.token_embedding(input_ids)

        # 2. Add position embeddings
        hidden_states = self.position_embedding(token_embeds)

        # 3. Transform through stacked transformer blocks
        # Returns TransformerOutput with output and attention_weights
        transformer_output = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
        )

        # 4. Project to vocabulary
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(transformer_output.output)

        # 5. Optional: compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            # We want to predict token at position i+1 given tokens up to position i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=transformer_output.output,
        )

    def get_parameter_count(self) -> Dict[str, int]:
        """Count trainable parameters in the model.

        Returns a dictionary with total parameter count and breakdown by component.

        Returns:
            Dictionary with 'total' count and counts for each component
        """
        params = {}
        total = 0

        # Count parameters by component
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            params[name] = count
            total += count

        params["total"] = total
        return params

    @torch.no_grad()
    def estimate_memory_mb(self) -> float:
        """Estimate model memory usage in MB.

        Only estimates parameter memory, not activation memory.

        Returns:
            Estimated memory usage in MB
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Each float32 parameter takes 4 bytes
        bytes_per_param = 4 if self.parameters().__next__().dtype == torch.float32 else 2
        return (total_params * bytes_per_param) / (1024 * 1024)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask for autoregressive decoding.

    The mask prevents attention to future tokens (upper triangular).

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len) with -inf for future positions
    """
    mask = torch.ones((seq_len, seq_len), device=device)
    mask = torch.triu(mask, diagonal=1)
    # Replace 1s (future positions) with large negative number for softmax
    mask = mask.masked_fill(mask == 1, float("-inf"))
    # Add batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)
