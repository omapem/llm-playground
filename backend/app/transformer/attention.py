"""Attention mechanism implementation with visualization support."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionOutput:
    """Output from attention mechanism."""

    output: torch.Tensor
    attention_weights: torch.Tensor
    """Shape: (batch_size, num_heads, seq_len, seq_len)"""


class Attention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        hidden_size: Dimension of the model
        num_heads: Number of attention heads
        dropout_rate: Dropout probability for attention weights

    Attributes:
        hidden_size: Model dimension
        num_heads: Number of heads
        head_dim: Dimension per head (hidden_size // num_heads)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        """Forward pass for attention mechanism.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch_size, 1, 1, seq_len) or
                          (batch_size, 1, seq_len, seq_len)
            past_key_value: Optional cached key and value tensors for efficiency
            use_cache: Whether to return cached key and value tensors

        Returns:
            AttentionOutput with output and attention_weights
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Linear projections and reshape for multi-head attention
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape for multi-head attention
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Handle past key values (for caching in generation)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        # Compute attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, value)

        # Reshape back to (batch_size, seq_len, hidden_size)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.output_proj(context)

        # Prepare cache if requested
        cache = (key, value) if use_cache else None

        return AttentionOutput(
            output=output,
            attention_weights=attention_weights,
        )

    def get_parameter_count(self) -> int:
        """Calculate number of parameters in attention module.

        Returns:
            Total number of parameters
        """
        total = 0
        # Q, K, V projections: 3 * (hidden_size * hidden_size + hidden_size)
        total += 3 * (self.hidden_size * self.hidden_size + self.hidden_size)
        # Output projection: hidden_size * hidden_size + hidden_size
        total += self.hidden_size * self.hidden_size + self.hidden_size
        return total


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask for autoregressive generation.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len) with -inf for future positions
    """
    mask = torch.ones((seq_len, seq_len), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(
    attention_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Convert attention mask to format for multi-head attention.

    Args:
        attention_mask: Binary mask of shape (batch_size, seq_len) where 1=attend, 0=ignore
        num_heads: Number of attention heads

    Returns:
        Mask of shape (batch_size, 1, 1, seq_len) for broadcasting to attention scores
    """
    # Reshape to (batch_size, 1, 1, seq_len) for broadcasting
    # Set 0 to -inf (don't attend) and 1 to 0 (attend normally)
    mask = attention_mask.unsqueeze(1).unsqueeze(2)
    mask = mask.to(dtype=torch.float32)
    mask = (1.0 - mask) * -10000.0
    return mask
