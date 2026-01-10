"""Position encoding implementations for transformer models."""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from "Attention is All You Need").

    This encoding adds sinusoidal patterns at different frequencies to represent
    the absolute position of tokens in the sequence.

    Args:
        hidden_size: Model dimension
        max_position_embeddings: Maximum sequence length supported
        dropout_rate: Dropout probability

    Reference:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_size % 2 != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be even for sinusoidal encoding"
            )

        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        # Create position indices: shape (1, max_position_embeddings)
        positions = torch.arange(max_position_embeddings).unsqueeze(1)

        # Create dimension indices
        # Using log space for better numerical stability
        # dim_indices: shape (hidden_size // 2)
        dim_indices = torch.arange(0, hidden_size, 2).float()
        div_term = torch.exp(dim_indices * -(math.log(10000.0) / hidden_size))

        # Compute sinusoidal encodings
        # pe[pos, 2i] = sin(pos / (10000^(2i/d)))
        # pe[pos, 2i+1] = cos(pos / (10000^(2i/d)))
        pe = torch.zeros(max_position_embeddings, hidden_size)
        pe[:, 0::2] = torch.sin(positions.float() * div_term)
        pe[:, 1::2] = torch.cos(positions.float() * div_term)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        # Add positional encoding to input
        # pe[:seq_len] has shape (seq_len, hidden_size)
        # Broadcast to (batch_size, seq_len, hidden_size)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)

    def get_parameter_count(self) -> int:
        """Positional encoding doesn't have learnable parameters."""
        return 0


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (learnable absolute position representation).

    This module learns position embeddings as regular parameters, which can be
    more expressive than fixed sinusoidal patterns.

    Args:
        hidden_size: Model dimension
        max_position_embeddings: Maximum sequence length supported
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        # Learnable position embeddings
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        # Get position indices: shape (seq_len,)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)

        # Get position embeddings: shape (seq_len, hidden_size)
        position_embeddings = self.embeddings(position_ids)

        # Add to input
        x = x + position_embeddings.unsqueeze(0)
        return self.dropout(x)

    def get_parameter_count(self) -> int:
        """Calculate number of learnable parameters."""
        return self.max_position_embeddings * self.hidden_size


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer.

    This applies rotation matrices in the complex plane to encode positions.
    RoPE has shown better length extrapolation properties than absolute encodings.

    Args:
        dim: Dimension per head
        max_position_embeddings: Maximum sequence length supported
        base: Base for the exponential (default: 10000)

    Reference:
        https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even for RoPE")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Optional position indices. If None, uses range(seq_len)

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)

        # Compute frequency matrix: (seq_len, dim // 2)
        freqs = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq)

        # Add dimension for batch and num_heads
        # freqs: (1, 1, seq_len, dim // 2)
        freqs = freqs.unsqueeze(0).unsqueeze(0)

        # Compute rotation matrix in complex form
        # (batch_size, num_heads, seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Apply rotation to x
        # Interleave real and imaginary parts
        x_rotated = (x * cos) + self._rotate_half(x) * sin
        return x_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def get_parameter_count(self) -> int:
        """RoPE doesn't have learnable parameters."""
        return 0
