"""Layer normalization implementations for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):
    """Layer Normalization implementation.

    Normalizes activations across the feature dimension (hidden size) for each
    position in the sequence. This helps stabilize training and improve convergence.

    The formula is:
    LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β

    Where:
    - μ and σ² are mean and variance computed over the feature dimension
    - γ (weight) and β (bias) are learnable parameters
    - ε is a small constant for numerical stability

    Args:
        hidden_size: Model dimension (feature size)
        eps: Small constant for numerical stability (default: 1e-6)
        bias: Whether to include bias term (default: True)

    Reference:
        https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute mean and variance over last dimension (feature dimension)
        # mean shape: (..., 1)
        mean = x.mean(dim=-1, keepdim=True)

        # variance shape: (..., 1)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Scale and shift
        output = normalized * self.weight

        if self.bias is not None:
            output = output + self.bias

        return output

    def get_parameter_count(self) -> int:
        """Calculate number of learnable parameters."""
        count = self.hidden_size  # weight
        if self.bias is not None:
            count += self.hidden_size  # bias
        return count


class RMSNormalization(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    A simplified variant of layer normalization that only normalizes based on
    the root mean square (RMS) instead of computing both mean and variance.
    This is computationally more efficient while maintaining similar benefits.

    The formula is:
    RMSNorm(x) = x / RMS(x) * γ

    Where:
    - RMS(x) = sqrt(E[x²] + ε)
    - γ is a learnable scale parameter

    This is used in models like T5 and LLaMA.

    Args:
        hidden_size: Model dimension
        eps: Small constant for numerical stability (default: 1e-6)

    Reference:
        https://arxiv.org/abs/1910.07468 (T5)
        https://github.com/facebookresearch/llama
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS over the feature dimension
        # rms shape: (..., 1)
        rms = torch.sqrt(
            (x.pow(2).mean(dim=-1, keepdim=True)) + self.eps
        )

        # Normalize and scale
        output = (x / rms) * self.weight

        return output

    def get_parameter_count(self) -> int:
        """Calculate number of learnable parameters."""
        return self.hidden_size


class GroupNormalization(nn.Module):
    """Group Normalization for transformers.

    Alternative to layer normalization that divides channels into groups
    and normalizes within each group. Can be useful in low-batch-size settings.

    Args:
        hidden_size: Model dimension
        num_groups: Number of groups (default: 32)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        hidden_size: int,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps

        # Use PyTorch's GroupNorm
        self.norm = nn.GroupNorm(num_groups, hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization.

        Note: GroupNorm expects shape (batch, channels, ...)
        For transformer inputs (batch, seq_len, hidden_size), we need to reshape.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Store original shape
        batch_size, seq_len, hidden_size = x.shape

        # Reshape for GroupNorm: (batch_size, hidden_size, seq_len)
        x = x.transpose(1, 2)

        # Apply group norm
        x = self.norm(x)

        # Reshape back: (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)

        return x

    def get_parameter_count(self) -> int:
        """Calculate number of learnable parameters."""
        # weight + bias for each group
        return 2 * self.hidden_size
