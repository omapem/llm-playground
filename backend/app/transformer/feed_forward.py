"""Feed-forward network component for transformer blocks."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN).

    As described in "Attention is All You Need", the FFN consists of two
    linear transformations with a ReLU activation in between:

    FFN(x) = max(0, xW1 + b1)W2 + b2

    This can also be viewed as a 1x1 convolutional layer with an intermediate
    expansion factor.

    Args:
        hidden_size: Model dimension
        intermediate_size: Dimension of hidden layer (typically 4x hidden_size)
        activation: Activation function (default: ReLU)
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 2048,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # First linear transformation: expand dimension
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "gelu_approximate":
            self.activation = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout after activation
        self.dropout = nn.Dropout(dropout_rate)

        # Second linear transformation: contract dimension back to hidden_size
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First linear transformation
        hidden_states = self.linear_1(hidden_states)

        # Activation
        hidden_states = self.activation(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Second linear transformation
        hidden_states = self.linear_2(hidden_states)

        return hidden_states

    def get_parameter_count(self) -> int:
        """Calculate number of parameters in FFN.

        Returns:
            Total number of parameters
        """
        # linear_1: hidden_size * intermediate_size + intermediate_size
        count = self.hidden_size * self.intermediate_size + self.intermediate_size

        # linear_2: intermediate_size * hidden_size + hidden_size
        count += self.intermediate_size * self.hidden_size + self.hidden_size

        return count


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) variant of feed-forward network.

    GLU combines two linear transformations with a gating mechanism:
    GLU(x) = (xW + b) ⊗ σ(xV + c)

    Where ⊗ is element-wise multiplication and σ is a sigmoid function.

    Args:
        hidden_size: Model dimension
        intermediate_size: Dimension of intermediate layer
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Linear transformation for main branch
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)

        # Activation and dropout for gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Output linear transformation
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through gated linear unit.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Main branch
        gate_values = self.linear_1(hidden_states)

        # Gate branch
        gate_output = self.gate(hidden_states)

        # Element-wise multiplication (gating)
        gated = gate_values * gate_output

        # Dropout
        gated = self.dropout(gated)

        # Output projection
        output = self.linear_2(gated)

        return output

    def get_parameter_count(self) -> int:
        """Calculate number of parameters in GLU."""
        # linear_1: hidden_size * intermediate_size + intermediate_size
        count = self.hidden_size * self.intermediate_size + self.intermediate_size

        # Gate linear: hidden_size * intermediate_size + intermediate_size
        count += self.hidden_size * self.intermediate_size + self.intermediate_size

        # linear_2: intermediate_size * hidden_size + hidden_size
        count += self.intermediate_size * self.hidden_size + self.hidden_size

        return count
