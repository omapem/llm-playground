"""Transformer block combining attention and feed-forward networks."""

from typing import Optional

import torch
import torch.nn as nn

from .attention import Attention, AttentionOutput
from .feed_forward import FeedForward
from .layer_norm import LayerNormalization


class TransformerBlock(nn.Module):
    """A single transformer encoder block.

    Combines multi-head self-attention with a position-wise feed-forward network,
    using layer normalization and residual connections around each sub-layer.

    Architecture:
    1. Layer Norm
    2. Multi-head Attention
    3. Residual connection
    4. Layer Norm
    5. Feed-forward Network
    6. Residual connection

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        intermediate_size: Dimension of feed-forward hidden layer
        activation: Activation function for FFN ('relu', 'gelu', 'gelu_approximate')
        dropout_rate: Dropout probability
        norm_first: If True, use pre-norm (norm before sublayer).
                   If False, use post-norm (norm after sublayer, before residual)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int = 2048,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Layer normalization
        self.norm1 = LayerNormalization(hidden_size)
        self.norm2 = LayerNormalization(hidden_size)

        # Attention sublayer
        self.attention = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # Feed-forward sublayer
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        # Layer dropout (drop entire residual connection)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_first = norm_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """Forward pass through transformer block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            AttentionOutput containing:
            - output: Transformed tensor of shape (batch_size, seq_len, hidden_size)
            - attention_weights: Attention weights for visualization
        """
        if self.norm_first:
            # Pre-norm variant
            # Attention with residual
            attention_output = self.attention(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
            )
            hidden_states = hidden_states + self.dropout(attention_output.output)

            # Feed-forward with residual
            ff_output = self.feed_forward(self.norm2(hidden_states))
            hidden_states = hidden_states + self.dropout(ff_output)
        else:
            # Post-norm variant
            # Attention with residual
            attention_output = self.attention(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = self.norm1(
                hidden_states + self.dropout(attention_output.output)
            )

            # Feed-forward with residual
            ff_output = self.feed_forward(hidden_states)
            hidden_states = self.norm2(hidden_states + self.dropout(ff_output))

        return AttentionOutput(
            output=hidden_states,
            attention_weights=attention_output.attention_weights,
        )

    def get_parameter_count(self) -> int:
        """Calculate total number of parameters in the block.

        Returns:
            Total parameter count
        """
        count = 0

        # Layer normalization
        count += self.norm1.get_parameter_count()
        count += self.norm2.get_parameter_count()

        # Attention
        count += self.attention.get_parameter_count()

        # Feed-forward
        count += self.feed_forward.get_parameter_count()

        return count


class StackedTransformerBlocks(nn.Module):
    """Stack of multiple transformer blocks.

    Args:
        hidden_size: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        intermediate_size: FFN hidden dimension
        activation: Activation function
        dropout_rate: Dropout probability
        norm_first: Use pre-norm if True, post-norm if False
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int = 2048,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout_rate=dropout_rate,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm (if pre-norm is used)
        if norm_first:
            self.final_norm = LayerNormalization(hidden_size)
        else:
            self.final_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """Forward pass through stacked blocks.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            AttentionOutput from the last block
        """
        attention_weights = None

        for block in self.blocks:
            output = block(hidden_states, attention_mask=attention_mask)
            hidden_states = output.output
            attention_weights = output.attention_weights

        # Apply final normalization if using pre-norm
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        return AttentionOutput(
            output=hidden_states,
            attention_weights=attention_weights,
        )

    def get_parameter_count(self) -> int:
        """Calculate total parameters in all blocks."""
        count = 0
        for block in self.blocks:
            count += block.get_parameter_count()
        if self.final_norm is not None:
            count += self.final_norm.get_parameter_count()
        return count
