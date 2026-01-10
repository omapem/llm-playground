"""Visualization utilities for transformer components."""

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch


class AttentionVisualization:
    """Utilities for visualizing attention weights.

    Provides methods to process and format attention weights for visualization
    in the frontend.
    """

    @staticmethod
    def extract_attention_head(
        attention_weights: torch.Tensor,
        layer: int,
        head: int,
    ) -> np.ndarray:
        """Extract a single attention head from attention weights.

        Args:
            attention_weights: Tensor of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
                             or (batch_size, num_heads, seq_len, seq_len)
            layer: Layer index
            head: Head index

        Returns:
            Attention matrix of shape (seq_len, seq_len) as numpy array
        """
        if attention_weights.dim() == 5:
            # (num_layers, batch_size, num_heads, seq_len, seq_len)
            head_attention = attention_weights[layer, 0, head, :, :].detach().cpu().numpy()
        elif attention_weights.dim() == 4:
            # (batch_size, num_heads, seq_len, seq_len)
            head_attention = attention_weights[0, head, :, :].detach().cpu().numpy()
        else:
            raise ValueError(
                f"Expected 4D or 5D tensor, got {attention_weights.dim()}D"
            )

        return head_attention

    @staticmethod
    def average_attention_heads(
        attention_weights: torch.Tensor,
        layer: int,
    ) -> np.ndarray:
        """Average attention weights across all heads for a layer.

        Args:
            attention_weights: Attention tensor
            layer: Layer index

        Returns:
            Averaged attention matrix of shape (seq_len, seq_len)
        """
        if attention_weights.dim() == 5:
            # (num_layers, batch_size, num_heads, seq_len, seq_len)
            layer_attention = attention_weights[layer, 0, :, :, :]  # (num_heads, seq_len, seq_len)
        elif attention_weights.dim() == 4:
            layer_attention = attention_weights[0, :, :, :]  # (num_heads, seq_len, seq_len)
        else:
            raise ValueError(
                f"Expected 4D or 5D tensor, got {attention_weights.dim()}D"
            )

        # Average across heads
        averaged = layer_attention.mean(dim=0).detach().cpu().numpy()
        return averaged

    @staticmethod
    def get_top_attention_positions(
        attention_weights: np.ndarray,
        top_k: int = 5,
        sequence: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get positions with highest attention weights.

        Args:
            attention_weights: Attention matrix of shape (seq_len, seq_len)
            top_k: Number of top positions to return
            sequence: Optional list of tokens for context

        Returns:
            List of dictionaries with top attention positions
        """
        # Flatten and get top-k indices
        flat_weights = attention_weights.flatten()
        top_indices = np.argsort(flat_weights)[-top_k:][::-1]

        results = []
        seq_len = attention_weights.shape[0]

        for idx in top_indices:
            query_pos = idx // seq_len
            key_pos = idx % seq_len
            weight = flat_weights[idx]

            result = {
                "query_position": int(query_pos),
                "key_position": int(key_pos),
                "attention_weight": float(weight),
            }

            if sequence is not None:
                if query_pos < len(sequence):
                    result["query_token"] = sequence[query_pos]
                if key_pos < len(sequence):
                    result["key_token"] = sequence[key_pos]

            results.append(result)

        return results

    @staticmethod
    def compute_attention_entropy(
        attention_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute entropy of attention distributions.

        Measures how focused or diffuse the attention is.
        Low entropy = focused (attending to few positions)
        High entropy = diffuse (attending to many positions)

        Args:
            attention_weights: Attention matrix of shape (seq_len, seq_len)

        Returns:
            Entropy values of shape (seq_len,) - one entropy per query position
        """
        # Compute entropy for each query position
        # Entropy = -sum(p * log(p))
        epsilon = 1e-10
        attention_weights = np.clip(attention_weights, epsilon, 1.0)
        entropy = -np.sum(attention_weights * np.log(attention_weights), axis=-1)
        return entropy

    @staticmethod
    def find_attention_patterns(
        attention_weights: torch.Tensor,
        layer: int,
    ) -> Dict[str, Any]:
        """Identify attention patterns in a layer.

        Args:
            attention_weights: Attention tensor
            layer: Layer index

        Returns:
            Dictionary describing attention patterns
        """
        head_attention = AttentionVisualization.average_attention_heads(
            attention_weights, layer
        )

        # Pattern detection
        seq_len = head_attention.shape[0]
        diag = np.diag(head_attention)
        diag_mean = np.mean(diag)

        # Check for common patterns
        patterns = {
            "type": "unknown",
            "characteristics": [],
        }

        # Pattern 1: Diagonal (position-specific attention)
        if diag_mean > 0.5:
            patterns["type"] = "positional"
            patterns["characteristics"].append("Strong diagonal (positional attention)")

        # Pattern 2: Broad (distributed attention)
        entropy = AttentionVisualization.compute_attention_entropy(head_attention)
        if np.mean(entropy) > np.log(seq_len) * 0.8:
            patterns["type"] = "distributed"
            patterns["characteristics"].append("High entropy (distributed attention)")

        # Pattern 3: First token (CLS token or similar)
        first_col = head_attention[:, 0]
        if np.mean(first_col) > 0.3:
            patterns["characteristics"].append("High attention to first token")

        # Pattern 4: Previous tokens (local attention)
        local_attention = 0
        for i in range(1, min(5, seq_len)):
            local_attention += np.mean(np.diagonal(head_attention, offset=-i))
        if local_attention > 0.3:
            patterns["type"] = "local"
            patterns["characteristics"].append("Local attention pattern (nearby tokens)")

        return patterns

    @staticmethod
    def format_for_visualization(
        attention_weights: torch.Tensor,
        layer: int,
        head: Optional[int] = None,
        sequence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format attention weights for frontend visualization.

        Args:
            attention_weights: Attention tensor
            layer: Layer index
            head: Optional specific head to extract. If None, averages all heads
            sequence: Optional token sequence for labels

        Returns:
            Formatted visualization data
        """
        if head is not None:
            matrix = AttentionVisualization.extract_attention_head(
                attention_weights, layer, head
            )
            title = f"Layer {layer}, Head {head}"
        else:
            matrix = AttentionVisualization.average_attention_heads(
                attention_weights, layer
            )
            title = f"Layer {layer} (averaged across heads)"

        # Convert to 0-100 scale for visualization
        matrix_scaled = (matrix * 100).astype(np.float32)

        # Create labels
        seq_len = matrix.shape[0]
        if sequence is None:
            x_labels = y_labels = [f"Pos {i}" for i in range(seq_len)]
        else:
            x_labels = y_labels = sequence[:seq_len]

        # Detect patterns
        patterns = AttentionVisualization.find_attention_patterns(
            attention_weights, layer
        )

        return {
            "title": title,
            "matrix": matrix_scaled.tolist(),
            "x_labels": x_labels,
            "y_labels": y_labels,
            "entropy": AttentionVisualization.compute_attention_entropy(matrix).tolist(),
            "patterns": patterns,
            "top_connections": AttentionVisualization.get_top_attention_positions(
                matrix, top_k=5, sequence=sequence
            ),
        }

    @staticmethod
    def batch_visualize_layers(
        attention_weights: torch.Tensor,
        num_layers: Optional[int] = None,
        sequence: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create visualizations for multiple layers.

        Args:
            attention_weights: Tensor of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
            num_layers: Optional number of layers to visualize. If None, uses all layers
            sequence: Optional token sequence

        Returns:
            List of visualization data dictionaries, one per layer
        """
        if num_layers is None:
            if attention_weights.dim() == 5:
                num_layers = attention_weights.shape[0]
            else:
                num_layers = 1

        visualizations = []
        for layer in range(num_layers):
            viz = AttentionVisualization.format_for_visualization(
                attention_weights, layer, sequence=sequence
            )
            visualizations.append(viz)

        return visualizations


class ActivationVisualization:
    """Utilities for visualizing activation patterns."""

    @staticmethod
    def compute_activation_statistics(
        activations: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute statistics for activation tensor.

        Args:
            activations: Activation tensor of any shape

        Returns:
            Dictionary with activation statistics
        """
        act_np = activations.detach().cpu().numpy()

        return {
            "mean": float(np.mean(act_np)),
            "std": float(np.std(act_np)),
            "min": float(np.min(act_np)),
            "max": float(np.max(act_np)),
            "percentile_25": float(np.percentile(act_np, 25)),
            "percentile_50": float(np.percentile(act_np, 50)),
            "percentile_75": float(np.percentile(act_np, 75)),
            "sparsity": float(np.mean(act_np == 0)),
        }

    @staticmethod
    def compute_activation_flow(
        layer_outputs: List[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """Analyze how activations flow through layers.

        Args:
            layer_outputs: List of output tensors from each layer

        Returns:
            List of activation flow statistics per layer
        """
        flow = []
        for i, output in enumerate(layer_outputs):
            stats = ActivationVisualization.compute_activation_statistics(output)
            stats["layer"] = i
            flow.append(stats)
        return flow
