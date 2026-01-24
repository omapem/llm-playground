"""Training metrics computation and tracking.

Provides utilities for computing common metrics (perplexity, gradient norms),
tracking metrics with windowed averaging, and computing training throughput.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss.

    Perplexity = exp(loss). For numerical stability, clamps loss to prevent
    overflow when exp(loss) would exceed float32 range.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value (exp(loss), clamped at exp(20))
    """
    # Clamp loss to prevent overflow in exp
    clamped_loss = min(loss, 20.0)
    return math.exp(clamped_loss)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute L2 norm of all gradients in the model.

    Computes the L2 norm across all parameters that have gradients.
    Parameters with requires_grad=False are ignored.

    Args:
        model: PyTorch model

    Returns:
        L2 norm of all gradients, or 0.0 if no gradients exist
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm ** 2

    return math.sqrt(total_norm)


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in MB.

    Returns dict with 'allocated' and 'reserved' keys representing
    currently allocated and reserved GPU memory in MB. If CUDA is
    not available, returns zeros.

    Returns:
        Dict with 'allocated' and 'reserved' keys (values in MB)
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB

    return {
        "allocated": allocated,
        "reserved": reserved,
    }


class MetricsTracker:
    """Tracks training metrics with windowed averaging.

    Maintains a sliding window of recent metric values, computing averages
    over the window. Useful for smoothing noisy metrics like loss.

    Args:
        window_size: Number of values to keep in sliding window (default: 100)

    Example:
        >>> tracker = MetricsTracker(window_size=100)
        >>> for step in range(1000):
        ...     tracker.update({"loss": loss.item(), "grad_norm": grad_norm})
        ...     if step % 100 == 0:
        ...         avg = tracker.get_averages()
        ...         print(f"Loss: {avg['loss']:.4f}")
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize metrics tracker."""
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.total_steps = 0
        self.start_time = time.time()

    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dict mapping metric names to values
        """
        self.total_steps += 1

        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)

            self.metrics[name].append(value)

    def get_averages(self) -> Dict[str, float]:
        """Get averaged metrics over the window.

        Returns:
            Dict mapping metric names to their windowed averages
        """
        averages = {}

        for name, values in self.metrics.items():
            if len(values) > 0:
                averages[name] = sum(values) / len(values)

        return averages

    def get_throughput(self, tokens_per_batch: int) -> Dict[str, float]:
        """Compute training throughput.

        Args:
            tokens_per_batch: Number of tokens processed per batch

        Returns:
            Dict with 'steps_per_second' and 'tokens_per_second'
        """
        elapsed_time = time.time() - self.start_time
        steps_per_second = self.total_steps / elapsed_time if elapsed_time > 0 else 0.0
        tokens_per_second = steps_per_second * tokens_per_batch

        return {
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
        }

    def reset(self) -> None:
        """Reset all metrics and start time."""
        self.metrics = {}
        self.total_steps = 0
        self.start_time = time.time()
