"""Tests for training metrics computation and tracking."""

import math

import pytest
import torch
import torch.nn as nn
from app.training.metrics import (
    compute_perplexity,
    compute_gradient_norm,
    get_gpu_memory_usage,
    MetricsTracker,
)


class TestPerplexityComputation:
    """Tests for perplexity computation."""

    def test_perplexity_basic(self):
        """Test basic perplexity computation."""
        loss = 1.0
        perplexity = compute_perplexity(loss)
        expected = math.exp(1.0)
        assert abs(perplexity - expected) < 1e-6

    def test_perplexity_zero_loss(self):
        """Test perplexity with zero loss."""
        perplexity = compute_perplexity(0.0)
        assert abs(perplexity - 1.0) < 1e-6

    def test_perplexity_large_loss(self):
        """Test perplexity doesn't overflow with large loss."""
        perplexity = compute_perplexity(100.0)
        # Should be capped at exp(20) to prevent overflow
        assert perplexity == math.exp(20)

    def test_perplexity_negative_loss(self):
        """Test perplexity with negative loss (edge case)."""
        perplexity = compute_perplexity(-1.0)
        expected = math.exp(-1.0)
        assert abs(perplexity - expected) < 1e-6

    def test_perplexity_sequence(self):
        """Test perplexity decreases with lower loss."""
        ppl_high = compute_perplexity(2.0)
        ppl_low = compute_perplexity(1.0)
        assert ppl_low < ppl_high


class TestGradientNormComputation:
    """Tests for gradient norm computation."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Linear(10, 1)

    def test_gradient_norm_after_backward(self, simple_model):
        """Test gradient norm computation after backward pass."""
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        grad_norm = compute_gradient_norm(simple_model)
        assert grad_norm > 0
        assert math.isfinite(grad_norm)

    def test_gradient_norm_no_gradients(self, simple_model):
        """Test gradient norm is zero when no gradients computed."""
        grad_norm = compute_gradient_norm(simple_model)
        assert grad_norm == 0.0

    def test_gradient_norm_zero_gradients(self, simple_model):
        """Test gradient norm with zero gradients."""
        # Initialize gradients to zero
        for param in simple_model.parameters():
            param.grad = torch.zeros_like(param)

        grad_norm = compute_gradient_norm(simple_model)
        assert abs(grad_norm) < 1e-10

    def test_gradient_norm_consistency(self, simple_model):
        """Test gradient norm computation is consistent."""
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        norm1 = compute_gradient_norm(simple_model)
        norm2 = compute_gradient_norm(simple_model)

        assert abs(norm1 - norm2) < 1e-10

    def test_gradient_norm_multiple_backward_passes(self, simple_model):
        """Test gradient norm with multiple backward passes."""
        for i in range(3):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()

            grad_norm = compute_gradient_norm(simple_model)
            assert grad_norm > 0

    def test_gradient_norm_layer_wise(self):
        """Test gradient norm includes all layers."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        assert grad_norm > 0

    def test_gradient_norm_with_no_grad_params(self):
        """Test gradient norm ignores parameters with requires_grad=False."""
        model = nn.Linear(10, 1)
        # Freeze first layer
        for param in model.parameters():
            param.requires_grad = False

        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()

        # Should not raise error even though no gradients
        grad_norm = compute_gradient_norm(model)
        assert grad_norm >= 0


class TestGPUMemoryUsage:
    """Tests for GPU memory usage computation."""

    def test_gpu_memory_usage_returns_dict(self):
        """Test GPU memory usage returns expected structure."""
        memory = get_gpu_memory_usage()
        assert isinstance(memory, dict)

    def test_gpu_memory_has_allocated_key(self):
        """Test GPU memory dict has 'allocated' key."""
        memory = get_gpu_memory_usage()
        assert "allocated" in memory

    def test_gpu_memory_has_reserved_key(self):
        """Test GPU memory dict has 'reserved' key."""
        memory = get_gpu_memory_usage()
        assert "reserved" in memory

    def test_gpu_memory_values_non_negative(self):
        """Test GPU memory values are non-negative."""
        memory = get_gpu_memory_usage()
        assert memory["allocated"] >= 0
        assert memory["reserved"] >= 0

    def test_gpu_memory_reserved_gte_allocated(self):
        """Test reserved memory is >= allocated memory."""
        memory = get_gpu_memory_usage()
        # Reserved should be >= allocated (reserved is the total pool size)
        assert memory["reserved"] >= memory["allocated"]


class TestMetricsTracker:
    """Tests for metrics tracking system."""

    def test_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker(window_size=10)
        assert tracker.window_size == 10

    def test_update_single_metric(self):
        """Test updating a single metric."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 1.5})
        assert "loss" in tracker.metrics

    def test_update_multiple_metrics(self):
        """Test updating multiple metrics at once."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 1.5, "accuracy": 0.85})
        assert "loss" in tracker.metrics
        assert "accuracy" in tracker.metrics

    def test_get_averages(self):
        """Test getting averaged metrics."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 1.0})
        tracker.update({"loss": 2.0})
        tracker.update({"loss": 3.0})

        averages = tracker.get_averages()
        assert abs(averages["loss"] - 2.0) < 1e-6

    def test_windowed_averaging(self):
        """Test windowed averaging behavior."""
        tracker = MetricsTracker(window_size=3)

        # Add 5 values, but only last 3 should be averaged
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.update({"loss": value})

        averages = tracker.get_averages()
        # Average of last 3: (3 + 4 + 5) / 3 = 4.0
        assert abs(averages["loss"] - 4.0) < 1e-6

    def test_multiple_metrics_independent_windows(self):
        """Test different metrics track independently."""
        tracker = MetricsTracker(window_size=3)

        # Add values for two metrics
        for i in range(5):
            tracker.update({"loss": float(i + 1), "accuracy": float(i * 2)})

        averages = tracker.get_averages()
        # Loss: avg of [3, 4, 5] = 4.0
        assert abs(averages["loss"] - 4.0) < 1e-6
        # Accuracy: for i=2,3,4: [4, 6, 8], avg = 6.0
        assert abs(averages["accuracy"] - 6.0) < 1e-6

    def test_get_throughput(self):
        """Test throughput computation."""
        tracker = MetricsTracker(window_size=10)

        # Add 100 updates
        for _ in range(100):
            tracker.update({"loss": 1.0})

        throughput = tracker.get_throughput(tokens_per_batch=256)
        assert "steps_per_second" in throughput
        assert "tokens_per_second" in throughput

    def test_throughput_values_positive(self):
        """Test throughput values are positive."""
        tracker = MetricsTracker(window_size=10)

        for _ in range(50):
            tracker.update({"loss": 1.0})

        throughput = tracker.get_throughput(tokens_per_batch=256)
        assert throughput["steps_per_second"] > 0
        assert throughput["tokens_per_second"] > 0

    def test_throughput_relationship(self):
        """Test throughput relationship: tokens/sec = steps/sec * tokens_per_batch."""
        tracker = MetricsTracker(window_size=10)

        for _ in range(50):
            tracker.update({"loss": 1.0})

        tokens_per_batch = 512
        throughput = tracker.get_throughput(tokens_per_batch=tokens_per_batch)

        expected_tokens = throughput["steps_per_second"] * tokens_per_batch
        assert abs(throughput["tokens_per_second"] - expected_tokens) < 0.1

    def test_reset(self):
        """Test resetting metrics."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 1.5})
        tracker.reset()

        averages = tracker.get_averages()
        assert averages == {}

    def test_reset_clears_all_metrics(self):
        """Test reset clears all metrics."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 1.0, "accuracy": 0.9})
        tracker.reset()

        assert len(tracker.metrics) == 0

    def test_update_with_zero_loss(self):
        """Test updating with zero loss."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"loss": 0.0})

        averages = tracker.get_averages()
        assert abs(averages["loss"]) < 1e-10

    def test_update_with_negative_metrics(self):
        """Test updating with negative metrics."""
        tracker = MetricsTracker(window_size=10)
        tracker.update({"accuracy": -0.5})

        averages = tracker.get_averages()
        assert averages["accuracy"] < 0

    def test_step_counter(self):
        """Test step counter tracking."""
        tracker = MetricsTracker(window_size=10)

        for i in range(5):
            tracker.update({"loss": float(i)})

        assert tracker.total_steps == 5

    def test_multiple_updates_same_step(self):
        """Test handling multiple updates per step."""
        tracker = MetricsTracker(window_size=10)

        # Multiple metrics in one update
        tracker.update({"loss": 1.0, "grad_norm": 0.5, "lr": 1e-4})
        tracker.update({"loss": 1.1, "grad_norm": 0.6, "lr": 1e-4})

        averages = tracker.get_averages()
        assert "loss" in averages
        assert "grad_norm" in averages
        assert "lr" in averages

    def test_window_size_one(self):
        """Test metrics tracker with window size 1."""
        tracker = MetricsTracker(window_size=1)

        tracker.update({"loss": 1.0})
        tracker.update({"loss": 2.0})
        tracker.update({"loss": 3.0})

        averages = tracker.get_averages()
        # Should only have the most recent value
        assert abs(averages["loss"] - 3.0) < 1e-6


class TestMetricsIntegration:
    """Integration tests for metrics with training loop."""

    def test_metrics_during_training_loop(self):
        """Test metrics tracking during actual training."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tracker = MetricsTracker(window_size=10)

        for step in range(20):
            # Forward pass
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            grad_norm = compute_gradient_norm(model)
            perplexity = compute_perplexity(loss.item())

            tracker.update({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "perplexity": perplexity,
            })

        # Verify metrics were collected
        averages = tracker.get_averages()
        assert "loss" in averages
        assert "grad_norm" in averages
        assert "perplexity" in averages

        # Verify values are reasonable
        assert averages["grad_norm"] > 0
        assert averages["perplexity"] > 0

    def test_throughput_during_training(self):
        """Test throughput computation during training."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tracker = MetricsTracker(window_size=5)

        batch_size = 256
        tokens_per_batch = batch_size * 128  # Assuming 128 tokens per sample

        for step in range(10):
            x = torch.randn(batch_size, 10)
            y = model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.update({"loss": loss.item()})

        # Compute throughput
        throughput = tracker.get_throughput(tokens_per_batch=tokens_per_batch)

        assert throughput["steps_per_second"] > 0
        assert throughput["tokens_per_second"] > 0
        # Check relationship
        expected = throughput["steps_per_second"] * tokens_per_batch
        assert abs(throughput["tokens_per_second"] - expected) < 0.1
