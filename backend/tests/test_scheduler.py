"""Tests for learning rate schedulers."""

import pytest
import torch
import torch.nn as nn
from app.training.scheduler import (
    CosineSchedulerWithWarmup,
    LinearSchedulerWithWarmup,
    ConstantScheduler,
    get_scheduler,
)


class TestCosineSchedulerWithWarmup:
    """Tests for cosine scheduler with warmup."""

    @pytest.fixture
    def optimizer_and_params(self):
        """Create simple model and optimizer."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return optimizer, model

    def test_initialization(self, optimizer_and_params):
        """Test scheduler initialization."""
        optimizer, _ = optimizer_and_params
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1,
        )
        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.min_lr_ratio == 0.1
        assert scheduler.current_step == 0

    def test_warmup_phase(self, optimizer_and_params):
        """Test learning rate increases during warmup."""
        optimizer, _ = optimizer_and_params
        base_lr = optimizer.defaults["lr"]
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1,
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase during warmup
        assert lrs[-1] > lrs[0]
        # Last warmup LR should be close to base_lr
        assert abs(lrs[-1] - base_lr) < 0.01 * base_lr

    def test_cosine_decay_phase(self, optimizer_and_params):
        """Test learning rate decays after warmup."""
        optimizer, _ = optimizer_and_params
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1,
        )

        # Warmup phase
        for _ in range(100):
            scheduler.step()

        # Record LR at start of cosine phase
        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Cosine decay phase
        for _ in range(900):
            scheduler.step()

        # LR should decay
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < lr_at_warmup_end

        # Final LR should be above min_lr
        min_lr = optimizer.defaults["lr"] * 0.1
        assert final_lr >= min_lr * 0.99  # Allow small numerical error

    def test_min_lr_boundary(self, optimizer_and_params):
        """Test LR doesn't go below min_lr."""
        optimizer, _ = optimizer_and_params
        base_lr = optimizer.defaults["lr"]
        min_lr = base_lr * 0.05

        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.05,
        )

        for _ in range(1000):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr >= min_lr * 0.99  # Allow small numerical error

    def test_state_dict(self, optimizer_and_params):
        """Test saving and loading state."""
        optimizer, _ = optimizer_and_params
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Take some steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert "current_step" in state
        assert state["current_step"] == 50

        # Create new scheduler and load state
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        scheduler2.load_state_dict(state)
        assert scheduler2.current_step == 50

    def test_step_increments_counter(self, optimizer_and_params):
        """Test that step() increments current_step."""
        optimizer, _ = optimizer_and_params
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        assert scheduler.current_step == 0
        scheduler.step()
        assert scheduler.current_step == 1
        scheduler.step()
        assert scheduler.current_step == 2


class TestLinearSchedulerWithWarmup:
    """Tests for linear scheduler with warmup."""

    @pytest.fixture
    def optimizer_and_params(self):
        """Create simple model and optimizer."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return optimizer, model

    def test_initialization(self, optimizer_and_params):
        """Test scheduler initialization."""
        optimizer, _ = optimizer_and_params
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.current_step == 0

    def test_warmup_phase(self, optimizer_and_params):
        """Test learning rate increases during warmup."""
        optimizer, _ = optimizer_and_params
        base_lr = optimizer.defaults["lr"]
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase during warmup
        assert lrs[-1] > lrs[0]
        # Last warmup LR should be close to base_lr
        assert abs(lrs[-1] - base_lr) < 0.01 * base_lr

    def test_linear_decay_phase(self, optimizer_and_params):
        """Test learning rate decays linearly after warmup."""
        optimizer, _ = optimizer_and_params
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Warmup phase
        for _ in range(100):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Decay phase (should be linear, so slope should be consistent)
        lrs_decay = []
        for _ in range(900):
            scheduler.step()
            lrs_decay.append(optimizer.param_groups[0]["lr"])

        # LRs should be monotonically decreasing
        for i in range(1, len(lrs_decay)):
            assert lrs_decay[i] <= lrs_decay[i - 1]

        # Final LR should be near zero
        final_lr = lrs_decay[-1]
        assert final_lr < 1e-6

    def test_state_dict(self, optimizer_and_params):
        """Test saving and loading state."""
        optimizer, _ = optimizer_and_params
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Take some steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert "current_step" in state
        assert state["current_step"] == 50

        # Create new scheduler and load state
        scheduler2 = LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        scheduler2.load_state_dict(state)
        assert scheduler2.current_step == 50


class TestConstantScheduler:
    """Tests for constant scheduler (warmup then constant)."""

    @pytest.fixture
    def optimizer_and_params(self):
        """Create simple model and optimizer."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return optimizer, model

    def test_initialization(self, optimizer_and_params):
        """Test scheduler initialization."""
        optimizer, _ = optimizer_and_params
        scheduler = ConstantScheduler(
            optimizer,
            warmup_steps=100,
        )
        assert scheduler.warmup_steps == 100
        assert scheduler.current_step == 0

    def test_warmup_phase(self, optimizer_and_params):
        """Test learning rate increases during warmup."""
        optimizer, _ = optimizer_and_params
        base_lr = optimizer.defaults["lr"]
        scheduler = ConstantScheduler(
            optimizer,
            warmup_steps=100,
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase during warmup
        assert lrs[-1] > lrs[0]
        # Last warmup LR should be close to base_lr
        assert abs(lrs[-1] - base_lr) < 0.01 * base_lr

    def test_constant_phase(self, optimizer_and_params):
        """Test learning rate stays constant after warmup."""
        optimizer, _ = optimizer_and_params
        scheduler = ConstantScheduler(
            optimizer,
            warmup_steps=100,
        )

        # Warmup phase
        for _ in range(100):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Constant phase
        lrs_const = []
        for _ in range(100):
            scheduler.step()
            lrs_const.append(optimizer.param_groups[0]["lr"])

        # All LRs should be approximately the same
        for lr in lrs_const:
            assert abs(lr - lr_at_warmup_end) < 1e-6

    def test_state_dict(self, optimizer_and_params):
        """Test saving and loading state."""
        optimizer, _ = optimizer_and_params
        scheduler = ConstantScheduler(
            optimizer,
            warmup_steps=100,
        )

        # Take some steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert "current_step" in state
        assert state["current_step"] == 50

        # Create new scheduler and load state
        scheduler2 = ConstantScheduler(
            optimizer,
            warmup_steps=100,
        )
        scheduler2.load_state_dict(state)
        assert scheduler2.current_step == 50


class TestGetSchedulerFactory:
    """Tests for scheduler factory function."""

    @pytest.fixture
    def optimizer_and_params(self):
        """Create simple model and optimizer."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return optimizer, model

    def test_get_cosine_scheduler(self, optimizer_and_params):
        """Test factory creates cosine scheduler."""
        optimizer, _ = optimizer_and_params
        scheduler = get_scheduler(
            "cosine",
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        assert isinstance(scheduler, CosineSchedulerWithWarmup)

    def test_get_linear_scheduler(self, optimizer_and_params):
        """Test factory creates linear scheduler."""
        optimizer, _ = optimizer_and_params
        scheduler = get_scheduler(
            "linear",
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        assert isinstance(scheduler, LinearSchedulerWithWarmup)

    def test_get_constant_scheduler(self, optimizer_and_params):
        """Test factory creates constant scheduler."""
        optimizer, _ = optimizer_and_params
        scheduler = get_scheduler(
            "constant",
            optimizer,
            warmup_steps=100,
        )
        assert isinstance(scheduler, ConstantScheduler)

    def test_invalid_scheduler_type(self, optimizer_and_params):
        """Test factory raises error for invalid type."""
        optimizer, _ = optimizer_and_params
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            get_scheduler(
                "invalid_type",
                optimizer,
                warmup_steps=100,
                total_steps=1000,
            )

    def test_cosine_requires_total_steps(self, optimizer_and_params):
        """Test cosine scheduler requires total_steps."""
        optimizer, _ = optimizer_and_params
        with pytest.raises(
            ValueError, match="total_steps required for cosine scheduler"
        ):
            get_scheduler(
                "cosine",
                optimizer,
                warmup_steps=100,
                total_steps=None,
            )

    def test_linear_requires_total_steps(self, optimizer_and_params):
        """Test linear scheduler requires total_steps."""
        optimizer, _ = optimizer_and_params
        with pytest.raises(
            ValueError, match="total_steps required for linear scheduler"
        ):
            get_scheduler(
                "linear",
                optimizer,
                warmup_steps=100,
                total_steps=None,
            )

    def test_constant_doesnt_require_total_steps(self, optimizer_and_params):
        """Test constant scheduler works without total_steps."""
        optimizer, _ = optimizer_and_params
        scheduler = get_scheduler(
            "constant",
            optimizer,
            warmup_steps=100,
            total_steps=None,
        )
        assert isinstance(scheduler, ConstantScheduler)

    def test_factory_with_kwargs(self, optimizer_and_params):
        """Test factory passes kwargs to scheduler."""
        optimizer, _ = optimizer_and_params
        scheduler = get_scheduler(
            "cosine",
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.05,
        )
        assert scheduler.min_lr_ratio == 0.05


class TestSchedulerIntegration:
    """Integration tests for schedulers with training loop."""

    def test_multiple_param_groups(self):
        """Test scheduler works with multiple parameter groups."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))

        # Different learning rates for different layers
        optimizer = torch.optim.AdamW(
            [
                {"params": model[0].parameters(), "lr": 1e-3},
                {"params": model[1].parameters(), "lr": 1e-4},
            ]
        )

        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=50,
            total_steps=500,
        )

        # Take some steps
        for _ in range(100):
            scheduler.step()

        # Both param groups should have learning rates
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] > 0
        assert optimizer.param_groups[1]["lr"] > 0

    def test_scheduler_with_gradient_updates(self):
        """Test scheduler works with actual gradient updates."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        # Simulate training loop
        for _ in range(100):
            # Forward pass
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            scheduler.step()

        # Verify scheduler is at final step
        assert scheduler.current_step == 100

    def test_scheduler_consistency_across_restarts(self):
        """Test scheduler produces same LR sequence when restarted."""
        model1 = nn.Linear(10, 1)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        scheduler1 = CosineSchedulerWithWarmup(
            optimizer1,
            warmup_steps=50,
            total_steps=200,
        )

        # Get first 50 LRs
        lrs1 = []
        for _ in range(50):
            scheduler1.step()
            lrs1.append(optimizer1.param_groups[0]["lr"])

        # Create new scheduler and load state at step 50
        model2 = nn.Linear(10, 1)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer2,
            warmup_steps=50,
            total_steps=200,
        )
        scheduler2.load_state_dict(scheduler1.state_dict())

        # Get next 50 LRs from both
        lrs1_next = []
        lrs2_next = []
        for _ in range(50):
            scheduler1.step()
            scheduler2.step()
            lrs1_next.append(optimizer1.param_groups[0]["lr"])
            lrs2_next.append(optimizer2.param_groups[0]["lr"])

        # LRs should be identical (within numerical precision)
        for lr1, lr2 in zip(lrs1_next, lrs2_next):
            assert abs(lr1 - lr2) < 1e-9
