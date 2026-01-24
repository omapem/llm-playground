"""Learning rate schedulers for training.

Provides flexible learning rate scheduling strategies including warmup phases,
cosine annealing, linear decay, and constant learning rates. Supports state
persistence for checkpoint/resume workflows.
"""

import math
from typing import Dict, Optional

import torch
from torch.optim import Optimizer


class CosineSchedulerWithWarmup:
    """Cosine annealing learning rate scheduler with linear warmup.

    Learning rate schedule:
    1. Linear warmup phase: LR increases from 0 to base_lr over warmup_steps
    2. Cosine decay phase: LR decays from base_lr to min_lr following cosine curve

    The cosine decay follows: LR = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    where progress goes from 0 to 1 during the decay phase.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup phase
        total_steps: Total training steps (used to calculate decay phase duration)
        min_lr_ratio: Minimum LR as ratio of base LR (default: 0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        """Initialize cosine scheduler."""
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

        # Store base learning rates for each parameter group
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self) -> None:
        """Update learning rate for current step."""
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Linear warmup phase
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Cosine decay phase
                progress = (self.current_step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                progress = min(progress, 1.0)  # Clamp to [0, 1]

                min_lr = base_lr * self.min_lr_ratio
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, int]:
        """Return state for checkpointing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]


class LinearSchedulerWithWarmup:
    """Linear decay learning rate scheduler with linear warmup.

    Learning rate schedule:
    1. Linear warmup phase: LR increases from 0 to base_lr over warmup_steps
    2. Linear decay phase: LR decreases linearly from base_lr to 0

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup phase
        total_steps: Total training steps (used to calculate decay slope)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        """Initialize linear scheduler."""
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

        # Store base learning rates for each parameter group
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self) -> None:
        """Update learning rate for current step."""
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Linear warmup phase
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Linear decay phase
                remaining_steps = self.total_steps - self.current_step
                total_decay_steps = self.total_steps - self.warmup_steps
                progress = (self.current_step - self.warmup_steps) / total_decay_steps
                progress = min(progress, 1.0)  # Clamp to [0, 1]

                lr = base_lr * (1 - progress)

            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, int]:
        """Return state for checkpointing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]


class ConstantScheduler:
    """Constant learning rate scheduler with linear warmup.

    Learning rate schedule:
    1. Linear warmup phase: LR increases from 0 to base_lr over warmup_steps
    2. Constant phase: LR stays at base_lr for remaining steps

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup phase
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
    ) -> None:
        """Initialize constant scheduler."""
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Store base learning rates for each parameter group
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self) -> None:
        """Update learning rate for current step."""
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Linear warmup phase
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Constant phase
                lr = base_lr

            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, int]:
        """Return state for checkpointing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]


def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: Optional[int] = None,
    **kwargs,
):
    """Factory function to create scheduler instances.

    Args:
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (required for cosine and linear)
        **kwargs: Additional arguments passed to scheduler (e.g., min_lr_ratio for cosine)

    Returns:
        Scheduler instance

    Raises:
        ValueError: If scheduler_type is unknown or required args are missing
    """
    if scheduler_type == "cosine":
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        return CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            **kwargs,
        )
    elif scheduler_type == "linear":
        if total_steps is None:
            raise ValueError("total_steps required for linear scheduler")
        return LinearSchedulerWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            **kwargs,
        )
    elif scheduler_type == "constant":
        return ConstantScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
