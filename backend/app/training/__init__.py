"""Training module for LLM Playground.

Provides complete training infrastructure including:
- Training configuration and management
- Learning rate schedulers
- Checkpoint management
- Training metrics and tracking
- Main training orchestrator
"""

from .config import TrainingConfig
from .scheduler import (
    CosineSchedulerWithWarmup,
    LinearSchedulerWithWarmup,
    ConstantScheduler,
    get_scheduler,
)
from .checkpoint import CheckpointManager
from .checkpoint_cleaner import CheckpointCleaner
from .metrics import (
    compute_perplexity,
    compute_gradient_norm,
    get_gpu_memory_usage,
    MetricsTracker,
)
from .trainer import Trainer

__all__ = [
    "TrainingConfig",
    "CosineSchedulerWithWarmup",
    "LinearSchedulerWithWarmup",
    "ConstantScheduler",
    "get_scheduler",
    "CheckpointManager",
    "CheckpointCleaner",
    "compute_perplexity",
    "compute_gradient_norm",
    "get_gpu_memory_usage",
    "MetricsTracker",
    "Trainer",
]
