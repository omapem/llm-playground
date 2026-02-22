"""Training module for LLM Playground.

Provides complete training infrastructure including:
- Training configuration and management
- Learning rate schedulers
- Checkpoint management
- Training metrics and tracking
- Main training orchestrator
- Distributed training (DDP) utilities
"""

from .config import TrainingConfig
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloader,
    reduce_mean,
)
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
    reduce_metrics,
    MetricsTracker,
)
from .trainer import Trainer

__all__ = [
    "TrainingConfig",
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "create_distributed_dataloader",
    "reduce_mean",
    "CosineSchedulerWithWarmup",
    "LinearSchedulerWithWarmup",
    "ConstantScheduler",
    "get_scheduler",
    "CheckpointManager",
    "CheckpointCleaner",
    "compute_perplexity",
    "compute_gradient_norm",
    "get_gpu_memory_usage",
    "reduce_metrics",
    "MetricsTracker",
    "Trainer",
]
