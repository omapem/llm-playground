"""Checkpoint management for training state persistence and resumption.

Provides robust checkpoint saving and loading with automatic rotation of old
checkpoints to manage disk space. Supports saving model state, optimizer state,
scheduler state, training configuration, and metadata.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .config import TrainingConfig


class CheckpointManager:
    """Manager for training checkpoints with automatic rotation.

    Saves complete training state including model, optimizer, scheduler, and config.
    Automatically rotates old checkpoints to maintain a maximum number of recent
    checkpoints, preventing unbounded disk usage during long training runs.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (default: 5)

    Example:
        >>> manager = CheckpointManager('./checkpoints', max_checkpoints=3)
        >>> path = manager.save_checkpoint(model, optimizer, scheduler, step=10, loss=1.5, config=config)
        >>> step, loss, config = manager.load_checkpoint(path, model, optimizer, scheduler)
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5) -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints

        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        step: int,
        loss: float,
        config: TrainingConfig,
    ) -> str:
        """Save training checkpoint.

        Saves all training state to a checkpoint file with the naming pattern:
        checkpoint_step_{step}.pt

        After saving, old checkpoints are rotated to keep only the most recent
        max_checkpoints checkpoints.

        Args:
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint
            scheduler: The learning rate scheduler to checkpoint
            step: Current training step
            loss: Current loss value
            config: Training configuration

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "loss": loss,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        # Create checkpoint filename
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_step_{step}.pt",
        )

        # Save checkpoint (atomic write via temp file)
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
    ) -> Tuple[int, float, TrainingConfig]:
        """Load training checkpoint.

        Restores model, optimizer, and scheduler state from checkpoint. The scheduler's
        current_step is updated to match the checkpoint step for correct resumption.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to restore state into
            optimizer: Optimizer to restore state into
            scheduler: Scheduler to restore state into

        Returns:
            Tuple of (step, loss, config) from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Restore state
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["step"], checkpoint["loss"], checkpoint["config"]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_step_*.pt"))

        if not checkpoints:
            return None

        # Sort by step number (highest step is most recent)
        checkpoints.sort(
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )

        return str(checkpoints[0])

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints sorted by step.

        Returns:
            List of dicts with 'step' and 'path' keys, sorted by step (ascending)
        """
        checkpoints = []

        for checkpoint_file in Path(self.checkpoint_dir).glob("checkpoint_step_*.pt"):
            step = int(checkpoint_file.stem.split("_")[-1])
            checkpoints.append(
                {
                    "step": step,
                    "path": str(checkpoint_file),
                }
            )

        # Sort by step (ascending)
        checkpoints.sort(key=lambda x: x["step"])

        return checkpoints

    def _rotate_checkpoints(self) -> None:
        """Remove oldest checkpoints to maintain max_checkpoints limit.

        This is called automatically after saving a checkpoint. It identifies
        which checkpoints to keep based on recency (step number) and deletes
        any excess checkpoints.
        """
        checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_step_*.pt"))

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by step number (ascending)
        checkpoints.sort(
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint_file in checkpoints[:num_to_remove]:
            checkpoint_file.unlink()
