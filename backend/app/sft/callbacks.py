"""Custom callbacks for SFT training.

Provides validation, W&B logging, and checkpointing functionality for SFT training.
Integrates with TRL's SFTTrainer to provide enhanced monitoring and model management.
"""

import os
import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class SFTCallback(ABC):
    """Base class for SFT training callbacks.

    Callbacks receive training events and can log metrics,
    save checkpoints, or perform validation.

    Example:
        >>> class CustomCallback(SFTCallback):
        ...     def on_step_end(self, trainer, step, metrics, **kwargs):
        ...         print(f"Step {step}: loss={metrics['loss']}")
    """

    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training.

        Args:
            trainer: The trainer instance
            **kwargs: Additional keyword arguments
        """
        pass

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training.

        Args:
            trainer: The trainer instance
            **kwargs: Additional keyword arguments
        """
        pass

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            **kwargs: Additional keyword arguments
        """
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
            **kwargs: Additional keyword arguments
        """
        pass

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs):
        """Called after each training step.

        Args:
            trainer: The trainer instance
            step: Current training step
            metrics: Dictionary of metrics for this step
            **kwargs: Additional keyword arguments
        """
        pass


class ValidationCallback(SFTCallback):
    """Performs validation on a held-out dataset at specified intervals.

    Runs validation every `assess_steps` steps and tracks the best validation loss.
    Optionally runs validation at the end of each epoch.

    Args:
        val_dataset: Validation dataset (optional)
        assess_steps: Run validation every N steps (default: 500)
        assess_on_epoch_end: Whether to run validation at epoch end (default: True)

    Example:
        >>> from datasets import Dataset
        >>> val_data = Dataset.from_dict({"text": ["example 1", "example 2"]})
        >>> callback = ValidationCallback(val_dataset=val_data, assess_steps=100)
        >>> trainer.add_callback(callback)
    """

    def __init__(
        self,
        val_dataset: Optional[Any] = None,
        eval_steps: int = 500,
        eval_on_epoch_end: bool = True,
    ):
        """Initialize validation callback."""
        self.val_dataset = val_dataset
        self.eval_steps = eval_steps
        self.eval_on_epoch_end = eval_on_epoch_end
        self.best_val_loss = float("inf")

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs):
        """Run validation at specified intervals.

        Args:
            trainer: The trainer instance
            step: Current training step
            metrics: Dictionary of metrics for this step
            **kwargs: Additional keyword arguments
        """
        if self.val_dataset is None:
            return

        if step % self.eval_steps == 0:
            val_loss = self._run_validation(trainer)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(f"New best validation loss: {val_loss:.4f} at step {step}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Run validation at epoch end if enabled.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
            **kwargs: Additional keyword arguments
        """
        if self.val_dataset is None or not self.eval_on_epoch_end:
            return

        val_loss = self._run_validation(trainer)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            logger.info(f"New best validation loss: {val_loss:.4f} at epoch {epoch}")

    def _run_validation(self, trainer) -> float:
        """Run validation and compute loss.

        Args:
            trainer: The trainer instance

        Returns:
            Validation loss
        """
        if self.val_dataset is None:
            return float("inf")

        model = trainer.model
        original_mode = model.training
        model.eval()

        total_loss = 0.0
        num_batches = 0

        try:
            # Create dataloader if trainer has one
            if hasattr(trainer, "val_dataloader") and trainer.val_dataloader is not None:
                dataloader = trainer.val_dataloader
            else:
                # Create a simple dataloader from dataset
                from torch.utils.data import DataLoader
                dataloader = DataLoader(self.val_dataset, batch_size=1)

            device = trainer.device if hasattr(trainer, "device") else "cpu"

            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(device)

                    # Compute loss
                    if hasattr(model, "forward"):
                        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
                        if hasattr(outputs, "loss"):
                            loss = outputs.loss
                        elif isinstance(outputs, dict) and "loss" in outputs:
                            loss = outputs["loss"]
                        else:
                            # Skip if no loss available
                            continue
                    else:
                        continue

                    total_loss += loss.item()
                    num_batches += 1

        finally:
            model.train(original_mode)

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss


class WandBCallback(SFTCallback):
    """Logs metrics and artifacts to Weights & Biases.

    Initializes a W&B run at training start, logs metrics during training,
    and optionally logs model artifacts at checkpoints.

    Args:
        project: W&B project name (default: "llm-playground-sft")
        run_name: W&B run name (optional, auto-generated if None)
        config: Configuration dict to log (optional)
        log_model: Whether to log model checkpoints to W&B (default: False)

    Example:
        >>> config = {"learning_rate": 1e-4, "batch_size": 8}
        >>> callback = WandBCallback(project="my-sft", config=config, log_model=True)
        >>> trainer.add_callback(callback)
    """

    def __init__(
        self,
        project: str = "llm-playground-sft",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = False,
    ):
        """Initialize W&B callback."""
        self.project = project
        self.run_name = run_name
        self.config = config
        self.log_model = log_model
        self.run = None

    def on_train_begin(self, trainer, **kwargs):
        """Initialize W&B run at training start.

        Args:
            trainer: The trainer instance
            **kwargs: Additional keyword arguments
        """
        try:
            import wandb

            self.run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
            )
            logger.info(f"Initialized W&B run: {self.run.name}")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.run = None

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs):
        """Log step metrics to W&B.

        Args:
            trainer: The trainer instance
            step: Current training step
            metrics: Dictionary of metrics for this step
            **kwargs: Additional keyword arguments
        """
        if self.run is None:
            return

        try:
            import wandb

            log_data = {**metrics, "step": step}
            wandb.log(log_data)
        except Exception as e:
            logger.warning(f"Failed to log step metrics to W&B: {e}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log epoch metrics to W&B.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
            **kwargs: Additional keyword arguments
        """
        if self.run is None:
            return

        try:
            import wandb

            log_data = {**metrics, "epoch": epoch}
            wandb.log(log_data)
        except Exception as e:
            logger.warning(f"Failed to log epoch metrics to W&B: {e}")

    def on_train_end(self, trainer, **kwargs):
        """Finish W&B run at training end.

        Args:
            trainer: The trainer instance
            **kwargs: Additional keyword arguments
        """
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("Finished W&B run")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")


class CheckpointCallback(SFTCallback):
    """Saves model checkpoints based on validation performance.

    Saves checkpoints every `save_steps` steps and optionally keeps only the best
    model based on validation loss. Automatically rotates old checkpoints to
    maintain a maximum number of saved checkpoints.

    Args:
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps (default: 500)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        save_best_only: Only save checkpoint if validation improves (default: False)

    Example:
        >>> callback = CheckpointCallback(
        ...     output_dir="./checkpoints",
        ...     save_steps=100,
        ...     save_total_limit=5,
        ...     save_best_only=True,
        ... )
        >>> trainer.add_callback(callback)
    """

    def __init__(
        self,
        output_dir: str,
        save_steps: int = 500,
        save_total_limit: int = 3,
        save_best_only: bool = False,
    ):
        """Initialize checkpoint callback."""
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.best_val_loss = float("inf")
        self.checkpoints: List[str] = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float], **kwargs):
        """Save checkpoint at specified intervals.

        Args:
            trainer: The trainer instance
            step: Current training step
            metrics: Dictionary of metrics for this step
            **kwargs: Additional keyword arguments
        """
        if step % self.save_steps != 0:
            return

        # Check if we should save based on validation loss
        val_loss = metrics.get("val_loss", None)
        should_save = True

        if self.save_best_only and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                should_save = True
                logger.info(f"New best validation loss: {val_loss:.4f}, saving checkpoint")
            else:
                should_save = False

        if should_save:
            self._save_checkpoint(trainer, step, metrics)

    def _save_checkpoint(self, trainer, step: int, metrics: Dict[str, float]):
        """Save checkpoint to disk.

        Args:
            trainer: The trainer instance
            step: Current training step
            metrics: Dictionary of metrics for this step
        """
        checkpoint_path = self._get_checkpoint_path(step)

        try:
            # Save model state
            if hasattr(trainer, "model"):
                # If using PEFT/LoRA, save adapter weights
                model = trainer.model
                if hasattr(model, "save_pretrained"):
                    # For PEFT models
                    adapter_path = checkpoint_path.replace(".pt", "")
                    model.save_pretrained(adapter_path)
                    logger.info(f"Saved checkpoint adapter to {adapter_path}")
                else:
                    # For regular models
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "metrics": metrics,
                        },
                        checkpoint_path,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                self.checkpoints.append(checkpoint_path)
                self._rotate_checkpoints()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _get_checkpoint_path(self, step: int) -> str:
        """Get checkpoint file path for a given step.

        Args:
            step: Training step number

        Returns:
            Path to checkpoint file
        """
        return os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")

    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit.

        Keeps only the most recent `save_total_limit` checkpoints and deletes older ones.
        """
        if len(self.checkpoints) <= self.save_total_limit:
            return

        # Sort checkpoints by modification time (oldest first)
        checkpoints_to_delete = self.checkpoints[: -self.save_total_limit]

        for checkpoint_path in checkpoints_to_delete:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logger.info(f"Deleted old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")

        # Keep only recent checkpoints in the list
        self.checkpoints = self.checkpoints[-self.save_total_limit :]
