"""Main training orchestrator for LLM training.

Coordinates all training components including model, optimizer, scheduler,
checkpointing, and metrics tracking. Supports distributed training, mixed
precision, gradient accumulation, and training resumption.
"""

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

from app.transformer.model import GPTModel
from .config import TrainingConfig
from .scheduler import get_scheduler
from .checkpoint import CheckpointManager
from .metrics import (
    compute_perplexity,
    compute_gradient_norm,
    get_gpu_memory_usage,
    MetricsTracker,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator.

    Coordinates model training with all supporting infrastructure including
    optimization, scheduling, checkpointing, and metrics tracking.

    Args:
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Optional validation dataset

    Example:
        >>> config = TrainingConfig.from_yaml('config.yaml')
        >>> train_data = MyDataset('train.txt')
        >>> trainer = Trainer(config, train_data)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """Initialize trainer."""
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.current_step = 0

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.num_devices > 0 else "cpu"
        )

        # Initialize model
        self.model = GPTModel(config.model_config)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = get_scheduler(
            config.scheduler_type,
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=0.1,
        )

        # Initialize data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints_to_keep,
        )

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=100)

        # Initialize mixed precision training
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        if self.use_amp:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.scaler = GradScaler(device_type, enabled=(config.mixed_precision == "fp16"))
        else:
            self.scaler = None

        # Initialize W&B if configured
        self.use_wandb = False
        if config.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name,
                    config=config.to_dict(),
                )
                self.use_wandb = True
                logger.info("W&B logging enabled")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")

        # Resume from checkpoint if requested
        if config.resume_from_checkpoint:
            self._resume_from_checkpoint()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        if self.config.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _resume_from_checkpoint(self) -> None:
        """Resume training from latest checkpoint."""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()

        if latest_checkpoint is None:
            logger.warning("No checkpoint found to resume from")
            return

        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        step, loss, config = self.checkpoint_manager.load_checkpoint(
            latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.current_step = step
        logger.info(f"Resumed from step {step}, loss {loss:.4f}")

    def train(self, cancellation_event: Optional['threading.Event'] = None) -> None:
        """Run training loop.

        Args:
            cancellation_event: Optional event to signal training cancellation
        """
        import threading

        self.model.train()
        logger.info(f"Starting training from step {self.current_step}")

        # Create infinite data iterator
        train_iter = iter(self.train_loader)

        step_count = 0
        while self.current_step < self.config.max_steps:
            # Check for cancellation:
            # - At the start of each iteration
            # - Every 10 steps within the loop
            if cancellation_event and cancellation_event.is_set():
                logger.info(f"Training cancelled at step {self.current_step}")
                return

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            loss = self._training_step(batch)

            # Update step counter
            self.current_step += 1
            step_count += 1

            # Check for cancellation every 10 steps
            if cancellation_event and step_count >= 10:
                step_count = 0
                if cancellation_event.is_set():
                    logger.info(f"Training cancelled at step {self.current_step}")
                    return

            # Log metrics
            if self.current_step % self.config.logging_steps == 0:
                self._log_metrics(loss)

            # Save checkpoint
            if self.current_step % self.config.save_steps == 0:
                self._save_checkpoint(loss)

            # Validation
            if (
                self.val_loader is not None
                and self.config.eval_steps > 0
                and self.current_step % self.config.eval_steps == 0
            ):
                self._validate()

        logger.info(f"Training completed at step {self.current_step}")

        # Save final checkpoint
        self._save_checkpoint(loss)

        # Close W&B if used
        if self.use_wandb:
            import wandb

            wandb.finish()

    def _training_step(self, batch: torch.Tensor) -> float:
        """Execute single training step.

        Args:
            batch: Input batch of token IDs

        Returns:
            Loss value for this step
        """
        # Move batch to device
        batch = batch.to(self.device)

        # Create input and target sequences
        # For language modeling: input = [0:seq_len-1], target = [1:seq_len]
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # Forward pass with mixed precision
        if self.use_amp:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
            with autocast(device_type, dtype=dtype):
                output = self.model(input_ids, labels=target_ids)
                loss = output.loss / self.config.gradient_accumulation_steps
        else:
            output = self.model(input_ids, labels=target_ids)
            loss = output.loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Scheduler step
            self.scheduler.step()

        # Return unscaled loss
        return loss.item() * self.config.gradient_accumulation_steps

    def _log_metrics(self, loss: float) -> None:
        """Log training metrics.

        Args:
            loss: Current loss value
        """
        # Compute metrics
        grad_norm = compute_gradient_norm(self.model)
        perplexity = compute_perplexity(loss)
        lr = self.optimizer.param_groups[0]["lr"]

        # Update metrics tracker
        self.metrics_tracker.update({
            "loss": loss,
            "perplexity": perplexity,
            "grad_norm": grad_norm,
            "lr": lr,
        })

        # Get averages
        averages = self.metrics_tracker.get_averages()

        # Log to console
        logger.info(
            f"Step {self.current_step}/{self.config.max_steps} | "
            f"Loss: {averages['loss']:.4f} | "
            f"PPL: {averages['perplexity']:.2f} | "
            f"GradNorm: {averages['grad_norm']:.4f} | "
            f"LR: {lr:.2e}"
        )

        # Log to W&B
        if self.use_wandb:
            import wandb

            wandb.log({
                "train/loss": averages["loss"],
                "train/perplexity": averages["perplexity"],
                "train/grad_norm": averages["grad_norm"],
                "train/learning_rate": lr,
                "train/step": self.current_step,
            })

    def _save_checkpoint(self, loss: float) -> None:
        """Save training checkpoint.

        Args:
            loss: Current loss value
        """
        path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.current_step,
            loss=loss,
            config=self.config,
        )
        logger.info(f"Saved checkpoint to {path}")

    def _validate(self) -> None:
        """Run validation loop."""
        if self.val_loader is None:
            return

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]

                if self.use_amp:
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
                    with autocast(device_type, dtype=dtype):
                        output = self.model(input_ids, labels=target_ids)
                        loss = output.loss
                else:
                    output = self.model(input_ids, labels=target_ids)
                    loss = output.loss

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = compute_perplexity(avg_loss)

        logger.info(
            f"Validation at step {self.current_step} | "
            f"Loss: {avg_loss:.4f} | "
            f"PPL: {perplexity:.2f}"
        )

        if self.use_wandb:
            import wandb

            wandb.log({
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
                "train/step": self.current_step,
            })

        self.model.train()

    def get_throughput(self) -> Dict[str, float]:
        """Compute training throughput.

        Returns:
            Dict with 'steps_per_second' and 'tokens_per_second'
        """
        tokens_per_batch = self.config.batch_size * (self.config.model_config.max_position_embeddings - 1)
        return self.metrics_tracker.get_throughput(tokens_per_batch)
