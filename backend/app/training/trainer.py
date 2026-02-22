"""Main training orchestrator for LLM training.

Coordinates all training components including model, optimizer, scheduler,
checkpointing, and metrics tracking. Supports distributed training (DDP),
mixed precision, gradient accumulation, and training resumption.
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
from .distributed import DistributedConfig, create_distributed_dataloader, reduce_mean
from .scheduler import get_scheduler
from .checkpoint import CheckpointManager
from .checkpoint_cleaner import CheckpointCleaner
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
    optimization, scheduling, checkpointing, and metrics tracking. Supports
    single-process and distributed (DDP) training.

    Args:
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        dist_config: Optional distributed training configuration. When provided,
            the model is wrapped in DistributedDataParallel and data is
            partitioned across ranks via DistributedSampler.

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
        dist_config: Optional[DistributedConfig] = None,
    ) -> None:
        """Initialize trainer."""
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dist_config = dist_config
        self.current_step = 0

        # Set device (respect distributed local_rank when available)
        if dist_config is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{dist_config.local_rank}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and config.num_devices > 0 else "cpu"
            )

        # Initialize model
        self.model = GPTModel(config.model_config)
        self.model.to(self.device)

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning(
                    "gradient_checkpointing=True but model does not support it"
                )

        # Wrap model in DDP if distributed training is configured.
        # This must happen AFTER model.to(device) and gradient checkpointing.
        if self.dist_config is not None:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[dist_config.local_rank] if torch.cuda.is_available() else None,
            )
            logger.info(f"Model wrapped in DDP (rank {dist_config.rank})")

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

        # Initialize data loader (with distributed sampler when DDP is active)
        self.train_loader = create_distributed_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            dist_config=dist_config,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        if val_dataset is not None:
            self.val_loader = create_distributed_dataloader(
                val_dataset,
                batch_size=config.batch_size,
                dist_config=dist_config,
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

        # Initialize checkpoint cleaner for quality-based retention
        self.checkpoint_cleaner = CheckpointCleaner(
            checkpoint_dir=config.checkpoint_dir,
            save_total_limit=config.max_checkpoints_to_keep,
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

        # Initialize W&B if configured (only on main process)
        self.use_wandb = False
        if config.wandb_project and self.is_main_process:
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

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0 or non-distributed)."""
        return self.dist_config is None or self.dist_config.is_main_process

    def _get_unwrapped_model(self) -> nn.Module:
        """Get the unwrapped model (without DDP wrapper).

        When the model is wrapped in DistributedDataParallel, the actual
        model is accessible via `model.module`. This helper returns the
        underlying model regardless of wrapping.

        Returns:
            The unwrapped nn.Module
        """
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

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
        """Resume training from a checkpoint.

        Supports two modes:
        - ``resume_from_checkpoint="latest"``: finds the most recent checkpoint
          in ``checkpoint_dir``.
        - ``resume_from_checkpoint="<path>"``: loads the specific checkpoint file.

        Restores model weights, optimizer state, scheduler state, and the
        training step counter so that training continues seamlessly.
        """
        resume_path = self.config.resume_from_checkpoint

        if resume_path == "latest" or resume_path is True:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No checkpoint found to resume from")
                return
        else:
            checkpoint_path = str(resume_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        step, loss, _ = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self._get_unwrapped_model(),
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
        epoch = 0
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
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
                epoch += 1
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)
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

            # Log metrics (main process only in distributed mode)
            if self.current_step % self.config.logging_steps == 0:
                if self.is_main_process:
                    self._log_metrics(loss)

            # Log GPU memory every 100 steps (main process only)
            if self.current_step % 100 == 0:
                if self.is_main_process:
                    self._log_gpu_memory(self.current_step)

            # Save checkpoint (main process only)
            saved_ckpt_path = None
            if self.current_step % self.config.save_steps == 0:
                if self.is_main_process:
                    saved_ckpt_path = self._save_checkpoint(loss)

            # Validation
            if (
                self.val_loader is not None
                and self.config.eval_steps > 0
                and self.current_step % self.config.eval_steps == 0
            ):
                val_loss = self._validate()

                # Register checkpoint with its validation loss only if a
                # checkpoint was saved at this exact step (avoids mismatch
                # when save_steps and eval_steps don't align)
                if self.is_main_process and saved_ckpt_path is not None and val_loss is not None:
                    self.checkpoint_cleaner.register(
                        checkpoint_path=saved_ckpt_path,
                        val_loss=val_loss,
                        step=self.current_step,
                    )

        logger.info(f"Training completed at step {self.current_step}")

        # Save final checkpoint (main process only)
        if self.is_main_process:
            self._save_checkpoint(loss)

            # Run quality-based checkpoint cleanup
            self._run_checkpoint_cleanup()

        # Close W&B if used
        if self.use_wandb:
            import wandb

            wandb.finish()

    def _training_step(self, batch) -> float:
        """Execute single training step.

        Args:
            batch: Input batch of token IDs (tensor or tuple from TensorDataset)

        Returns:
            Loss value for this step
        """
        # Handle TensorDataset tuples: (tensor,)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

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

        # Compute unscaled loss value for logging
        loss_val = loss.item() * self.config.gradient_accumulation_steps

        # Reduce loss across ranks for accurate metrics in distributed mode
        if self.dist_config is not None:
            loss_tensor = torch.tensor(loss_val, device=self.device)
            loss_tensor = reduce_mean(loss_tensor)
            loss_val = loss_tensor.item()

        return loss_val

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

    def _log_gpu_memory(self, step: int) -> None:
        """Log GPU memory usage.

        Args:
            step: Current training step
        """
        if not torch.cuda.is_available():
            return

        # Get GPU memory usage
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)

        # Log to console
        logger.info(
            f"Step {step} | "
            f"GPU Memory - Allocated: {allocated_gb:.2f}GB | "
            f"Reserved: {reserved_gb:.2f}GB"
        )

        # Log to W&B
        if self.use_wandb:
            import wandb

            wandb.log({
                "train/gpu_memory_allocated_gb": allocated_gb,
                "train/gpu_memory_reserved_gb": reserved_gb,
                "train/step": step,
            })

    def _save_checkpoint(self, loss: float) -> str:
        """Save training checkpoint.

        Uses the unwrapped model (without DDP wrapper) to avoid the
        ``module.`` prefix in state dict keys.

        Args:
            loss: Current loss value

        Returns:
            Path to the saved checkpoint file
        """
        path = self.checkpoint_manager.save_checkpoint(
            model=self._get_unwrapped_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.current_step,
            loss=loss,
            config=self.config,
        )
        logger.info(f"Saved checkpoint to {path}")
        return path

    def _validate(self) -> Optional[float]:
        """Run validation loop.

        Returns:
            Average validation loss, or None if no validation data is available
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Handle TensorDataset tuples: (tensor,)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

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

        # Reduce validation loss across ranks for consistent metrics
        if self.dist_config is not None:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            loss_tensor = reduce_mean(loss_tensor)
            avg_loss = loss_tensor.item()
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

        return avg_loss

    def _run_checkpoint_cleanup(self) -> None:
        """Run quality-based checkpoint cleanup after training.

        Uses the CheckpointCleaner to remove checkpoints beyond the
        save_total_limit, keeping only the N best by validation loss.
        Only runs if checkpoints have been registered with the cleaner
        (i.e., validation was performed during training).
        """
        tracked = self.checkpoint_cleaner.get_best_checkpoints()
        if not tracked:
            logger.info(
                "No checkpoints registered with validation loss; "
                "skipping quality-based cleanup"
            )
            return

        removed = self.checkpoint_cleaner.cleanup()
        if removed:
            logger.info(
                f"Checkpoint cleanup removed {len(removed)} checkpoint(s): "
                f"{removed}"
            )

        best = self.checkpoint_cleaner.get_best_checkpoint()
        if best is not None:
            logger.info(f"Best checkpoint (lowest val loss): {best}")

    def get_throughput(self) -> Dict[str, float]:
        """Compute training throughput.

        Returns:
            Dict with 'steps_per_second' and 'tokens_per_second'
        """
        tokens_per_batch = self.config.batch_size * (self.config.model_config.max_position_embeddings - 1)
        return self.metrics_tracker.get_throughput(tokens_per_batch)
