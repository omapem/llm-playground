"""Tests for main training orchestrator."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from app.training.trainer import Trainer
from app.training.config import TrainingConfig
from app.transformer import GPT2Small, TransformerConfig


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 100, seq_len: int = 128, vocab_size: int = 50257):
        """Initialize dataset."""
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Return random token sequence."""
        return torch.randint(0, self.vocab_size, (self.seq_len,))


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_config(temp_checkpoint_dir):
    """Create simple training config for testing."""
    model_config = TransformerConfig(
        vocab_size=50257,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
    )
    return TrainingConfig(
        model_config=model_config,
        batch_size=4,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=2,
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=5,
        logging_steps=2,
    )


@pytest.fixture
def simple_dataset():
    """Create simple dataset for testing."""
    return SimpleDataset(size=50, seq_len=128)


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_initialization(self, simple_config, simple_dataset):
        """Test Trainer initialization."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.config == simple_config
        assert trainer.train_dataset == simple_dataset

    def test_model_created(self, simple_config, simple_dataset):
        """Test model is created from config."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.model is not None
        assert isinstance(trainer.model, nn.Module)

    def test_optimizer_created(self, simple_config, simple_dataset):
        """Test optimizer is created."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.optimizer is not None

    def test_scheduler_created(self, simple_config, simple_dataset):
        """Test scheduler is created."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.scheduler is not None

    def test_dataloader_created(self, simple_config, simple_dataset):
        """Test dataloader is created with correct batch size."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.train_loader is not None
        batch = next(iter(trainer.train_loader))
        assert batch.shape[0] == simple_config.batch_size

    def test_checkpoint_manager_created(self, simple_config, simple_dataset):
        """Test checkpoint manager is created."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.checkpoint_manager is not None

    def test_metrics_tracker_created(self, simple_config, simple_dataset):
        """Test metrics tracker is created."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        assert trainer.metrics_tracker is not None


class TestTrainerTrainingLoop:
    """Tests for training loop functionality."""

    def test_train_runs_successfully(self, simple_config, simple_dataset):
        """Test basic training loop runs without errors."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()
        # Should complete without errors

    def test_train_updates_step_counter(self, simple_config, simple_dataset):
        """Test step counter is updated during training."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()
        assert trainer.current_step == simple_config.max_steps

    def test_train_computes_loss(self, simple_config, simple_dataset):
        """Test loss is computed during training."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()
        # Metrics should contain loss
        averages = trainer.metrics_tracker.get_averages()
        assert "loss" in averages
        assert averages["loss"] > 0

    def test_train_updates_gradients(self, simple_config, simple_dataset):
        """Test gradients are computed and applied."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)

        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        trainer.train()

        # Parameters should have changed
        final_params = list(trainer.model.parameters())
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, final_params)
        )
        assert params_changed

    def test_train_saves_checkpoints(self, simple_config, simple_dataset):
        """Test checkpoints are saved during training."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()

        # Check checkpoint directory
        checkpoints = list(Path(simple_config.checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) > 0

    def test_checkpoint_saved_at_save_steps(self, simple_config, simple_dataset):
        """Test checkpoint is saved at specified intervals."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()

        # Should have checkpoint at step 5 and step 10
        checkpoints = list(Path(simple_config.checkpoint_dir).glob("checkpoint_*.pt"))
        checkpoint_steps = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
        assert 5 in checkpoint_steps
        assert 10 in checkpoint_steps


class TestTrainerGradientAccumulation:
    """Tests for gradient accumulation."""

    def test_gradient_accumulation(self, temp_checkpoint_dir, simple_dataset):
        """Test gradient accumulation works correctly."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            max_steps=4,
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=simple_dataset)
        trainer.train()

        # Should complete successfully
        assert trainer.current_step == 4

    def test_effective_batch_size(self, temp_checkpoint_dir, simple_dataset):
        """Test effective batch size with accumulation."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-3,
            max_steps=2,
            checkpoint_dir=temp_checkpoint_dir,
        )

        effective_batch_size = config.get_effective_batch_size()
        assert effective_batch_size == 16  # 4 * 4


class TestTrainerGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self, temp_checkpoint_dir, simple_dataset):
        """Test gradient clipping is applied."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            max_grad_norm=1.0,
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=simple_dataset)
        trainer.train()

        # Training should complete with clipping enabled
        assert trainer.current_step == 5


class TestTrainerMixedPrecision:
    """Tests for mixed precision training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_training(self, temp_checkpoint_dir, simple_dataset):
        """Test FP16 mixed precision training."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            mixed_precision="fp16",
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=simple_dataset)
        trainer.train()

        assert trainer.current_step == 5

    def test_no_mixed_precision(self, temp_checkpoint_dir, simple_dataset):
        """Test training without mixed precision."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            mixed_precision=None,
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=simple_dataset)
        trainer.train()

        assert trainer.current_step == 5


class TestTrainerResume:
    """Tests for training resumption."""

    def test_resume_from_checkpoint(self, temp_checkpoint_dir, simple_dataset):
        """Test resuming training from checkpoint."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=10,
            save_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # First training session
        trainer1 = Trainer(config=config, train_dataset=simple_dataset)
        trainer1.train()

        # Resume from checkpoint
        config.resume_from_checkpoint = True
        trainer2 = Trainer(config=config, train_dataset=simple_dataset)

        # Should resume from step 10
        assert trainer2.current_step == 10

    def test_resume_continues_training(self, temp_checkpoint_dir, simple_dataset):
        """Test resumed training continues from checkpoint step."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            save_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Train to step 5
        trainer1 = Trainer(config=config, train_dataset=simple_dataset)
        trainer1.train()
        assert trainer1.current_step == 5

        # Resume and train to step 10
        config.max_steps = 10
        config.resume_from_checkpoint = True
        trainer2 = Trainer(config=config, train_dataset=simple_dataset)
        trainer2.train()

        assert trainer2.current_step == 10


class TestTrainerMetrics:
    """Tests for metrics tracking."""

    def test_metrics_logged(self, simple_config, simple_dataset):
        """Test metrics are logged during training."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()

        averages = trainer.metrics_tracker.get_averages()
        assert "loss" in averages
        assert "grad_norm" in averages
        assert "lr" in averages

    def test_perplexity_computed(self, simple_config, simple_dataset):
        """Test perplexity is computed from loss."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()

        averages = trainer.metrics_tracker.get_averages()
        assert "perplexity" in averages
        assert averages["perplexity"] > 0

    def test_throughput_computed(self, simple_config, simple_dataset):
        """Test throughput is computed."""
        trainer = Trainer(config=simple_config, train_dataset=simple_dataset)
        trainer.train()

        throughput = trainer.get_throughput()
        assert "steps_per_second" in throughput
        assert "tokens_per_second" in throughput


class TestTrainerIntegration:
    """Integration tests for complete training workflow."""

    def test_complete_training_workflow(self, temp_checkpoint_dir, simple_dataset):
        """Test complete end-to-end training workflow."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
                max_position_embeddings=128,
            ),
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=6e-4,
            max_steps=20,
            warmup_steps=5,
            scheduler_type="cosine",
            max_grad_norm=1.0,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=10,
            logging_steps=5,
        )

        trainer = Trainer(config=config, train_dataset=simple_dataset)
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 20

        # Verify checkpoints were saved
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 2  # At steps 10 and 20

        # Verify metrics were collected
        averages = trainer.metrics_tracker.get_averages()
        assert "loss" in averages
        assert "grad_norm" in averages
        assert "lr" in averages
        assert "perplexity" in averages

    def test_training_with_validation(self, temp_checkpoint_dir):
        """Test training with validation dataset."""
        train_dataset = SimpleDataset(size=100, seq_len=128)
        val_dataset = SimpleDataset(size=20, seq_len=128)

        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=10,
            checkpoint_dir=temp_checkpoint_dir,
            eval_steps=5,
        )

        trainer = Trainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        trainer.train()

        # Should complete successfully with validation
        assert trainer.current_step == 10
