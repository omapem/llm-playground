"""Tests for checkpoint management system."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from app.training.checkpoint import CheckpointManager
from app.training.scheduler import CosineSchedulerWithWarmup
from app.training.config import TrainingConfig
from app.transformer import GPT2Small

# Register TrainingConfig as safe for torch.load with weights_only=True
torch.serialization.add_safe_globals([TrainingConfig])


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_optimizer_scheduler(self):
        """Create model, optimizer, and scheduler."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=10,
            total_steps=100,
        )
        return model, optimizer, scheduler

    def test_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.max_checkpoints == 5
        assert os.path.isdir(temp_checkpoint_dir)

    def test_save_checkpoint_creates_file(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test saving checkpoint creates file."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.5,
            config=config,
        )

        assert os.path.exists(path)
        assert "checkpoint_step_10" in path

    def test_checkpoint_contains_all_state(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test checkpoint contains all necessary state."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.5,
            config=config,
        )

        # Load checkpoint (disable weights_only for TrainingConfig loading)
        checkpoint = torch.load(path, weights_only=False)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "step" in checkpoint
        assert "loss" in checkpoint
        assert "config" in checkpoint
        assert "timestamp" in checkpoint

        # Verify values
        assert checkpoint["step"] == 10
        assert checkpoint["loss"] == 1.5

    def test_load_checkpoint_restores_state(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test loading checkpoint restores state correctly."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        # Step scheduler to step 50 before saving
        for _ in range(50):
            scheduler.step()

        # Save checkpoint
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=50,
            loss=1.5,
            config=config,
        )

        # Create new model/optimizer/scheduler
        model2 = nn.Linear(10, 10)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer2,
            warmup_steps=10,
            total_steps=100,
        )

        # Load checkpoint
        step, loss, loaded_config = manager.load_checkpoint(
            path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        assert step == 50
        assert loss == 1.5
        assert loaded_config is not None
        assert scheduler2.current_step == 50

    def test_checkpoint_rotation(self, temp_checkpoint_dir, model_optimizer_scheduler):
        """Test checkpoint rotation keeps only max_checkpoints."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=3)

        # Save 5 checkpoints
        for step in [10, 20, 30, 40, 50]:
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=float(step) / 10,
                config=config,
            )

        # Only 3 most recent should exist
        checkpoints = sorted(Path(temp_checkpoint_dir).glob("checkpoint_step_*.pt"))
        assert len(checkpoints) == 3

        # Verify they are the most recent ones
        checkpoint_steps = [
            int(cp.stem.split("_")[-1]) for cp in checkpoints
        ]
        assert set(checkpoint_steps) == {30, 40, 50}

    def test_get_latest_checkpoint(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test finding the latest checkpoint."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)

        # Save multiple checkpoints
        for step in [10, 20, 30]:
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=float(step) / 10,
                config=config,
            )

        # Get latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "checkpoint_step_30" in latest

    def test_get_latest_checkpoint_empty_dir(self, temp_checkpoint_dir):
        """Test get_latest_checkpoint with no checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        latest = manager.get_latest_checkpoint()
        assert latest is None

    def test_list_checkpoints(self, temp_checkpoint_dir, model_optimizer_scheduler):
        """Test listing all checkpoints."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)

        # Save checkpoints
        for step in [10, 20, 30]:
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=float(step) / 10,
                config=config,
            )

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        # Should be sorted by step (ascending)
        steps = [cp["step"] for cp in checkpoints]
        assert steps == [10, 20, 30]

    def test_checkpoint_metadata(self, temp_checkpoint_dir, model_optimizer_scheduler):
        """Test checkpoint metadata is preserved."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=42,
            loss=2.71,
            config=config,
        )

        # Load and verify metadata
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["step"] == 42
        assert abs(checkpoint["loss"] - 2.71) < 1e-6

    def test_checkpoint_with_config(self, temp_checkpoint_dir, model_optimizer_scheduler):
        """Test checkpoint preserves training config."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=16,
            learning_rate=1e-3,
            max_steps=1000,
        )

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.5,
            config=config,
        )

        # Load checkpoint
        model2 = nn.Linear(10, 10)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer2,
            warmup_steps=10,
            total_steps=100,
        )

        step, loss, loaded_config = manager.load_checkpoint(
            path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        assert loaded_config.batch_size == 16
        assert loaded_config.learning_rate == 1e-3
        assert loaded_config.max_steps == 1000

    def test_checkpoint_atomic_write(self, temp_checkpoint_dir, model_optimizer_scheduler):
        """Test checkpoint is written atomically."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.5,
            config=config,
        )

        # File should exist and be readable
        assert os.path.exists(path)
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint is not None

    def test_multiple_checkpoints_dont_interfere(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test multiple checkpoints don't interfere with each other."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=10)

        # Save and load multiple checkpoints
        paths = []
        for step in [10, 20, 30, 40, 50]:
            path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=float(step) / 10,
                config=config,
            )
            paths.append((path, step))

        # Verify each checkpoint has correct step
        for path, expected_step in paths:
            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["step"] == expected_step

    def test_resume_from_latest_checkpoint(
        self, temp_checkpoint_dir, model_optimizer_scheduler
    ):
        """Test typical resume workflow."""
        model, optimizer, scheduler = model_optimizer_scheduler
        config = TrainingConfig(model_config=GPT2Small())

        # Step scheduler to 100 before saving
        for _ in range(100):
            scheduler.step()

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)

        # Save checkpoint at step 100
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            loss=1.23,
            config=config,
        )

        # Simulate resume: get latest checkpoint
        latest_path = manager.get_latest_checkpoint()
        assert latest_path is not None

        # Create new model/optimizer/scheduler
        model2 = nn.Linear(10, 10)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer2,
            warmup_steps=10,
            total_steps=200,
        )

        # Load checkpoint
        resume_step, resume_loss, resume_config = manager.load_checkpoint(
            latest_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        assert resume_step == 100
        assert abs(resume_loss - 1.23) < 1e-6
        assert scheduler2.current_step == 100

    def test_checkpoint_device_compatibility(self, temp_checkpoint_dir):
        """Test checkpoint works across devices (CPU)."""
        # Create and save on CPU
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            warmup_steps=10,
            total_steps=100,
        )
        config = TrainingConfig(model_config=GPT2Small())

        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.5,
            config=config,
        )

        # Load on CPU (should work)
        model2 = nn.Linear(10, 10)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineSchedulerWithWarmup(
            optimizer2,
            warmup_steps=10,
            total_steps=100,
        )

        step, loss, loaded_config = manager.load_checkpoint(
            path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        assert step == 10
        assert abs(loss - 1.5) < 1e-6


class TestCheckpointIntegration:
    """Integration tests for checkpoint system."""

    def test_training_loop_with_checkpointing(self):
        """Test checkpoint system in a training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            model = nn.Linear(10, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = CosineSchedulerWithWarmup(
                optimizer,
                warmup_steps=10,
                total_steps=100,
            )
            config = TrainingConfig(model_config=GPT2Small())
            manager = CheckpointManager(tmpdir, max_checkpoints=5)

            # Train and save checkpoints
            checkpoint_steps = []
            for step in range(1, 101):
                # Simulate training step
                x = torch.randn(32, 10)
                y = model(x)
                loss = y.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Save checkpoint every 25 steps
                if step % 25 == 0:
                    manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        loss=loss.item(),
                        config=config,
                    )
                    checkpoint_steps.append(step)

            # Verify checkpoints were saved
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 4
            assert [cp["step"] for cp in checkpoints] == checkpoint_steps

            # Resume from checkpoint at step 50
            resume_path = [
                cp["path"]
                for cp in checkpoints
                if cp["step"] == 50
            ][0]

            model2 = nn.Linear(10, 1)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            scheduler2 = CosineSchedulerWithWarmup(
                optimizer2,
                warmup_steps=10,
                total_steps=100,
            )

            resume_step, _, _ = manager.load_checkpoint(
                resume_path,
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
            )

            assert resume_step == 50
            assert scheduler2.current_step == 50
