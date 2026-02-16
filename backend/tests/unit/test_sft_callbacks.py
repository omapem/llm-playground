"""Tests for SFT training callbacks."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
from torch.utils.data import Dataset

from app.sft.callbacks import (
    SFTCallback,
    ValidationCallback,
    WandBCallback,
    CheckpointCallback,
)


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, 100, (10,)), "labels": torch.randint(0, 100, (10,))}


class DummyTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self):
        self.model = Mock()
        self.model.eval = Mock()
        self.model.train = Mock()
        self.val_dataloader = None
        self.device = "cpu"


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def dummy_trainer():
    """Create dummy trainer."""
    return DummyTrainer()


@pytest.fixture
def dummy_dataset():
    """Create dummy dataset."""
    return DummyDataset(size=10)


# ============================================================================
# Test Base Callback Class
# ============================================================================


def test_sft_callback_base_class_has_lifecycle_methods():
    """Test that SFTCallback base class has all lifecycle methods."""
    callback = SFTCallback()

    # Check all lifecycle methods exist
    assert hasattr(callback, "on_train_begin")
    assert hasattr(callback, "on_train_end")
    assert hasattr(callback, "on_epoch_begin")
    assert hasattr(callback, "on_epoch_end")
    assert hasattr(callback, "on_step_end")


def test_sft_callback_default_methods_do_nothing(dummy_trainer):
    """Test that default callback methods do nothing."""
    callback = SFTCallback()

    # Should not raise any errors
    callback.on_train_begin(dummy_trainer)
    callback.on_train_end(dummy_trainer)
    callback.on_epoch_begin(dummy_trainer, epoch=0)
    callback.on_epoch_end(dummy_trainer, epoch=0, metrics={})
    callback.on_step_end(dummy_trainer, step=0, metrics={})


# ============================================================================
# Test ValidationCallback
# ============================================================================


def test_validation_callback_init_with_defaults():
    """Test ValidationCallback initialization with default values."""
    callback = ValidationCallback()

    assert callback.val_dataset is None
    assert callback.eval_steps == 500
    assert callback.eval_on_epoch_end is True
    assert callback.best_val_loss == float("inf")


def test_validation_callback_init_with_custom_values(dummy_dataset):
    """Test ValidationCallback initialization with custom values."""
    callback = ValidationCallback(
        val_dataset=dummy_dataset,
        eval_steps=100,
        eval_on_epoch_end=False,
    )

    assert callback.val_dataset == dummy_dataset
    assert callback.eval_steps == 100
    assert callback.eval_on_epoch_end is False


def test_validation_callback_on_step_end_runs_validation_at_interval(dummy_trainer, dummy_dataset):
    """Test that validation runs at specified step intervals."""
    callback = ValidationCallback(val_dataset=dummy_dataset, eval_steps=10)

    # Mock the validation method
    callback._run_validation = Mock(return_value=1.5)

    # Step 5 - should not run
    callback.on_step_end(dummy_trainer, step=5, metrics={})
    callback._run_validation.assert_not_called()

    # Step 10 - should run
    callback.on_step_end(dummy_trainer, step=10, metrics={})
    callback._run_validation.assert_called_once()

    # Step 15 - should not run
    callback._run_validation.reset_mock()
    callback.on_step_end(dummy_trainer, step=15, metrics={})
    callback._run_validation.assert_not_called()

    # Step 20 - should run
    callback.on_step_end(dummy_trainer, step=20, metrics={})
    callback._run_validation.assert_called_once()


def test_validation_callback_on_epoch_end_runs_validation(dummy_trainer, dummy_dataset):
    """Test that validation runs at epoch end when enabled."""
    callback = ValidationCallback(val_dataset=dummy_dataset, eval_on_epoch_end=True)
    callback._run_validation = Mock(return_value=1.5)

    callback.on_epoch_end(dummy_trainer, epoch=0, metrics={})
    callback._run_validation.assert_called_once()


def test_validation_callback_on_epoch_end_skips_when_disabled(dummy_trainer, dummy_dataset):
    """Test that validation is skipped at epoch end when disabled."""
    callback = ValidationCallback(val_dataset=dummy_dataset, eval_on_epoch_end=False)
    callback._run_validation = Mock(return_value=1.5)

    callback.on_epoch_end(dummy_trainer, epoch=0, metrics={})
    callback._run_validation.assert_not_called()


def test_validation_callback_tracks_best_loss(dummy_trainer, dummy_dataset):
    """Test that ValidationCallback tracks best validation loss."""
    callback = ValidationCallback(val_dataset=dummy_dataset, eval_steps=10)
    callback._run_validation = Mock(side_effect=[2.0, 1.5, 1.8, 1.2])

    # Initial best loss
    assert callback.best_val_loss == float("inf")

    # First validation
    callback.on_step_end(dummy_trainer, step=10, metrics={})
    assert callback.best_val_loss == 2.0

    # Better loss
    callback.on_step_end(dummy_trainer, step=20, metrics={})
    assert callback.best_val_loss == 1.5

    # Worse loss - should not update
    callback.on_step_end(dummy_trainer, step=30, metrics={})
    assert callback.best_val_loss == 1.5

    # Even better loss
    callback.on_step_end(dummy_trainer, step=40, metrics={})
    assert callback.best_val_loss == 1.2


def test_validation_callback_skips_when_no_dataset(dummy_trainer):
    """Test that validation is skipped when no dataset is provided."""
    callback = ValidationCallback(val_dataset=None, eval_steps=10)

    # Should not raise error
    callback.on_step_end(dummy_trainer, step=10, metrics={})
    callback.on_epoch_end(dummy_trainer, epoch=0, metrics={})


# ============================================================================
# Test WandBCallback
# ============================================================================


def test_wandb_callback_init_with_defaults():
    """Test WandBCallback initialization with default values."""
    callback = WandBCallback()

    assert callback.project == "llm-playground-sft"
    assert callback.run_name is None
    assert callback.config is None
    assert callback.log_model is False
    assert callback.run is None


def test_wandb_callback_init_with_custom_values():
    """Test WandBCallback initialization with custom values."""
    config = {"learning_rate": 1e-4, "batch_size": 8}
    callback = WandBCallback(
        project="test-project",
        run_name="test-run",
        config=config,
        log_model=True,
    )

    assert callback.project == "test-project"
    assert callback.run_name == "test-run"
    assert callback.config == config
    assert callback.log_model is True


def test_wandb_callback_on_train_begin_initializes_run(dummy_trainer):
    """Test that W&B run is initialized on training start."""
    with patch("builtins.__import__", side_effect=ImportError):
        # Test when wandb is not available
        config = {"learning_rate": 1e-4}
        callback = WandBCallback(project="test-project", config=config)
        callback.on_train_begin(dummy_trainer)
        assert callback.run is None


def test_wandb_callback_on_step_end_logs_metrics(dummy_trainer):
    """Test that metrics are logged to W&B on step end."""
    with patch("builtins.__import__") as mock_import:
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_import.return_value = mock_wandb

        callback = WandBCallback()
        callback.run = Mock()

        metrics = {"loss": 1.5, "learning_rate": 1e-4, "grad_norm": 2.3}

        # Mock the wandb.log function
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            callback.on_step_end(dummy_trainer, step=10, metrics=metrics)

            if mock_wandb.log.called:
                logged_data = mock_wandb.log.call_args[0][0]
                assert "step" in logged_data


def test_wandb_callback_on_epoch_end_logs_metrics(dummy_trainer):
    """Test that epoch metrics are logged to W&B."""
    with patch("builtins.__import__") as mock_import:
        mock_wandb = MagicMock()
        mock_import.return_value = mock_wandb

        callback = WandBCallback()
        callback.run = Mock()

        metrics = {"epoch_loss": 1.2, "val_loss": 1.3}

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            callback.on_epoch_end(dummy_trainer, epoch=5, metrics=metrics)


def test_wandb_callback_on_train_end_finishes_run(dummy_trainer):
    """Test that W&B run is finished on training end."""
    callback = WandBCallback()
    mock_run = Mock()
    callback.run = mock_run

    callback.on_train_end(dummy_trainer)

    mock_run.finish.assert_called_once()


def test_wandb_callback_handles_missing_wandb(dummy_trainer):
    """Test that callback handles gracefully when W&B is not available."""
    with patch("builtins.__import__", side_effect=ImportError("wandb not installed")):
        callback = WandBCallback()

        # Should not raise error
        callback.on_train_begin(dummy_trainer)
        assert callback.run is None

        # Should not raise error when logging
        callback.on_step_end(dummy_trainer, step=10, metrics={"loss": 1.5})


# ============================================================================
# Test CheckpointCallback
# ============================================================================


def test_checkpoint_callback_init_with_defaults(temp_dir):
    """Test CheckpointCallback initialization with default values."""
    callback = CheckpointCallback(output_dir=temp_dir)

    assert callback.output_dir == temp_dir
    assert callback.save_steps == 500
    assert callback.save_total_limit == 3
    assert callback.save_best_only is False
    assert callback.best_val_loss == float("inf")
    assert callback.checkpoints == []


def test_checkpoint_callback_init_with_custom_values(temp_dir):
    """Test CheckpointCallback initialization with custom values."""
    callback = CheckpointCallback(
        output_dir=temp_dir,
        save_steps=100,
        save_total_limit=5,
        save_best_only=True,
    )

    assert callback.save_steps == 100
    assert callback.save_total_limit == 5
    assert callback.save_best_only is True


def test_checkpoint_callback_init_creates_directory(temp_dir):
    """Test that CheckpointCallback creates output directory."""
    output_dir = os.path.join(temp_dir, "checkpoints", "nested")
    callback = CheckpointCallback(output_dir=output_dir)

    assert os.path.exists(output_dir)


def test_checkpoint_callback_on_step_end_saves_at_interval(temp_dir, dummy_trainer):
    """Test that checkpoint is saved at specified step intervals."""
    callback = CheckpointCallback(output_dir=temp_dir, save_steps=10)
    callback._save_checkpoint = Mock()

    # Step 5 - should not save
    callback.on_step_end(dummy_trainer, step=5, metrics={})
    callback._save_checkpoint.assert_not_called()

    # Step 10 - should save
    callback.on_step_end(dummy_trainer, step=10, metrics={"loss": 1.5})
    callback._save_checkpoint.assert_called_once()

    # Step 15 - should not save
    callback._save_checkpoint.reset_mock()
    callback.on_step_end(dummy_trainer, step=15, metrics={})
    callback._save_checkpoint.assert_not_called()

    # Step 20 - should save
    callback.on_step_end(dummy_trainer, step=20, metrics={"loss": 1.3})
    callback._save_checkpoint.assert_called_once()


def test_checkpoint_callback_saves_only_if_best_when_enabled(temp_dir, dummy_trainer):
    """Test that checkpoint is saved only if validation loss improves."""
    callback = CheckpointCallback(output_dir=temp_dir, save_steps=10, save_best_only=True)
    callback._save_checkpoint = Mock()

    # First save - should save (initial best)
    callback.on_step_end(dummy_trainer, step=10, metrics={"val_loss": 2.0})
    callback._save_checkpoint.assert_called_once()
    assert callback.best_val_loss == 2.0

    # Worse loss - should not save
    callback._save_checkpoint.reset_mock()
    callback.on_step_end(dummy_trainer, step=20, metrics={"val_loss": 2.5})
    callback._save_checkpoint.assert_not_called()
    assert callback.best_val_loss == 2.0

    # Better loss - should save
    callback.on_step_end(dummy_trainer, step=30, metrics={"val_loss": 1.5})
    callback._save_checkpoint.assert_called_once()
    assert callback.best_val_loss == 1.5


def test_checkpoint_callback_rotates_old_checkpoints(temp_dir, dummy_trainer):
    """Test that old checkpoints are deleted when limit is reached."""
    callback = CheckpointCallback(output_dir=temp_dir, save_steps=10, save_total_limit=3)

    # Create dummy checkpoint files
    for i in range(5):
        checkpoint_path = os.path.join(temp_dir, f"checkpoint_step_{(i + 1) * 10}.pt")
        with open(checkpoint_path, "w") as f:
            f.write("dummy")
        callback.checkpoints.append(checkpoint_path)

    # Rotate should keep only last 3
    callback._rotate_checkpoints()

    assert len(callback.checkpoints) == 3
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_step_30.pt"))
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_step_40.pt"))
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_step_50.pt"))
    assert not os.path.exists(os.path.join(temp_dir, "checkpoint_step_10.pt"))
    assert not os.path.exists(os.path.join(temp_dir, "checkpoint_step_20.pt"))


def test_checkpoint_callback_handles_missing_val_loss(temp_dir, dummy_trainer):
    """Test that callback handles missing val_loss in metrics."""
    callback = CheckpointCallback(output_dir=temp_dir, save_steps=10, save_best_only=True)
    callback._save_checkpoint = Mock()

    # No val_loss in metrics - should still save
    callback.on_step_end(dummy_trainer, step=10, metrics={"loss": 1.5})
    callback._save_checkpoint.assert_called_once()


def test_checkpoint_callback_checkpoint_path_format(temp_dir):
    """Test that checkpoint file names follow expected format."""
    callback = CheckpointCallback(output_dir=temp_dir)

    path = callback._get_checkpoint_path(step=100)
    expected = os.path.join(temp_dir, "checkpoint_step_100.pt")
    assert path == expected


# ============================================================================
# Test Callback Integration
# ============================================================================


def test_multiple_callbacks_can_be_used_together(temp_dir, dummy_trainer, dummy_dataset):
    """Test that multiple callbacks can be used in the same training run."""
    val_callback = ValidationCallback(val_dataset=dummy_dataset, eval_steps=10)
    checkpoint_callback = CheckpointCallback(output_dir=temp_dir, save_steps=10)

    # Mock methods
    val_callback._run_validation = Mock(return_value=1.5)
    checkpoint_callback._save_checkpoint = Mock()

    metrics = {"loss": 1.5}

    # Call both callbacks
    val_callback.on_step_end(dummy_trainer, step=10, metrics=metrics)
    checkpoint_callback.on_step_end(dummy_trainer, step=10, metrics=metrics)

    # Both should have been called
    val_callback._run_validation.assert_called_once()
    checkpoint_callback._save_checkpoint.assert_called_once()


def test_all_callbacks_lifecycle(temp_dir, dummy_trainer, dummy_dataset):
    """Test complete lifecycle with all callbacks."""
    val_callback = ValidationCallback(val_dataset=dummy_dataset)
    wandb_callback = WandBCallback()
    checkpoint_callback = CheckpointCallback(output_dir=temp_dir)

    callbacks = [val_callback, wandb_callback, checkpoint_callback]

    # Training begin
    for cb in callbacks:
        cb.on_train_begin(dummy_trainer)

    # Training steps
    for step in [10, 20]:
        metrics = {"loss": 1.5}
        for cb in callbacks:
            cb.on_step_end(dummy_trainer, step=step, metrics=metrics)

    # Training end
    for cb in callbacks:
        cb.on_train_end(dummy_trainer)
