"""Tests for training resume from checkpoint.

Verifies that the Trainer can:
- Resume from the latest checkpoint in checkpoint_dir
- Resume from a specific checkpoint path
- Correctly restore model, optimizer, scheduler state and step counter
- Handle missing checkpoint gracefully
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from app.training import Trainer, TrainingConfig
from app.transformer import TransformerConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def small_config(temp_dir):
    """Create a small training config for fast tests."""
    return TrainingConfig(
        model_config=TransformerConfig(
            vocab_size=50257,
            hidden_size=64,
            num_layers=1,
            num_heads=1,
            max_position_embeddings=128,
        ),
        batch_size=4,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=2,
        checkpoint_dir=temp_dir,
        save_steps=5,
        logging_steps=5,
    )


@pytest.fixture
def train_dataset():
    """Create a small training dataset."""
    return TensorDataset(torch.randint(0, 50257, (40, 128)))


class TestTrainingResume:
    """Tests for training resume from checkpoint."""

    def test_resume_latest_restores_step(self, small_config, train_dataset, temp_dir):
        """Resuming from 'latest' should restore the training step counter."""
        # Phase 1: Train for 10 steps, saving at step 5 and 10
        trainer1 = Trainer(config=small_config, train_dataset=train_dataset)
        trainer1.train()
        assert trainer1.current_step == 10

        # Verify checkpoints exist
        checkpoints = list(Path(temp_dir).glob("checkpoint_step_*.pt"))
        assert len(checkpoints) >= 1

        # Phase 2: Resume from latest checkpoint
        resume_config = TrainingConfig(
            model_config=small_config.model_config,
            batch_size=small_config.batch_size,
            learning_rate=small_config.learning_rate,
            max_steps=20,
            warmup_steps=small_config.warmup_steps,
            checkpoint_dir=temp_dir,
            save_steps=5,
            logging_steps=5,
            resume_from_checkpoint="latest",
        )
        trainer2 = Trainer(config=resume_config, train_dataset=train_dataset)

        # Step counter should be restored from checkpoint
        assert trainer2.current_step == 10

    def test_resume_specific_path(self, small_config, train_dataset, temp_dir):
        """Resuming from a specific checkpoint path should work correctly."""
        # Phase 1: Train for 10 steps
        trainer1 = Trainer(config=small_config, train_dataset=train_dataset)
        trainer1.train()

        # Find the step-5 checkpoint
        step5_path = Path(temp_dir) / "checkpoint_step_5.pt"
        assert step5_path.exists(), "checkpoint_step_5.pt should exist"

        # Phase 2: Resume from specific checkpoint
        resume_config = TrainingConfig(
            model_config=small_config.model_config,
            batch_size=small_config.batch_size,
            learning_rate=small_config.learning_rate,
            max_steps=20,
            warmup_steps=small_config.warmup_steps,
            checkpoint_dir=temp_dir,
            save_steps=5,
            logging_steps=5,
            resume_from_checkpoint=str(step5_path),
        )
        trainer2 = Trainer(config=resume_config, train_dataset=train_dataset)

        # Should resume from step 5
        assert trainer2.current_step == 5

    def test_resume_continues_training(self, small_config, train_dataset, temp_dir):
        """Resumed training should continue from the restored step to max_steps."""
        # Phase 1: Train for 10 steps
        trainer1 = Trainer(config=small_config, train_dataset=train_dataset)
        trainer1.train()
        assert trainer1.current_step == 10

        # Phase 2: Resume and train to 20 steps
        resume_config = TrainingConfig(
            model_config=small_config.model_config,
            batch_size=small_config.batch_size,
            learning_rate=small_config.learning_rate,
            max_steps=20,
            warmup_steps=small_config.warmup_steps,
            checkpoint_dir=temp_dir,
            save_steps=5,
            logging_steps=5,
            resume_from_checkpoint="latest",
        )
        trainer2 = Trainer(config=resume_config, train_dataset=train_dataset)
        trainer2.train()

        # Should have completed to step 20
        assert trainer2.current_step == 20

    def test_resume_no_checkpoint_warns(self, train_dataset):
        """Resuming with 'latest' when no checkpoints exist should warn, not crash."""
        with tempfile.TemporaryDirectory() as empty_dir:
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=64,
                    num_layers=1,
                    num_heads=1,
                    max_position_embeddings=128,
                ),
                batch_size=4,
                learning_rate=1e-3,
                max_steps=5,
                warmup_steps=2,
                checkpoint_dir=empty_dir,
                save_steps=5,
                logging_steps=5,
                resume_from_checkpoint="latest",
            )
            # Should not raise, should warn and start from step 0
            trainer = Trainer(config=config, train_dataset=train_dataset)
            assert trainer.current_step == 0

    def test_resume_missing_path_raises(self, train_dataset, temp_dir):
        """Resuming from a nonexistent specific path should raise FileNotFoundError."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=64,
                num_layers=1,
                num_heads=1,
                max_position_embeddings=128,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=5,
            warmup_steps=2,
            checkpoint_dir=temp_dir,
            save_steps=5,
            logging_steps=5,
            resume_from_checkpoint="/nonexistent/path/checkpoint.pt",
        )
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            Trainer(config=config, train_dataset=train_dataset)

    def test_resume_restores_optimizer_state(self, small_config, train_dataset, temp_dir):
        """Resumed training should have the same optimizer state as the checkpoint."""
        # Phase 1: Train for 10 steps
        trainer1 = Trainer(config=small_config, train_dataset=train_dataset)
        trainer1.train()

        # Capture optimizer state at step 10
        orig_opt_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in trainer1.optimizer.state_dict()["state"][0].items()
        }

        # Phase 2: Resume from latest
        resume_config = TrainingConfig(
            model_config=small_config.model_config,
            batch_size=small_config.batch_size,
            learning_rate=small_config.learning_rate,
            max_steps=20,
            warmup_steps=small_config.warmup_steps,
            checkpoint_dir=temp_dir,
            save_steps=5,
            logging_steps=5,
            resume_from_checkpoint="latest",
        )
        trainer2 = Trainer(config=resume_config, train_dataset=train_dataset)

        # Optimizer state should match
        resumed_opt_state = trainer2.optimizer.state_dict()["state"][0]
        for key in orig_opt_state:
            if isinstance(orig_opt_state[key], torch.Tensor):
                assert torch.allclose(
                    orig_opt_state[key],
                    resumed_opt_state[key],
                ), f"Optimizer state mismatch for key {key}"
