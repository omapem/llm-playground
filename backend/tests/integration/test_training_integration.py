"""End-to-end integration tests for complete training system."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from app.training import Trainer, TrainingConfig
from app.transformer import TransformerConfig


class SimpleDataset(Dataset):
    """Simple dataset for integration testing."""

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
def training_dataset():
    """Create training dataset."""
    return SimpleDataset(size=100, seq_len=128)


@pytest.fixture
def validation_dataset():
    """Create validation dataset."""
    return SimpleDataset(size=20, seq_len=128)


class TestCompleteTrainingWorkflow:
    """Test complete training workflows end-to-end."""

    def test_basic_training_workflow(self, temp_checkpoint_dir, training_dataset):
        """Test basic training from start to finish."""
        # Create configuration
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
                max_position_embeddings=128,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=20,
            warmup_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=10,
            logging_steps=5,
        )

        # Create trainer
        trainer = Trainer(config=config, train_dataset=training_dataset)

        # Verify initial state
        assert trainer.current_step == 0
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 20

        # Verify checkpoints were saved
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 2  # At steps 10 and 20

        # Verify metrics were collected
        metrics = trainer.metrics_tracker.get_averages()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "grad_norm" in metrics
        assert "lr" in metrics

    def test_training_with_validation(self, temp_checkpoint_dir, training_dataset, validation_dataset):
        """Test training with validation dataset."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            learning_rate=1e-3,
            max_steps=15,
            warmup_steps=3,
            checkpoint_dir=temp_checkpoint_dir,
            eval_steps=5,
        )

        trainer = Trainer(
            config=config,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
        )

        # Run training with validation
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 15

        # Validation should have run at steps 5, 10, 15
        # (We can't directly verify this without logging capture,
        # but we can verify the trainer completed without errors)

    def test_checkpoint_and_resume_workflow(self, temp_checkpoint_dir, training_dataset):
        """Test checkpointing and resuming training."""
        # Initial training
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
            warmup_steps=2,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=5,
        )

        trainer1 = Trainer(config=config, train_dataset=training_dataset)
        trainer1.train()

        # Verify checkpoint exists
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 2

        # Resume training
        config.max_steps = 20
        config.resume_from_checkpoint = True

        trainer2 = Trainer(config=config, train_dataset=training_dataset)

        # Should resume from step 10
        assert trainer2.current_step == 10

        # Continue training
        trainer2.train()

        # Should reach step 20
        assert trainer2.current_step == 20

        # Should have more checkpoints
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 4  # Steps 5, 10, 15, 20

    def test_gradient_accumulation_workflow(self, temp_checkpoint_dir, training_dataset):
        """Test training with gradient accumulation."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-3,
            max_steps=10,
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 10

        # Effective batch size should be 8 (2 * 4)
        assert config.get_effective_batch_size() == 8

    def test_mixed_precision_workflow(self, temp_checkpoint_dir, training_dataset):
        """Test training with mixed precision."""
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
            mixed_precision="bf16",
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)

        # Verify AMP is enabled
        assert trainer.use_amp is True
        assert trainer.scaler is not None

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 10


class TestComponentIntegration:
    """Test integration between different components."""

    def test_model_config_integration(self, temp_checkpoint_dir, training_dataset):
        """Test model configuration integration with trainer."""
        # Test different model configurations
        configs = [
            TransformerConfig(vocab_size=50257, hidden_size=64, num_layers=1, num_heads=1),
            TransformerConfig(vocab_size=50257, hidden_size=128, num_layers=2, num_heads=2),
        ]

        for model_config in configs:
            training_config = TrainingConfig(
                model_config=model_config,
                batch_size=4,
                max_steps=5,
                checkpoint_dir=temp_checkpoint_dir,
            )

            trainer = Trainer(config=training_config, train_dataset=training_dataset)
            trainer.train()

            # Verify training completed
            assert trainer.current_step == 5

    def test_scheduler_integration(self, temp_checkpoint_dir, training_dataset):
        """Test different schedulers integrate correctly."""
        scheduler_types = ["cosine", "linear", "constant"]

        for scheduler_type in scheduler_types:
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=128,
                    num_layers=2,
                    num_heads=2,
                ),
                batch_size=4,
                max_steps=10,
                warmup_steps=3,
                scheduler_type=scheduler_type,
                checkpoint_dir=temp_checkpoint_dir,
            )

            trainer = Trainer(config=config, train_dataset=training_dataset)
            trainer.train()

            # Verify training completed
            assert trainer.current_step == 10

    def test_checkpoint_manager_integration(self, temp_checkpoint_dir, training_dataset):
        """Test checkpoint manager integrates correctly."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=25,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=5,
            max_checkpoints_to_keep=3,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        # Verify checkpoint rotation worked
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) == 3  # Only kept last 3

        # Verify they are the most recent
        checkpoint_steps = sorted([
            int(cp.stem.split("_")[-1])
            for cp in checkpoints
        ])
        assert checkpoint_steps == [15, 20, 25]

    def test_metrics_tracker_integration(self, temp_checkpoint_dir, training_dataset):
        """Test metrics tracker integrates correctly."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=20,
            logging_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        # Verify metrics are available
        metrics = trainer.metrics_tracker.get_averages()
        assert len(metrics) > 0

        # Verify throughput can be computed
        throughput = trainer.get_throughput()
        assert "steps_per_second" in throughput
        assert "tokens_per_second" in throughput
        assert throughput["steps_per_second"] > 0


class TestConfigurationIntegration:
    """Test configuration management integration."""

    def test_yaml_config_workflow(self, temp_checkpoint_dir, training_dataset):
        """Test loading config from YAML and training."""
        import yaml

        # Create YAML config
        config_dict = {
            "model_config": {
                "vocab_size": 50257,
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 2,
                "max_position_embeddings": 128,
            },
            "batch_size": 4,
            "learning_rate": 1e-3,
            "max_steps": 10,
            "warmup_steps": 2,
            "checkpoint_dir": temp_checkpoint_dir,
        }

        # Save to YAML
        yaml_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load and train
        config = TrainingConfig.from_yaml(yaml_path)
        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 10

    def test_config_serialization_integration(self, temp_checkpoint_dir, training_dataset):
        """Test config can be serialized and deserialized."""
        # Create config
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=10,
            warmup_steps=2,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Serialize to dict
        config_dict = config.to_dict()

        # Deserialize
        config2 = TrainingConfig.from_dict(config_dict)

        # Train with deserialized config
        trainer = Trainer(config=config2, train_dataset=training_dataset)
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 10


class TestErrorHandlingIntegration:
    """Test error handling across components."""

    def test_invalid_config_validation(self):
        """Test invalid config is caught."""
        with pytest.raises(ValueError):
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=128,
                    num_layers=2,
                    num_heads=2,
                ),
                batch_size=-1,  # Invalid
            )
            config.validate()

    def test_checkpoint_dir_creation(self, training_dataset):
        """Test checkpoint directory is created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints", "nested")

            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=128,
                    num_layers=2,
                    num_heads=2,
                ),
                batch_size=4,
                max_steps=5,
                checkpoint_dir=checkpoint_dir,
            )

            trainer = Trainer(config=config, train_dataset=training_dataset)
            trainer.train()

            # Verify directory was created
            assert os.path.isdir(checkpoint_dir)


class TestRealWorldScenarios:
    """Test realistic training scenarios."""

    def test_small_model_quick_training(self, temp_checkpoint_dir, training_dataset):
        """Test quick training of small model."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=64,
                num_layers=1,
                num_heads=1,
                max_position_embeddings=128,
            ),
            batch_size=8,
            learning_rate=1e-3,
            max_steps=50,
            warmup_steps=10,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=25,
            logging_steps=10,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        assert trainer.current_step == 50

    def test_medium_model_with_features(self, temp_checkpoint_dir, training_dataset, validation_dataset):
        """Test medium model with various features."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=256,
                num_layers=4,
                num_heads=4,
                max_position_embeddings=128,
            ),
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=6e-4,
            max_steps=30,
            warmup_steps=5,
            scheduler_type="cosine",
            max_grad_norm=1.0,
            mixed_precision="bf16",
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=10,
            logging_steps=5,
            eval_steps=10,
            max_checkpoints_to_keep=2,
        )

        trainer = Trainer(
            config=config,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
        )
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 30

        # Verify features worked
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2  # max_checkpoints_to_keep=2

    def test_long_running_training_simulation(self, temp_checkpoint_dir, training_dataset):
        """Test longer training run with multiple checkpoints."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=100,
            warmup_steps=20,
            checkpoint_dir=temp_checkpoint_dir,
            save_steps=20,
            logging_steps=10,
        )

        trainer = Trainer(config=config, train_dataset=training_dataset)
        trainer.train()

        # Verify training completed
        assert trainer.current_step == 100

        # Verify multiple checkpoints
        checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoints) == 5  # Steps 20, 40, 60, 80, 100

        # Verify metrics collected over long run
        metrics = trainer.metrics_tracker.get_averages()
        assert metrics["loss"] > 0
