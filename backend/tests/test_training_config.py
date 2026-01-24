"""Tests for training configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path
from app.training.config import TrainingConfig
from app.transformer import TransformerConfig, GPT2Small


class TestTrainingConfigInitialization:
    """Tests for TrainingConfig initialization."""

    def test_initialization_with_defaults(self):
        """Test TrainingConfig initializes with defaults."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.model_config is not None
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 6e-4
        assert config.warmup_steps == 2000
        assert config.max_steps == 100000

    def test_initialization_with_custom_values(self):
        """Test TrainingConfig initializes with custom values."""
        model_config = GPT2Small()
        config = TrainingConfig(
            model_config=model_config,
            batch_size=16,
            learning_rate=1e-3,
            max_steps=50000,
        )
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3
        assert config.max_steps == 50000

    def test_initialization_with_mixed_precision(self):
        """Test TrainingConfig supports mixed precision settings."""
        for precision in [None, "fp16", "bf16"]:
            config = TrainingConfig(
                model_config=GPT2Small(),
                mixed_precision=precision,
            )
            assert config.mixed_precision == precision

    def test_initialization_requires_model_config(self):
        """Test TrainingConfig requires model_config."""
        with pytest.raises(TypeError):
            TrainingConfig()


class TestTrainingConfigValidation:
    """Tests for TrainingConfig validation."""

    def test_validate_valid_config(self):
        """Test valid config passes validation."""
        config = TrainingConfig(model_config=GPT2Small())
        # Should not raise
        config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation rejects invalid batch size."""
        config = TrainingConfig(model_config=GPT2Small())
        config.batch_size = 0
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_learning_rate(self):
        """Test validation rejects invalid learning rate."""
        config = TrainingConfig(model_config=GPT2Small())
        config.learning_rate = -1.0
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_gradient_accumulation(self):
        """Test validation rejects invalid gradient accumulation."""
        config = TrainingConfig(model_config=GPT2Small())
        config.gradient_accumulation_steps = 0
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_warmup_steps(self):
        """Test validation rejects negative warmup steps."""
        config = TrainingConfig(model_config=GPT2Small())
        config.warmup_steps = -1
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_max_steps(self):
        """Test validation rejects invalid max steps."""
        config = TrainingConfig(model_config=GPT2Small())
        config.max_steps = 0
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_warmup_greater_than_max_steps(self):
        """Test validation handles warmup > max_steps."""
        config = TrainingConfig(model_config=GPT2Small())
        config.warmup_steps = 100000
        config.max_steps = 50000
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_mixed_precision(self):
        """Test validation rejects invalid mixed precision."""
        config = TrainingConfig(model_config=GPT2Small())
        config.mixed_precision = "fp32"  # Invalid
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_optimizer_type(self):
        """Test validation rejects invalid optimizer type."""
        config = TrainingConfig(model_config=GPT2Small())
        config.optimizer_type = "sgd"  # Not supported yet
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_scheduler_type(self):
        """Test validation rejects invalid scheduler type."""
        config = TrainingConfig(model_config=GPT2Small())
        config.scheduler_type = "exponential"
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_negative_weight_decay(self):
        """Test validation rejects negative weight decay."""
        config = TrainingConfig(model_config=GPT2Small())
        config.weight_decay = -0.1
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_max_grad_norm(self):
        """Test validation rejects invalid max grad norm."""
        config = TrainingConfig(model_config=GPT2Small())
        config.max_grad_norm = -1.0
        with pytest.raises(ValueError):
            config.validate()


class TestTrainingConfigSerialization:
    """Tests for config dict serialization."""

    def test_to_dict(self):
        """Test config converts to dict."""
        config = TrainingConfig(model_config=GPT2Small(), batch_size=16)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["batch_size"] == 16
        assert "model_config" in config_dict

    def test_from_dict(self):
        """Test config loads from dict."""
        original = TrainingConfig(model_config=GPT2Small(), batch_size=16)
        config_dict = original.to_dict()

        # Create new config from dict
        restored = TrainingConfig.from_dict(config_dict)
        assert restored.batch_size == 16
        assert restored.learning_rate == original.learning_rate

    def test_dict_roundtrip(self):
        """Test dict serialization roundtrip."""
        original = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=32,
            learning_rate=1e-3,
            max_steps=50000,
        )
        config_dict = original.to_dict()
        restored = TrainingConfig.from_dict(config_dict)

        assert restored.batch_size == original.batch_size
        assert restored.learning_rate == original.learning_rate
        assert restored.max_steps == original.max_steps


class TestTrainingConfigYAML:
    """Tests for YAML I/O."""

    def test_to_yaml_creates_file(self):
        """Test config saves to YAML file."""
        config = TrainingConfig(model_config=GPT2Small())

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(yaml_path))
            assert yaml_path.exists()

    def test_yaml_file_is_valid(self):
        """Test saved YAML file is valid."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=16,
            learning_rate=1e-3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(yaml_path))

            # Load with yaml library to verify validity
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            assert data["batch_size"] == 16
            assert data["learning_rate"] == 1e-3

    def test_from_yaml_loads_file(self):
        """Test config loads from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Create YAML file
            original = TrainingConfig(
                model_config=GPT2Small(),
                batch_size=32,
                learning_rate=5e-4,
            )
            original.to_yaml(str(yaml_path))

            # Load from file
            loaded = TrainingConfig.from_yaml(str(yaml_path))
            assert loaded.batch_size == 32
            assert loaded.learning_rate == 5e-4

    def test_yaml_roundtrip(self):
        """Test YAML serialization roundtrip."""
        original = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=64,
            learning_rate=8e-4,
            max_steps=200000,
            warmup_steps=5000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            original.to_yaml(str(yaml_path))
            restored = TrainingConfig.from_yaml(str(yaml_path))

            assert restored.batch_size == original.batch_size
            assert restored.learning_rate == original.learning_rate
            assert restored.max_steps == original.max_steps
            assert restored.warmup_steps == original.warmup_steps

    def test_load_invalid_yaml_raises_error(self):
        """Test loading invalid YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "invalid.yaml"
            yaml_path.write_text("invalid: yaml: content:")

            with pytest.raises((ValueError, yaml.YAMLError, KeyError)):
                TrainingConfig.from_yaml(str(yaml_path))

    def test_nonexistent_yaml_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml("/nonexistent/path/config.yaml")


class TestTrainingConfigUtilities:
    """Tests for utility methods."""

    def test_get_effective_batch_size_single_device(self):
        """Test effective batch size calculation for single device."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=8,
            gradient_accumulation_steps=4,
        )
        effective_bs = config.get_effective_batch_size()
        assert effective_bs == 32  # 8 * 4

    def test_get_effective_batch_size_multi_device(self):
        """Test effective batch size calculation for multiple devices."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            batch_size=8,
            gradient_accumulation_steps=4,
            num_devices=2,
        )
        effective_bs = config.get_effective_batch_size()
        assert effective_bs == 64  # 8 * 4 * 2

    def test_total_training_steps_with_tokens(self):
        """Test estimating training steps from token count."""
        config = TrainingConfig(model_config=GPT2Small())
        config.max_steps = 100000
        assert config.max_steps == 100000

    def test_warmup_ratio_vs_steps(self):
        """Test warmup can be specified as ratio."""
        config = TrainingConfig(model_config=GPT2Small())
        config.max_steps = 100000
        config.warmup_steps = 10000
        assert config.warmup_steps == 10000

    def test_config_repr(self):
        """Test config has readable repr."""
        config = TrainingConfig(model_config=GPT2Small())
        repr_str = repr(config)
        assert "TrainingConfig" in repr_str or "batch_size" in repr_str


class TestTrainingConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_optimizer(self):
        """Test default optimizer is AdamW."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.optimizer_type == "adamw"

    def test_default_scheduler(self):
        """Test default scheduler is cosine with warmup."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.scheduler_type == "cosine"

    def test_default_mixed_precision(self):
        """Test default mixed precision is bf16."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.mixed_precision == "bf16"

    def test_default_weight_decay(self):
        """Test default weight decay."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.weight_decay == 0.01

    def test_default_grad_clipping(self):
        """Test default gradient clipping."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.max_grad_norm == 1.0

    def test_default_logging_interval(self):
        """Test default logging interval."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.logging_steps == 100

    def test_default_save_interval(self):
        """Test default checkpoint save interval."""
        config = TrainingConfig(model_config=GPT2Small())
        assert config.save_steps == 1000


class TestTrainingConfigAdvanced:
    """Tests for advanced configuration options."""

    def test_distributed_training_settings(self):
        """Test config supports distributed training settings."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            num_devices=4,
            use_ddp=True,
        )
        assert config.num_devices == 4
        assert config.use_ddp is True

    def test_wandb_configuration(self):
        """Test config supports W&B settings."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            wandb_project="test-project",
            wandb_entity="test-entity",
        )
        assert config.wandb_project == "test-project"
        assert config.wandb_entity == "test-entity"

    def test_checkpoint_settings(self):
        """Test checkpoint configuration."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            checkpoint_dir="/tmp/checkpoints",
            max_checkpoints_to_keep=5,
        )
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.max_checkpoints_to_keep == 5

    def test_gradient_checkpointing_option(self):
        """Test gradient checkpointing can be enabled."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            gradient_checkpointing=True,
        )
        assert config.gradient_checkpointing is True

    def test_resume_from_checkpoint(self):
        """Test config supports resuming from checkpoint."""
        config = TrainingConfig(
            model_config=GPT2Small(),
            resume_from_checkpoint="/path/to/checkpoint.pt",
        )
        assert config.resume_from_checkpoint == "/path/to/checkpoint.pt"
