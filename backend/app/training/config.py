"""Training configuration system for LLM Playground.

Provides YAML-based configuration management with validation for training experiments.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from app.transformer import TransformerConfig


@dataclass
class TrainingConfig:
    """Configuration for training experiments.

    Supports:
    - Model architecture configuration
    - Training hyperparameters
    - Optimizer and scheduler settings
    - Mixed precision and distributed training
    - Checkpoint and logging settings
    - W&B experiment tracking

    Args:
        model_config: TransformerConfig for the model architecture
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping max norm
        optimizer_type: Optimizer type ("adamw" supported)
        scheduler_type: LR scheduler type ("cosine", "linear", "constant")
        mixed_precision: Mixed precision mode (None, "fp16", "bf16")
        gradient_checkpointing: Enable gradient checkpointing
        logging_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Run evaluation every N steps
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints_to_keep: Number of checkpoints to keep
        num_devices: Number of devices for distributed training
        use_ddp: Use DistributedDataParallel
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        resume_from_checkpoint: Path to checkpoint to resume from
    """

    # Model and data
    model_config: TransformerConfig
    max_seq_length: int = 1024

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-4
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer and scheduler
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Mixed precision and optimization
    mixed_precision: Optional[str] = "bf16"
    gradient_checkpointing: bool = True

    # Logging and checkpointing
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: Optional[int] = None
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints_to_keep: int = 5

    # Distributed training
    num_devices: int = 1
    use_ddp: bool = False

    # Weights & Biases integration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Batch size validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        # Learning rate validation
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        # Gradient accumulation validation
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0, "
                f"got {self.gradient_accumulation_steps}"
            )

        # Warmup validation
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")

        # Max steps validation
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")

        # Warmup must be less than max steps
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be <= "
                f"max_steps ({self.max_steps})"
            )

        # Weight decay validation
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

        # Max grad norm validation
        if self.max_grad_norm < 0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # Mixed precision validation
        if self.mixed_precision not in [None, "fp16", "bf16"]:
            raise ValueError(
                f"mixed_precision must be None, 'fp16', or 'bf16', "
                f"got {self.mixed_precision}"
            )

        # Optimizer type validation
        if self.optimizer_type not in ["adamw"]:
            raise ValueError(
                f"optimizer_type must be 'adamw', got {self.optimizer_type}"
            )

        # Scheduler type validation
        if self.scheduler_type not in ["cosine", "linear", "constant"]:
            raise ValueError(
                f"scheduler_type must be 'cosine', 'linear', or 'constant', "
                f"got {self.scheduler_type}"
            )

        # Validate model config
        try:
            self.model_config.validate()
        except ValueError as e:
            raise ValueError(f"Invalid model_config: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        config_dict = asdict(self)
        # Convert model_config dataclass to dict
        config_dict["model_config"] = asdict(self.model_config)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Load config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            TrainingConfig instance

        Raises:
            ValueError: If config is invalid
        """
        config_dict = config_dict.copy()

        # Handle model_config separately
        if isinstance(config_dict.get("model_config"), dict):
            model_config_dict = config_dict.pop("model_config")
            model_config = TransformerConfig(**model_config_dict)
        else:
            model_config = config_dict.pop("model_config")

        config = cls(model_config=model_config, **config_dict)
        config.validate()
        return config

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file.

        Args:
            path: File path to save to

        Raises:
            IOError: If file cannot be written
        """
        config_dict = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file.

        Args:
            path: File path to load from

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If config is invalid
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError("YAML file must contain a dictionary")

        return cls.from_dict(config_dict)

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size for training.

        Accounts for gradient accumulation and number of devices.

        Returns:
            Effective batch size
        """
        return self.batch_size * self.gradient_accumulation_steps * self.num_devices

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TrainingConfig("
            f"batch_size={self.batch_size}, "
            f"learning_rate={self.learning_rate}, "
            f"max_steps={self.max_steps}, "
            f"mixed_precision={self.mixed_precision})"
        )
