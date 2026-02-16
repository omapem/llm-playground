"""SFT configuration system for supervised fine-tuning.

Provides YAML-based configuration management for SFT experiments with:
- LoRA/QLoRA parameter configuration
- Dataset format and processing settings
- Training hyperparameters
- Hardware and optimization settings
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning experiments.

    Supports:
    - LoRA/QLoRA parameter-efficient fine-tuning
    - Multiple dataset formats (Alpaca, Chat, Completion)
    - Template-based instruction formatting
    - Mixed precision and gradient optimization
    - W&B experiment tracking

    Required Args:
        base_model: Name or path of base model to fine-tune
        dataset_name: Name of HuggingFace dataset or local path
        dataset_format: Format of dataset ("alpaca", "chat", "completion")

    Optional Args (LoRA):
        use_lora: Enable LoRA fine-tuning
        use_qlora: Enable QLoRA (4-bit quantized LoRA)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA

    Optional Args (Dataset):
        dataset_split: Dataset split to use
        dataset_text_field: Field name for input text
        validation_split: Validation split ratio
        max_seq_length: Maximum sequence length

    Optional Args (Training):
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides num_epochs if set)
        warmup_ratio: Warmup ratio of total steps
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping max norm
        optimizer_type: Optimizer type
        scheduler_type: LR scheduler type
        logging_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Run evaluation every N steps

    Optional Args (Hardware):
        mixed_precision: Mixed precision mode (None, "fp16", "bf16")
        gradient_checkpointing: Enable gradient checkpointing
        num_devices: Number of devices for distributed training
        packing: Enable sequence packing for efficiency

    Optional Args (Tracking):
        output_dir: Directory to save checkpoints and outputs
        checkpoint_dir: Directory for checkpoints (defaults to output_dir)
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        wandb_run_name: Weights & Biases run name
    """

    # Required fields
    base_model: str
    dataset_name: str
    dataset_format: str

    # LoRA configuration
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Dataset configuration
    dataset_split: str = "train"
    dataset_text_field: Optional[str] = None
    validation_split: float = 0.1
    max_seq_length: int = 2048
    template_name: Optional[str] = None  # Explicit template name override

    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3

    # Optimizer and scheduler
    optimizer_type: str = "paged_adamw_32bit"
    scheduler_type: str = "cosine"

    # Hardware and optimization
    mixed_precision: Optional[str] = "bf16"
    gradient_checkpointing: bool = True
    num_devices: int = 1
    packing: bool = False
    device_map: str = "auto"  # Device mapping strategy for model loading

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: Optional[int] = None
    output_dir: str = "./sft_outputs"
    checkpoint_dir: Optional[str] = None
    save_total_limit: int = 3  # Maximum number of checkpoints to keep
    save_merged_model: bool = False  # Whether to save merged LoRA+base model

    # Weights & Biases integration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    run_name: Optional[str] = None  # General run name (used if wandb_run_name not set)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

        # Auto-set checkpoint_dir if not provided
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

        # Auto-configure paged optimizer for QLoRA
        if self.use_qlora and self.optimizer_type == "paged_adamw_32bit":
            pass  # Already set correctly

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Required fields validation
        if not self.base_model:
            raise ValueError("base_model is required")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        if not self.dataset_format:
            raise ValueError("dataset_format is required")

        # Dataset format validation
        valid_formats = ["alpaca", "chat", "completion"]
        if self.dataset_format not in valid_formats:
            raise ValueError(
                f"dataset_format must be one of {valid_formats}, "
                f"got {self.dataset_format}"
            )

        # Validation split validation
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError(
                f"validation_split must be in [0.0, 1.0), got {self.validation_split}"
            )

        # LoRA rank validation
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be > 0, got {self.lora_r}")

        # LoRA alpha validation
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")

        # LoRA dropout validation
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}"
            )

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

        # Epochs validation
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {self.num_epochs}")

        # Max steps validation
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")

        # Warmup validation
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0.0, 1.0), got {self.warmup_ratio}"
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
        valid_optimizers = ["adamw", "paged_adamw_32bit", "paged_adamw_8bit"]
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"optimizer_type must be one of {valid_optimizers}, "
                f"got {self.optimizer_type}"
            )

        # Scheduler type validation
        valid_schedulers = ["cosine", "linear", "constant"]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"scheduler_type must be one of {valid_schedulers}, "
                f"got {self.scheduler_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SFTConfig":
        """Load config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            SFTConfig instance

        Raises:
            ValueError: If config is invalid
        """
        config = cls(**config_dict)
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
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from YAML file.

        Args:
            path: File path to load from

        Returns:
            SFTConfig instance

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
            f"SFTConfig("
            f"base_model={self.base_model}, "
            f"dataset_format={self.dataset_format}, "
            f"use_lora={self.use_lora}, "
            f"lora_r={self.lora_r})"
        )
