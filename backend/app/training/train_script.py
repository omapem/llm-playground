"""Standalone training script for distributed training.

Launch with:
    torchrun --nproc_per_node=NUM_GPUS -m app.training.train_script --config path/to/config.yaml

Or for single-GPU:
    python -m app.training.train_script --config path/to/config.yaml
"""

import argparse
import logging

from .config import TrainingConfig
from .distributed import setup_distributed, cleanup_distributed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Entry point for standalone training."""
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_yaml(args.config)

    # Set up distributed training (returns None if not launched with torchrun)
    dist_config = setup_distributed()

    if dist_config is not None:
        logger.info(
            f"Running distributed training: rank {dist_config.rank}/{dist_config.world_size}"
        )
    else:
        logger.info("Running single-process training")

    try:
        # Import here to avoid circular imports
        from .trainer import Trainer

        # Create dummy dataset for now (real usage would load actual data)
        # Users should modify this or pass dataset through config
        import torch
        from torch.utils.data import TensorDataset

        # Placeholder: create random data matching config
        seq_len = config.model_config.max_position_embeddings
        num_samples = max(config.batch_size * 10, 100)
        data = torch.randint(0, config.model_config.vocab_size, (num_samples, seq_len))
        train_dataset = TensorDataset(data)

        # Create trainer
        trainer = Trainer(
            config=config,
            train_dataset=train_dataset,
            dist_config=dist_config,
        )

        # Train
        trainer.train()

    finally:
        if dist_config is not None:
            cleanup_distributed()


if __name__ == "__main__":
    main()
