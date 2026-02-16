#!/usr/bin/env python3
"""Example script for SFT training using the SFTTrainer.

This script demonstrates:
1. Loading a configuration from YAML
2. Creating an SFT trainer with callbacks
3. Running training
4. Inspecting results

Usage:
    python examples/train_sft.py [--config CONFIG_PATH]

Example:
    python examples/train_sft.py --config config/examples/sft_alpaca.yaml
"""

import argparse
import logging
from pathlib import Path

from app.sft import (
    SFTConfig,
    SFTTrainer,
    ValidationCallback,
    WandBCallback,
    CheckpointCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/examples/sft_alpaca.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = SFTConfig.from_yaml(args.config)

    # Display key settings
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Base Model: {config.base_model}")
    logger.info(f"  Dataset: {config.dataset_name} ({config.dataset_format})")
    logger.info(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info(f"  QLoRA: {config.use_qlora}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective Batch Size: {config.get_effective_batch_size()}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Output Dir: {config.output_dir}")
    logger.info("=" * 60)

    # Create callbacks
    callbacks = []

    # Validation callback (if validation split configured)
    if config.validation_split > 0:
        validation_callback = ValidationCallback(
            eval_steps=config.eval_steps or 100,
            eval_on_epoch_end=True,
        )
        callbacks.append(validation_callback)
        logger.info("✓ Added ValidationCallback")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        output_dir=config.checkpoint_dir or f"{config.output_dir}/checkpoints",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_best_only=False,
    )
    callbacks.append(checkpoint_callback)
    logger.info("✓ Added CheckpointCallback")

    # Weights & Biases callback (if configured and not disabled)
    if config.wandb_project and not args.no_wandb:
        wandb_callback = WandBCallback(
            project=config.wandb_project,
            run_name=config.wandb_run_name or config.run_name,
            config=config.to_dict(),
            log_model=False,
        )
        callbacks.append(wandb_callback)
        logger.info("✓ Added WandBCallback")

    # Create trainer
    logger.info("Creating SFTTrainer...")
    trainer = SFTTrainer(config, callbacks=callbacks)

    # Start training
    logger.info("Starting training...")
    try:
        result = trainer.train()

        # Display results
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Final Loss: {result.get('train_loss', 'N/A')}")
        logger.info(f"  Output Directory: {result['output_dir']}")
        logger.info("=" * 60)

        # Show saved files
        output_path = Path(result["output_dir"])
        if output_path.exists():
            logger.info("\nSaved files:")
            for file in sorted(output_path.rglob("*")):
                if file.is_file():
                    logger.info(f"  {file.relative_to(output_path)}")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
