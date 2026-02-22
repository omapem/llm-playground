"""SFT training orchestrator using TRL's SFTTrainer.

Coordinates model loading, dataset preparation, and training execution
for supervised fine-tuning with LoRA/QLoRA support.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer as TRLSFTTrainer
from datasets import Dataset

from .config import SFTConfig
from .templates import TemplateRegistry, PromptTemplate
from .dataset import SFTDatasetProcessor
from .lora_config import LoRAConfigManager
from .callbacks import SFTCallback, ValidationCallback

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Main SFT training orchestrator.

    Coordinates model loading, dataset preparation, and training execution
    using TRL's SFTTrainer internally with custom callbacks and configuration.

    Args:
        config: SFT configuration
        callbacks: Optional list of training callbacks

    Example:
        >>> config = SFTConfig.from_yaml('config/sft_alpaca.yaml')
        >>> callbacks = [ValidationCallback(), WandBCallback(), CheckpointCallback()]
        >>> trainer = SFTTrainer(config, callbacks=callbacks)
        >>> trainer.train()
    """

    def __init__(self, config: SFTConfig, callbacks: Optional[List[SFTCallback]] = None):
        """Initialize SFT trainer.

        Args:
            config: SFT configuration
            callbacks: Optional list of training callbacks
        """
        self.config = config
        self.callbacks = callbacks or []
        self.dataset_processor = SFTDatasetProcessor()
        self.template = None  # Loaded lazily

    def train(self) -> Dict[str, Any]:
        """Execute complete training workflow.

        Steps:
        1. Load model and tokenizer with LoRA/QLoRA config
        2. Prepare datasets with template formatting
        3. Create TRL SFTTrainer instance
        4. Run training with callbacks
        5. Save adapter weights and optionally merged model

        Returns:
            Dict with training metrics and output paths

        Example:
            >>> trainer = SFTTrainer(config)
            >>> result = trainer.train()
            >>> print(f"Final loss: {result['train_loss']}")
        """
        logger.info(f"Starting SFT training with config: {self.config}")

        # 1. Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = self._load_model()

        # 2. Prepare datasets
        logger.info("Preparing datasets...")
        self.template = self._load_template()
        train_dataset, eval_dataset = self._prepare_datasets()
        logger.info(
            f"Train size: {len(train_dataset)}, "
            f"Eval size: {len(eval_dataset) if eval_dataset else 0}"
        )

        # 3. Auto-add ValidationCallback if eval_dataset is available
        #    and no ValidationCallback is already in the callbacks list
        has_validation_cb = any(
            isinstance(cb, ValidationCallback) for cb in self.callbacks
        )
        if eval_dataset is not None and not has_validation_cb:
            eval_steps = self.config.eval_steps or 500
            logger.info(
                f"Auto-adding ValidationCallback (eval_steps={eval_steps})"
            )
            self.callbacks.append(
                ValidationCallback(
                    val_dataset=eval_dataset,
                    eval_steps=eval_steps,
                )
            )

        # 4. Create training arguments
        training_args = self._create_training_args()

        # 5. Create TRL trainer
        logger.info("Creating TRL SFTTrainer...")
        trl_trainer = self._create_trl_trainer(
            model, tokenizer, train_dataset, eval_dataset, training_args
        )

        # 6. Add custom callbacks
        for callback in self.callbacks:
            trl_trainer.add_callback(callback)

        # 7. Train
        logger.info("Starting training...")
        train_result = trl_trainer.train()

        # 8. Save adapter
        logger.info(f"Saving adapter to {self.config.output_dir}")
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        # 9. Optionally merge and save full model
        if hasattr(self.config, 'save_merged_model') and self.config.save_merged_model:
            merged_dir = Path(self.config.output_dir) / "merged"
            logger.info(f"Saving merged model to {merged_dir}")
            self._save_merged_model(model, str(merged_dir))

        logger.info("Training complete!")

        return {
            "train_loss": train_result.training_loss,
            "metrics": train_result.metrics,
            "output_dir": self.config.output_dir,
        }

    def _load_template(self) -> PromptTemplate:
        """Load template from registry based on config.

        Uses config.template_name if specified, otherwise auto-detects
        from config.dataset_format (alpaca → alpaca, chat → chatml).

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If template name not found in registry
            ValueError: If dataset format is unknown
        """
        registry = TemplateRegistry()

        if hasattr(self.config, 'template_name') and self.config.template_name:
            return registry.get(self.config.template_name)
        else:
            # Auto-detect from dataset format
            if self.config.dataset_format == "alpaca":
                return registry.get("alpaca")
            elif self.config.dataset_format == "chat":
                return registry.get("chatml")
            else:
                raise ValueError(
                    f"Unknown dataset format: {self.config.dataset_format}. "
                    "Expected 'alpaca' or 'chat'."
                )

    def _load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load base model and tokenizer with LoRA/QLoRA config.

        Steps:
        1. Load tokenizer
        2. Create LoRA config from SFTConfig
        3. Optionally create QLoRA config (4-bit quantization)
        4. Load base model with quantization config if QLoRA
        5. Apply PEFT/LoRA to model if enabled

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.config.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        # Create quantization config if QLoRA
        quantization_config = None
        if self.config.use_qlora:
            logger.info("Creating QLoRA quantization config")
            quantization_config = LoRAConfigManager.create_qlora_config(self.config)

        # Determine torch dtype
        torch_dtype = torch.float32
        if self.config.mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            torch_dtype = torch.float16

        # Determine device map
        device_map = getattr(self.config, 'device_map', 'auto')

        # Load base model
        logger.info(f"Loading base model from {self.config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info("Applying LoRA configuration")
            lora_config = LoRAConfigManager.create_lora_config(self.config)
            model = get_peft_model(model, lora_config)
            logger.info(f"LoRA applied: r={self.config.lora_r}, alpha={self.config.lora_alpha}")

        return model, tokenizer

    def _prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and format datasets.

        Uses SFTDatasetProcessor to load from HuggingFace Hub or local files,
        applies template formatting, and splits into train/validation if configured.

        Returns:
            Tuple of (train_dataset, eval_dataset). eval_dataset is None if no split.
        """
        # Ensure template is loaded
        if self.template is None:
            self.template = self._load_template()

        # Load and format with template
        result = self.dataset_processor.load_and_format(
            dataset_name=self.config.dataset_name,
            template=self.template,
            split=self.config.dataset_split,
            validation_split=self.config.validation_split,
        )

        # Handle single dataset or train/val split
        if isinstance(result, tuple):
            return result
        else:
            return result, None

    def _create_training_args(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from SFTConfig.

        Maps all relevant training hyperparameters, hardware settings,
        and logging configuration to TrainingArguments.

        Returns:
            TrainingArguments instance
        """
        # Determine report_to
        report_to = []
        if hasattr(self.config, 'wandb_project') and self.config.wandb_project:
            report_to.append("wandb")

        # Determine run name
        run_name = None
        if hasattr(self.config, 'wandb_run_name') and self.config.wandb_run_name:
            run_name = self.config.wandb_run_name
        elif hasattr(self.config, 'run_name') and self.config.run_name:
            run_name = self.config.run_name

        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=getattr(self.config, 'save_total_limit', 3),
            report_to=report_to if report_to else ["none"],
            run_name=run_name,
            fp16=(self.config.mixed_precision == "fp16"),
            bf16=(self.config.mixed_precision == "bf16"),
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optimizer_type,
            max_steps=self.config.max_steps if self.config.max_steps else -1,
        )

    def _create_trl_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: TrainingArguments,
    ) -> TRLSFTTrainer:
        """Create TRL SFTTrainer instance.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Optional validation dataset
            training_args: Training arguments

        Returns:
            TRLSFTTrainer instance
        """
        return TRLSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",  # Our formatted column name
        )

    def _save_merged_model(self, model: PreTrainedModel, output_dir: str) -> None:
        """Merge LoRA weights with base model and save.

        Args:
            model: PEFT model with LoRA weights
            output_dir: Directory to save merged model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if model has PEFT merge method
        if hasattr(model, 'merge_and_unload'):
            logger.info("Merging LoRA weights with base model")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(output_path))
            logger.info(f"Merged model saved to {output_path}")
        else:
            logger.warning(
                "Model does not have merge_and_unload method. "
                "Saving model as-is (may not be merged)."
            )
            model.save_pretrained(str(output_path))
