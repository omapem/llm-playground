"""Dataset loading and processing for SFT."""

import logging
from pathlib import Path
from typing import Optional, Union
from datasets import Dataset, load_dataset

from .templates import PromptTemplate

logger = logging.getLogger(__name__)


class SFTDatasetProcessor:
    """Processor for loading and formatting SFT datasets.

    Handles dataset loading from HuggingFace Hub or local files,
    applies prompt templates, and prepares data for TRL SFTTrainer.

    Example:
        >>> processor = SFTDatasetProcessor()
        >>> # Load from HuggingFace Hub
        >>> dataset = processor.load_dataset("tatsu-lab/alpaca")
        >>>
        >>> # Load with validation split
        >>> train, val = processor.load_dataset("tatsu-lab/alpaca", validation_split=0.05)
        >>>
        >>> # Apply template formatting
        >>> from app.sft.templates import AlpacaTemplate
        >>> template = AlpacaTemplate()
        >>> formatted = processor.format_dataset(dataset, template)
        >>>
        >>> # Or combine loading and formatting
        >>> train, val = processor.load_and_format(
        ...     "tatsu-lab/alpaca",
        ...     template,
        ...     validation_split=0.05
        ... )
    """

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        validation_split: Optional[float] = None,
    ) -> Union[Dataset, tuple[Dataset, Dataset]]:
        """Load dataset from HuggingFace Hub or local path.

        Args:
            dataset_name: HF dataset name or local file path
            split: Dataset split to load (default: "train")
            validation_split: If provided, split dataset for validation (0.0 to 1.0)

        Returns:
            Dataset if validation_split is None, otherwise (train_dataset, val_dataset)

        Raises:
            ValueError: If file format is unsupported or validation_split is invalid

        Example:
            >>> processor = SFTDatasetProcessor()
            >>> # Load entire dataset
            >>> dataset = processor.load_dataset("tatsu-lab/alpaca")
            >>>
            >>> # Load with validation split
            >>> train, val = processor.load_dataset("tatsu-lab/alpaca", validation_split=0.05)
            >>>
            >>> # Load from local file
            >>> dataset = processor.load_dataset("/path/to/data.json")
        """
        # Validate validation_split if provided
        if validation_split is not None:
            if not (0 < validation_split < 1):
                raise ValueError(
                    f"validation_split must be in (0, 1), got {validation_split}"
                )

        # Check if local file
        path = Path(dataset_name)
        if path.exists():
            logger.info(f"Loading dataset from local path: {dataset_name}")
            if path.suffix == ".json":
                dataset = Dataset.from_json(str(path))
            elif path.suffix == ".jsonl":
                dataset = Dataset.from_json(str(path))
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
            dataset = load_dataset(dataset_name, split=split)

        # Handle validation split
        if validation_split is not None:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            logger.info(
                f"Split dataset into train ({len(split_dataset['train'])}) "
                f"and validation ({len(split_dataset['test'])}) sets"
            )
            return split_dataset["train"], split_dataset["test"]

        return dataset

    def format_dataset(
        self,
        dataset: Dataset,
        template: PromptTemplate,
        text_column: str = "text",
    ) -> Dataset:
        """Apply prompt template to format dataset examples.

        Args:
            dataset: Input dataset
            template: PromptTemplate to apply
            text_column: Name of output column (default: "text")

        Returns:
            Formatted dataset with text column (original columns removed)

        Example:
            >>> from app.sft.templates import AlpacaTemplate
            >>> template = AlpacaTemplate()
            >>> formatted = processor.format_dataset(dataset, template)
            >>>
            >>> # Custom output column name
            >>> formatted = processor.format_dataset(dataset, template, text_column="prompt")
        """

        def format_example(example):
            """Apply template to single example."""
            return {text_column: template.format(example)}

        logger.info(
            f"Formatting {len(dataset)} examples with template "
            f"{template.__class__.__name__}"
        )
        formatted = dataset.map(format_example, remove_columns=dataset.column_names)

        return formatted

    def load_and_format(
        self,
        dataset_name: str,
        template: PromptTemplate,
        split: str = "train",
        validation_split: Optional[float] = None,
        text_column: str = "text",
    ) -> Union[Dataset, tuple[Dataset, Dataset]]:
        """Load dataset and apply template formatting.

        Convenience method combining load_dataset() and format_dataset().

        Args:
            dataset_name: HF dataset name or local file path
            template: PromptTemplate to apply
            split: Dataset split to load
            validation_split: If provided, split dataset for validation
            text_column: Name of output column

        Returns:
            Formatted dataset if validation_split is None,
            otherwise (train_dataset, val_dataset) tuple

        Example:
            >>> from app.sft.templates import AlpacaTemplate
            >>> template = AlpacaTemplate()
            >>>
            >>> # Load and format in one step
            >>> train, val = processor.load_and_format(
            ...     "tatsu-lab/alpaca",
            ...     template,
            ...     validation_split=0.05
            ... )
        """
        # Load dataset
        result = self.load_dataset(dataset_name, split, validation_split)

        # Format based on whether we have train/val split
        if isinstance(result, tuple):
            train_dataset, val_dataset = result
            train_formatted = self.format_dataset(train_dataset, template, text_column)
            val_formatted = self.format_dataset(val_dataset, template, text_column)
            return train_formatted, val_formatted
        else:
            return self.format_dataset(result, template, text_column)
