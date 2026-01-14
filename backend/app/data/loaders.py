"""Dataset loading and collection functionality."""

from typing import Optional, Dict, Any, List
from loguru import logger
import datasets


class DatasetLoader:
    """Load datasets from HuggingFace Hub or local sources."""

    SUPPORTED_DATASETS = {
        "wikitext": {
            "configs": ["wikitext-103-v1", "wikitext-103", "wikitext-2-v1", "wikitext-2"],
            "text_column": "text",
            "description": "WikiText language modeling dataset (Wikipedia articles)",
        },
        "openwebtext": {
            "configs": ["default"],
            "text_column": "text",
            "description": "OpenWebText dataset (web text corpus)",
        },
        "c4": {
            "configs": ["en", "multilingual"],
            "text_column": "text",
            "description": "C4 dataset (Colossal Clean Crawled Corpus)",
        },
    }

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: Optional[str] = None,
        max_examples: Optional[int] = None,
        streaming: bool = False,
    ):
        """Initialize DatasetLoader.

        Args:
            dataset_name: Name of dataset (wikitext, openwebtext, c4)
            dataset_config: Configuration/variant of the dataset
            split: Dataset split to load (train, validation, test)
            max_examples: Maximum number of examples to load
            streaming: Whether to stream the dataset (for large datasets)
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Choose from: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_examples = max_examples
        self.streaming = streaming

    def get_supported_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported datasets."""
        return self.SUPPORTED_DATASETS

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{self.dataset_name}' not found")

        info = self.SUPPORTED_DATASETS[self.dataset_name].copy()
        info["selected_config"] = self.dataset_config
        return info

    def get_text_column(self) -> str:
        """Get the name of the text column for this dataset."""
        return self.SUPPORTED_DATASETS[self.dataset_name]["text_column"]

    def load(self) -> Any:
        """Load the dataset from HuggingFace Hub.

        Returns:
            Loaded dataset object from datasets library

        Raises:
            ValueError: If dataset configuration is invalid
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info(
                f"Loading dataset: {self.dataset_name} ({self.dataset_config}) "
                f"split={self.split}, streaming={self.streaming}"
            )

            # Load from HuggingFace Hub
            dataset = datasets.load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )

            # Apply max_examples limit if specified
            if self.max_examples and not self.streaming:
                if isinstance(dataset, datasets.DatasetDict):
                    dataset = {k: v.select(range(min(len(v), self.max_examples)))
                              for k, v in dataset.items()}
                else:
                    dataset = dataset.select(range(min(len(dataset), self.max_examples)))

            logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        dataset = self.load()

        stats = {
            "dataset": self.dataset_name,
            "config": self.dataset_config,
            "split": self.split,
        }

        if isinstance(dataset, datasets.DatasetDict):
            stats["splits"] = list(dataset.keys())
            stats["total_examples"] = sum(len(d) for d in dataset.values())
            stats["splits_count"] = {k: len(v) for k, v in dataset.items()}
        else:
            stats["total_examples"] = len(dataset)

        return stats


def load_dataset(
    dataset_name: str,
    dataset_config: str,
    split: Optional[str] = None,
    max_examples: Optional[int] = None,
    streaming: bool = False,
) -> Any:
    """Helper function to load a dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration of the dataset
        split: Which split to load
        max_examples: Maximum examples to load
        streaming: Whether to stream the dataset

    Returns:
        Loaded dataset
    """
    loader = DatasetLoader(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_examples=max_examples,
        streaming=streaming,
    )
    return loader.load()
