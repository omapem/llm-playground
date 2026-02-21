"""Dataset loading and collection functionality."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
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


class CustomDatasetLoader:
    """Load and manage custom user-uploaded datasets.

    Supports JSON (array of objects) and JSONL (one object per line) formats.
    Each record must have either a "text" field or instruction/input/output fields.

    Args:
        upload_dir: Directory to store uploaded datasets.
    """

    # Valid record formats
    TEXT_FIELDS = {"text"}
    INSTRUCTION_FIELDS = {"instruction", "output"}
    # UUID4 pattern for dataset_id validation (path traversal prevention)
    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def __init__(self, upload_dir: str = "./data/uploaded_datasets") -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load records from a JSON or JSONL file.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of record dictionaries.

        Raises:
            ValueError: If the file contains invalid JSON.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("Invalid JSON: file is empty")

        # Try JSON array first
        if content.startswith("["):
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("Invalid JSON: expected an array of objects")
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e

        # Try JSONL (one object per line)
        records: List[Dict[str, Any]] = []
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Invalid JSON: line {i} is not a JSON object"
                    )
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e

        if not records:
            raise ValueError("Invalid JSON: no records found")
        return records

    def validate_format(self, data: List[Dict[str, Any]]) -> bool:
        """Check whether all records have the required fields.

        Accepts two formats:
        - **text**: each record has a ``"text"`` key.
        - **instruction**: each record has ``"instruction"`` and ``"output"`` keys.

        Args:
            data: List of record dictionaries.

        Returns:
            True if valid, False otherwise.
        """
        if not data:
            return False

        first_keys = set(data[0].keys())

        # Determine format from first record
        if self.TEXT_FIELDS.issubset(first_keys):
            required = self.TEXT_FIELDS
        elif self.INSTRUCTION_FIELDS.issubset(first_keys):
            required = self.INSTRUCTION_FIELDS
        else:
            return False

        # Verify all records match
        return all(required.issubset(set(r.keys())) for r in data)

    def detect_format(self, data: List[Dict[str, Any]]) -> str:
        """Detect the format of the dataset.

        Args:
            data: List of record dictionaries.

        Returns:
            ``"text"`` or ``"instruction"``.

        Raises:
            ValueError: If format cannot be determined.
        """
        if not data:
            raise ValueError("Cannot detect format of empty dataset")

        first_keys = set(data[0].keys())
        if self.TEXT_FIELDS.issubset(first_keys):
            return "text"
        if self.INSTRUCTION_FIELDS.issubset(first_keys):
            return "instruction"
        raise ValueError("Unrecognized dataset format")

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Clean text by stripping and normalizing whitespace.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text with normalized whitespace.
        """
        # Replace tabs, newlines, and multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def sanitize_records(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize all text fields in the dataset records.

        Applies whitespace normalization to every string field that is part
        of the record format (text, instruction, input, output).

        Args:
            data: List of record dictionaries.

        Returns:
            Sanitized copy of the data.
        """
        text_keys = {"text", "instruction", "input", "output"}
        sanitized = []
        for record in data:
            clean = dict(record)
            for key in text_keys:
                if key in clean and isinstance(clean[key], str):
                    clean[key] = self.sanitize_text(clean[key])
            sanitized.append(clean)
        return sanitized

    def save_dataset(
        self,
        data: List[Dict[str, Any]],
        filename: str = "dataset.json",
    ) -> str:
        """Save a dataset to the upload directory.

        Args:
            data: List of record dictionaries.
            filename: Original filename for metadata.

        Returns:
            Unique dataset ID.
        """
        dataset_id = str(uuid.uuid4())
        dataset_dir = self.upload_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = dataset_dir / "data.json"
        data_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # Save metadata (sanitize filename to prevent path traversal in metadata)
        import os
        safe_filename = os.path.basename(filename or "upload.json")[:255]
        fmt = self.detect_format(data)
        metadata = {
            "dataset_id": dataset_id,
            "filename": safe_filename,
            "record_count": len(data),
            "format": fmt,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = dataset_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        logger.info(f"Saved dataset {dataset_id}: {len(data)} records ({fmt} format)")
        return dataset_id

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all uploaded datasets.

        Returns:
            List of metadata dictionaries.
        """
        result: List[Dict[str, Any]] = []
        if not self.upload_dir.exists():
            return result

        for meta_path in sorted(self.upload_dir.glob("*/metadata.json")):
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                result.append(metadata)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping corrupt metadata: {meta_path}: {e}")
        return result

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific uploaded dataset.

        Args:
            dataset_id: The dataset identifier (must be a valid UUID4).

        Returns:
            Metadata dictionary or None if not found or invalid ID.
        """
        # Validate UUID format to prevent path traversal
        if not self._UUID_RE.match(dataset_id):
            return None

        meta_path = (self.upload_dir / dataset_id / "metadata.json").resolve()
        # Double-check: resolved path must stay within upload_dir
        if not meta_path.is_relative_to(self.upload_dir.resolve()):
            return None

        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
