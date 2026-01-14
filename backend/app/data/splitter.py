"""Train/validation/test data splitting functionality."""

import random
from typing import List, Tuple, Optional, Any
from loguru import logger


class TrainValSplitter:
    """Split data into train/validation/test sets."""

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: Optional[int] = None,
        stratify: bool = False,
    ):
        """Initialize TrainValSplitter.

        Args:
            train_ratio: Proportion of data for training (0-1)
            val_ratio: Proportion of data for validation (0-1)
            test_ratio: Proportion of data for testing (0-1)
            seed: Random seed for reproducibility
            stratify: Whether to perform stratified split (preserve class distribution)

        Raises:
            ValueError: If ratios are invalid or don't sum to valid amount
        """
        if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Ratios must be non-negative")

        total = train_ratio + val_ratio + test_ratio
        if total > 1.0 + 1e-6:  # Allow small floating point error
            raise ValueError(
                f"Ratios sum to {total}, must be <= 1.0 "
                f"({train_ratio} + {val_ratio} + {test_ratio})"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify = stratify

        if seed is not None:
            random.seed(seed)

    def split(
        self, items: List[Any]
    ) -> Tuple[List[Any], List[Any]] | Tuple[List[Any], List[Any], List[Any]]:
        """Split items into train/validation (and test) sets.

        Args:
            items: List of items to split

        Returns:
            Tuple of (train, val) or (train, val, test) if test_ratio > 0
        """
        if len(items) == 0:
            if self.test_ratio > 0:
                return [], [], []
            return [], []

        # Create indices and shuffle
        indices = list(range(len(items)))
        random.shuffle(indices)

        # Calculate split points (ensure at least 1 item per set if possible)
        train_size = max(1, int(len(items) * self.train_ratio)) if len(items) > 1 else int(len(items) * self.train_ratio)
        val_size = max(0, len(items) - train_size) if self.test_ratio == 0 else int(len(items) * self.val_ratio)

        # Split indices
        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]

        # Convert indices to actual items
        train = [items[i] for i in train_idx]
        val = [items[i] for i in val_idx]

        if self.test_ratio > 0:
            test = [items[i] for i in test_idx]
            logger.info(
                f"Split {len(items)} items into "
                f"train({len(train)}), val({len(val)}), test({len(test)})"
            )
            return train, val, test
        else:
            logger.info(
                f"Split {len(items)} items into "
                f"train({len(train)}), val({len(val)})"
            )
            return train, val

    def get_split_indices(
        self, total_size: int
    ) -> Tuple[List[int], List[int]] | Tuple[List[int], List[int], List[int]]:
        """Get indices for splits without needing actual items.

        Args:
            total_size: Total number of items

        Returns:
            Tuple of (train_indices, val_indices) or (train_indices, val_indices, test_indices)
        """
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)

        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]

        if self.test_ratio > 0:
            return train_idx, val_idx, test_idx
        else:
            return train_idx, val_idx

    def split_dataset(self, dataset: Any) -> dict:
        """Split a HuggingFace dataset object.

        Args:
            dataset: HuggingFace dataset object

        Returns:
            Dictionary with splits
        """
        if hasattr(dataset, 'train_test_split'):
            # Use HF's native train_test_split for efficiency
            test_size = self.test_ratio if self.test_ratio > 0 else self.val_ratio

            split_result = dataset.train_test_split(
                test_size=test_size,
                seed=self.seed,
                stratified_by_column=None  # Can be set if dataset has labels
            )

            result = {
                'train': split_result['train'],
                'val': split_result['test'],
            }

            if self.test_ratio > 0:
                # Further split train set for validation
                val_split = result['train'].train_test_split(
                    test_size=self.val_ratio / (1 - self.test_ratio),
                    seed=self.seed
                )
                result['train'] = val_split['train']
                result['val'] = val_split['test']

            return result
        else:
            raise TypeError("Dataset does not have train_test_split method")

    def validate_split(
        self, train: List[Any], val: List[Any], test: Optional[List[Any]] = None
    ) -> dict:
        """Validate that split is correct.

        Args:
            train: Training set
            val: Validation set
            test: Test set (optional)

        Returns:
            Dictionary with validation results
        """
        total = len(train) + len(val) + (len(test) if test else 0)

        results = {
            "valid": True,
            "total_items": total,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test) if test else 0,
            "train_ratio": len(train) / total if total > 0 else 0,
            "val_ratio": len(val) / total if total > 0 else 0,
            "test_ratio": (len(test) / total if test else 0) if total > 0 else 0,
        }

        # Check ratios match expected
        train_tol = abs(results["train_ratio"] - self.train_ratio) < 0.05
        val_tol = abs(results["val_ratio"] - self.val_ratio) < 0.05

        if test:
            test_tol = abs(results["test_ratio"] - self.test_ratio) < 0.05
            results["valid"] = train_tol and val_tol and test_tol
        else:
            results["valid"] = train_tol and val_tol

        return results
